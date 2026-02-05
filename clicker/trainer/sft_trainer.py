# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from accelerate import PartialState, logging
from datasets import Dataset, IterableDataset
from transformers import (
    AutoProcessor,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ..data_utils import (
    VALID_TRUNCATION_STRATEGIES,
    add_system_message_to_example,
    apply_chat_template,
    apply_truncation_strategy_to_example,
    convert_binary_preference_to_sft,
    convert_preference_to_sft,
    expand_split_chunks,
    fix_example_turn_order,
    is_binary_preference_dataset,
    is_conversational,
    is_conversational_from_value,
    is_preference_dataset,
    maybe_convert_to_chatml,
    pack_dataset,
    prepare_multimodal_messages,
    truncate_conversation_by_turns,
    truncate_dataset,
)
from ..models import clone_chat_template, get_act_offloading_ctx_manager, prepare_peft_model
from .base_trainer import BaseTrainer
from .sft_config import SFTConfig
from .utils import (
    create_model_from_path,
    entropy_from_logits,
    flush_left,
    pad,
    remove_none_values,
    selective_log_softmax,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, PeftType


logger = logging.get_logger(__name__)


def mask_to_last_segment_only(mask: list[int]) -> list[int]:
    """
    Transform an assistant mask to only keep the last contiguous segment of 1s.

    This is used for `last_assistant_only_loss` where we only want to train
    on the final assistant response in a multi-turn conversation.

    Args:
        mask: List of 0s and 1s where 1 indicates assistant tokens.

    Returns:
        Modified mask with only the last contiguous segment of 1s preserved.

    Example:
        >>> mask_to_last_segment_only([0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    """
    if not mask or 1 not in mask:
        return mask

    result = [0] * len(mask)

    # Find the last segment of 1s by scanning backwards
    last_one_idx = None
    for i in range(len(mask) - 1, -1, -1):
        if mask[i] == 1:
            last_one_idx = i
            break

    if last_one_idx is None:
        return result

    # Find the start of this segment
    start_idx = last_one_idx
    while start_idx > 0 and mask[start_idx - 1] == 1:
        start_idx -= 1

    # Set only this segment to 1
    for i in range(start_idx, last_one_idx + 1):
        result[i] = 1

    return result


def remove_trailing_eos(input_ids: list[int], eos_token_id: int) -> list[int]:
    """
    Remove trailing EOS token(s) from input_ids.

    This is used for `train_on_incomplete_assistant` where we don't want
    the model to learn to generate EOS on truncated data.

    Args:
        input_ids: List of token IDs.
        eos_token_id: The EOS token ID to remove.

    Returns:
        input_ids with trailing EOS token(s) removed.
    """
    while input_ids and input_ids[-1] == eos_token_id:
        input_ids = input_ids[:-1]
    return input_ids


FLASH_ATTENTION_VARIANTS = {
    "flash_attention_2",
    "flash_attention_3",
    "kernels-community/flash-attn",
    "kernels-community/vllm-flash-attn3",
    "kernels-community/flash-attn3",
}


def get_dataset_column_names(dataset: Union[Dataset, IterableDataset]) -> list[str]:
    return list(next(iter(dataset)).keys()) if dataset.column_names is None else dataset.column_names


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch.

    This collator expects each example in the input list to be a dictionary containing at least the `"input_ids"` key.
    If the input contains a `"completion_mask"`, it is used to set the labels to `-100` for tokens that are not in the
    completion. If `"assistant_masks"` are present, they are used to set the labels to `-100` for tokens that are not
    in the assistant part of the sequence. The collator returns a dictionary containing the following keys:
    - `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch.
    - `"labels"`: Tensor of labels, padded to the maximum length of the batch. If `completion_only_loss` is set to
    `True`, tokens that are not in the completion are set to -100. If `assistant_masks` are present, tokens that are
    not in the assistant part of the sequence are set to -100. If `padding_free` is set to `False`, the following key
    is also returned:
    - `"attention_mask"`: Tensor of attention masks, padded to the maximum length of the batch.
    If `padding_free` is set to `True`, the following key is also returned:
    - `"position_ids"`: Tensor of position IDs, padded to the maximum length of the batch.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        completion_only_loss (`bool`, *optional*, defaults to `True`):
            When the input contains a completion mask (`completion_mask`), the labels are set to -100 for the tokens
            that are no in the completion.
        padding_free (`bool`, *optional*, defaults to `False`):
            If set to `True`, the sequences will be flattened into a single sequence, and the position IDs will be
            generated accordingly and returned instead of the attention mask.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
    >>> examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'labels': tensor([[   1,    2,    3],
                       [   4,    5, -100]])}

    >>> # With completion mask
    >>> examples = [
    ...     {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
    ...     {"input_ids": [4, 5], "completion_mask": [0, 1]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'labels': tensor([[-100,    2,    3],
                       [-100,    5, -100]])}

    >>> # With padding_free
    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
    >>> collator(examples)
    {'input_ids': tensor([[ 1, 2, 3, 4, 5]]),
     'position_ids': tensor([[0, 1, 2, 0, 1]]),
     'labels': tensor([[1, 2, 3, 4, 5]])}
    ```
    """

    pad_token_id: int
    completion_only_loss: bool = True
    padding_free: bool = False
    pad_to_multiple_of: Optional[int] = None
    pass_through_assistant_masks: bool = False
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        if "labels" in examples[0]:
            labels = [torch.tensor(example["labels"]) for example in examples]
        else:
            labels = [torch.tensor(example["input_ids"]) for example in examples]

        # For padding-free, we should NOT create attention_mask as it causes FlashAttention to ignore position_ids and
        # compute wrong cu_seq_lens from the all-1s mask
        if self.padding_free:
            if "seq_lengths" in examples[0]:
                position_ids = self.get_position_ids_from_packed_seq_lengths(
                    [example["seq_lengths"] for example in examples]
                )
            else:
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
        else:
            attention_mask = [torch.ones_like(ids) for ids in input_ids]
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = [torch.tensor(example["completion_mask"]) for example in examples]
        if "assistant_masks" in examples[0]:
            assistant_masks = [torch.tensor(example["assistant_masks"]) for example in examples]

        # If padding_free, flatten everything into a single sequence
        output = {}
        if self.padding_free:
            input_ids = [torch.cat(input_ids, dim=0)]
            labels = [torch.cat(labels, dim=0)]
            position_ids = [torch.cat(position_ids, dim=0)]
            if self.completion_only_loss and "completion_mask" in examples[0]:
                completion_mask = [torch.cat(completion_mask, dim=0)]
            if "assistant_masks" in examples[0]:
                assistant_masks = [torch.cat(assistant_masks, dim=0)]

        # Pad
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["labels"] = pad(
            labels, padding_value=-100, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        )
        if self.padding_free:
            output["position_ids"] = pad(
                position_ids, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][output["position_ids"] == 0] = -100
        else:
            output["attention_mask"] = pad(
                attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = pad(
                completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][completion_mask == 0] = -100  # mask everything that is not in the completion
        if "assistant_masks" in examples[0]:
            assistant_masks = pad(
                assistant_masks, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][assistant_masks == 0] = -100
            if self.pass_through_assistant_masks:
                output["assistant_masks"] = assistant_masks
        return output

    @staticmethod
    def get_position_ids_from_packed_seq_lengths(batch_seq_lengths: list[list[int]]) -> list[torch.Tensor]:
        """
        Get position IDs for packed sequences.

        Args:
            batch_seq_lengths (`list[list[int]]`):
                A list of lists containing the lengths of each individual document in the packed batch.

        Return:
            `list[torch.Tensor]`:
                A list of tensors containing the position IDs for each packed sequence.
        """
        # Get lengths per row
        example_lengths = [sum(seq_lengths) for seq_lengths in batch_seq_lengths]
        # Flat list of lengths
        batch_seq_lengths = torch.tensor(
            [seq_length for seq_lengths in batch_seq_lengths for seq_length in seq_lengths]
        )
        position_ids = torch.ones(sum(example_lengths), dtype=batch_seq_lengths.dtype)
        position_ids[0] = 0
        # Reset position ids to 0 at the start of each sequence
        position_ids[batch_seq_lengths[:-1].cumsum(0)] = -(batch_seq_lengths[:-1] - 1)
        position_ids = position_ids.cumsum(0)
        # Split back into one tensor per example
        return list(position_ids.split(example_lengths))


@dataclass
class DataCollatorForVisionLanguageModeling(DataCollatorMixin):
    """
    Data collator for vision-language modeling tasks.

    Unlike text-only datasets—where the collator typically receives pre-tokenized inputs ready for batching,
    vision-language data processing involves converting images into pixel values. This conversion is disk-intensive,
    making upfront preprocessing of the entire dataset impractical. Therefore, this collator performs tokenization and
    image processing on-the-fly to efficiently prepare batches.

    Each input example should be a dictionary containing at least:
    - An `"images"` key holding the image data.
    - [language modeling](#language-modeling) type: either a `"messages"` key for conversational inputs or a `"text"`
      key for standard text inputs.
    - [prompt-completion](#prompt-completion) type: keys `"prompt"` and `"completion"` for the prompt and completion.

    The collator outputs a dictionary including:
    - `"input_ids"`: Tensor of token IDs.
    - `"attention_mask"`: Tensor indicating attention mask.
    - `"pixel_values"`: Tensor representing image pixel values.
    - `"labels"`: Tensor for training labels.

    Additional keys may be present depending on the processor, such as `"image_grid_thw"`.

    Args:
        processor (`ProcessorMixin`):
            The processor used to tokenize text and process images. It must be a subclass of `ProcessorMixin` and
            include a `tokenizer` with a defined `pad_token_id`.
        max_length (`int` or `None`, optional, defaults to `None`):
            Maximum sequence length for input tokens. If `None`, no truncation is applied.
        completion_only_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute loss only on the completion part of the sequence. When `True`, the labels for the prompt
            part are set to -100. It requires the dataset type to be prompt-completion.
        pad_to_multiple_of (`int` or `None`, optional, defaults to `None`):
            If set, the sequences will be padded to a multiple of this value.
        dataset_text_field (`str`, optional, defaults to `"text"`):
            Name of the column that contains text data in the dataset. This parameter is only relevant for [standard
            datasets format](dataset_formats#standard).
        return_tensors (`str`, optional, defaults to `"pt"`):
            The tensor type to return. Currently, only `"pt"` (PyTorch tensors) is supported.

    Example:
    ```python
    >>> from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
    >>> from transformers import AutoProcessor

    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    >>> collator = DataCollatorForVisionLanguageModeling(processor)
    >>> examples = [
    ...     {"images": [Image.open("image_0.png")], "messages": [{"role": "user", "content": "What is this?"}]},
    ...     {"images": [Image.open("image_1.png")], "messages": [{"role": "user", "content": "Describe this image."}]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                           151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,   3838,    374,
                              419,     30, 151645,    198],
                          [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                           151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,  74785,    419,
                             2168,     13, 151645,    198]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
     'pixel_values': tensor([[-0.9893,  0.1785,  1.5362,  ..., -0.0582,  0.8661, -0.2431],
                             [-0.2302,  0.9522, -1.1061,  ...,  0.0555,  1.3354, -0.6412],
                             [ 1.2150,  0.9084,  0.7041,  ...,  0.2404, -0.8403, -0.5133],
                             ...,
                             [ 0.6895,  0.2807,  0.2515,  ..., -0.2004, -1.2100,  0.0555],
                             [ 0.8209, -0.9748,  1.5654,  ...,  1.6055, -0.4706,  0.5817],
                             [-1.0915,  0.4559,  0.9230,  ...,  0.5106,  0.0982, -0.1720]]),
     'image_grid_thw': tensor([[1, 4, 4],
                               [1, 4, 4]]),
     'labels': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                        151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,   3838,    374,
                           419,     30, 151645,    198],
                        [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                         151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,  74785,    419,
                           2168,     13, 151645,    198]])}
    ```
    """

    processor: ProcessorMixin
    max_length: Optional[int] = None
    completion_only_loss: bool = False  # default not used in practice; SFTTrainer always passes the relevant value
    pad_to_multiple_of: Optional[int] = None
    dataset_text_field: str = "text"
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if "messages" in examples[0] or self.dataset_text_field in examples[0]:
            if self.completion_only_loss:
                raise ValueError(
                    "The `completion_only_loss` argument is not supported for language modeling datasets."
                )
            return self._collate_language_modeling(examples)
        elif "prompt" in examples[0] and "completion" in examples[0]:
            return self._collate_prompt_completion(examples)
        else:
            raise KeyError(f"Unexpected input keys in examples: {list(examples[0].keys())}.")

    def _collate_language_modeling(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = [example["images"] for example in examples]
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if all(img_list == [] for img_list in images):
            images = None

        if "messages" in examples[0]:  # conversational case
            for example in examples:
                prepare_multimodal_messages(example["messages"], len(example["images"]))
            messages = [example["messages"] for example in examples]
            texts = self.processor.apply_chat_template(messages)
        elif self.dataset_text_field in examples[0]:  # standard case
            texts = [example[self.dataset_text_field] for example in examples]
        else:
            raise KeyError(
                "The input examples must contain either 'messages' for conversational data or 'text' for standard "
                "data."
            )

        output = self.processor(
            images=images,
            text=texts,
            padding=True,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )
        labels = output["input_ids"].clone()
        labels[output["attention_mask"] == 0] = -100
        # We mask only padding tokens (-100) in the labels. Vision tokens are left unchanged because their handling in
        # loss computation has to be done by the model, and masking them here would be infeasible in practice as vision
        # token definitions vary across architectures.
        output["labels"] = labels
        return output

    def _collate_prompt_completion(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.pad_to_multiple_of is not None:
            raise NotImplementedError(
                "Padding to a multiple of a value is not yet implemented for vision-language modeling and "
                "prompt-completion data yet."
            )
        images = [example["images"] for example in examples]
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if all(img_list == [] for img_list in images):
            images = None
        if is_conversational(examples[0]):  # conversational case
            for example in examples:
                prepare_multimodal_messages(example["prompt"] + example["completion"], len(example["images"]))
            examples = [apply_chat_template(example, self.processor) for example in examples]

        prompts = [example["prompt"] for example in examples]
        completions = [example["completion"] for example in examples]

        processed_prompts = self.processor(
            images=images,
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )
        processed_completions = self.processor(
            text=completions,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )

        # Concatenate prompts and completions
        prompt_ids, completion_ids = processed_prompts["input_ids"], processed_completions["input_ids"]
        prompt_mask, completion_mask = processed_prompts["attention_mask"], processed_completions["attention_mask"]
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        completion_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)
        if "token_type_ids" in processed_prompts:  # special case for Gemma
            prompt_token_type_ids = processed_prompts["token_type_ids"]
            completion_token_type_ids = processed_completions["token_type_ids"]
            token_type_ids = torch.cat((prompt_token_type_ids, completion_token_type_ids), dim=1)

        # Flush left to reduce padding
        if "token_type_ids" in processed_prompts:
            attention_mask, input_ids, completion_mask, token_type_ids = flush_left(
                attention_mask, input_ids, completion_mask, token_type_ids
            )
        else:
            attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)

        # Truncate if necessary
        if self.max_length is not None:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            completion_mask = completion_mask[:, : self.max_length]
            if "token_type_ids" in processed_prompts:
                token_type_ids = token_type_ids[:, : self.max_length]

        # Create labels and mask padding tokens
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if self.completion_only_loss:
            labels[completion_mask == 0] = -100

        # Build the output dictionary
        output = processed_prompts  # we take processed_prompts because it contains the images
        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        output["labels"] = labels
        if "token_type_ids" in processed_prompts:
            output["token_type_ids"] = token_type_ids
        return output


def dft_loss(outputs, labels, num_items_in_batch=None):
    """
    DFT loss function, as presented in [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward
    Rectification](https://huggingface.co/papers/2508.05629)
    """
    labels = nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = shift_labels != -100
    shift_labels[~loss_mask] = 0
    logprobs = selective_log_softmax(outputs.logits, shift_labels)
    per_token_loss = -logprobs.exp().detach() * logprobs
    if num_items_in_batch is None:
        num_items_in_batch = loss_mask.sum()
    loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
    return loss


def aux_eos_calibration_loss(logits, labels, assistant_masks, eos_token_id):
    """
    EOS calibration auxiliary loss.

    Adds extra cross-entropy on EOS token at positions where the model should stop generating
    (end of assistant turns). This encourages the model to assign higher probability to EOS
    at turn boundaries.

    The target positions are the last token of each contiguous assistant segment in the mask.
    At those positions, we compute cross-entropy loss specifically for the EOS token.

    Args:
        logits: Model logits, shape (batch, seq_len, vocab_size)
        labels: Shifted labels, shape (batch, seq_len) — only used for alignment
        assistant_masks: Binary mask, shape (batch, seq_len), 1 = assistant token
        eos_token_id: The EOS token ID to target

    Returns:
        Scalar loss (mean cross-entropy on EOS at turn-end positions), or 0 if no turn boundaries found.
    """
    # Shift assistant_masks to align with shifted labels (logits[:-1] predicts labels[1:])
    shift_masks = assistant_masks[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Find turn-end positions: where assistant_mask transitions from 1 to 0 (or ends at sequence boundary)
    # A turn ends at position i if mask[i] == 1 and (mask[i+1] == 0 or i is the last position)
    shift_labels = labels[..., 1:].contiguous()
    padded = torch.nn.functional.pad(shift_masks, (0, 1), value=0)  # pad right with 0
    turn_ends = (shift_masks == 1) & (padded[..., 1:] == 0) & (shift_labels != -100)  # shape: (batch, seq_len-1)

    if not turn_ends.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # At turn-end positions, compute cross-entropy targeting EOS
    # We want the model to predict EOS at these positions
    eos_targets = torch.full_like(shift_labels, fill_value=-100)
    eos_targets[turn_ends] = eos_token_id

    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        eos_targets.view(-1),
        ignore_index=-100,
    )
    return loss


def aux_repetition_penalty_loss(logits, labels, window_size=64):
    """
    Repetition penalty auxiliary loss.

    Penalizes the model for assigning high probability to tokens that have already appeared
    within a sliding window of recent positions. This discourages repetitive generation patterns
    at the weight level rather than only at inference time.

    For each position, we look back `window_size` tokens in the ground-truth labels.
    For each previous token found, we gather the model's predicted probability for that token
    at the current position. The sum of these probabilities (normalized by window size) is the
    penalty — the more probability the model puts on recently-seen tokens, the higher the loss.

    Args:
        logits: Model logits, shape (batch, seq_len, vocab_size)
        labels: Shifted labels, shape (batch, seq_len), -100 for masked positions
        window_size: How many previous tokens to consider as "recent"

    Returns:
        Scalar loss (mean probability mass on repeated tokens across trainable positions).
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = shift_labels != -100

    batch_size, seq_len, vocab_size = shift_logits.shape

    if not loss_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Compute softmax once — all gathers are differentiable through this
    probs = torch.softmax(shift_logits, dim=-1)

    clean_labels = shift_labels.clone()
    clean_labels[~loss_mask] = 0

    # For each offset in the window, shift labels and gather the probability
    # the model assigns to that token at the current position.
    # This is O(window_size) gathers, each O(batch * seq_len).
    rep_prob = torch.zeros(batch_size, seq_len, device=logits.device)
    win = min(window_size, seq_len - 1)

    for offset in range(1, win + 1):
        # Token that appeared `offset` positions ago
        prev_tok = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
        prev_valid = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=logits.device)
        prev_tok[:, offset:] = clean_labels[:, :seq_len - offset]
        prev_valid[:, offset:] = loss_mask[:, :seq_len - offset]

        # What probability does the model assign to that previous token at this position?
        gathered = torch.gather(probs, dim=-1, index=prev_tok.unsqueeze(-1)).squeeze(-1)
        rep_prob += gathered * prev_valid.float()

    # Normalize by window size so the magnitude is independent of window choice
    rep_prob = rep_prob / max(win, 1)

    loss = (rep_prob * loss_mask.float()).sum() / loss_mask.sum()
    return loss


def aux_vocabulary_diversity_loss(logits, labels, max_weight_ratio=5.0):
    """
    Vocabulary diversity auxiliary loss.

    Upweights loss on rare tokens and downweights common tokens (within the batch) to encourage
    the model to maintain a diverse vocabulary distribution rather than collapsing to a narrow
    set of high-frequency outputs.

    Token frequencies are computed per-batch from the ground-truth labels. Rare tokens get
    higher weight, common tokens get lower weight. The weighting uses inverse frequency
    (smoothed to avoid extreme weights on hapax legomena).

    Args:
        logits: Model logits, shape (batch, seq_len, vocab_size)
        labels: Shifted labels, shape (batch, seq_len), -100 for masked positions
        max_weight_ratio: Maximum ratio for per-token weights after normalization. Weights are
            clamped to [1/max_weight_ratio, max_weight_ratio] to prevent extremely rare tokens
            from having outsized influence. Set to 0 to disable clamping.

    Returns:
        Scalar loss (frequency-weighted cross-entropy on trainable positions).
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = shift_labels != -100

    if not loss_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Compute token frequencies within the batch (only trainable positions)
    valid_tokens = shift_labels[loss_mask]  # 1D tensor of all trainable token IDs
    num_valid = valid_tokens.numel()

    # Count occurrences of each token in the batch
    vocab_size = shift_logits.size(-1)
    counts = torch.zeros(vocab_size, device=logits.device)
    counts.scatter_add_(0, valid_tokens, torch.ones_like(valid_tokens, dtype=counts.dtype))

    # Inverse frequency weighting with smoothing: weight = log(N / (count + 1)) + 1
    # This gives rare tokens higher weight without extreme outliers
    weights = torch.log(num_valid / (counts + 1.0)) + 1.0

    # Look up the weight for each position's target token
    clean_labels = shift_labels.clone()
    clean_labels[~loss_mask] = 0
    per_token_weights = weights[clean_labels]  # (batch, seq_len)
    per_token_weights[~loss_mask] = 0.0

    # Normalize weights to mean=1 so this loss is on the same scale as regular CE
    weight_sum = per_token_weights[loss_mask].sum()
    weight_mean = weight_sum / loss_mask.sum()
    per_token_weights = per_token_weights / (weight_mean + 1e-8)

    # Clamp weights to prevent extreme values from dominating
    if max_weight_ratio > 0:
        per_token_weights = per_token_weights.clamp(min=1.0 / max_weight_ratio, max=max_weight_ratio)

    # Compute per-token cross-entropy
    per_token_ce = nn.functional.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)

    # Apply diversity weights
    loss = (per_token_ce * per_token_weights * loss_mask.float()).sum() / loss_mask.sum()
    return loss


def aux_confidence_regularization_loss(logits, labels):
    """
    Confidence regularization auxiliary loss.

    Penalizes the model for being too confident (low entropy) on its predictions at trainable
    positions. This acts as targeted entropy regularization — encouraging the model to maintain
    some uncertainty rather than collapsing to point predictions.

    Unlike generic label smoothing, this only applies at positions where we compute loss
    (respecting the existing masking). The loss is the negative entropy of the predicted
    distribution, so maximizing entropy = minimizing this loss.

    Args:
        logits: Model logits, shape (batch, seq_len, vocab_size)
        labels: Shifted labels, shape (batch, seq_len), -100 for masked positions

    Returns:
        Scalar loss (negative mean entropy at trainable positions).
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = shift_labels != -100

    if not loss_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Compute entropy at each position: H = -sum(p * log(p))
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    probs = log_probs.exp()
    per_token_entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)

    # The loss is negative entropy (we want to maximize entropy = minimize negative entropy)
    neg_entropy = -per_token_entropy
    loss = (neg_entropy * loss_mask.float()).sum() / loss_mask.sum()
    return loss


class SFTTrainer(BaseTrainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) method.

    This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    from datasets import load_dataset
    from trl import SFTTrainer

    dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

    trainer = SFTTrainer(model="Qwen/Qwen2-0.5B-Instruct", train_dataset=dataset)
    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object.
            If you're training a model with an MoE architecture and want to include the load balancing/auxilliary loss
            as a part of the final loss, remember to set the `output_router_logits` config of the model to `True`.
        args ([`SFTConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`~trainer.sft_trainer.DataCollatorForLanguageModeling`] if the model is a language model
            and [`~trainer.sft_trainer.DataCollatorForVisionLanguageModeling`] if the model is a vision-language model.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. SFT supports both [language modeling](#language-modeling) type and
            [prompt-completion](#prompt-completion) type. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            The trainer also supports processed datasets (tokenized) as long as they contain an `input_ids` field.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoProcessor.from_pretrained`]. A padding token, `tokenizer.pad_token`, must be set.
            If the processing class has not set a padding token, `tokenizer.eos_token` will be used as the default.
        compute_loss_func (`Callable`, *optional*):
            A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
            batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default [loss
            function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618)
            used by [`Trainer`].
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a
            [`~transformers.EvalPrediction`] and return a dictionary string to metric values. When passing
            [`SFTConfig`] with `batch_eval_metrics` set to `True`, your `compute_metrics` function must take a boolean
            `compute_result` argument. This will be triggered after the last eval batch to signal that the function
            needs to calculate and return the global summary statistics rather than accumulating the batch-level
            statistics.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.

            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before
            initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        formatting_func (`Callable`, *optional*):
            Formatting function applied to the dataset before tokenization. Applying the formatting function explicitly
            converts the dataset into a [language modeling](#language-modeling) type.
    """

    _tag_names = ["trl", "sft"]
    _name = "SFT"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable[[dict], str]] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = SFTConfig(f"{model_name}-SFT")
        elif isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            dict_args.pop("push_to_hub_token")
            args = SFTConfig(**dict_args)

        # Model
        if isinstance(model, str):
            model = create_model_from_path(model, **args.model_init_kwargs or {})
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `SFTConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
        model_id = model.config._name_or_path

        # Processing class
        if processing_class is None:
            trust_remote_code = getattr(model.config, "auto_map", None) is not None
            processing_class = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                raise ValueError(
                    f"The specified `eos_token` ('{eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            tokenizer.eos_token_id = eos_token_id

        if args.chat_template_path is not None:
            if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
                with open(args.chat_template_path, encoding="utf-8") as chat_template_file:
                    processing_class.chat_template = chat_template_file.read()
                added_tokens = []
            else:
                model, processing_class, added_tokens = clone_chat_template(
                    model, processing_class, args.chat_template_path
                )
        else:
            added_tokens = []

        # Catch some wrong configurations related to VLMs
        if self._is_vlm and args.packing:
            raise ValueError(
                "Packing is not supported for vision-language models. Please set `packing=False` in the SFTConfig."
            )
        if self._is_vlm and args.padding_free:
            raise ValueError(
                "Padding-free training is yet not supported for vision-language models. Please set "
                "`padding_free=False` in the `SFTConfig`."
            )
        if self._is_vlm and args.assistant_only_loss:
            raise ValueError(
                "Assistant-only loss is not yet supported for vision-language models. Please set "
                "`assistant_only_loss=False` in the `SFTConfig`."
            )

        # PEFT configuration and model wrapping
        if peft_config is not None:
            if added_tokens:
                # Ensure that the added tokens are trainable
                if peft_config.trainable_token_indices is None:
                    peft_config.trainable_token_indices = {"embed_tokens": added_tokens}
                elif "embed_tokens" not in peft_config.trainable_token_indices:
                    peft_config.trainable_token_indices["embed_tokens"] = added_tokens
                else:
                    peft_config.trainable_token_indices["embed_tokens"].extend(added_tokens)

                # Ensure that the lm_head is trainable
                if peft_config.modules_to_save is None or "lm_head" not in peft_config.modules_to_save:
                    logger.warning(
                        "Cloning chat template added new tokens to the tokenizer, but 'lm_head' is not in PEFT's "
                        "`modules_to_save`. As a result, the model may not learn to generate outputs with these new "
                        "tokens, leading to degraded generation quality. To fix this, add "
                        "`modules_to_save=['lm_head']` to your PEFT configuration."
                    )

                    if peft_config.modules_to_save is None:
                        peft_config.modules_to_save = ["lm_head"]
                    else:
                        peft_config.modules_to_save.append("lm_head")

        # In Prompt Tuning a small set of trainable virtual tokens (continuous prompt embeddings) is prepended to the
        # input. We store the number of these tokens so we can account for them correctly when calculating accuracy.
        self.num_virtual_tokens = 0

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)
            if model.active_adapter in model.peft_config:
                peft_model_config = model.peft_config[model.active_adapter]
                self.num_virtual_tokens = getattr(peft_model_config, "num_virtual_tokens", 0)

        # Data collator
        # BFD packing requires padding-free mode; otherwise, the collator outputs padded attention masks, causing
        # FlashAttention to ignore position_ids and recompute them incorrectly from the padded attention mask.
        self.padding_free = args.padding_free or (args.packing and args.packing_strategy == "bfd")
        use_flash_attention = model.config._attn_implementation in FLASH_ATTENTION_VARIANTS
        if self.padding_free:
            if data_collator is not None:
                raise ValueError("Passing a custom data collator is not supported when using padding-free.")
            if args.packing and args.packing_strategy == "wrapped":
                logger.warning(
                    "You are passing `padding_free=True` with the 'wrapped' packing strategy, which is not "
                    "recommended. Please refer to the documentation to understand why this is not recommended."
                )
            if not use_flash_attention:
                logger.warning(
                    "Padding-free training is enabled, but the attention implementation is not set to a supported "
                    "flash attention variant. Padding-free training flattens batches into a single sequence, and only "
                    "the following implementations are known to reliably support this: "
                    f"{', '.join(sorted(FLASH_ATTENTION_VARIANTS))}. Using other implementations may lead to "
                    "unexpected behavior. To ensure compatibility, set `attn_implementation` in the model "
                    "configuration to one of these supported options or verify that your attention mechanism can "
                    "handle flattened sequences."
                )

            if args.per_device_train_batch_size == 1 and not args.packing:
                logger.warning(
                    "You are using a per_device_train_batch_size of 1 with padding-free training. Using a batch size "
                    "of 1 anihilate the benefits of padding-free training. Please consider increasing the batch size "
                    "to at least 2."
                )

        # Decide whether to use completion-only loss: if not specified, then it is set to True if the dataset format
        # is prompt-completion, and False if the dataset format is language modeling.
        dataset_sample = next(iter(train_dataset))
        if args.completion_only_loss is None:
            self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample
        else:
            self.completion_only_loss = args.completion_only_loss

        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample
        if self._is_vision_dataset and not self._is_vlm:
            raise ValueError(
                "The dataset appears to be vision-related (contains 'image' or 'images' keys), but the provided "
                "model does not seem to be a vision-language model. Please check your model and dataset."
            )

        if data_collator is None and not self._is_vision_dataset:
            # Get the pad token: if not provided, use the one from the processing class or the eos token
            # if the processing class does not have a pad token.
            pad_token = args.pad_token or tokenizer.pad_token or tokenizer.eos_token
            pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
            if pad_token_id is None:
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )
            # Pass assistant_masks through to compute_loss when EOS calibration is enabled
            needs_assistant_masks = args.aux_loss_eos_weight > 0
            data_collator = DataCollatorForLanguageModeling(
                pad_token_id=pad_token_id,
                completion_only_loss=self.completion_only_loss,
                padding_free=self.padding_free,
                pad_to_multiple_of=args.pad_to_multiple_of,
                pass_through_assistant_masks=needs_assistant_masks,
            )
        elif data_collator is None and self._is_vision_dataset:
            data_collator = DataCollatorForVisionLanguageModeling(
                processor=processing_class,
                max_length=args.max_length,
                completion_only_loss=self.completion_only_loss,
                pad_to_multiple_of=args.pad_to_multiple_of,
                dataset_text_field=args.dataset_text_field,
            )

        if args.packing and args.packing_strategy == "bfd" and not use_flash_attention:
            logger.warning(
                "You are using packing, but the attention implementation is not set to a supported flash attention "
                "variant. Packing gathers multiple samples into a single sequence, and only the following "
                f"implementations are known to reliably support this: {', '.join(sorted(FLASH_ATTENTION_VARIANTS))}. "
                "Using other implementations may lead to cross-contamination between samples. To avoid this, either "
                "disable packing by setting `packing=False`, or set `attn_implementation` in the model configuration "
                "to one of these supported options."
            )
        if args.assistant_only_loss and not is_conversational(dataset_sample):
            raise ValueError(
                "You set `assistant_only_loss=True`, but the dataset is not conversational. This option is only "
                "supported for conversational datasets."
            )
        if args.last_assistant_only_loss and not is_conversational(dataset_sample):
            raise ValueError(
                "You set `last_assistant_only_loss=True`, but the dataset is not conversational. This option is only "
                "supported for conversational datasets."
            )
        if args.train_on_incomplete_assistant and not is_conversational(dataset_sample):
            raise ValueError(
                "You set `train_on_incomplete_assistant=True`, but the dataset is not conversational. This option is "
                "only supported for conversational datasets."
            )

        # Dataset
        # Skip dataset preparation if `skip_prepare_dataset=True` in `dataset_kwargs`, or if it's a VLM, where
        # preprocessing (e.g., image-to-pixel conversion) is too costly and done on the fly instead.
        skip_prepare_dataset = (
            args.dataset_kwargs is not None
            and args.dataset_kwargs.get("skip_prepare_dataset", False)
            or self._is_vision_dataset
        )
        if not skip_prepare_dataset:
            if self.completion_only_loss and formatting_func:
                raise ValueError(
                    "A formatting function was provided while `completion_only_loss=True`, which is incompatible. "
                    "Using a formatter converts the dataset to a language modeling type, conflicting with "
                    "completion-only loss. To resolve this, apply your formatting function before passing the "
                    "dataset, or disable `completion_only_loss` in `SFTConfig`."
                )
            train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, args.packing, formatting_func, "train"
            )
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, packing, formatting_func, "eval"
                    )

        # Loss function
        if args.loss_type == "nll":
            pass  # use the default loss
        elif args.loss_type == "dft":
            if compute_loss_func is not None:
                raise ValueError(
                    "You passed a `compute_loss_func` together with `loss_type='dft'` to the `SFTTrainer`. "
                    "When using `loss_type='dft'`, the loss function is internally set to the DFT loss, so passing a "
                    "`compute_loss_func` is not allowed."
                )
            compute_loss_func = dft_loss
        else:
            raise ValueError(f"Invalid `loss_type` {args.loss_type} passed. Supported values are 'nll' and 'dft'.")

        # Validate auxiliary loss configuration
        if args.aux_loss_eos_weight > 0 and not (args.assistant_only_loss or args.last_assistant_only_loss):
            logger.warning(
                "EOS calibration loss (`aux_loss_eos_weight > 0`) works best with `assistant_only_loss=True` "
                "so that assistant turn boundaries are available. Without it, no turn-end positions can be "
                "identified and the EOS loss will have no effect."
            )
        if args.aux_loss_rep_weight > 0:
            logger.info(f"Repetition penalty loss enabled (weight={args.aux_loss_rep_weight}, window={args.aux_loss_rep_window})")
        if args.aux_loss_diversity_weight > 0:
            logger.info(f"Vocabulary diversity loss enabled (weight={args.aux_loss_diversity_weight})")
        if args.aux_loss_confidence_weight > 0:
            logger.info(f"Confidence regularization loss enabled (weight={args.aux_loss_confidence_weight})")

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Initialize the Trainer. Parent class will handle:
        # - DeepSpeed configuration (through create_accelerator_and_postprocess)
        # - FSDP setup
        # - Distributed training setup
        # - Optimizer and scheduler creation

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Initialize activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)
        self.aux_loss_coef = getattr(model.config, "router_aux_loss_coef", 0.0)
        if self.aux_loss_enabled and self.aux_loss_coef == 0.0:
            logger.warning(
                "You set `output_router_logits` to `True` in the model config, but `router_aux_loss_coef` is "
                "set to `0.0`, meaning the MoE load-balancing loss will not be added to the training loss. "
                "Either set `router_aux_loss_coef` to a value greater than `0.0`, or set "
                "`output_router_logits` to `False` if you don't want to use the MoE auxiliary loss."
            )

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures. Clean them from
        # sampled data.
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = get_dataset_column_names(dataset)
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                logger.warning(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            if not is_processed:
                # Auto-convert preference datasets to SFT format
                first_example = next(iter(dataset))
                if is_preference_dataset(first_example):
                    if isinstance(dataset, Dataset):
                        map_kwargs["desc"] = f"Converting preference dataset {dataset_name} to SFT format"
                    logger.info(
                        f"Detected preference dataset format (chosen/rejected). "
                        f"Converting to SFT format using 'chosen' responses."
                    )
                    column_names = get_dataset_column_names(dataset)
                    remove_cols = [c for c in ["prompt", "chosen", "rejected"] if c in column_names]
                    dataset = dataset.map(
                        convert_preference_to_sft,
                        remove_columns=remove_cols,
                        **map_kwargs,
                    )

                elif is_binary_preference_dataset(first_example):
                    if isinstance(dataset, Dataset):
                        map_kwargs["desc"] = f"Converting binary preference dataset {dataset_name} to SFT format"
                    logger.info(
                        f"Detected binary preference dataset format (completion/label). "
                        f"Converting to SFT format using good (label=True) responses only."
                    )
                    column_names = get_dataset_column_names(dataset)
                    remove_cols = [c for c in ["prompt", "completion", "label"] if c in column_names]

                    # Map and filter in one pass
                    def convert_and_filter(example):
                        result = convert_binary_preference_to_sft(example)
                        # Return empty dict for bad examples (will be filtered)
                        return result if result is not None else {}

                    dataset = dataset.map(
                        convert_and_filter,
                        remove_columns=remove_cols,
                        **map_kwargs,
                    )

                    # Filter out empty examples (bad responses)
                    if isinstance(dataset, Dataset):
                        original_len = len(dataset)
                        dataset = dataset.filter(lambda x: "messages" in x and len(x["messages"]) > 0)
                        filtered_len = len(dataset)
                        if filtered_len < original_len:
                            logger.info(
                                f"Filtered out {original_len - filtered_len} bad (label=False) examples "
                                f"from binary preference dataset."
                            )

                # Convert the dataset to ChatML if needed
                first_example = next(iter(dataset))
                if is_conversational_from_value(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                    column_names = get_dataset_column_names(dataset)
                    dataset = dataset.map(
                        maybe_convert_to_chatml,
                        remove_columns="conversations" if "conversations" in column_names else None,
                        **map_kwargs,
                    )

                # Add default system message if configured or per-dataset system messages exist
                column_names = get_dataset_column_names(dataset)
                has_per_dataset_system_msg = "_system_message" in column_names
                if (args.default_system_message or has_per_dataset_system_msg) and is_conversational(
                    next(iter(dataset))
                ):
                    if isinstance(dataset, Dataset):
                        map_kwargs["desc"] = f"Adding default system message to {dataset_name} dataset"

                    def add_system_message_fn(example, system_message):
                        return add_system_message_to_example(example, system_message)

                    # Remove _system_message column after processing (it was added by dataset mixer)
                    remove_cols = "_system_message" if has_per_dataset_system_msg else None

                    dataset = dataset.map(
                        add_system_message_fn,
                        fn_kwargs={"system_message": args.default_system_message or ""},
                        remove_columns=remove_cols,
                        **map_kwargs,
                    )

                # Fix turn order if requested (for models with strict turn order requirements)
                if args.fix_turn_order and is_conversational(next(iter(dataset))):
                    if isinstance(dataset, Dataset):
                        map_kwargs["desc"] = f"Fixing turn order in {dataset_name} dataset"

                    def fix_turn_order_fn(example, filler_message):
                        return fix_example_turn_order(example, filler_message)

                    dataset = dataset.map(
                        fix_turn_order_fn,
                        fn_kwargs={"filler_message": args.fix_turn_order_filler},
                        **map_kwargs,
                    )

                    # Filter out empty conversations (where all messages were dropped)
                    def has_valid_conversation(example):
                        for key in ["messages", "prompt", "completion"]:
                            if key in example:
                                val = example[key]
                                if isinstance(val, list) and len(val) > 0:
                                    return True
                        return False

                    if isinstance(dataset, Dataset):
                        original_len = len(dataset)
                        dataset = dataset.filter(has_valid_conversation)
                        filtered_len = len(dataset)
                        if filtered_len < original_len:
                            logger.warning(
                                f"fix_turn_order: Filtered out {original_len - filtered_len} examples "
                                f"with invalid turn order that couldn't be fixed."
                            )

                # Apply truncate_turns strategy before tokenization (if applicable)
                # Get effective truncation strategy (per-dataset overrides global)
                column_names = get_dataset_column_names(dataset)
                has_per_dataset_strategy = "_truncation_strategy" in column_names
                effective_strategy = args.truncation_strategy

                if (
                    effective_strategy == "truncate_turns" or has_per_dataset_strategy
                ) and args.max_length is not None:
                    first_example = next(iter(dataset))
                    if is_conversational(first_example):
                        if isinstance(dataset, Dataset):
                            map_kwargs["desc"] = f"Truncating {dataset_name} by turns"

                        def truncate_turns_fn(example, tokenizer, max_length, default_strategy):
                            # Per-dataset strategy overrides global
                            strategy = example.pop("_truncation_strategy", None) or default_strategy
                            if strategy != "truncate_turns":
                                return example  # Will be handled post-tokenization
                            truncated = truncate_conversation_by_turns(
                                example.get("messages", []),
                                tokenizer,
                                max_length,
                            )
                            if truncated is None:
                                # Mark for filtering
                                example["_truncation_drop"] = True
                            else:
                                example["messages"] = truncated
                            return example

                        remove_cols = "_truncation_strategy" if has_per_dataset_strategy else None
                        dataset = dataset.map(
                            truncate_turns_fn,
                            fn_kwargs={
                                "tokenizer": processing_class,
                                "max_length": args.max_length,
                                "default_strategy": effective_strategy,
                            },
                            remove_columns=remove_cols,
                            **map_kwargs,
                        )

                        # Filter out dropped samples
                        if isinstance(dataset, Dataset):
                            original_len = len(dataset)
                            dataset = dataset.filter(lambda x: not x.get("_truncation_drop", False))
                            filtered_len = len(dataset)
                            if filtered_len < original_len:
                                logger.info(
                                    f"truncate_turns: Dropped {original_len - filtered_len} samples "
                                    f"that couldn't fit even one turn pair in max_length={args.max_length}."
                                )
                            # Remove the marker column
                            if "_truncation_drop" in get_dataset_column_names(dataset):
                                dataset = dataset.remove_columns(["_truncation_drop"])

                # Apply the chat template if needed
                first_example = next(iter(dataset))
                if not is_conversational(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if "text" in example and not example["text"].endswith(eos_token):  # language modeling case
                            example["text"] = example["text"] + eos_token
                        elif "completion" in example and not example["completion"].endswith(eos_token):
                            example["completion"] = example["completion"] + eos_token
                        return example

                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": processing_class.eos_token},
                        remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                        **map_kwargs,
                    )

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize_fn(
                    example,
                    processing_class,
                    dataset_text_field,
                    assistant_only_loss,
                    last_assistant_only_loss,
                    train_on_incomplete_assistant,
                    eos_token_id,
                ):
                    # Determine if we need assistant masks (for any assistant loss mode)
                    need_assistant_masks = assistant_only_loss or last_assistant_only_loss

                    # Check if last message is from assistant (for train_on_incomplete_assistant)
                    last_role_is_assistant = False
                    if train_on_incomplete_assistant:
                        messages = example.get("messages") or (
                            example.get("prompt", []) + example.get("completion", [])
                        )
                        if messages and isinstance(messages, list) and len(messages) > 0:
                            last_msg = messages[-1]
                            if isinstance(last_msg, dict):
                                role = last_msg.get("role") or last_msg.get("from", "")
                                last_role_is_assistant = role.lower() in ("assistant", "gpt")

                    if "prompt" in example:  # prompt-completion case
                        output = {}
                        if is_conversational(example):
                            if self._is_vlm:
                                prepare_multimodal_messages(example["prompt"], num_images=0)
                                prepare_multimodal_messages(example["completion"], num_images=0)
                            prompt_ids = processing_class.apply_chat_template(
                                example["prompt"],
                                tokenize=True,
                                add_generation_prompt=True,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
                            prompt_completion_processed = processing_class.apply_chat_template(
                                example["prompt"] + example["completion"],
                                return_dict=True,
                                tokenize=True,
                                return_assistant_tokens_mask=need_assistant_masks,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            prompt_completion_processed = {
                                k: v[0] if isinstance(v[0], list) else v
                                for k, v in prompt_completion_processed.items()
                            }
                            prompt_completion_ids = prompt_completion_processed["input_ids"]
                            if "assistant_masks" in prompt_completion_processed:
                                output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
                        else:
                            prompt_ids = processing_class(text=example["prompt"])["input_ids"]
                            prompt_completion_ids = processing_class(text=example["prompt"] + example["completion"])[
                                "input_ids"
                            ]

                        # Check if the tokenized prompt starts with the tokenized prompt+completion
                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                            logger.warning(
                                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                                "token handling. Verify that the tokenizer is processing text consistently."
                            )

                        # Create completion mask
                        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                        output["input_ids"] = prompt_completion_ids
                        output["completion_mask"] = completion_mask

                    else:  # language modeling case
                        if is_conversational(example):
                            if self._is_vlm:
                                prepare_multimodal_messages(example["messages"], num_images=0)
                            processed = processing_class.apply_chat_template(
                                example["messages"],
                                return_dict=True,
                                tokenize=True,
                                return_assistant_tokens_mask=need_assistant_masks,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            processed = {k: v[0] if isinstance(v[0], list) else v for k, v in processed.items()}
                            output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
                        else:
                            output = {"input_ids": processing_class(text=example[dataset_text_field])["input_ids"]}

                    # Apply last_assistant_only_loss: mask all but the last assistant turn
                    if last_assistant_only_loss and "assistant_masks" in output:
                        output["assistant_masks"] = mask_to_last_segment_only(output["assistant_masks"])

                    # Apply train_on_incomplete_assistant: remove trailing EOS if last role is assistant
                    if train_on_incomplete_assistant and last_role_is_assistant and eos_token_id is not None:
                        output["input_ids"] = remove_trailing_eos(output["input_ids"], eos_token_id)
                        # Also truncate masks to match
                        if "assistant_masks" in output:
                            output["assistant_masks"] = output["assistant_masks"][: len(output["input_ids"])]
                        if "completion_mask" in output:
                            output["completion_mask"] = output["completion_mask"][: len(output["input_ids"])]

                    if "assistant_masks" in output and 1 not in output["assistant_masks"]:
                        raise RuntimeError(
                            "You're using `assistant_only_loss=True` or `last_assistant_only_loss=True`, but at least "
                            "one example has no assistant tokens. This usually means the tokenizer's chat template "
                            "doesn't generate assistant masks — it may be missing the `{% generation %}` keyword. "
                            "Please check the template and ensure it's correctly configured to support assistant masking."
                        )
                    return output

                dataset = dataset.map(
                    tokenize_fn,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "assistant_only_loss": args.assistant_only_loss,
                        "last_assistant_only_loss": args.last_assistant_only_loss,
                        "train_on_incomplete_assistant": args.train_on_incomplete_assistant,
                        "eos_token_id": processing_class.eos_token_id,
                    },
                    **map_kwargs,
                )

            # Pack or truncate
            # Skip truncation for pre-tokenized blend data (already truncated/split)
            _pretokenized = getattr(args, "_pretokenized", False)
            if _pretokenized and is_processed and not packing:
                logger.info(
                    f"Skipping truncation for {dataset_name} — data is pre-tokenized from blend."
                )
            elif packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                columns = ["input_ids"]
                if "completion_mask" in get_dataset_column_names(dataset):
                    columns.append("completion_mask")
                if "assistant_masks" in get_dataset_column_names(dataset):
                    columns.append("assistant_masks")

                dataset = dataset.select_columns(columns)

                # Packing adds new column "seq_lengths" needed for document aware FlashAttention
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            elif args.max_length is not None:
                # Apply truncation strategy (truncate, drop, or split)
                # Note: truncate_turns is handled before tokenization
                strategy = args.truncation_strategy
                if strategy == "truncate_turns":
                    # Already handled pre-tokenization, fall back to regular truncate
                    strategy = "truncate"

                # Check for per-example max_length column
                column_names = get_dataset_column_names(dataset)
                has_per_example_max_length = "_max_length" in column_names

                if strategy == "drop":
                    # Filter out samples exceeding max_length
                    if isinstance(dataset, Dataset):
                        original_len = len(dataset)
                        if has_per_example_max_length:
                            # Use per-example max_length if available
                            dataset = dataset.filter(
                                lambda x: len(x.get("input_ids", [])) <= x.get("_max_length", args.max_length)
                            )
                        else:
                            dataset = dataset.filter(lambda x: len(x.get("input_ids", [])) <= args.max_length)
                        filtered_len = len(dataset)
                        if filtered_len < original_len:
                            logger.info(
                                f"drop strategy: Filtered out {original_len - filtered_len} samples "
                                f"exceeding max_length."
                            )
                        # Clean up _max_length column after drop
                        if has_per_example_max_length and "_max_length" in get_dataset_column_names(dataset):
                            dataset = dataset.remove_columns(["_max_length"])
                elif strategy == "split":
                    # Split long sequences into chunks
                    if isinstance(dataset, Dataset):
                        map_kwargs["desc"] = f"Splitting {dataset_name} dataset into chunks"

                    def apply_split(example, tokenizer, default_max_length):
                        # Use per-example max_length if present, otherwise use default
                        effective_max_length = example.pop("_max_length", None) or default_max_length
                        return apply_truncation_strategy_to_example(
                            example, tokenizer, effective_max_length, strategy="split"
                        )

                    dataset = dataset.map(
                        apply_split,
                        fn_kwargs={"tokenizer": processing_class, "default_max_length": args.max_length},
                        remove_columns=["_max_length"] if has_per_example_max_length else None,
                        **map_kwargs,
                    )
                    # Expand chunks into separate rows
                    if isinstance(dataset, Dataset):
                        dataset = expand_split_chunks(dataset)
                else:
                    # Default truncate behavior (with EOS removal for truncated samples)
                    if isinstance(dataset, Dataset):
                        map_kwargs["desc"] = f"Truncating {dataset_name} dataset"

                    def apply_truncate(example, tokenizer, default_max_length):
                        # Use per-example max_length if present, otherwise use default
                        effective_max_length = example.pop("_max_length", None) or default_max_length
                        return apply_truncation_strategy_to_example(
                            example, tokenizer, effective_max_length, strategy="truncate"
                        )

                    dataset = dataset.map(
                        apply_truncate,
                        fn_kwargs={"tokenizer": processing_class, "default_max_length": args.max_length},
                        remove_columns=["_max_length"] if has_per_example_max_length else None,
                        **map_kwargs,
                    )
            # For Liger kernel, ensure only the essential columns
            if args.use_liger_kernel:
                collator_expected_keys = {"input_ids", "seq_lengths", "completion_mask", "assistant_masks"}
                column_names = get_dataset_column_names(dataset)
                dataset = dataset.select_columns(collator_expected_keys.intersection(column_names))

        return dataset

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). When using `train_on_completion_only` we add a "completion_mask" column to the
        # dataset. So we need to override the default signature columns to include "completion_mask" as well.
        if self._signature_columns is None:
            if self._is_vision_dataset:
                self._signature_columns = ["messages", "prompt", "completion", "images"]
            else:
                self._signature_columns = ["input_ids", "labels", "seq_lengths", "completion_mask", "assistant_masks"]

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "train" if self.model.training else "eval"

        # Set aside labels as it will be dropped by super().compute_loss() if a custom `compute_loss_func` is used.
        # This can be removed when this issue is fixed.
        labels = inputs["labels"]

        # Pop assistant_masks before forward pass — the model doesn't expect them
        assistant_masks = inputs.pop("assistant_masks", None)

        # If not set, defaults from model config and may warn since cache isn't compatible with gradient checkpointing
        inputs["use_cache"] = False
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Add MoE router load-balancing auxiliary loss to the training objective
        if self.aux_loss_enabled and self.aux_loss_coef > 0:
            aux_loss = getattr(outputs, "aux_loss", None)
            if aux_loss is not None:
                loss = loss + self.aux_loss_coef * aux_loss

        # Compute auxiliary losses (only when logits are available)
        if not self.args.use_liger_kernel:
            logits = outputs.logits
            has_any_aux = (
                self.args.aux_loss_eos_weight > 0
                or self.args.aux_loss_rep_weight > 0
                or self.args.aux_loss_diversity_weight > 0
                or self.args.aux_loss_confidence_weight > 0
            )
            if has_any_aux:
                # EOS calibration
                if self.args.aux_loss_eos_weight > 0 and assistant_masks is not None:
                    eos_token_id = self.processing_class.eos_token_id
                    eos_loss = aux_eos_calibration_loss(logits, labels, assistant_masks, eos_token_id)
                    loss = loss + self.args.aux_loss_eos_weight * eos_loss
                    self._metrics[mode]["aux_loss_eos"].append(
                        self.accelerator.gather_for_metrics(eos_loss.detach()).mean().item()
                    )

                # Repetition penalty
                if self.args.aux_loss_rep_weight > 0:
                    rep_loss = aux_repetition_penalty_loss(logits, labels, self.args.aux_loss_rep_window)
                    loss = loss + self.args.aux_loss_rep_weight * rep_loss
                    self._metrics[mode]["aux_loss_rep"].append(
                        self.accelerator.gather_for_metrics(rep_loss.detach()).mean().item()
                    )

                # Vocabulary diversity
                if self.args.aux_loss_diversity_weight > 0:
                    div_loss = aux_vocabulary_diversity_loss(
                        logits, labels, self.args.aux_loss_diversity_max_ratio
                    )
                    loss = loss + self.args.aux_loss_diversity_weight * div_loss
                    self._metrics[mode]["aux_loss_diversity"].append(
                        self.accelerator.gather_for_metrics(div_loss.detach()).mean().item()
                    )

                # Confidence regularization
                if self.args.aux_loss_confidence_weight > 0:
                    conf_loss = aux_confidence_regularization_loss(logits, labels)
                    loss = loss + self.args.aux_loss_confidence_weight * conf_loss
                    self._metrics[mode]["aux_loss_confidence"].append(
                        self.accelerator.gather_for_metrics(conf_loss.detach()).mean().item()
                    )

        # Compute entropy
        if not self.args.use_liger_kernel:  # liger doesn't return logits
            with torch.no_grad():
                per_token_entropy = entropy_from_logits(outputs.logits)
                # When using Prompt Tuning, skip the virtual tokens in logits before entropy computation, since they
                # do not correspond to actual input tokens.
                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
                ):
                    per_token_entropy = per_token_entropy[:, self.num_virtual_tokens :]
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"]
                    entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
                elif "position_ids" in inputs:
                    entropy = torch.mean(per_token_entropy)
                else:
                    raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
                entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
            self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if not self.args.use_liger_kernel:
            with torch.no_grad():
                if "shift_labels" in inputs:
                    # When using CP, labels are pre-shifted. We must use these (and cannot manually shift) because:
                    # - The first discarded token from inputs["labels"] actually belongs to process n-1
                    # - The last logits require the label from process n+1
                    shift_logits = outputs.logits.contiguous()
                    shift_labels = inputs["shift_labels"]
                else:
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                # Prompt Tuning and P-Tuning output logits for virtual tokens but Prefix-Tuning does not.
                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
                ):
                    shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

                # Get predictions
                predictions = shift_logits.argmax(dim=-1)

                # Create mask for non-padding tokens (assuming ignore_index is -100)
                mask = shift_labels != -100

                # Calculate accuracy only on non-padding tokens
                correct_predictions = (predictions == shift_labels) & mask
                total_tokens = mask.sum()
                correct_tokens = correct_predictions.sum()

                # Gather the correct_tokens and total_tokens across all processes
                correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
                total_tokens = self.accelerator.gather_for_metrics(total_tokens)

                # Compute the mean token accuracy and log it
                total_sum = total_tokens.sum()
                accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
                self._metrics[mode]["mean_token_accuracy"].append(accuracy)
                if self.aux_loss_enabled:
                    aux_loss = outputs.aux_loss
                    aux_loss = self.accelerator.gather_for_metrics(aux_loss).mean().item()
                    self._metrics[mode]["aux_loss"].append(aux_loss)

        return (loss, outputs) if return_outputs else loss

    # Override training step to add activation offloading context.
    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            return super().training_step(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Override to handle PEFT + DeepSpeed ZeRO-3 saving.

        The default Trainer.save_model() consolidates the entire ZeRO-3 sharded model
        into a single state dict before saving, which OOMs for large models even when
        only LoRA adapter weights need to be saved. This override detects the PEFT +
        DeepSpeed case and gathers only the small trainable adapter parameters.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        is_peft = is_peft_available() and isinstance(
            self.accelerator.unwrap_model(self.model, keep_torch_compile=False), PeftModel
        )

        if self.is_deepspeed_enabled and is_peft:
            # Gather only trainable (adapter) parameters instead of the full model.
            # Under ZeRO-3, all params are partitioned; we use GatheredParameters to
            # temporarily materialize just the trainable ones on rank 0.
            import deepspeed

            unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            trainable_params = [p for p in unwrapped.parameters() if p.requires_grad]

            with deepspeed.zero.GatheredParameters(trainable_params):
                if self.args.should_save:
                    unwrapped.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)

            # Save tokenizer / processor on main process
            if self.args.should_save:
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(output_dir)

            if self.args.push_to_hub and not _internal_call:
                self.push_to_hub(commit_message="Model save", revision=self.args.hub_revision)
        else:
            super().save_model(output_dir, _internal_call)

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
