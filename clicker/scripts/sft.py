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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# Full training
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

# LoRA
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```
"""

import argparse
import os
from typing import Optional

from accelerate import logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from clicker import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from clicker.import_utils import is_cce_available
from clicker.scripts.utils import (
    build_blend_config,
    load_prepared_dataset,
    get_tokenized_cache_path,
    is_pretokenized_blend,
    needs_blend,
    prompt_blend_overwrite,
    run_auto_blend,
    validate_blend_metadata,
)


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def main(script_args, training_args, model_args, dataset_args):
    ################
    # Model init kwargs
    ################
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    # Create model
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Apply CCE (Cut Cross-Entropy) patching for memory-efficient loss computation
    if training_args.use_cce:
        if not is_cce_available():
            raise ImportError(
                "CCE (Cut Cross-Entropy) is not available. Please install it with: pip install cut-cross-entropy"
            )
        try:
            from cut_cross_entropy.transformers import cce_patch
        except ImportError as e:
            raise ImportError(
                f"CCE import failed due to version incompatibility: {e}\n"
                "This typically occurs when cut-cross-entropy is incompatible with your transformers version.\n"
                "Try: pip install --upgrade cut-cross-entropy transformers"
            ) from e

        model = cce_patch(model)
        logger.info("Applied CCE (Cut Cross-Entropy) patch for memory-efficient loss computation.")

    # Load the dataset
    if training_args.prepared_dataset:
        _prepared = training_args.prepared_dataset
        _has_data_config = bool(training_args.data_config)

        # New-style: data_config is set, so we can auto-blend if needed
        if _has_data_config:
            if needs_blend(_prepared):
                # Path is missing/empty — auto-run blend
                logger.info(f"Prepared dataset not found at {_prepared} — running blend automatically.")
                run_auto_blend(training_args, model_args)
            else:
                # Path exists — validate metadata matches current config
                blend_config = build_blend_config(training_args, model_args)
                mismatches = validate_blend_metadata(_prepared, blend_config)
                if mismatches:
                    # Prompt user to overwrite (rank 0 only, exits on 'n')
                    prompt_blend_overwrite(_prepared, mismatches, force=training_args.force_blend)
                    # User said yes (or force_blend=True) — re-blend
                    run_auto_blend(training_args, model_args)

        # Load the prepared dataset (now guaranteed to exist if data_config was set)
        logger.info(f"Loading prepared dataset from {_prepared}")
        dataset = load_prepared_dataset(_prepared)
        logger.info(
            f"Loaded prepared dataset: {len(dataset['train'])} train"
            + (f", {len(dataset['test'])} test" if 'test' in dataset else "")
        )
        # If this is a new-style pre-tokenized blend, tell the trainer to skip truncation
        if is_pretokenized_blend(_prepared):
            training_args._pretokenized = True
            logger.info("Detected pre-tokenized blend — skipping tokenization and truncation in trainer.")
    elif training_args.data_config and not training_args.prepared_dataset:
        # data_config is set but no prepared_dataset path — error with helpful message
        raise ValueError(
            "Training config has 'data_config' but no 'prepared_dataset' path. "
            "Set 'prepared_dataset' to a directory where the blended data should be stored. "
            "It will be created automatically if it doesn't exist."
        )
    elif dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `prepared_dataset`, `datasets`, or `dataset_name` must be provided.")

    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
