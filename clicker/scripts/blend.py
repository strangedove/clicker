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

"""
Dataset blending/preprocessing script.

Prepares datasets with full tokenization, truncation/splitting, and eval splitting.
Takes a training config (which references a data config) and produces a pre-tokenized
dataset ready for training.

Usage:
    clicker blend --config configs/my-training.yaml
    clicker blend --config configs/my-training.yaml --output data/prepared/my-blend
"""

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Optional

import yaml
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from clicker.data_utils import (
    add_system_message_to_example,
    apply_truncation_to_dataset,
    convert_binary_preference_to_sft,
    convert_preference_to_sft,
    fix_example_turn_order,
    is_binary_preference_dataset,
    is_conversational,
    is_conversational_from_value,
    is_preference_dataset,
    maybe_convert_to_chatml,
    tokenize_sft_example,
    truncate_conversation_by_turns,
)
from clicker.scripts.utils import DatasetMixtureConfig, get_dataset


logger = logging.getLogger(__name__)

# Fun messages for the blend process
BLEND_MESSAGES = {
    "start": "🍹 Starting the blender...",
    "loading_data": "📦 Loading data config from {path}...",
    "loading_model": "🤖 Loading tokenizer from {model}...",
    "loading_datasets": "📦 Gathering ingredients from {n} dataset(s)...",
    "preprocessing": "🔄 Preprocessing (format conversion, system messages, turn order)...",
    "tokenizing": "🔤 Tokenizing with {model}...",
    "truncating": "✂️  Applying truncation strategy: {strategy} (max_length={max_length})...",
    "splitting": "📊 Splitting eval: {eval_split:.1%} ({eval} eval, {train} train)...",
    "no_eval": "📊 No eval split requested.",
    "saving": "💾 Saving to {path}...",
    "done": "✅ Dataset ready! Saved to {path}",
    "stats": "📊 Final: {train} train samples, {eval} eval samples, {total} total",
}


def _load_data_config(data_config_path: str) -> DatasetMixtureConfig:
    """Load a data config YAML and return a DatasetMixtureConfig."""
    with open(data_config_path) as f:
        raw = yaml.safe_load(f)

    # Force eval_split to 0 — we do eval splitting AFTER tokenization
    raw["eval_split"] = 0.0

    return DatasetMixtureConfig(**raw)


def _load_training_config(config_path: str) -> dict:
    """Load a training config YAML and return the raw dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def preprocess_dataset(
    dataset: Dataset,
    trainer_type: str = "sft",
    default_system_message: Optional[str] = None,
    fix_turn_order: bool = False,
    fix_turn_order_filler: str = "Let's begin.",
    num_proc: Optional[int] = None,
) -> Dataset:
    """
    Apply model-agnostic preprocessing to a dataset.

    Handles format conversion, system messages, turn order fixing.
    """
    map_kwargs = {}
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc

    if trainer_type != "sft":
        # For now, only SFT preprocessing is supported in blend
        logger.warning(f"trainer_type={trainer_type} is not fully supported in blend yet. Proceeding with SFT logic.")

    # Auto-convert preference datasets to SFT format
    first_example = next(iter(dataset))
    if is_preference_dataset(first_example):
        logger.info("Detected preference dataset format — converting to SFT format.")
        column_names = dataset.column_names
        remove_cols = [c for c in ["prompt", "chosen", "rejected"] if c in column_names]
        dataset = dataset.map(convert_preference_to_sft, remove_columns=remove_cols, **map_kwargs)

    elif is_binary_preference_dataset(first_example):
        logger.info("Detected binary preference dataset — converting to SFT format.")
        column_names = dataset.column_names
        remove_cols = [c for c in ["prompt", "completion", "label"] if c in column_names]

        def convert_and_filter(example):
            result = convert_binary_preference_to_sft(example)
            return result if result is not None else {}

        dataset = dataset.map(convert_and_filter, remove_columns=remove_cols, **map_kwargs)
        original_len = len(dataset)
        dataset = dataset.filter(lambda x: "messages" in x and len(x["messages"]) > 0, **map_kwargs)
        if len(dataset) < original_len:
            logger.info(f"Filtered out {original_len - len(dataset)} bad examples from binary preference dataset.")

    # Convert legacy conversation format to ChatML
    first_example = next(iter(dataset))
    if is_conversational_from_value(first_example):
        column_names = dataset.column_names
        dataset = dataset.map(
            maybe_convert_to_chatml,
            remove_columns="conversations" if "conversations" in column_names else None,
            desc="Converting to ChatML",
            **map_kwargs,
        )

    # Add system messages
    column_names = dataset.column_names
    has_per_dataset_system_msg = "_system_message" in column_names
    if (default_system_message or has_per_dataset_system_msg) and is_conversational(next(iter(dataset))):
        remove_cols = "_system_message" if has_per_dataset_system_msg else None
        dataset = dataset.map(
            add_system_message_to_example,
            fn_kwargs={"system_message": default_system_message or ""},
            remove_columns=remove_cols,
            desc="Adding system messages",
            **map_kwargs,
        )

    # Fix turn order
    if fix_turn_order and is_conversational(next(iter(dataset))):
        dataset = dataset.map(
            fix_example_turn_order,
            fn_kwargs={"filler_message": fix_turn_order_filler},
            desc="Fixing turn order",
            **map_kwargs,
        )
        original_len = len(dataset)
        dataset = dataset.filter(
            lambda x: any(
                isinstance(x.get(k), list) and len(x.get(k, [])) > 0
                for k in ["messages", "prompt", "completion"]
            ),
            **map_kwargs,
        )
        if len(dataset) < original_len:
            logger.warning(f"fix_turn_order: Dropped {original_len - len(dataset)} invalid examples.")

    return dataset


def tokenize_dataset(
    dataset: Dataset,
    processing_class,
    dataset_text_field: str = "text",
    truncation_strategy: str = "truncate",
    max_length: Optional[int] = None,
    assistant_only_loss: bool = False,
    last_assistant_only_loss: bool = False,
    train_on_incomplete_assistant: bool = False,
    num_proc: Optional[int] = None,
) -> Dataset:
    """
    Tokenize a preprocessed dataset and apply truncation.

    Handles:
    - truncate_turns (pre-tokenization, on messages)
    - EOS addition for plain text
    - Tokenization via chat template or plain tokenizer
    - Truncation strategy (split/drop/truncate)
    """
    map_kwargs = {}
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc

    # Handle truncate_turns before tokenization (needs message-level structure)
    column_names = dataset.column_names
    has_per_dataset_strategy = "_truncation_strategy" in column_names

    if (truncation_strategy == "truncate_turns" or has_per_dataset_strategy) and max_length is not None:
        first_example = next(iter(dataset))
        if is_conversational(first_example):
            def truncate_turns_fn(example, tokenizer, _max_length, default_strategy):
                strategy = example.pop("_truncation_strategy", None) or default_strategy
                if strategy != "truncate_turns":
                    return example
                truncated = truncate_conversation_by_turns(
                    example.get("messages", []), tokenizer, _max_length
                )
                if truncated is None:
                    example["_truncation_drop"] = True
                else:
                    example["messages"] = truncated
                return example

            remove_cols = "_truncation_strategy" if has_per_dataset_strategy else None
            dataset = dataset.map(
                truncate_turns_fn,
                fn_kwargs={
                    "tokenizer": processing_class,
                    "_max_length": max_length,
                    "default_strategy": truncation_strategy,
                },
                remove_columns=remove_cols,
                desc="Truncating by turns",
                **map_kwargs,
            )
            original_len = len(dataset)
            dataset = dataset.filter(lambda x: not x.get("_truncation_drop", False), **map_kwargs)
            if len(dataset) < original_len:
                logger.info(
                    f"truncate_turns: Dropped {original_len - len(dataset)} samples that couldn't fit "
                    f"even one turn pair in max_length={max_length}."
                )
            column_names = dataset.column_names
            if "_truncation_drop" in column_names:
                dataset = dataset.remove_columns(["_truncation_drop"])

    # Add EOS for plain text datasets
    first_example = next(iter(dataset))
    if not is_conversational(first_example):
        eos_token = processing_class.eos_token

        def add_eos(example, _eos_token):
            if "text" in example and not example["text"].endswith(_eos_token):
                example["text"] = example["text"] + _eos_token
            elif "completion" in example and not example["completion"].endswith(_eos_token):
                example["completion"] = example["completion"] + _eos_token
            return example

        dataset = dataset.map(
            add_eos,
            fn_kwargs={"_eos_token": eos_token},
            desc="Adding EOS",
            **map_kwargs,
        )

    # Tokenize
    dataset = dataset.map(
        tokenize_sft_example,
        fn_kwargs={
            "processing_class": processing_class,
            "dataset_text_field": dataset_text_field,
            "assistant_only_loss": assistant_only_loss,
            "last_assistant_only_loss": last_assistant_only_loss,
            "train_on_incomplete_assistant": train_on_incomplete_assistant,
            "eos_token_id": processing_class.eos_token_id,
        },
        desc="Tokenizing",
        **map_kwargs,
    )

    # Apply truncation strategy
    if max_length is not None:
        effective_strategy = truncation_strategy
        if effective_strategy == "truncate_turns":
            effective_strategy = "truncate"  # already handled above
        dataset = apply_truncation_to_dataset(
            dataset, processing_class, max_length, strategy=effective_strategy, num_proc=num_proc
        )

    return dataset


def prepare_dataset(
    training_config: dict,
    output_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Full pipeline: load data → preprocess → tokenize → truncate → eval split.

    Args:
        training_config: Raw dict from the training YAML config.
        output_dir: Override for output directory.

    Returns:
        DatasetDict with "train" and optionally "test" splits.
    """
    # Extract settings from training config
    data_config_path = training_config.get("data_config")
    if not data_config_path:
        raise ValueError(
            "Training config must have a 'data_config' field pointing to the data config YAML. "
            "Example: data_config: data/marvin.yaml"
        )

    model_name_or_path = training_config.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("Training config must have 'model_name_or_path'.")

    trust_remote_code = training_config.get("trust_remote_code", False)
    max_length = training_config.get("max_length", 1024)
    truncation_strategy = training_config.get("truncation_strategy", "truncate")
    dataset_text_field = training_config.get("dataset_text_field", "text")
    eval_split = training_config.get("eval_split", 0.0)
    split_seed = training_config.get("split_seed", 42)

    # Preprocessing options (from data config or training config)
    default_system_message = training_config.get("default_system_message")
    fix_turn_order = training_config.get("fix_turn_order", False)
    fix_turn_order_filler = training_config.get("fix_turn_order_filler", "Let's begin.")
    assistant_only_loss = training_config.get("assistant_only_loss", False)
    last_assistant_only_loss = training_config.get("last_assistant_only_loss", False)
    train_on_incomplete_assistant = training_config.get("train_on_incomplete_assistant", False)
    num_proc = training_config.get("dataset_num_proc")
    chat_template_path = training_config.get("chat_template_path")

    print(BLEND_MESSAGES["start"])

    # Step 1: Load data config
    print(BLEND_MESSAGES["loading_data"].format(path=data_config_path))
    data_config = _load_data_config(data_config_path)
    print(BLEND_MESSAGES["loading_datasets"].format(n=len(data_config.datasets)))

    # Load datasets (NO eval split — we do that after tokenization)
    dataset_dict = get_dataset(data_config)
    dataset = dataset_dict["train"]
    logger.info(f"Loaded {len(dataset)} training examples.")

    # Step 2: Load tokenizer
    print(BLEND_MESSAGES["loading_model"].format(model=model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    # Apply chat template if specified
    if chat_template_path:
        if os.path.isfile(chat_template_path):
            with open(chat_template_path) as f:
                tokenizer.chat_template = f.read()
            logger.info(f"Loaded chat template from {chat_template_path}")
        else:
            # Treat as a model/tokenizer path on Hub
            from transformers import AutoTokenizer as _AT
            template_tokenizer = _AT.from_pretrained(chat_template_path, trust_remote_code=trust_remote_code)
            tokenizer.chat_template = template_tokenizer.chat_template
            logger.info(f"Loaded chat template from tokenizer: {chat_template_path}")

    # Step 3: Preprocess
    print(BLEND_MESSAGES["preprocessing"])
    dataset = preprocess_dataset(
        dataset,
        trainer_type="sft",
        default_system_message=default_system_message,
        fix_turn_order=fix_turn_order,
        fix_turn_order_filler=fix_turn_order_filler,
        num_proc=num_proc,
    )

    # Step 4: Tokenize + truncate
    print(BLEND_MESSAGES["tokenizing"].format(model=model_name_or_path))
    print(BLEND_MESSAGES["truncating"].format(strategy=truncation_strategy, max_length=max_length))
    dataset = tokenize_dataset(
        dataset,
        tokenizer,
        dataset_text_field=dataset_text_field,
        truncation_strategy=truncation_strategy,
        max_length=max_length,
        assistant_only_loss=assistant_only_loss,
        last_assistant_only_loss=last_assistant_only_loss,
        train_on_incomplete_assistant=train_on_incomplete_assistant,
        num_proc=num_proc,
    )

    # Step 5: Eval split (AFTER tokenization + chunking)
    if eval_split and eval_split > 0:
        split_result = dataset.train_test_split(test_size=eval_split, seed=split_seed)
        result = DatasetDict({"train": split_result["train"], "test": split_result["test"]})
        print(BLEND_MESSAGES["splitting"].format(
            eval_split=eval_split,
            eval=len(split_result["test"]),
            train=len(split_result["train"]),
        ))
    else:
        result = DatasetDict({"train": dataset})
        print(BLEND_MESSAGES["no_eval"])

    return result


def save_prepared_dataset(
    dataset_dict: DatasetDict,
    output_dir: str,
    training_config: dict,
) -> None:
    """Save the prepared dataset to disk with metadata."""
    os.makedirs(output_dir, exist_ok=True)

    print(BLEND_MESSAGES["saving"].format(path=output_dir))

    # Save each split as parquet
    for split_name, dataset in dataset_dict.items():
        split_path = os.path.join(output_dir, f"{split_name}.parquet")
        dataset.to_parquet(split_path)
        logger.info(f"Saved {split_name} split to {split_path}")

    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "tokenized": True,
        "model_name_or_path": training_config.get("model_name_or_path"),
        "max_length": training_config.get("max_length"),
        "truncation_strategy": training_config.get("truncation_strategy", "truncate"),
        "eval_split": training_config.get("eval_split", 0.0),
        "split_seed": training_config.get("split_seed", 42),
        "data_config": training_config.get("data_config"),
        "dataset_text_field": training_config.get("dataset_text_field", "text"),
        "preprocessing": {
            "assistant_only_loss": training_config.get("assistant_only_loss", False),
            "last_assistant_only_loss": training_config.get("last_assistant_only_loss", False),
            "train_on_incomplete_assistant": training_config.get("train_on_incomplete_assistant", False),
            "fix_turn_order": training_config.get("fix_turn_order", False),
            "default_system_message": training_config.get("default_system_message"),
        },
        "splits": {
            split_name: len(dataset) for split_name, dataset in dataset_dict.items()
        },
    }

    # Config hash for cache invalidation
    config_str = json.dumps(training_config, sort_keys=True, default=str)
    metadata["config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    metadata_path = os.path.join(output_dir, "blend_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    train_count = metadata["splits"].get("train", 0)
    eval_count = metadata["splits"].get("test", 0)
    total_count = train_count + eval_count
    print(BLEND_MESSAGES["stats"].format(train=train_count, eval=eval_count, total=total_count))
    print(BLEND_MESSAGES["done"].format(path=output_dir))


def main(config_path: str, output_override: Optional[str] = None):
    """Main entry point for the blend command."""
    # Load training config
    training_config = _load_training_config(config_path)

    # Determine output directory
    output_dir = output_override or training_config.get("prepared_dataset") or training_config.get("output_dir")
    if not output_dir:
        raise ValueError(
            "Output directory must be specified via --output, or as 'prepared_dataset' "
            "or 'output_dir' in the training config."
        )

    # Run the pipeline
    dataset_dict = prepare_dataset(training_config)

    # Save
    save_prepared_dataset(dataset_dict, output_dir, training_config)


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    """Create the argument parser for the blend command."""
    if subparsers is not None:
        parser = subparsers.add_parser(
            "blend",
            help="Blend, tokenize, and prepare datasets for training",
        )
    else:
        parser = argparse.ArgumentParser(
            description="Blend, tokenize, and prepare datasets for training",
        )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (must have data_config and model_name_or_path)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (overrides prepared_dataset/output_dir from config)",
    )

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = make_parser()
    args = parser.parse_args()
    main(args.config, args.output)
