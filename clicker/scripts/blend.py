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

Prepares datasets in a model-agnostic way for later tokenization at training time.

Usage:
    clicker blend --config data/my_blend.yaml
    clicker blend --config data/my_blend.yaml --output data/my_blend_prepared
"""

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from datasets import Dataset, DatasetDict

from clicker.data_utils import (
    add_system_message_to_example,
    convert_binary_preference_to_sft,
    convert_preference_to_sft,
    fix_example_turn_order,
    is_binary_preference_dataset,
    is_conversational,
    is_preference_dataset,
    maybe_convert_to_chatml,
)
from clicker.scripts.utils import DataPrepConfig, TrlParser, get_dataset


logger = logging.getLogger(__name__)

# Fun messages for the blend process
BLEND_MESSAGES = {
    "start": "🍹 Starting the blender...",
    "loading": "📦 Gathering ingredients from {n} dataset(s)...",
    "processing": "🔄 Mixing and processing...",
    "sft_convert": "🔀 Converting preference data to SFT format...",
    "fix_turns": "🔧 Fixing conversation turn order...",
    "system_msg": "💬 Adding system messages...",
    "metadata": "🏷️  Adding metadata for training...",
    "saving": "💾 Pouring into container at {path}...",
    "done": "✅ Your data smoothie is ready! Saved to {path}",
    "stats": "📊 Recipe stats: {train} train samples, {eval} eval samples",
}


def preprocess_example_for_sft(
    example: dict,
    config: DataPrepConfig,
) -> Optional[dict]:
    """
    Preprocess a single example for SFT training.

    This handles:
    - Preference → SFT conversion
    - Binary preference → SFT conversion
    - Conversation format normalization (conversations → messages)
    - System message injection
    - Turn order fixing
    - Adding metadata columns for training

    Returns None if the example should be dropped.
    """
    # Check if this is a preference dataset that needs conversion
    if is_preference_dataset(example):
        example = convert_preference_to_sft(example)
    elif is_binary_preference_dataset(example):
        example = convert_binary_preference_to_sft(example)
        if example is None:
            return None  # Drop rejected examples in binary preference

    # Convert legacy conversation format to ChatML
    example = maybe_convert_to_chatml(example)

    # Add system message if needed
    if is_conversational(example):
        # Get per-example system message or fall back to global default
        system_message = example.pop("_system_message", None) or config.default_system_message
        if system_message:
            example = add_system_message_to_example(example, system_message)

        # Fix turn order if requested
        if config.fix_turn_order:
            example = fix_example_turn_order(example, filler_message=config.fix_turn_order_filler)
            if example is None:
                return None  # Dropped due to turn order issues

    return example


def preprocess_example_for_preference(
    example: dict,
    config: DataPrepConfig,
) -> Optional[dict]:
    """
    Preprocess a single example for preference training (DPO, ORPO).

    This handles:
    - Conversation format normalization for chosen/rejected
    - System message injection
    - Turn order fixing
    - Adding metadata columns

    Returns None if the example should be dropped.
    """
    # Convert legacy conversation format to ChatML for chosen/rejected
    if "chosen" in example:
        if isinstance(example["chosen"], list) and example["chosen"]:
            # Check if it's legacy format
            if "from" in example["chosen"][0] or "value" in example["chosen"][0]:
                temp = {"messages": example["chosen"]}
                temp = maybe_convert_to_chatml(temp)
                example["chosen"] = temp.get("messages", example["chosen"])

    if "rejected" in example:
        if isinstance(example["rejected"], list) and example["rejected"]:
            if "from" in example["rejected"][0] or "value" in example["rejected"][0]:
                temp = {"messages": example["rejected"]}
                temp = maybe_convert_to_chatml(temp)
                example["rejected"] = temp.get("messages", example["rejected"])

    # Add system message if needed
    system_message = example.pop("_system_message", None) or config.default_system_message
    if system_message:
        for key in ["chosen", "rejected"]:
            if key in example and isinstance(example[key], list) and example[key]:
                # Check if first message is already a system message
                if example[key][0].get("role") != "system":
                    example[key] = [{"role": "system", "content": system_message}] + example[key]

    # Fix turn order for both chosen and rejected
    if config.fix_turn_order:
        for key in ["chosen", "rejected"]:
            if key in example and isinstance(example[key], list):
                temp = {"messages": example[key]}
                temp = fix_example_turn_order(temp, filler_message=config.fix_turn_order_filler)
                if temp is None:
                    return None  # Drop if turn order can't be fixed
                example[key] = temp["messages"]

    return example


def preprocess_example_for_kto(
    example: dict,
    config: DataPrepConfig,
) -> Optional[dict]:
    """
    Preprocess a single example for KTO training.

    KTO expects completion and label columns. This normalizes the format.
    """
    # Convert legacy conversation format if completion is a list of messages
    if "completion" in example and isinstance(example["completion"], list):
        if example["completion"] and ("from" in example["completion"][0] or "value" in example["completion"][0]):
            temp = {"messages": example["completion"]}
            temp = maybe_convert_to_chatml(temp)
            example["completion"] = temp.get("messages", example["completion"])

    # Add system message to completion if it's conversational
    system_message = example.pop("_system_message", None) or config.default_system_message
    if system_message and "completion" in example and isinstance(example["completion"], list):
        if example["completion"] and example["completion"][0].get("role") != "system":
            example["completion"] = [{"role": "system", "content": system_message}] + example["completion"]

    return example


def add_metadata_columns(
    dataset: Dataset,
    config: DataPrepConfig,
) -> Dataset:
    """Add metadata columns for training-time processing."""

    def add_metadata(example):
        # Only add metadata if it differs from defaults or is explicitly set
        # Per-example truncation strategy takes precedence (already in _truncation_strategy)
        if "_truncation_strategy" not in example or example["_truncation_strategy"] is None:
            example["_truncation_strategy"] = config.truncation_strategy

        # Add loss masking metadata
        if config.assistant_only_loss:
            example["_assistant_only_loss"] = True
        if config.last_assistant_only_loss:
            example["_last_assistant_only_loss"] = True
        if config.train_on_incomplete_assistant:
            example["_train_on_incomplete_assistant"] = True

        return example

    return dataset.map(add_metadata, num_proc=config.num_proc, desc="Adding metadata")


def prepare_dataset(config: DataPrepConfig) -> DatasetDict:
    """
    Prepare a dataset blend according to the configuration.

    This performs all model-agnostic preprocessing:
    - Dataset loading, shuffling, subsetting, eval splitting (via get_dataset)
    - Format conversion (preference → SFT, legacy → ChatML)
    - System message injection
    - Turn order fixing
    - Metadata column addition

    The output can then be tokenized for any model at training time.
    """
    print(BLEND_MESSAGES["start"])
    print(BLEND_MESSAGES["loading"].format(n=len(config.datasets)))

    # Load and combine datasets using existing infrastructure
    dataset_dict = get_dataset(config)

    print(BLEND_MESSAGES["processing"])

    # Determine preprocessing function based on trainer type
    if config.trainer_type == "sft":
        preprocess_fn = lambda ex: preprocess_example_for_sft(ex, config)
        print(BLEND_MESSAGES["sft_convert"])
    elif config.trainer_type in ("dpo", "orpo"):
        preprocess_fn = lambda ex: preprocess_example_for_preference(ex, config)
    elif config.trainer_type == "kto":
        preprocess_fn = lambda ex: preprocess_example_for_kto(ex, config)
    else:
        raise ValueError(f"Unknown trainer_type: {config.trainer_type}")

    # Process each split
    processed_dict = {}
    for split_name, dataset in dataset_dict.items():
        logger.info(f"Processing {split_name} split ({len(dataset)} examples)")

        # Apply preprocessing
        processed = dataset.map(
            preprocess_fn,
            num_proc=config.num_proc,
            desc=f"Preprocessing {split_name}",
            remove_columns=[],  # Keep all columns for now
        )

        # Filter out None results (dropped examples)
        original_len = len(processed)
        processed = processed.filter(
            lambda x: x is not None and (
                "messages" in x or "text" in x or  # SFT
                ("chosen" in x and "rejected" in x) or  # DPO/ORPO
                ("completion" in x and "label" in x)  # KTO
            ),
            num_proc=config.num_proc,
        )
        if len(processed) < original_len:
            logger.info(f"  Dropped {original_len - len(processed)} invalid examples")

        # Add metadata columns
        print(BLEND_MESSAGES["metadata"])
        processed = add_metadata_columns(processed, config)

        processed_dict[split_name] = processed

    return DatasetDict(processed_dict)


def save_prepared_dataset(
    dataset_dict: DatasetDict,
    output_dir: str,
    config: DataPrepConfig,
) -> None:
    """Save the prepared dataset to disk with metadata."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    print(BLEND_MESSAGES["saving"].format(path=output_dir))

    # Save each split as parquet
    for split_name, dataset in dataset_dict.items():
        split_path = os.path.join(output_dir, f"{split_name}.parquet")
        dataset.to_parquet(split_path)
        logger.info(f"Saved {split_name} split to {split_path}")

    # Save metadata/config for reference
    metadata = {
        "created_at": datetime.now().isoformat(),
        "trainer_type": config.trainer_type,
        "num_datasets": len(config.datasets),
        "dataset_paths": [d.path for d in config.datasets],
        "shuffle_seed": config.shuffle_seed,
        "split_seed": config.split_seed,
        "preprocessing": {
            "assistant_only_loss": config.assistant_only_loss,
            "last_assistant_only_loss": config.last_assistant_only_loss,
            "train_on_incomplete_assistant": config.train_on_incomplete_assistant,
            "fix_turn_order": config.fix_turn_order,
            "default_system_message": config.default_system_message,
            "truncation_strategy": config.truncation_strategy,
        },
        "splits": {
            split_name: len(dataset) for split_name, dataset in dataset_dict.items()
        },
    }

    # Create a config hash for cache invalidation
    config_str = json.dumps(asdict(config), sort_keys=True, default=str)
    metadata["config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    metadata_path = os.path.join(output_dir, "blend_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    train_count = metadata["splits"].get("train", 0)
    eval_count = metadata["splits"].get("test", 0)
    print(BLEND_MESSAGES["stats"].format(train=train_count, eval=eval_count))
    print(BLEND_MESSAGES["done"].format(path=output_dir))


def main(config: DataPrepConfig, output_override: Optional[str] = None):
    """Main entry point for the blend command."""
    # Determine output directory
    output_dir = output_override or config.output_dir
    if not output_dir:
        raise ValueError(
            "Output directory must be specified either in config (output_dir) or via --output CLI argument"
        )

    # Prepare the dataset
    dataset_dict = prepare_dataset(config)

    # Save to disk
    save_prepared_dataset(dataset_dict, output_dir, config)


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    """Create the argument parser for the blend command."""
    if subparsers is not None:
        parser = subparsers.add_parser(
            "blend",
            help="Blend and preprocess datasets for training",
            dataclass_types=(DataPrepConfig,),
        )
    else:
        parser = TrlParser(dataclass_types=(DataPrepConfig,))

    # Add output override argument
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (overrides config's output_dir)",
    )

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = make_parser()

    # Parse with remaining strings to capture --output
    args, remaining = parser.parse_args_and_config(return_remaining_strings=True)

    # Handle output override from remaining args
    output_override = None
    if "--output" in remaining:
        idx = remaining.index("--output")
        output_override = remaining[idx + 1]
    elif "-o" in remaining:
        idx = remaining.index("-o")
        output_override = remaining[idx + 1]

    # args is a tuple with DataPrepConfig
    config = args[0]
    main(config, output_override)
