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
Unified training script that dispatches to the appropriate trainer based on config.

Usage:
    clicker train --config config.yaml

The config file must include a `trainer` field specifying which trainer to use:
    - sft
    - dpo
    - orpo
    - grpo
    - kto
    - rloo
    - reward

Example config.yaml:
```yaml
trainer: sft
model_name_or_path: Qwen/Qwen2-0.5B
dataset_name: trl-lib/Capybara
output_dir: ./output
num_train_epochs: 1
per_device_train_batch_size: 2
```
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

import yaml

# Handle both relative and absolute imports for when run as script vs module
try:
    from .utils import TrlParser
except ImportError:
    from clicker.scripts.utils import TrlParser


# Valid trainer types - these are the ones currently wired up
TRAINER_TYPES = Literal["sft", "dpo", "orpo", "grpo", "kto", "rloo", "reward"]

TRAINER_HELP = """
Trainer type to use. Options:
  sft    - Supervised Fine-Tuning
  dpo    - Direct Preference Optimization
  orpo   - Odds Ratio Preference Optimization
  grpo   - Group Relative Policy Optimization
  kto    - Kahneman-Tversky Optimization
  rloo   - REINFORCE Leave-One-Out
  reward - Reward model training
"""


@dataclass
class TrainConfig:
    """Minimal config just to extract trainer type."""

    trainer: Optional[str] = field(
        default=None,
        metadata={"help": TRAINER_HELP},
    )
    config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to YAML config file"},
    )


def get_trainer_type_from_config(config_path: str) -> str:
    """Extract trainer type from config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    trainer = config.get("trainer")
    if not trainer:
        raise ValueError(
            f"Config file {config_path} must specify a 'trainer' field. "
            f"Valid options: sft, dpo, orpo, grpo, kto, rloo, reward"
        )

    valid_trainers = ["sft", "dpo", "orpo", "grpo", "kto", "rloo", "reward"]
    if trainer not in valid_trainers:
        raise ValueError(
            f"Unknown trainer type: {trainer}. "
            f"Valid options: {', '.join(valid_trainers)}"
        )

    return trainer


def main():
    """
    Unified train command that dispatches to appropriate trainer.

    Reads trainer type from config, then delegates to the specific trainer's main().
    """
    # First, we need to peek at the config to get trainer type
    # Use a simple argparse just to get --config
    peek_parser = argparse.ArgumentParser(add_help=False)
    peek_parser.add_argument("--config", type=str, required=False)
    peek_parser.add_argument("--trainer", type=str, required=False)
    peek_args, remaining = peek_parser.parse_known_args()

    # Determine trainer type
    trainer_type = None

    if peek_args.trainer:
        trainer_type = peek_args.trainer
    elif peek_args.config:
        trainer_type = get_trainer_type_from_config(peek_args.config)
    else:
        # No config or trainer specified - show help
        print("Error: Must specify --config <path> or --trainer <type>")
        print(TRAINER_HELP)
        sys.exit(1)

    # Now dispatch to the appropriate trainer
    # Handle both relative and absolute imports
    try:
        if trainer_type == "sft":
            from .sft import main as trainer_main, make_parser
        elif trainer_type == "dpo":
            from .dpo import main as trainer_main, make_parser
        elif trainer_type == "orpo":
            from .orpo import main as trainer_main, make_parser
        elif trainer_type == "grpo":
            from .grpo import main as trainer_main, make_parser
        elif trainer_type == "kto":
            from .kto import main as trainer_main, make_parser
        elif trainer_type == "rloo":
            from .rloo import main as trainer_main, make_parser
        elif trainer_type == "reward":
            from .reward import main as trainer_main, make_parser
        else:
            raise ValueError(f"Unknown trainer type: {trainer_type}")
    except ImportError:
        # Fallback to absolute imports when run as script
        if trainer_type == "sft":
            from clicker.scripts.sft import main as trainer_main, make_parser
        elif trainer_type == "dpo":
            from clicker.scripts.dpo import main as trainer_main, make_parser
        elif trainer_type == "orpo":
            from clicker.scripts.orpo import main as trainer_main, make_parser
        elif trainer_type == "grpo":
            from clicker.scripts.grpo import main as trainer_main, make_parser
        elif trainer_type == "kto":
            from clicker.scripts.kto import main as trainer_main, make_parser
        elif trainer_type == "rloo":
            from clicker.scripts.rloo import main as trainer_main, make_parser
        elif trainer_type == "reward":
            from clicker.scripts.reward import main as trainer_main, make_parser
        else:
            raise ValueError(f"Unknown trainer type: {trainer_type}")

    # Parse with the trainer-specific parser
    parser = make_parser()

    # Re-inject the args (parse_args_and_config will handle --config)
    parsed = parser.parse_args_and_config(return_remaining_strings=True)

    # The parsed tuple has format: (script_args, training_args, model_args, dataset_args, remaining)
    # Call the trainer's main with the appropriate args (excluding remaining strings)
    trainer_main(*parsed[:-1])


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    """Create parser for the train command."""
    if subparsers is not None:
        parser = subparsers.add_parser(
            "train",
            help="Run training with trainer type specified in config",
        )
        parser.add_argument("--config", type=str, help="Path to YAML config file")
        parser.add_argument("--trainer", type=str, help=TRAINER_HELP)
    else:
        parser = argparse.ArgumentParser(
            description="Unified training command - reads trainer type from config"
        )
        parser.add_argument("--config", type=str, help="Path to YAML config file")
        parser.add_argument("--trainer", type=str, help=TRAINER_HELP)
    return parser


if __name__ == "__main__":
    main()
