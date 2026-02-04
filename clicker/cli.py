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

import importlib.resources as resources
import os
import sys

import torch
from accelerate import logging
from accelerate.commands.launch import launch_command, launch_command_parser

from .scripts.blend import main as blend_main
from .scripts.blend import make_parser as make_blend_parser
from .scripts.dpo import make_parser as make_dpo_parser
from .scripts.env import print_env
from .scripts.grpo import make_parser as make_grpo_parser
from .scripts.kto import make_parser as make_kto_parser
from .scripts.merge_lora import main as merge_main
from .scripts.merge_lora import make_parser as make_merge_parser
from .scripts.orpo import make_parser as make_orpo_parser
from .scripts.reward import make_parser as make_reward_parser
from .scripts.rloo import make_parser as make_rloo_parser
from .scripts.sft import make_parser as make_sft_parser
from .scripts.train import make_parser as make_train_parser
from .scripts.utils import TrlParser
from .scripts.vllm_serve import main as vllm_serve_main
from .scripts.vllm_serve import make_parser as make_vllm_serve_parser


logger = logging.get_logger(__name__)


def main():
    parser = TrlParser(prog="TRL CLI", usage="trl", allow_abbrev=False)

    # Add the subparsers
    subparsers = parser.add_subparsers(help="available commands", dest="command", parser_class=TrlParser)

    # Add the subparsers for every script
    make_blend_parser(subparsers)
    make_dpo_parser(subparsers)
    subparsers.add_parser("env", help="Print the environment information")
    make_grpo_parser(subparsers)
    make_kto_parser(subparsers)
    make_merge_parser(subparsers)
    make_orpo_parser(subparsers)
    make_reward_parser(subparsers)
    make_rloo_parser(subparsers)
    make_sft_parser(subparsers)
    make_train_parser(subparsers)
    make_vllm_serve_parser(subparsers)

    # Parse the arguments; the remaining ones (`launch_args`) are passed to the 'accelerate launch' subparser.
    # Duplicates may occur if the same argument is provided in both the config file and CLI.
    # For example: launch_args = `["--num_processes", "4", "--num_processes", "8"]`.
    # Deduplication and precedence (CLI over config) are handled later by launch_command_parser.
    args, launch_args = parser.parse_args_and_config(return_remaining_strings=True)

    # Replace `--accelerate_config foo` with `--config_file trl/accelerate_configs/foo.yaml` if it is present in the
    # launch_args. It allows the user to use predefined accelerate configs from the `trl` package.
    if "--accelerate_config" in launch_args:
        # Get the index of the '--accelerate_config' argument and the corresponding config name
        config_index = launch_args.index("--accelerate_config")
        config_name = launch_args[config_index + 1]

        # If the config_name correspond to a path in the filesystem, we don't want to override it
        if os.path.isfile(config_name):
            accelerate_config_path = config_name
        elif resources.files("clicker.accelerate_configs").joinpath(f"{config_name}.yaml").exists():  # pyright: ignore[reportAttributeAccessIssue]
            # Get the predefined accelerate config path from the package resources
            accelerate_config_path = resources.files("clicker.accelerate_configs").joinpath(f"{config_name}.yaml")
        else:
            raise ValueError(
                f"Accelerate config {config_name} is neither a file nor a valid config in the `trl` package. "
                "Please provide a valid config name or a path to a config file."
            )

        # Remove '--accelerate_config' and its corresponding config name
        launch_args.pop(config_index)
        launch_args.pop(config_index)

        # Insert '--config_file' and the absolute path to the front of the list
        launch_args = ["--config_file", str(accelerate_config_path)] + launch_args

    if args.command == "blend":
        # Blend/preprocess datasets - doesn't need accelerate launch
        blend_parser = make_blend_parser()
        blend_args = blend_parser.parse_args(sys.argv[2:])
        blend_main(blend_args.config, blend_args.output)

    elif args.command == "dpo":
        # Get the default args for the launch command
        dpo_training_script = resources.files("clicker.scripts").joinpath("dpo.py")
        args = launch_command_parser().parse_args([str(dpo_training_script)])

        # Feed the args to the launch command
        args.training_script_args = sys.argv[2:]  # remove "trl" and "dpo"
        launch_command(args)  # launch training

    elif args.command == "env":
        print_env()

    elif args.command == "grpo":
        # Get the default args for the launch command
        grpo_training_script = resources.files("clicker.scripts").joinpath("grpo.py")
        args = launch_command_parser().parse_args([str(grpo_training_script)])

        # Feed the args to the launch command
        args.training_script_args = sys.argv[2:]  # remove "trl" and "grpo"
        launch_command(args)  # launch training

    elif args.command == "kto":
        # Get the default args for the launch command
        kto_training_script = resources.files("clicker.scripts").joinpath("kto.py")
        args = launch_command_parser().parse_args([str(kto_training_script)])

        # Feed the args to the launch command
        args.training_script_args = sys.argv[2:]  # remove "trl" and "kto"
        launch_command(args)  # launch training

    elif args.command == "merge":
        # Merge LoRA adapter into base model - doesn't need accelerate launch
        from .scripts.merge_lora import MergeArguments
        merge_parser = make_merge_parser()
        merge_args = merge_parser.parse_args(sys.argv[2:])
        # Convert namespace to MergeArguments
        merge_arguments = MergeArguments(
            config=merge_args.config,
            base_model=merge_args.base_model,
            lora_path=merge_args.lora_path,
            output_path=merge_args.output_path,
            weight=merge_args.weight,
            no_gpu=merge_args.no_gpu,
        )
        merge_main(merge_arguments)

    elif args.command == "orpo":
        # Get the default args for the launch command
        orpo_training_script = resources.files("clicker.scripts").joinpath("orpo.py")
        args = launch_command_parser().parse_args([str(orpo_training_script)])

        # Feed the args to the launch command
        args.training_script_args = sys.argv[2:]  # remove "clicker" and "orpo"
        launch_command(args)  # launch training

    elif args.command == "reward":
        # Get the default args for the launch command
        reward_training_script = resources.files("clicker.scripts").joinpath("reward.py")
        args = launch_command_parser().parse_args([str(reward_training_script)])

        # Feed the args to the launch command
        args.training_script_args = sys.argv[2:]  # remove "trl" and "reward"
        launch_command(args)  # launch training

    elif args.command == "rloo":
        # Get the default args for the launch command
        rloo_training_script = resources.files("clicker.scripts").joinpath("rloo.py")
        args = launch_command_parser().parse_args([str(rloo_training_script)])

        # Feed the args to the launch command
        args.training_script_args = sys.argv[2:]  # remove "trl" and "rloo"
        launch_command(args)  # launch training

    elif args.command == "sft":
        # Get the path to the training script
        sft_training_script = resources.files("clicker.scripts").joinpath("sft.py")

        # This simulates running: `accelerate launch <launch args> sft.py <training script args>`.
        # Note that the training script args may include launch-related arguments (e.g., `--num_processes`),
        # but we rely on the script to ignore any that don't apply to it.
        training_script_args = sys.argv[2:]  # Remove "trl" and "sft"
        args = launch_command_parser().parse_args(launch_args + [str(sft_training_script)] + training_script_args)
        launch_command(args)  # launch training

    elif args.command == "train":
        # Unified train command - dispatches based on trainer type in config
        train_script = resources.files("clicker.scripts").joinpath("train.py")
        training_script_args = sys.argv[2:]  # Remove "clicker" and "train"
        args = launch_command_parser().parse_args(launch_args + [str(train_script)] + training_script_args)
        launch_command(args)

    elif args.command == "vllm-serve":
        (script_args,) = parser.parse_args_and_config()

        # Known issue: Using DeepSpeed with tensor_parallel_size=1 and data_parallel_size>1 may cause a crash when
        # launched via the CLI. Suggest running the module directly.
        # More information: https://github.com/vllm-project/vllm/issues/17079
        if script_args.tensor_parallel_size == 1 and script_args.data_parallel_size > 1 and torch.cuda.is_available():
            logger.warning(
                "Detected configuration: tensor_parallel_size=1 and data_parallel_size>1. This setup is known to "
                "cause a crash when using the `trl vllm-serve` CLI entry point. As a workaround, please run the "
                "server using the module path instead: `python -m clicker.scripts.vllm_serve`",
            )

        vllm_serve_main(script_args)


if __name__ == "__main__":
    main()
