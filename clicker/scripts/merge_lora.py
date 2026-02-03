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
Memory-efficient LoRA merging script for clicker.

Usage:
    # From config file (reads model_name_or_path and output_dir):
    python -m clicker.cli merge --config configs/my-training.yaml

    # With explicit paths (overrides config):
    python -m clicker.cli merge --config configs/my-training.yaml --lora_path ./my-lora --output_path ./merged

    # Without config (all paths explicit):
    python -m clicker.cli merge --base_model Qwen/Qwen3-0.6B --lora_path ./my-lora --output_path ./merged

    # Force CPU merging:
    python -m clicker.cli merge --config configs/my-training.yaml --no-gpu
"""

import argparse
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import safetensors
import safetensors.torch
import torch
from huggingface_hub import snapshot_download

try:
    from logbar import LogBar
    log = LogBar.shared()
except ImportError:
    # Fallback if logbar not installed
    class FakeLog:
        def info(self, msg): print(f"[INFO] {msg}")
        def debug(self, msg): print(f"[DEBUG] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
        def pb(self, iterable, **kwargs):
            return iterable
    log = FakeLog()

import peft


@dataclass
class MergeArguments:
    """Arguments for LoRA merging."""
    config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to clicker training config YAML file. Reads model_name_or_path and output_dir."}
    )
    base_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path or HuggingFace repo ID of the base model. Overrides config if provided."}
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path or HuggingFace repo ID of the LoRA adapter. Defaults to output_dir from config."}
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for merged model. Defaults to {output_dir}-merged."}
    )
    weight: float = field(
        default=1.0,
        metadata={"help": "Multiplier for LoRA weight (applied on top of alpha/r scaling). Default 1.0."}
    )
    no_gpu: bool = field(
        default=False,
        metadata={"help": "Force CPU merging (useful when GPUs are busy)."}
    )


def load_config(config_path: str) -> dict:
    """Load a clicker YAML config file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_path(path_or_repo_id: str) -> Path:
    """
    Resolves a path or Hugging Face Hub repo ID to a local directory path.
    If it's a repo ID, it downloads the repo and returns the cached path.
    If it's a local path, it returns the Path object.
    """
    path = Path(path_or_repo_id)
    if path.is_dir():
        log.info(f"'{path_or_repo_id}' is a local directory. Using it directly.")
        return path
    # A simple check for a repo ID. Hugging Face repo IDs contain a '/'.
    elif '/' in path_or_repo_id and not path.exists():
        log.info(f"'{path_or_repo_id}' is not a local directory. Assuming it's a Hugging Face repo ID and downloading.")
        try:
            downloaded_path = snapshot_download(
                repo_id=path_or_repo_id,
                local_dir_use_symlinks=False,
                repo_type='model',
            )
            log.info(f"Repo '{path_or_repo_id}' downloaded to: {downloaded_path}")
            return Path(downloaded_path)
        except Exception as e:
            log.error(f"Failed to download repo '{path_or_repo_id}'. Error: {e}")
            raise SystemExit(1)
    else:
        log.error(f"Input '{path_or_repo_id}' is not a valid local directory or a recognizable Hugging Face repo ID.")
        raise SystemExit(1)


def merge_lora(base_model_path: str, lora_path: str, output_path: str, use_gpu: bool = True, weight: float = 1.0):
    """
    Merge a LoRA adapter into a base model using memory-efficient shard-by-shard processing.

    Args:
        base_model_path: Path or HF repo ID of the base model
        lora_path: Path or HF repo ID of the LoRA adapter
        output_path: Output directory for merged model
        use_gpu: Whether to use GPU for merging (faster but requires VRAM)
        weight: Multiplier for LoRA weight (applied on top of alpha/r scaling)
    """
    # Resolve paths
    log.info("Resolving base model and LoRA paths...")
    input_path = resolve_path(base_model_path)
    lora_path = resolve_path(lora_path)
    output_path = Path(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Load LoRA config
    lora_config = peft.LoraConfig.from_json_file(lora_path / 'adapter_config.json')
    try:
        if lora_config.get("use_rslora", False):
            base_scale = lora_config['lora_alpha'] / math.sqrt(lora_config['r'])
        else:
            base_scale = lora_config['lora_alpha'] / lora_config['r']
    except Exception:
        base_scale = lora_config['lora_alpha'] / lora_config['r']

    # Apply user weight multiplier
    scale = base_scale * weight
    log.debug(f"LoRA base scale (alpha/r): x{base_scale}")
    if weight != 1.0:
        log.info(f"Applying weight multiplier: x{weight} -> final scale: x{scale}")

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    log.info(f"Using device: {device}")

    log.info('Loading LoRA model...')

    # Check if we have adapter_model.bin or adapter_model.safetensors
    if (lora_path / 'adapter_model.safetensors').exists():
        lora_state = safetensors.torch.load_file(lora_path / 'adapter_model.safetensors')
        if device == 'cuda':
            # Move mapped entries to cuda
            for key, value in lora_state.items():
                lora_state[key] = value.to('cuda')
    else:
        lora_state = torch.load(lora_path / 'adapter_model.bin', map_location=device, weights_only=True)

    def find_lora_weights(key):
        lora_A = None
        lora_B = None
        # Adjust for PEFT's potential prefixes
        lora_key_prefix = 'base_model.model.'
        for lora_key, lora_weight in lora_state.items():
            if lora_key.startswith(lora_key_prefix):
                lora_key_unprefixed = lora_key[len(lora_key_prefix):]
            else:
                lora_key_unprefixed = lora_key

            if key.strip('.weight') in lora_key_unprefixed:
                if 'lora_A' in lora_key:
                    lora_A = lora_weight
                elif 'lora_B' in lora_key:
                    lora_B = lora_weight
        if lora_A is not None and lora_B is not None:
            return lora_A, lora_B
        return None, None

    shards = list(input_path.glob('model*.safetensors'))

    log.info('Copying non-model files to output')
    for filepath in input_path.glob('*'):
        if filepath in shards:
            continue
        filepath = Path(filepath)
        if filepath.is_dir():
            continue
        if filepath.name.startswith('.'):
            continue
        if filepath.suffix == '.gguf':
            continue
        if filepath.suffix == '.md':
            continue
        if filepath.suffix == '.safetensors' and 'model' in filepath.name:
            continue
        log.debug(f'copying {filepath.name} to output')
        shutil.copy(filepath, output_path)

    log.info(f'Merging {len(shards)} shards...')
    found = 0
    for shard in shards:
        log.info(f'Processing {shard.name}...')
        tensors = {}
        with safetensors.safe_open(shard, framework='pt', device=device) as f:
            metadata = f.metadata()
            for key in f.keys():
                tensor = f.get_tensor(key)
                # PEFT models often have a 'language_model.' prefix that base models don't
                lora_key_lookup = re.sub(r'^(model|language_model)\.', '', key)

                lora_A, lora_B = find_lora_weights(lora_key_lookup)
                if lora_A is not None:
                    found += 1
                    log.debug(f'found LoRA weights for {key}')
                    old_type = tensor.dtype
                    tensor = tensor.to(torch.float32)
                    tensor += scale * lora_B.to(torch.float32) @ lora_A.to(torch.float32)
                    tensor = tensor.to(old_type)
                tensors[key] = tensor
            safetensors.torch.save_file(tensors, output_path / shard.name, metadata=metadata)

    log.info(f"Applied LoRA to {found} tensors.")

    # Copy tokenizer files from the LoRA repo
    log.info(f"Attempting to copy tokenizer from {lora_path} to {output_path}")
    for tokenizer_file in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.model']:
        try:
            shutil.copy(lora_path / tokenizer_file, output_path)
            log.debug(f"Copied {tokenizer_file}")
        except FileNotFoundError:
            log.debug(f"{tokenizer_file} not found in LoRA directory, skipping.")

    log.info(f"Merge complete. Output saved to: {output_path}")


def main(args: Optional[MergeArguments] = None):
    """Main entry point for merge command."""
    if args is None:
        # Parse from command line
        parser = make_parser()
        args = parser.parse_args()

    # Resolve paths from config and/or CLI arguments
    base_model = args.base_model
    lora_path = args.lora_path
    output_path = args.output_path

    if args.config:
        config = load_config(args.config)

        # Use config values as defaults, CLI args override
        if base_model is None:
            base_model = config.get('model_name_or_path')
        if lora_path is None:
            lora_path = config.get('output_dir')
        if output_path is None:
            output_dir = config.get('output_dir', './output')
            output_path = f"{output_dir}-merged"

    # Validate required arguments
    if not base_model:
        raise ValueError("base_model is required. Provide via --base_model or in config file (model_name_or_path).")
    if not lora_path:
        raise ValueError("lora_path is required. Provide via --lora_path or in config file (output_dir).")
    if not output_path:
        raise ValueError("output_path is required. Provide via --output_path.")

    log.info(f"Base model: {base_model}")
    log.info(f"LoRA path: {lora_path}")
    log.info(f"Output path: {output_path}")

    use_gpu = not args.no_gpu
    merge_lora(base_model, lora_path, output_path, use_gpu=use_gpu, weight=args.weight)


def make_parser(subparsers=None):
    """Create argument parser for merge command."""
    if subparsers is not None:
        parser = subparsers.add_parser("merge", help="Merge a LoRA adapter into a base model")
    else:
        parser = argparse.ArgumentParser(
            description="Merge a LoRA adapter into a base model. Can read paths from a clicker config file."
        )

    parser.add_argument('--config', type=str, default=None,
                        help='Path to clicker training config YAML file. Reads model_name_or_path and output_dir.')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Path or HuggingFace repo ID of the base model. Overrides config if provided.')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path or HuggingFace repo ID of the LoRA adapter. Defaults to output_dir from config.')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output directory for merged model. Defaults to {output_dir}-merged.')
    parser.add_argument('--weight', type=float, default=1.0,
                        help='Multiplier for LoRA weight (applied on top of alpha/r scaling). Default 1.0.')
    parser.add_argument('--no-gpu', action='store_true', dest='no_gpu',
                        help='Force CPU merging (useful when GPUs are busy).')

    return parser


if __name__ == "__main__":
    main()
