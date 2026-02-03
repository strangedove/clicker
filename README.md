# Clicker

A streamlined CLI tool for LLM post-training. Supports SFT, DPO, ORPO, KTO, GRPO, RLOO, and reward model training.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [Configuration Reference](#configuration-reference)
  - [Common Parameters](#common-parameters)
  - [SFT (Supervised Fine-Tuning)](#sft-supervised-fine-tuning)
  - [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
  - [ORPO (Odds Ratio Preference Optimization)](#orpo-odds-ratio-preference-optimization)
  - [KTO (Kahneman-Tversky Optimization)](#kto-kahneman-tversky-optimization)
  - [GRPO (Group Relative Policy Optimization)](#grpo-group-relative-policy-optimization)
  - [RLOO (REINFORCE Leave-One-Out)](#rloo-reinforce-leave-one-out)
  - [Reward Model Training](#reward-model-training)
- [Dataset Configuration](#dataset-configuration)
- [Dataset Blending](#dataset-blending)
- [LoRA Configuration](#lora-configuration)
- [Reward Functions](#reward-functions)
- [Distributed Training](#distributed-training)

---

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-org/clicker.git
cd clicker

# Install with pip
pip install -e .

# Or with uv (recommended)
uv sync
```

### Optional Dependencies

```bash
# Cut Cross-Entropy (memory-efficient SFT loss)
pip install -e ".[cce]"

# Liger Kernel (memory-efficient kernels)
pip install -e ".[liger]"

# DeepSpeed support
pip install -e ".[deepspeed]"

# Quantization (bitsandbytes)
pip install -e ".[quantization]"

# vLLM serving
pip install -e ".[vllm]"

# All dev dependencies
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA-capable GPU (recommended)

---

## Quick Start

```bash
# Run training with a YAML config
clicker train --config configs/sft.yaml

# Or use specific trainer commands
clicker sft --config configs/sft.yaml
clicker dpo --config configs/dpo.yaml
clicker orpo --config configs/orpo.yaml

# Merge LoRA adapter after training
clicker merge --config configs/sft.yaml
```

---

## CLI Commands

### `clicker train`

Unified training command that reads the trainer type from config.

```bash
clicker train --config <config.yaml> [--trainer <type>]
```

| Option | Description |
|--------|-------------|
| `--config` | Path to YAML config file |
| `--trainer` | Override trainer type: `sft`, `dpo`, `orpo`, `kto`, `grpo`, `rloo`, `reward` |

### `clicker sft`

Run supervised fine-tuning.

```bash
clicker sft --config <config.yaml>
```

### `clicker dpo`

Run Direct Preference Optimization training.

```bash
clicker dpo --config <config.yaml>
```

### `clicker orpo`

Run Odds Ratio Preference Optimization training.

```bash
clicker orpo --config <config.yaml>
```

### `clicker kto`

Run Kahneman-Tversky Optimization training.

```bash
clicker kto --config <config.yaml>
```

### `clicker grpo`

Run Group Relative Policy Optimization training.

```bash
clicker grpo --config <config.yaml>
```

### `clicker rloo`

Run REINFORCE Leave-One-Out training.

```bash
clicker rloo --config <config.yaml>
```

### `clicker reward`

Train a reward model.

```bash
clicker reward --config <config.yaml>
```

### `clicker merge`

Merge a LoRA adapter into the base model.

```bash
clicker merge --config <config.yaml> [options]
```

| Option | Description |
|--------|-------------|
| `--config` | Training config (reads `model_name_or_path` and `output_dir`) |
| `--base_model` | Override base model path |
| `--lora_path` | Override LoRA adapter path |
| `--output_path` | Override output directory (default: `{output_dir}-merged`) |
| `--weight` | LoRA weight multiplier (default: 1.0) |
| `--no-gpu` | Force CPU merging |

### `clicker blend`

Blend and preprocess datasets for training. Creates a prepared dataset that can be reused across different training runs and models.

```bash
clicker blend --config <data_config.yaml> [--output <output_dir>]
```

| Option | Description |
|--------|-------------|
| `--config` | Path to data blend config file |
| `--output`, `-o` | Output directory (overrides config's `output_dir`) |

See [Dataset Blending](#dataset-blending) for detailed documentation.

### `clicker env`

Print environment information for debugging.

```bash
clicker env
```

### `clicker vllm-serve`

Start a vLLM inference server.

```bash
clicker vllm-serve --model <model_path> [options]
```

---

## Configuration Reference

All trainers share common parameters. Trainer-specific parameters are listed in each section.

### Common Parameters

#### Model Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | str | required | HuggingFace model ID or local path |
| `trust_remote_code` | bool | `false` | Trust remote code in model repo |
| `attn_implementation` | str | `null` | Attention implementation: `null`, `"flash_attention_2"`, `"sdpa"` |
| `dtype` | str | `"auto"` | Model dtype: `"auto"`, `"bfloat16"`, `"float16"`, `"float32"` |

#### Dataset Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | required | HuggingFace dataset ID or local path |
| `dataset_train_split` | str | `"train"` | Training split name |
| `dataset_test_split` | str | `"test"` | Evaluation split name |
| `dataset_config` | str | `null` | Dataset configuration name |

**Multiple datasets:**

```yaml
datasets:
  - path: dataset1
    split: train
  - path: dataset2
    split: train
test_split_size: 0.05  # Auto-split if no test set
```

#### Training Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | required | Output directory for checkpoints |
| `num_train_epochs` | int | `3` | Number of training epochs |
| `max_steps` | int | `-1` | Max training steps (overrides epochs if > 0) |
| `per_device_train_batch_size` | int | `8` | Batch size per GPU for training |
| `per_device_eval_batch_size` | int | `8` | Batch size per GPU for evaluation |
| `gradient_accumulation_steps` | int | `1` | Gradient accumulation steps |
| `gradient_checkpointing` | bool | `false` | Enable gradient checkpointing (saves memory) |
| `learning_rate` | float | `2e-5` | Initial learning rate |
| `lr_scheduler_type` | str | `"linear"` | LR scheduler: `"linear"`, `"cosine"`, `"constant"`, etc. |
| `warmup_ratio` | float | `0.0` | Warmup ratio (deprecated, use `warmup_steps`) |
| `warmup_steps` | int | `0` | Number of warmup steps |
| `weight_decay` | float | `0.0` | Weight decay for AdamW |
| `max_grad_norm` | float | `1.0` | Max gradient norm for clipping |
| `optim` | str | `"adamw_torch"` | Optimizer type |

#### Evaluation & Saving

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_strategy` | str | `"no"` | Evaluation strategy: `"no"`, `"steps"`, `"epoch"` |
| `eval_steps` | int | `500` | Evaluate every N steps |
| `save_strategy` | str | `"steps"` | Save strategy: `"no"`, `"steps"`, `"epoch"` |
| `save_steps` | int | `500` | Save checkpoint every N steps |
| `save_total_limit` | int | `null` | Max checkpoints to keep |

#### Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logging_steps` | int | `10` | Log every N steps |
| `report_to` | list | `[]` | Reporting integrations: `["wandb"]`, `["tensorboard"]`, etc. |
| `run_name` | str | `null` | Run name for logging |

#### Hardware

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bf16` | bool | `null` | Use bfloat16 mixed precision |
| `fp16` | bool | `false` | Use float16 mixed precision |
| `dataloader_num_workers` | int | `0` | Number of dataloader workers |

---

### SFT (Supervised Fine-Tuning)

Train a model to follow instructions using prompt-completion pairs.

**Dataset format:** Conversational (messages) or text field.

```yaml
trainer: sft

# SFT-specific parameters
max_length: 2048              # Max sequence length
packing: false                # Pack multiple examples per sequence
padding_free: false           # Padding-free training (requires flash attention)
dataset_text_field: null      # Text field name (auto-detected if null)
use_cce: false                # Use Cut Cross-Entropy loss (memory efficient)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_length` | int | `1024` | Maximum sequence length |
| `truncation_strategy` | str | `"truncate"` | How to handle long samples (see below) |
| `packing` | bool | `false` | Pack short examples together for efficiency |
| `padding_free` | bool | `false` | Padding-free training (requires flash_attention_2) |
| `dataset_text_field` | str | `null` | Column name for text data |
| `use_cce` | bool | `false` | Use Cut Cross-Entropy loss (significant VRAM reduction) |
| `completion_only_loss` | bool | `false` | Only compute loss on completions |
| `assistant_only_loss` | bool | `false` | Only compute loss on assistant turns |
| `last_assistant_only_loss` | bool | `false` | Only compute loss on the last assistant turn (multi-turn) |
| `train_on_incomplete_assistant` | bool | `false` | Don't add EOS on truncated assistant responses |
| `fix_turn_order` | bool | `false` | Fix conversation turn order for strict models |
| `fix_turn_order_filler` | str | `"Let's begin."` | Filler message when conversation starts with assistant |
| `default_system_message` | str | `null` | Default system message to add to conversations without one |
| `prepared_dataset` | str | `null` | Path to prepared dataset from `clicker blend` |
| `tokenized_cache_dir` | str | `null` | Custom cache directory for tokenized datasets |
| `force_retokenize` | bool | `false` | Ignore cache and re-tokenize prepared dataset |

**Truncation strategies:**

| Strategy | Applies To | Behavior |
|----------|-----------|----------|
| `truncate` | All | Cut off at max_length, no EOS on truncated samples (default) |
| `drop` | All | Filter out samples exceeding max_length |
| `split` | Text/CPT | Split into chunks (first gets BOS, last gets EOS) |
| `truncate_turns` | Conversational | Drop turn pairs from end, preserve system message |

```yaml
# Example: Drop samples that exceed max_length
truncation_strategy: drop
max_length: 2048

# Example: Split long text into chunks for CPT
truncation_strategy: split
max_length: 4096

# Example: Truncate by complete turns for multi-turn data
truncation_strategy: truncate_turns
max_length: 2048
```

Per-dataset `truncation_strategy` can be set in the dataset mixer to override the global default.

---

### DPO (Direct Preference Optimization)

Train from preference pairs (chosen vs rejected responses).

**Dataset format:** Must have `chosen` and `rejected` columns (or `prompt`/`chosen`/`rejected`).

```yaml
trainer: dpo

# DPO-specific parameters
beta: 0.1                     # KL penalty coefficient
max_length: 1024              # Max sequence length
max_prompt_length: 512        # Max prompt length
loss_type: sigmoid            # Loss type
remove_unused_columns: false  # Important for DPO!
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | `0.1` | KL divergence penalty coefficient |
| `max_length` | int | `1024` | Maximum sequence length |
| `max_prompt_length` | int | `512` | Maximum prompt length |
| `loss_type` | str | `"sigmoid"` | Loss type: `"sigmoid"`, `"hinge"`, `"ipo"`, `"robust"` |
| `label_smoothing` | float | `0.0` | Label smoothing factor |
| `generate_during_eval` | bool | `false` | Generate samples during evaluation |
| `train_on_incomplete_assistant` | bool | `false` | Don't add EOS on truncated responses |

**Important:** Always set `remove_unused_columns: false` for DPO.

---

### ORPO (Odds Ratio Preference Optimization)

Combines SFT and preference optimization without a reference model.

**Dataset format:** Same as DPO (chosen/rejected pairs).

```yaml
trainer: orpo

# ORPO-specific parameters
beta: 0.1                     # Odds ratio weight (lambda in paper)
max_length: 1024
max_prompt_length: 512
remove_unused_columns: false
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | `0.1` | Odds ratio weight (lambda in paper) |
| `max_length` | int | `1024` | Maximum sequence length |
| `max_prompt_length` | int | `512` | Maximum prompt length |
| `train_on_incomplete_assistant` | bool | `false` | Don't add EOS on truncated responses |

**Advantages over DPO:**
- No reference model needed (saves ~50% memory)
- Higher learning rates work better (e.g., `5e-6` vs `5e-7`)
- Can use Liger kernel for additional memory savings (`use_liger_loss: true`)

---

### KTO (Kahneman-Tversky Optimization)

Train from binary feedback (good/bad) rather than pairwise preferences.

**Dataset format:** Must have `completion` and `label` (bool) columns.

```yaml
trainer: kto

# KTO-specific parameters
beta: 0.1                     # KL penalty coefficient
max_length: 1024
max_prompt_length: 512
desirable_weight: 1.0         # Weight for positive examples
undesirable_weight: 1.0       # Weight for negative examples
remove_unused_columns: false
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | `0.1` | KL divergence penalty coefficient |
| `max_length` | int | `1024` | Maximum sequence length |
| `max_prompt_length` | int | `512` | Maximum prompt length |
| `desirable_weight` | float | `1.0` | Weight for desirable (good) examples |
| `undesirable_weight` | float | `1.0` | Weight for undesirable (bad) examples |
| `train_on_incomplete_assistant` | bool | `false` | Don't add EOS on truncated responses |

---

### GRPO (Group Relative Policy Optimization)

Online RL method that generates completions and learns from relative rewards.

**Dataset format:** Must have `prompt` column.

```yaml
trainer: grpo

# GRPO-specific parameters
num_generations: 4            # Completions per prompt
max_completion_length: 256    # Max generated tokens
max_prompt_length: 512
temperature: 0.7
top_p: 0.9
beta: 0.04                    # KL penalty coefficient

# Reward (choose one)
reward_model_name_or_path: null
reward_funcs:
  - think_format_reward
  - get_soft_overlong_punishment
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_generations` | int | `4` | Number of completions to generate per prompt |
| `max_completion_length` | int | `256` | Maximum tokens to generate |
| `max_prompt_length` | int | `512` | Maximum prompt length |
| `temperature` | float | `0.7` | Sampling temperature |
| `top_p` | float | `0.9` | Top-p (nucleus) sampling |
| `beta` | float | `0.04` | KL penalty coefficient |
| `reward_model_name_or_path` | str | `null` | Path to reward model |
| `reward_funcs` | list | `[]` | Reward functions to use |

---

### RLOO (REINFORCE Leave-One-Out)

Online RL similar to GRPO but uses leave-one-out baselines.

**Dataset format:** Must have `prompt` column.

```yaml
trainer: rloo

# RLOO-specific parameters
num_generations: 4            # Completions per prompt (for leave-one-out)
max_completion_length: 256
max_prompt_length: 512
temperature: 0.7
top_p: 0.9
beta: 0.05                    # KL penalty coefficient (was kl_coef)

# Reward
reward_model_name_or_path: null
reward_funcs:
  - get_soft_overlong_punishment
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_generations` | int | `4` | Completions per prompt |
| `max_completion_length` | int | `256` | Maximum tokens to generate |
| `max_prompt_length` | int | `512` | Maximum prompt length |
| `temperature` | float | `0.7` | Sampling temperature |
| `top_p` | float | `0.9` | Top-p sampling |
| `beta` | float | `0.05` | KL penalty coefficient |
| `reward_model_name_or_path` | str | `null` | Path to reward model |
| `reward_funcs` | list | `[]` | Reward functions to use |

---

### Reward Model Training

Train a reward model from preference data.

**Dataset format:** Same as DPO (chosen/rejected pairs).

```yaml
trainer: reward

# Reward-specific parameters
max_length: 1024
center_rewards_coefficient: null  # For reward centering
remove_unused_columns: false
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_length` | int | `1024` | Maximum sequence length |
| `center_rewards_coefficient` | float | `null` | Coefficient for reward centering |

---

## Dataset Configuration

Clicker supports various dataset formats and provides flexible configuration options.

### Dataset Formats

Clicker automatically detects and handles these dataset formats:

| Format | Required Columns | Used By |
|--------|------------------|---------|
| **Language Modeling** | `messages` | SFT |
| **Text** | `text` (or custom via `dataset_text_field`) | SFT |
| **Prompt-Completion** | `prompt`, `completion` | SFT |
| **Preference** | `prompt`, `chosen`, `rejected` | DPO, ORPO, Reward, SFT* |
| **Implicit Preference** | `chosen`, `rejected` | DPO, ORPO, Reward, SFT* |
| **Binary Feedback** | `prompt`, `completion`, `label` | KTO, SFT* |
| **Prompt-Only** | `prompt` | GRPO, RLOO |

*\*SFT auto-converts preference datasets - see [Using Preference Datasets for SFT](#using-preference-datasets-for-sft).*

### Conversational Format

Clicker supports two conversational formats and auto-converts between them:

**Standard format** (`messages` with `role`/`content`):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

**Legacy format** (`conversations` with `from`/`value`):

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "Hello!"},
    {"from": "gpt", "value": "Hi there!"}
  ]
}
```

Clicker automatically:
- Detects conversational format and applies the model's chat template
- Converts `conversations` → `messages`
- Converts `from`/`value` → `role`/`content`
- Maps role names: `human` → `user`, `gpt` → `assistant`

This means datasets using the common ShareGPT/Vicuna format work out of the box.

### Single Dataset

```yaml
# Simple single dataset
dataset_name: trl-lib/Capybara
dataset_train_split: train
dataset_test_split: test
dataset_config: null          # Optional: dataset configuration name
```

### Multiple Datasets (Dataset Mixer)

Combine multiple datasets with the `datasets` field:

```yaml
# Global data processing settings
shuffle_datasets: true        # Shuffle each dataset before subsetting
shuffle_combined: true        # Shuffle final combined dataset
eval_split: 0.05              # 5% eval from each dataset
shuffle_seed: 42
split_seed: 42

datasets:
  - path: nbeerbower/gutenberg-moderne-dpo
    split: train
    columns: [prompt, chosen, rejected]
    subset: 5000              # Take 5000 samples
  - path: nbeerbower/gutenberg2-dpo
    split: train
    columns: [prompt, chosen, rejected]
    subset: 0.5               # Take 50% of dataset
  - path: jondurbin/gutenberg-dpo-v0.1
    split: train
    columns: [prompt, chosen, rejected]  # Use all samples
```

**Per-dataset options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | required | HuggingFace dataset ID or local path |
| `name` | str | `null` | Dataset configuration name |
| `split` | str | `"train"` | Which split to load |
| `data_dir` | str | `null` | Data directory within the dataset |
| `data_files` | str/list | `null` | Specific data files to load |
| `columns` | list | `null` | Columns to select (filters dataset) |
| `system_message` | str | `null` | System message to add (overrides global `default_system_message`) |
| `truncation_strategy` | str | `null` | How to handle long samples (overrides global `truncation_strategy`) |
| `subset` | int/float | `null` | Number of samples (int) or fraction (float 0-1) to take |
| `shuffle` | bool | `null` | Whether to shuffle before subsetting (defaults to `shuffle_datasets`) |
| `eval_split` | float/false | `null` | Eval split fraction, or `false` to exclude from eval |
| `eval_before_subset` | bool | `null` | Whether to split eval before subsetting |

**Global data processing options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shuffle_datasets` | bool | `true` | Shuffle each dataset before subsetting |
| `shuffle_combined` | bool | `true` | Shuffle final combined dataset |
| `eval_split` | float | `0.0` | Default eval split fraction for all datasets |
| `eval_before_subset` | bool | `false` | Whether eval split happens before subsetting |
| `shuffle_seed` | int | `42` | Seed for shuffle operations |
| `split_seed` | int | `42` | Seed for train/eval splits |
| `streaming` | bool | `false` | Stream datasets instead of loading into memory |

### Column Selection

Use `columns` to select specific columns from a dataset:

```yaml
datasets:
  - path: my-dataset
    split: train
    columns: [prompt, chosen, rejected]  # Only keep these columns
```

This is useful when:
- Dataset has extra columns you don't need
- Dataset column names match expected format but has additional metadata

**Note:** `columns` **selects** columns, it doesn't rename them. Your dataset must already use the expected column names (`messages`, `prompt`, `chosen`, `rejected`, `completion`, `label`, `text`).

### Shuffle, Subset, and Eval Split

Clicker provides fine-grained control over dataset shuffling, subsampling, and evaluation splitting.

**Basic example - take 500 random samples with 5% eval:**

```yaml
shuffle_datasets: true
shuffle_combined: true
eval_split: 0.05
shuffle_seed: 42
split_seed: 42

datasets:
  - path: large-dataset
    subset: 500           # Take 500 samples
  - path: small-dataset   # Use all samples from this one
```

**Processing order:**

1. Load dataset
2. Shuffle (if `shuffle` or `shuffle_datasets` is true)
3. Split eval (if `eval_before_subset` is true)
4. Apply subset
5. Split eval (if `eval_before_subset` is false, the default)
6. Combine all train datasets, combine all eval datasets
7. Shuffle combined (if `shuffle_combined` is true)

**Eval split timing (`eval_before_subset`):**

- `false` (default): Eval size scales with subset. Good when you want proportional eval.
  - Example: 10k dataset → subset 1000 → 5% eval = 50 eval, 950 train
- `true`: Eval is representative of full dataset. Good for large datasets with small subsets.
  - Example: 10k dataset → 5% eval = 500 eval → subset 1000 = 1000 train

**Advanced example - mixed settings:**

```yaml
shuffle_datasets: true
shuffle_combined: true
eval_split: 0.05              # Global default: 5% eval from each dataset
shuffle_seed: 42
split_seed: 123               # Different seed for reproducible splits

datasets:
  - path: dataset-a
    subset: 10000             # Random 10k samples, 5% eval (global default)

  - path: dataset-b
    subset: 0.15              # Random 15% of dataset
    eval_split: 0.10          # Override: 10% eval for this one

  - path: dataset-c
    shuffle: false            # Keep original order (curriculum learning)
    subset: 5000              # First 5000 samples
    eval_split: false         # Exclude from eval entirely

  - path: pretraining-corpus
    subset: 0.01              # 1% of corpus for training
    eval_before_subset: true  # But eval from full corpus
    eval_split: 0.001         # 0.1% of full corpus for eval
```

**Seeds:**

- `shuffle_seed`: Controls dataset shuffling and combined dataset shuffling
- `split_seed`: Controls train/eval splits (separate for reproducibility)

Using separate seeds allows you to:
- Change the random sample selection without changing the train/eval split
- Reproduce the same eval set across different subset sizes

### SFT-Specific Dataset Options

```yaml
# For text datasets (non-conversational)
dataset_text_field: text      # Column containing the text data

# Loss computation options
completion_only_loss: null    # Compute loss only on completion (auto-detected)
assistant_only_loss: false    # Compute loss only on assistant turns
last_assistant_only_loss: false  # Compute loss only on LAST assistant turn
train_on_incomplete_assistant: false  # Don't add EOS on truncated data

# Processing
dataset_num_proc: 4           # Parallel processing workers
```

**Training on assistant turns only:**

For conversational datasets, set `assistant_only_loss: true` to only compute loss on assistant responses (not user messages or system prompts):

```yaml
trainer: sft
dataset_name: my-chat-dataset
assistant_only_loss: true     # Only train on assistant responses
```

This is equivalent to axolotl's `train_on_inputs: false`.

**Training on last assistant turn only:**

For multi-turn conversations, set `last_assistant_only_loss: true` to only compute loss on the final assistant response:

```yaml
trainer: sft
dataset_name: my-multiturn-dataset
last_assistant_only_loss: true  # Only train on final assistant response
```

This is useful when you want to teach a specific final behavior without influencing intermediate dialogue patterns.

**Training on incomplete/truncated responses:**

For datasets with intentionally truncated assistant responses, set `train_on_incomplete_assistant: true` to prevent the model from learning to generate EOS mid-thought:

```yaml
trainer: sft
dataset_name: my-truncated-dataset
train_on_incomplete_assistant: true  # Don't add EOS on last assistant turn
```

This removes the trailing EOS token when the last message is from the assistant, treating it as a continuation rather than a complete response. Useful for training on partial completions or long-form content that was cut off.

**Fixing conversation turn order:**

Some models (like Llama) have strict requirements about conversation structure. Set `fix_turn_order: true` to automatically fix conversations:

```yaml
trainer: sft
dataset_name: my-messy-dataset
fix_turn_order: true              # Fix conversation structure
fix_turn_order_filler: "Continue." # Custom filler message (optional)
```

This performs three fixes:
1. **Adds a filler user message** if the conversation starts with an assistant turn
2. **Merges consecutive same-role messages** by joining their content with newlines
3. **Drops trailing user messages** so conversations always end with an assistant turn

Conversations that become empty after fixing (e.g., user-only conversations) are filtered out automatically.

**Adding a default system message:**

Add a system message to conversations that don't have one:

```yaml
trainer: sft
dataset_name: my-dataset
default_system_message: "You are a helpful AI assistant."
```

For mixed datasets where different datasets need different system prompts, use per-dataset configuration:

```yaml
trainer: sft
default_system_message: "You are a helpful assistant."  # Global fallback

datasets:
  - path: dataset-with-code-focus
    split: train
    system_message: "You are an expert programmer."  # Overrides global
  - path: dataset-with-writing-focus
    split: train
    system_message: "You are a creative writing assistant."  # Overrides global
  - path: dataset-with-existing-system
    split: train
    # No system_message - uses global default (if conversation lacks one)
```

Per-dataset `system_message` takes priority over `default_system_message`. Conversations that already have a system message are not modified.

**Using a custom text field:**

If your dataset uses a different column name for text:

```yaml
trainer: sft
dataset_name: my-dataset
dataset_text_field: content   # Use 'content' column instead of 'text'
```

### Using Preference Datasets for SFT

Clicker automatically converts preference datasets to SFT format when using the SFT trainer. This lets you reuse DPO/ORPO/KTO datasets for supervised fine-tuning.

**Preference datasets (chosen/rejected):**

When you provide a dataset with `prompt`, `chosen`, and `rejected` columns, Clicker converts it to conversational format using only the `chosen` responses:

```yaml
trainer: sft
dataset_name: my-dpo-dataset  # Has prompt/chosen/rejected columns
# Automatically converts to: prompt -> user, chosen -> assistant
```

This works with both string and conversational formats:
- String: `{"prompt": "Hello", "chosen": "Hi!", "rejected": "Go away"}` → messages
- Conversational: Full message lists are preserved and combined

**Binary preference datasets (completion/label):**

KTO-style datasets with `prompt`, `completion`, and `label` (boolean) columns are also supported. Only good examples (`label=True`) are kept:

```yaml
trainer: sft
dataset_name: my-kto-dataset  # Has prompt/completion/label columns
# Automatically uses only label=True examples
```

This is useful when you want to:
- Fine-tune on the same data you used for preference training
- Use high-quality curated preference data for initial SFT
- Create a baseline model from preference annotations

### Preference Dataset Options

For DPO, ORPO, KTO, and Reward training:

```yaml
# Always set this for preference trainers
remove_unused_columns: false

# Sequence lengths
max_length: 1024              # Max total sequence length
max_prompt_length: 512        # Max prompt length (rest for response)
```

### Environment Variables

Set environment variables in config:

```yaml
env:
  WANDB_PROJECT: my-project
  HF_TOKEN: your-token
```

---

## Dataset Blending

Clicker provides a powerful dataset blending workflow that separates data preparation from model training. This allows you to:

- Create reusable data blends for different training experiments
- Prepare datasets once and tokenize for different models
- Cache tokenized datasets for faster subsequent runs

### Workflow Overview

```
1. Create data blend config → 2. Run `clicker blend` → 3. Reference in training config
```

### Step 1: Create a Data Blend Config

Create a YAML file that specifies your dataset blend:

```yaml
# data/my_blend.yaml
trainer_type: sft              # sft, dpo, orpo, or kto
output_dir: data/my_blend_prepared

# Data processing settings
shuffle_datasets: true
shuffle_combined: true
eval_split: 0.05
shuffle_seed: 42
split_seed: 42

# Preprocessing options (stored as metadata)
assistant_only_loss: true
fix_turn_order: true
default_system_message: "You are a helpful assistant."
truncation_strategy: truncate

# Datasets to blend
datasets:
  - path: dataset-a
    subset: 5000
  - path: dataset-b
    subset: 0.15
    system_message: "You are a coding assistant."
    truncation_strategy: drop
```

### Step 2: Run the Blend Command

```bash
clicker blend --config data/my_blend.yaml
```

Output:
```
🍹 Starting the blender...
📦 Gathering ingredients from 2 dataset(s)...
🔄 Mixing and processing...
🔀 Converting preference data to SFT format...
🏷️  Adding metadata for training...
💾 Pouring into container at data/my_blend_prepared...
📊 Recipe stats: 4750 train samples, 250 eval samples
✅ Your data smoothie is ready! Saved to data/my_blend_prepared
```

This creates:
- `data/my_blend_prepared/train.parquet` - Training data
- `data/my_blend_prepared/test.parquet` - Evaluation data
- `data/my_blend_prepared/blend_metadata.json` - Blend configuration metadata

### Step 3: Use in Training Config

Reference the prepared dataset in your training config:

```yaml
# configs/train.yaml
trainer: sft
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct

# Use prepared dataset
prepared_dataset: data/my_blend_prepared
max_length: 4096

# Training parameters
learning_rate: 2e-5
num_train_epochs: 3
per_device_train_batch_size: 4
```

Run training:
```bash
clicker sft --config configs/train.yaml
```

### Tokenization Caching

When using a prepared dataset, tokenized data is automatically cached. The cache key is based on:
- The blend's config hash
- The model name
- The max_length setting

This means:
- **Same model + max_length**: Loads from cache instantly
- **Different model or max_length**: Re-tokenizes and creates new cache

Control caching behavior:
```yaml
prepared_dataset: data/my_blend_prepared
tokenized_cache_dir: /fast-storage/tokenized  # Custom cache location
force_retokenize: false                        # Set true to ignore cache
```

### Data Blend Config Reference

**Core options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainer_type` | str | `"sft"` | Output format: `sft`, `dpo`, `orpo`, `kto` |
| `output_dir` | str | required | Directory to save prepared dataset |

**Preprocessing options (stored as metadata):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_system_message` | str | `null` | System message to add to conversations |
| `assistant_only_loss` | bool | `false` | Compute loss only on assistant turns |
| `last_assistant_only_loss` | bool | `false` | Compute loss only on last assistant turn |
| `train_on_incomplete_assistant` | bool | `false` | Don't add EOS on truncated responses |
| `fix_turn_order` | bool | `false` | Fix conversation turn order |
| `fix_turn_order_filler` | str | `"Let's begin."` | Filler for turn order fixing |
| `truncation_strategy` | str | `"truncate"` | Default truncation strategy |
| `num_proc` | int | `null` | Parallel processing workers |

**Data processing options:** (inherited from dataset mixer)

See [Global data processing options](#multiple-datasets-dataset-mixer) for `shuffle_datasets`, `shuffle_combined`, `eval_split`, etc.

---

## LoRA Configuration

Enable parameter-efficient fine-tuning with LoRA:

```yaml
use_peft: true
lora_r: 32                    # LoRA rank
lora_alpha: 64                # LoRA alpha (scaling = alpha/r)
lora_dropout: 0.05            # LoRA dropout
lora_target_modules: null     # Auto-detect, or specify: ["q_proj", "v_proj"]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_peft` | bool | `false` | Enable LoRA/PEFT |
| `lora_r` | int | `8` | LoRA rank |
| `lora_alpha` | int | `16` | LoRA alpha (effective scale = alpha/r) |
| `lora_dropout` | float | `0.0` | Dropout for LoRA layers |
| `lora_target_modules` | list | `null` | Target modules (auto-detected if null) |
| `use_rslora` | bool | `false` | Use rank-stabilized LoRA |
| `use_dora` | bool | `false` | Use DoRA (Weight-Decomposed LoRA) |

**Quantization with LoRA (QLoRA):**

```yaml
use_peft: true
load_in_4bit: true
bnb_4bit_quant_type: nf4
use_bnb_nested_quant: true
```

---

## Reward Functions

For GRPO and RLOO, you can specify reward functions:

### Built-in Functions

```yaml
reward_funcs:
  - think_format_reward        # Rewards <think>...</think> format
  - get_soft_overlong_punishment  # Penalizes overlong completions
```

| Function | Description |
|----------|-------------|
| `think_format_reward` | Returns 1.0 if completion has `<think>...</think>` tags, else 0.0 |
| `get_soft_overlong_punishment` | Soft penalty for completions exceeding length limits |

### Custom Reward Functions

Specify a dotted import path:

```yaml
reward_funcs:
  - my_module.rewards.custom_reward
```

Your function must have this signature:

```python
def custom_reward(
    prompts: list[str],
    completions: list[str],
    completion_ids: list[list[int]],
    **kwargs
) -> list[float]:
    # Return a reward for each completion
    return [1.0 for _ in completions]
```

### Reward Models

Use a trained reward model:

```yaml
reward_model_name_or_path: path/to/reward/model
```

---

## Distributed Training

### Multi-GPU

```bash
# Auto-detected
clicker train --config config.yaml

# Explicit accelerate config
clicker train --config config.yaml --accelerate_config multi_gpu
```

### DeepSpeed

```bash
clicker train --config config.yaml --accelerate_config zero2
```

### FSDP (Fully Sharded Data Parallel)

FSDP configs auto-detect the transformer layer class from the model's `_no_split_modules` attribute - no need to specify layer names for different architectures.

```bash
# FSDP with FULL_SHARD (aggressive memory optimization)
clicker train --config config.yaml --accelerate_config fsdp2

# FSDP with CPU offload (for large models on limited VRAM)
clicker train --config config.yaml --accelerate_config fsdp_offload
```

**Built-in accelerate configs:**

| Config | Description |
|--------|-------------|
| `single_gpu` | Single GPU training |
| `multi_gpu` | Multi-GPU with DDP |
| `zero1` | DeepSpeed ZeRO Stage 1 |
| `zero2` | DeepSpeed ZeRO Stage 2 |
| `zero3` | DeepSpeed ZeRO Stage 3 |
| `fsdp1` | FSDP with SHARD_GRAD_OP (less aggressive sharding) |
| `fsdp2` | FSDP with FULL_SHARD (shards params, grads, optimizer) |
| `fsdp_offload` | FSDP with CPU offload + activation checkpointing |

### Custom Accelerate Config

Create your own accelerate config and use:

```bash
clicker train --config config.yaml --accelerate_config path/to/accelerate_config.yaml
```

**FSDP config options:**

```yaml
fsdp_config:
  # Auto-wrap policy (auto-detects layer class from model)
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP

  # Sharding strategy: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
  fsdp_sharding_strategy: FULL_SHARD

  # Memory optimizations
  fsdp_offload_params: true           # Offload params to CPU
  fsdp_activation_checkpointing: true # Checkpoint activations
  fsdp_cpu_ram_efficient_loading: true

  # State management
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
  fsdp_limit_all_gathers: true
  fsdp_state_dict_type: FULL_STATE_DICT
```

---

## Example Configs

See `configs/` directory for complete examples:

- `sft.yaml` - Supervised fine-tuning
- `dpo.yaml` - Direct Preference Optimization
- `orpo.yaml` - Odds Ratio Preference Optimization
- `kto.yaml` - Kahneman-Tversky Optimization
- `grpo.yaml` - Group Relative Policy Optimization
- `rloo.yaml` - REINFORCE Leave-One-Out
- `reward.yaml` - Reward model training

---

## License

Apache 2.0
