# Examples

This directory contains example scripts showing how to use Clicker programmatically.

For YAML-based configuration, see the `configs/` directory in the repo root.

## Directory Structure

- `scripts/` - Example training scripts using the Python API
- `accelerate_configs/` - Example accelerate configurations for distributed training
- `datasets/` - Example dataset loading scripts

## Quick Start

### CLI (Recommended)

```bash
# Using YAML config
clicker train --config ../configs/sft.yaml

# Or using specific trainer commands
clicker sft --model_name_or_path Qwen/Qwen2-0.5B --dataset_name trl-lib/Capybara
```

### Python API

```python
from clicker import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
dataset = load_dataset("trl-lib/Capybara")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="./output"),
    train_dataset=dataset["train"],
)
trainer.train()
```

See `scripts/` for more complete examples.
