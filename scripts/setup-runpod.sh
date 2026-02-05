#!/bin/bash
# Clicker Setup Script for RunPod
# Usage: curl -sSL https://raw.githubusercontent.com/strangedove/clicker/main/scripts/setup-runpod.sh | bash
#
# Environment variables:
#   CLICKER_BRANCH - Git branch to checkout (default: main)
#   HF_KEY         - HuggingFace API token for model downloads
#   WANDB_KEY      - Weights & Biases API key for logging

set -e

BRANCH="${CLICKER_BRANCH:-main}"

echo "🚀 Setting up Clicker on RunPod..."
echo "   Branch: $BRANCH"

# Clone the repo
if [ -d "/workspace/clicker" ]; then
    echo "📁 /workspace/clicker already exists, pulling latest..."
    cd /workspace/clicker
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "📦 Cloning clicker repo..."
    git clone -b "$BRANCH" https://github.com/strangedove/clicker.git /workspace/clicker
    cd /workspace/clicker
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📥 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install clicker and dependencies
echo "📦 Installing clicker and dependencies..."
cd /workspace/clicker
uv sync

# Install flash-attn (often needed, compile can take a while)
echo "⚡ Installing flash-attention..."
uv pip install flash-attn --no-build-isolation || echo "⚠️  flash-attn install failed (may need manual install)"

# Authenticate with HuggingFace
if [ -n "$HF_KEY" ]; then
    echo "🤗 Logging into HuggingFace..."
    uv run huggingface-cli login --token "$HF_KEY"
else
    echo "⚠️  HF_KEY not set, skipping HuggingFace login"
fi

# Authenticate with Weights & Biases
if [ -n "$WANDB_KEY" ]; then
    echo "📊 Logging into Weights & Biases..."
    uv run wandb login "$WANDB_KEY"
else
    echo "⚠️  WANDB_KEY not set, skipping W&B login"
fi

# Add to PATH for convenience
echo 'export PATH="/workspace/clicker/.venv/bin:$PATH"' >> ~/.bashrc

echo ""
echo "✅ Clicker setup complete!"
echo ""
echo "Usage:"
echo "  cd /workspace/clicker"
echo "  uv run clicker --help"
echo ""
echo "Or activate the venv:"
echo "  source /workspace/clicker/.venv/bin/activate"
echo "  clicker --help"
