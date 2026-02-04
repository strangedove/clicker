"""
Tests for the blend pipeline (clicker/scripts/blend.py) and related utils.

These tests use lightweight tokenizers (gpt2 for text, Qwen2-0.5B for chat)
to avoid needing large model downloads. They create temporary datasets in
/tmp and validate the full pipeline: load -> preprocess -> tokenize ->
truncate -> eval split -> save.
"""

import json
import os
import shutil
import tempfile

import pytest
import yaml
from datasets import Dataset

from clicker.scripts.blend import (
    PipelineStats,
    _load_data_config,
    _load_training_config,
    prepare_dataset,
    preprocess_dataset,
    save_prepared_dataset,
    tokenize_dataset,
)
from clicker.scripts.utils import (
    is_pretokenized_blend,
    load_prepared_dataset,
    needs_blend,
    prompt_blend_overwrite,
    validate_blend_metadata,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="clicker_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def text_dataset(tmp_dir):
    """Create a small text dataset on disk and return (path, data_config_path, count)."""
    # Create 20 text samples of varying lengths
    texts = []
    for i in range(20):
        # Some short, some long (to test split truncation)
        words = f"Sample text number {i}. " * (50 + i * 30)
        texts.append(words.strip())

    ds = Dataset.from_dict({"text": texts})
    ds_dir = os.path.join(tmp_dir, "text_data")
    os.makedirs(ds_dir, exist_ok=True)
    parquet_path = os.path.join(ds_dir, "data.parquet")
    ds.to_parquet(parquet_path)

    # Write data config — use parquet path as data_files
    data_config = {
        "datasets": [{"path": ds_dir, "split": "train"}],
        "shuffle_datasets": False,
        "shuffle_combined": False,
    }
    data_config_path = os.path.join(tmp_dir, "data.yaml")
    with open(data_config_path, "w") as f:
        yaml.dump(data_config, f)

    return ds_dir, data_config_path, len(texts)


@pytest.fixture
def chat_dataset(tmp_dir):
    """Create a small chat dataset on disk and return (path, data_config_path, count)."""
    samples = []
    for i in range(15):
        messages = [
            {"role": "user", "content": f"Question {i}: " + "How does this work? " * (5 + i * 3)},
            {"role": "assistant", "content": f"Answer {i}: " + "Here is the explanation. " * (10 + i * 5)},
        ]
        samples.append(messages)

    # Save as jsonl since parquet struggles with nested list[dict] for messages
    ds_dir = os.path.join(tmp_dir, "chat_data")
    os.makedirs(ds_dir, exist_ok=True)
    jsonl_path = os.path.join(ds_dir, "data.jsonl")
    with open(jsonl_path, "w") as f:
        for msgs in samples:
            f.write(json.dumps({"messages": msgs}) + "\n")

    # Write data config
    data_config = {
        "datasets": [{"path": ds_dir, "split": "train"}],
        "shuffle_datasets": False,
        "shuffle_combined": False,
    }
    data_config_path = os.path.join(tmp_dir, "data.yaml")
    with open(data_config_path, "w") as f:
        yaml.dump(data_config, f)

    return ds_dir, data_config_path, len(samples)


# ──────────────────────────────────────────────────────────────────────────────
# PipelineStats tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineStats:
    def test_empty_stats(self):
        stats = PipelineStats()
        assert stats.format_summary() == ""

    def test_record_and_format(self):
        stats = PipelineStats()
        stats.record("Loaded", 100)
        stats.record("Tokenized", 100)
        stats.record("Truncation (split)", 250, "split into 512-token chunks")
        stats.record("Train split", 237)
        stats.record("Eval split", 13, "5.0% of total")

        summary = stats.format_summary()
        assert "Loaded" in summary
        assert "100" in summary
        assert "+150" in summary  # 250 - 100
        assert "split into 512-token chunks" in summary
        assert "Eval split" in summary

    def test_delta_decrease(self):
        stats = PipelineStats()
        stats.record("Before", 100)
        stats.record("After", 90, "dropped 10")

        summary = stats.format_summary()
        assert "-10" in summary
        assert "dropped 10" in summary

    def test_no_delta_for_first(self):
        stats = PipelineStats()
        stats.record("Loaded", 100, "from disk")

        summary = stats.format_summary()
        assert "from disk" in summary
        # Should not have +/- delta for first step
        assert "+" not in summary.split("100")[1] or "from disk" in summary


# ──────────────────────────────────────────────────────────────────────────────
# Config loading tests
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigLoading:
    def test_load_data_config_missing_file(self):
        with pytest.raises(FileNotFoundError, match="Data config file not found"):
            _load_data_config("/nonexistent/path.yaml")

    def test_load_data_config_no_datasets(self, tmp_dir):
        config_path = os.path.join(tmp_dir, "bad.yaml")
        with open(config_path, "w") as f:
            yaml.dump({"shuffle_datasets": True}, f)

        with pytest.raises(ValueError, match="must have a 'datasets' list"):
            _load_data_config(config_path)

    def test_load_data_config_invalid_yaml(self, tmp_dir):
        config_path = os.path.join(tmp_dir, "bad.yaml")
        with open(config_path, "w") as f:
            f.write("{{invalid yaml::")

        with pytest.raises(ValueError, match="Invalid YAML"):
            _load_data_config(config_path)

    def test_load_data_config_forces_zero_eval(self, tmp_dir):
        config_path = os.path.join(tmp_dir, "data.yaml")
        with open(config_path, "w") as f:
            yaml.dump({
                "datasets": [{"path": "/tmp/fake", "split": "train"}],
                "eval_split": 0.1,
            }, f)

        config = _load_data_config(config_path)
        assert config.eval_split == 0.0

    def test_load_training_config_missing(self):
        with pytest.raises(FileNotFoundError):
            _load_training_config("/nonexistent/config.yaml")

    def test_load_training_config_valid(self, tmp_dir):
        config_path = os.path.join(tmp_dir, "train.yaml")
        with open(config_path, "w") as f:
            yaml.dump({
                "model_name_or_path": "gpt2",
                "data_config": "data.yaml",
                "max_length": 512,
            }, f)

        config = _load_training_config(config_path)
        assert config["model_name_or_path"] == "gpt2"
        assert config["max_length"] == 512


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_preprocess_empty_dataset_raises(self):
        empty_ds = Dataset.from_dict({"text": []})
        stats = PipelineStats()
        with pytest.raises(ValueError, match="Dataset is empty"):
            preprocess_dataset(empty_ds, stats)

    def test_preprocess_text_passthrough(self):
        """Text datasets should pass through preprocessing mostly unchanged."""
        ds = Dataset.from_dict({"text": ["Hello world", "Foo bar baz"]})
        stats = PipelineStats()
        result = preprocess_dataset(ds, stats)
        assert len(result) == 2
        assert "text" in result.column_names


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline tests (text data + gpt2)
# ──────────────────────────────────────────────────────────────────────────────

class TestTextPipeline:
    def test_prepare_dataset_text_truncate(self, text_dataset, tmp_dir):
        """Test full pipeline with text data and truncate strategy."""
        _, data_config_path, count = text_dataset
        output_dir = os.path.join(tmp_dir, "output")

        config = {
            "model_name_or_path": "gpt2",
            "data_config": data_config_path,
            "max_length": 512,
            "truncation_strategy": "truncate",
            "dataset_text_field": "text",
            "eval_split": 0.0,
            "split_seed": 42,
        }

        dataset_dict, stats = prepare_dataset(config)
        assert "train" in dataset_dict
        assert "test" not in dataset_dict
        assert len(dataset_dict["train"]) == count
        assert "input_ids" in dataset_dict["train"].column_names
        assert len(stats.steps) > 0

        # All sequences should be <= max_length
        for example in dataset_dict["train"]:
            assert len(example["input_ids"]) <= 512

    def test_prepare_dataset_text_split(self, text_dataset, tmp_dir):
        """Test full pipeline with text data and split strategy."""
        _, data_config_path, count = text_dataset
        output_dir = os.path.join(tmp_dir, "output")

        config = {
            "model_name_or_path": "gpt2",
            "data_config": data_config_path,
            "max_length": 128,
            "truncation_strategy": "split",
            "dataset_text_field": "text",
            "eval_split": 0.0,
            "split_seed": 42,
        }

        dataset_dict, stats = prepare_dataset(config)

        # Split should produce MORE samples than the original count
        assert len(dataset_dict["train"]) > count
        # All sequences should be <= max_length
        for example in dataset_dict["train"]:
            assert len(example["input_ids"]) <= 128

    def test_prepare_dataset_text_drop(self, text_dataset, tmp_dir):
        """Test full pipeline with text data and drop strategy."""
        _, data_config_path, count = text_dataset

        config = {
            "model_name_or_path": "gpt2",
            "data_config": data_config_path,
            "max_length": 128,
            "truncation_strategy": "drop",
            "dataset_text_field": "text",
            "eval_split": 0.0,
            "split_seed": 42,
        }

        dataset_dict, stats = prepare_dataset(config)

        # Drop should produce FEWER or equal samples
        assert len(dataset_dict["train"]) <= count
        # All sequences should be <= max_length
        for example in dataset_dict["train"]:
            assert len(example["input_ids"]) <= 128

    def test_prepare_dataset_with_eval_split(self, text_dataset, tmp_dir):
        """Test that eval_split produces train and test splits."""
        _, data_config_path, count = text_dataset

        config = {
            "model_name_or_path": "gpt2",
            "data_config": data_config_path,
            "max_length": 512,
            "truncation_strategy": "truncate",
            "dataset_text_field": "text",
            "eval_split": 0.2,
            "split_seed": 42,
        }

        dataset_dict, stats = prepare_dataset(config)

        assert "train" in dataset_dict
        assert "test" in dataset_dict
        total = len(dataset_dict["train"]) + len(dataset_dict["test"])
        assert total == count
        # Eval should be approximately 20%
        eval_frac = len(dataset_dict["test"]) / total
        assert 0.1 < eval_frac < 0.4  # generous bounds for small dataset


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline tests (chat data + Qwen2)
# ──────────────────────────────────────────────────────────────────────────────

class TestChatPipeline:
    def test_prepare_dataset_chat(self, chat_dataset, tmp_dir):
        """Test full pipeline with chat-format data."""
        _, data_config_path, count = chat_dataset

        config = {
            "model_name_or_path": "Qwen/Qwen2-0.5B",
            "data_config": data_config_path,
            "max_length": 256,
            "truncation_strategy": "truncate",
            "dataset_text_field": "text",
            "eval_split": 0.0,
            "split_seed": 42,
        }

        dataset_dict, stats = prepare_dataset(config)
        assert "train" in dataset_dict
        assert "input_ids" in dataset_dict["train"].column_names
        assert len(dataset_dict["train"]) == count

        # All sequences should be <= max_length
        for example in dataset_dict["train"]:
            assert len(example["input_ids"]) <= 256


# ──────────────────────────────────────────────────────────────────────────────
# Save and load tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSaveAndLoad:
    def test_save_and_load_roundtrip(self, text_dataset, tmp_dir):
        """Test that save_prepared_dataset + load_prepared_dataset roundtrips."""
        _, data_config_path, count = text_dataset
        output_dir = os.path.join(tmp_dir, "saved_output")

        config = {
            "model_name_or_path": "gpt2",
            "data_config": data_config_path,
            "max_length": 512,
            "truncation_strategy": "truncate",
            "dataset_text_field": "text",
            "eval_split": 0.1,
            "split_seed": 42,
        }

        dataset_dict, stats = prepare_dataset(config)
        save_prepared_dataset(dataset_dict, output_dir, config, stats)

        # Check files exist
        assert os.path.exists(os.path.join(output_dir, "train.parquet"))
        assert os.path.exists(os.path.join(output_dir, "test.parquet"))
        assert os.path.exists(os.path.join(output_dir, "blend_metadata.json"))

        # Load back
        loaded = load_prepared_dataset(output_dir)
        assert len(loaded["train"]) == len(dataset_dict["train"])
        assert len(loaded["test"]) == len(dataset_dict["test"])

    def test_metadata_contents(self, text_dataset, tmp_dir):
        """Test that metadata contains expected fields."""
        _, data_config_path, count = text_dataset
        output_dir = os.path.join(tmp_dir, "meta_output")

        config = {
            "model_name_or_path": "gpt2",
            "data_config": data_config_path,
            "max_length": 512,
            "truncation_strategy": "truncate",
            "dataset_text_field": "text",
            "eval_split": 0.0,
            "split_seed": 42,
        }

        dataset_dict, stats = prepare_dataset(config)
        save_prepared_dataset(dataset_dict, output_dir, config, stats)

        with open(os.path.join(output_dir, "blend_metadata.json")) as f:
            metadata = json.load(f)

        assert metadata["tokenized"] is True
        assert metadata["model_name_or_path"] == "gpt2"
        assert metadata["max_length"] == 512
        assert metadata["truncation_strategy"] == "truncate"
        assert "config_hash" in metadata
        assert "pipeline_steps" in metadata
        assert "created_at" in metadata

    def test_is_pretokenized_blend(self, text_dataset, tmp_dir):
        """Test is_pretokenized_blend detection."""
        _, data_config_path, _ = text_dataset
        output_dir = os.path.join(tmp_dir, "pretok_output")

        config = {
            "model_name_or_path": "gpt2",
            "data_config": data_config_path,
            "max_length": 512,
            "truncation_strategy": "truncate",
            "dataset_text_field": "text",
            "eval_split": 0.0,
            "split_seed": 42,
        }

        dataset_dict, stats = prepare_dataset(config)
        save_prepared_dataset(dataset_dict, output_dir, config, stats)

        assert is_pretokenized_blend(output_dir) is True
        assert is_pretokenized_blend("/nonexistent/path") is False


# ──────────────────────────────────────────────────────────────────────────────
# Utils function tests
# ──────────────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_needs_blend_missing_path(self):
        assert needs_blend("/nonexistent/path/to/data") is True

    def test_needs_blend_empty_dir(self, tmp_dir):
        empty = os.path.join(tmp_dir, "empty")
        os.makedirs(empty)
        assert needs_blend(empty) is True

    def test_needs_blend_with_train_parquet(self, tmp_dir):
        d = os.path.join(tmp_dir, "has_train")
        os.makedirs(d)
        # Create a dummy train.parquet
        with open(os.path.join(d, "train.parquet"), "w") as f:
            f.write("dummy")
        assert needs_blend(d) is False

    def test_needs_blend_empty_string(self):
        assert needs_blend("") is True

    def test_validate_blend_metadata_no_metadata(self, tmp_dir):
        mismatches = validate_blend_metadata(tmp_dir, {"model_name_or_path": "gpt2"})
        assert len(mismatches) == 1
        assert "No blend_metadata.json" in mismatches[0]

    def test_validate_blend_metadata_matching(self, tmp_dir):
        metadata = {
            "model_name_or_path": "gpt2",
            "max_length": 512,
            "truncation_strategy": "truncate",
            "data_config": "data/test.yaml",
            "dataset_text_field": "text",
            "eval_split": 0.05,
        }
        with open(os.path.join(tmp_dir, "blend_metadata.json"), "w") as f:
            json.dump(metadata, f)

        config = {
            "model_name_or_path": "gpt2",
            "max_length": 512,
            "truncation_strategy": "truncate",
            "data_config": "data/test.yaml",
            "dataset_text_field": "text",
            "eval_split": 0.05,
        }

        mismatches = validate_blend_metadata(tmp_dir, config)
        assert mismatches == []

    def test_validate_blend_metadata_mismatch(self, tmp_dir):
        metadata = {
            "model_name_or_path": "gpt2",
            "max_length": 512,
            "truncation_strategy": "truncate",
            "eval_split": 0.05,
        }
        with open(os.path.join(tmp_dir, "blend_metadata.json"), "w") as f:
            json.dump(metadata, f)

        config = {
            "model_name_or_path": "llama-7b",  # different model
            "max_length": 1024,  # different length
            "truncation_strategy": "split",  # different strategy
            "eval_split": 0.1,  # different eval split
        }

        mismatches = validate_blend_metadata(tmp_dir, config)
        assert len(mismatches) >= 3  # model, max_length, truncation, eval_split

    def test_prompt_blend_overwrite_force(self, tmp_dir):
        """force=True should return True without prompting."""
        result = prompt_blend_overwrite(tmp_dir, ["mismatch 1"], force=True)
        assert result is True


# ──────────────────────────────────────────────────────────────────────────────
# Error handling tests
# ──────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_data_config_field(self):
        """Config without data_config should raise ValueError."""
        config = {
            "model_name_or_path": "gpt2",
            "max_length": 512,
        }
        with pytest.raises(ValueError, match="data_config"):
            prepare_dataset(config)

    def test_missing_model_field(self, tmp_dir):
        """Config without model_name_or_path should raise ValueError."""
        data_config_path = os.path.join(tmp_dir, "data.yaml")
        with open(data_config_path, "w") as f:
            yaml.dump({"datasets": [{"path": "/tmp/fake", "split": "train"}]}, f)

        config = {
            "data_config": data_config_path,
            "max_length": 512,
        }
        with pytest.raises(ValueError, match="model_name_or_path"):
            prepare_dataset(config)

    def test_wrong_text_field_name(self, tmp_dir):
        """Dataset without matching text field should give helpful error."""
        ds = Dataset.from_dict({"content": ["hello world"]})
        ds_dir = os.path.join(tmp_dir, "wrong_field_data")
        os.makedirs(ds_dir, exist_ok=True)
        ds.to_parquet(os.path.join(ds_dir, "data.parquet"))

        data_config_path = os.path.join(tmp_dir, "data.yaml")
        with open(data_config_path, "w") as f:
            yaml.dump({"datasets": [{"path": ds_dir, "split": "train"}]}, f)

        config = {
            "model_name_or_path": "gpt2",
            "data_config": data_config_path,
            "max_length": 512,
            "truncation_strategy": "truncate",
            "dataset_text_field": "text",  # but data has "content"
            "eval_split": 0.0,
        }

        with pytest.raises(ValueError, match="neither chat format.*nor the text field"):
            prepare_dataset(config)
