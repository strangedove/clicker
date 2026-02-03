# pyright: reportUnknownVariableType=none, reportPrivateLocalImportUsage=none

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

from pathlib import Path
from typing import TYPE_CHECKING

from .import_utils import _LazyModule

__version__ = '1!0.1.0.dev0'

_import_structure = {
    "scripts": ["DataPrepConfig", "DatasetMixtureConfig", "ScriptArguments", "TrlParser", "get_dataset", "init_zero_verbose"],
    "data_utils": [
        "apply_chat_template",
        "extract_prompt",
        "is_conversational",
        "is_conversational_from_value",
        "maybe_apply_chat_template",
        "maybe_convert_to_chatml",
        "maybe_extract_prompt",
        "maybe_unpair_preference_dataset",
        "pack_dataset",
        "prepare_multimodal_messages",
        "truncate_dataset",
        "unpair_preference_dataset",
    ],
    "extras": ["BestOfNSampler"],
    "models": [
        "SUPPORTED_ARCHITECTURES",
        "AutoModelForCausalLMWithValueHead",
        "AutoModelForSeq2SeqLMWithValueHead",
        "PreTrainedModelWrapper",
        "clone_chat_template",
        "create_reference_model",
        "setup_chat_format",
    ],
    "trainer": [
        "AllTrueJudge",
        "BaseBinaryJudge",
        "BaseJudge",
        "BasePairwiseJudge",
        "BaseRankJudge",
        "DPOConfig",
        "DPOTrainer",
        "FDivergenceConstants",
        "FDivergenceType",
        "GRPOConfig",
        "GRPOTrainer",
        "HfPairwiseJudge",
        "KTOConfig",
        "KTOTrainer",
        "LogCompletionsCallback",
        "ModelConfig",
        "OpenAIPairwiseJudge",
        "ORPOConfig",
        "ORPOTrainer",
        "PairRMJudge",
        "RewardConfig",
        "RewardTrainer",
        "RLOOConfig",
        "RLOOTrainer",
        "SFTConfig",
        "SFTTrainer",
        "WinRateCallback",
    ],
    "trainer.callbacks": [
        "BEMACallback",
        "MergeModelCallback",
        "RichProgressCallback",
        "SyncRefModelCallback",
        "WeaveCallback",
    ],
    "trainer.utils": ["get_kbit_device_map", "get_peft_config", "get_quantization_config"],
}

if TYPE_CHECKING:
    from .data_utils import (
        apply_chat_template,  
        extract_prompt,
        is_conversational,
        is_conversational_from_value,
        maybe_apply_chat_template,
        maybe_convert_to_chatml,
        maybe_extract_prompt,
        maybe_unpair_preference_dataset,
        pack_dataset,
        prepare_multimodal_messages,
        truncate_dataset,
        unpair_preference_dataset,
    )
    from .extras import BestOfNSampler
    from .models import (
        SUPPORTED_ARCHITECTURES,
        AutoModelForCausalLMWithValueHead,
        AutoModelForSeq2SeqLMWithValueHead,
        PreTrainedModelWrapper,
        clone_chat_template,
        create_reference_model,
        setup_chat_format,
    )
    from .scripts import DataPrepConfig, DatasetMixtureConfig, ScriptArguments, TrlParser, get_dataset, init_zero_verbose
    from .trainer import (
        AllTrueJudge,
        BaseBinaryJudge,
        BaseJudge,
        BasePairwiseJudge,
        BaseRankJudge,
        DPOConfig,
        DPOTrainer,
        FDivergenceConstants,
        FDivergenceType,
        GRPOConfig,
        GRPOTrainer,
        HfPairwiseJudge,
        KTOConfig,
        KTOTrainer,
        LogCompletionsCallback,
        ModelConfig,
        OpenAIPairwiseJudge,
        ORPOConfig,
        ORPOTrainer,
        PairRMJudge,
        RewardConfig,
        RewardTrainer,
        RLOOConfig,
        RLOOTrainer,
        SFTConfig,
        SFTTrainer,
        WinRateCallback,
    )
    from .trainer.callbacks import (
        BEMACallback,
        RichProgressCallback,
        SyncRefModelCallback,
        WeaveCallback,
    )
    from .trainer.utils import get_kbit_device_map, get_peft_config, get_quantization_config

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
