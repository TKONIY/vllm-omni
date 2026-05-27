# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expected-failing repro for deploy YAML parallel_config CLI overrides.

Run with:
    pytest tests/test_deploy_parallel_config_cli_override.py

These tests document that deploy-based diffusion models such as HunyuanImage3
DiT cannot currently override their YAML `parallel_config` through the same
top-level values a user would expect to pass from CLI/API construction.
"""

from pathlib import Path

import pytest

from vllm_omni.config.stage_config import StageConfigFactory

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HUNYUAN_IMAGE3_DIT_DEPLOY = Path("vllm_omni/deploy/hunyuan_image3_dit.yaml")


def _resolve_hunyuan_image3_dit_stage(cli_overrides: dict):
    stages = StageConfigFactory._create_from_registry(
        "hunyuan_image3_dit",
        cli_overrides=cli_overrides,
        deploy_config_path=str(_HUNYUAN_IMAGE3_DIT_DEPLOY),
    )
    assert len(stages) == 1
    return stages[0].to_omegaconf()


def test_deploy_parallel_config_override_should_replace_yaml_parallel_config():
    stage = _resolve_hunyuan_image3_dit_stage(
        {
            "parallel_config": {
                "tensor_parallel_size": 1,
                "cfg_parallel_size": 2,
            },
        }
    )

    # Expected behavior: a user-supplied parallel_config override should replace
    # the deploy YAML value `tensor_parallel_size=4, cfg_parallel_size=1`.
    # Current behavior: `parallel_config` is treated as orchestrator-owned and
    # filtered out before stage runtime overrides are applied.
    assert stage.engine_args.parallel_config.tensor_parallel_size == 1
    assert stage.engine_args.parallel_config.cfg_parallel_size == 2


def test_deploy_flat_parallel_fields_should_update_nested_parallel_config():
    stage = _resolve_hunyuan_image3_dit_stage(
        {
            "tensor_parallel_size": 1,
            "cfg_parallel_size": 2,
        }
    )

    # Expected behavior: flat parallel CLI fields should deep-merge into the
    # diffusion `parallel_config` used by HunyuanImage3 DiT.
    # Current behavior: they are added as top-level engine args while the
    # nested YAML parallel_config stays at TP=4, CFG=1; the diffusion path then
    # prefers the nested parallel_config and ignores the flat values.
    assert stage.engine_args.parallel_config.tensor_parallel_size == 1
    assert stage.engine_args.parallel_config.cfg_parallel_size == 2
