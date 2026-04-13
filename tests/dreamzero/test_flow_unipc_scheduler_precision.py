# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_SCHEDULER = REPO_ROOT / "vllm_omni/diffusion/models/schedulers/scheduling_flow_unipc_multistep.py"
UPSTREAM_SCHEDULER = Path(
    os.path.expanduser("~/code/dreamzero/groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.py")
)


def _load_module(name: str, path: Path, *, disable_compile: bool) -> object:
    compile_orig = torch.compile
    if disable_compile:

        def identity_compile(*args, **kwargs):
            def deco(fn):
                return fn

            return deco

        torch.compile = identity_compile

    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        torch.compile = compile_orig


def _new_scheduler(cls):
    return cls(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )


def _clone_scheduler_state(dst, src) -> None:
    dst.model_outputs = copy.deepcopy(src.model_outputs)
    dst.timestep_list = copy.deepcopy(src.timestep_list)
    dst.lower_order_nums = src.lower_order_nums
    dst.disable_corrector = copy.deepcopy(src.disable_corrector)
    dst.last_sample = None if src.last_sample is None else src.last_sample.clone()
    if hasattr(src, "this_order"):
        dst.this_order = src.this_order
    if hasattr(src, "_step_index"):
        dst._step_index = src._step_index
    if hasattr(src, "_begin_index"):
        dst._begin_index = src._begin_index


def _set_deterministic() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _run_step0_step1_pair(
    *,
    baseline_cls,
    upstream_eager_cls,
    upstream_compiled_cls,
) -> tuple[float, float, float, float, float, float]:
    baseline = _new_scheduler(baseline_cls)
    upstream_eager = _new_scheduler(upstream_eager_cls)
    upstream_compiled = _new_scheduler(upstream_compiled_cls)

    for scheduler in (baseline, upstream_eager, upstream_compiled):
        scheduler.set_timesteps(3, device="cuda")

    sample0 = torch.randn(1, 16, 7, device="cuda", dtype=torch.bfloat16)
    model_output0 = torch.randn_like(sample0)

    out_base0 = baseline.step(model_output0, baseline.timesteps[0], sample0, return_dict=False)[0]
    out_eager0 = upstream_eager.step(
        model_output0,
        upstream_eager.timesteps[0],
        sample0,
        step_index=0,
        return_dict=False,
    )[0]
    out_comp0 = upstream_compiled.step(
        model_output0,
        upstream_compiled.timesteps[0],
        sample0,
        step_index=0,
        return_dict=False,
    )[0]

    base_vs_eager_0 = (out_base0.float() - out_eager0.float()).abs().max().item()
    base_vs_comp_0 = (out_base0.float() - out_comp0.float()).abs().max().item()
    eager_vs_comp_0 = (out_eager0.float() - out_comp0.float()).abs().max().item()

    baseline1 = _new_scheduler(baseline_cls)
    upstream_eager1 = _new_scheduler(upstream_eager_cls)
    upstream_compiled1 = _new_scheduler(upstream_compiled_cls)
    for scheduler in (baseline1, upstream_eager1, upstream_compiled1):
        scheduler.set_timesteps(3, device="cuda")

    _clone_scheduler_state(baseline1, baseline)
    _clone_scheduler_state(upstream_eager1, upstream_compiled)
    _clone_scheduler_state(upstream_compiled1, upstream_compiled)

    sample1 = out_comp0.clone()
    model_output1 = torch.randn_like(sample1)

    out_base1 = baseline1.step(model_output1, baseline1.timesteps[1], sample1, return_dict=False)[0]
    out_eager1 = upstream_eager1.step(
        model_output1,
        upstream_eager1.timesteps[1],
        sample1,
        step_index=1,
        return_dict=False,
    )[0]
    out_comp1 = upstream_compiled1.step(
        model_output1,
        upstream_compiled1.timesteps[1],
        sample1,
        step_index=1,
        return_dict=False,
    )[0]

    base_vs_eager_1 = (out_base1.float() - out_eager1.float()).abs().max().item()
    base_vs_comp_1 = (out_base1.float() - out_comp1.float()).abs().max().item()
    eager_vs_comp_1 = (out_eager1.float() - out_comp1.float()).abs().max().item()

    return (
        base_vs_eager_0,
        base_vs_comp_0,
        eager_vs_comp_0,
        base_vs_eager_1,
        base_vs_comp_1,
        eager_vs_comp_1,
    )


def test_vllm_scheduler_matches_dreamzero_eager() -> None:
    if not UPSTREAM_SCHEDULER.exists():
        pytest.skip(f"Missing DreamZero scheduler source: {UPSTREAM_SCHEDULER}")

    _set_deterministic()
    baseline_mod = _load_module("vllm_flow_unipc_baseline", VLLM_SCHEDULER, disable_compile=False)
    upstream_eager_mod = _load_module("dreamzero_flow_unipc_eager", UPSTREAM_SCHEDULER, disable_compile=True)
    upstream_compiled_mod = _load_module("dreamzero_flow_unipc_compiled_aux", UPSTREAM_SCHEDULER, disable_compile=False)

    (
        base_vs_eager_0,
        _base_vs_comp_0,
        _eager_vs_comp_0,
        base_vs_eager_1,
        _base_vs_comp_1,
        _eager_vs_comp_1,
    ) = _run_step0_step1_pair(
        baseline_cls=baseline_mod.FlowUniPCMultistepScheduler,
        upstream_eager_cls=upstream_eager_mod.FlowUniPCMultistepScheduler,
        upstream_compiled_cls=upstream_compiled_mod.FlowUniPCMultistepScheduler,
    )

    assert base_vs_eager_0 == 0.0
    assert base_vs_eager_1 == 0.0


def test_dreamzero_scheduler_compile_differs_from_eager_on_cuda_bf16() -> None:
    if not UPSTREAM_SCHEDULER.exists():
        pytest.skip(f"Missing DreamZero scheduler source: {UPSTREAM_SCHEDULER}")

    _set_deterministic()
    baseline_mod = _load_module("vllm_flow_unipc_baseline_aux", VLLM_SCHEDULER, disable_compile=False)
    upstream_eager_mod = _load_module("dreamzero_flow_unipc_eager_aux", UPSTREAM_SCHEDULER, disable_compile=True)
    upstream_compiled_mod = _load_module("dreamzero_flow_unipc_compiled", UPSTREAM_SCHEDULER, disable_compile=False)

    (
        _base_vs_eager_0,
        _base_vs_comp_0,
        eager_vs_comp_0,
        _base_vs_eager_1,
        _base_vs_comp_1,
        eager_vs_comp_1,
    ) = _run_step0_step1_pair(
        baseline_cls=baseline_mod.FlowUniPCMultistepScheduler,
        upstream_eager_cls=upstream_eager_mod.FlowUniPCMultistepScheduler,
        upstream_compiled_cls=upstream_compiled_mod.FlowUniPCMultistepScheduler,
    )

    assert eager_vs_comp_0 >= 1e-3
    assert eager_vs_comp_1 >= 1e-3
