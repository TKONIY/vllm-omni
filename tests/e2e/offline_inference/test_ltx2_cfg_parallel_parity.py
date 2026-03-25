# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils import hardware_test

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = REPO_ROOT / "examples" / "offline_inference" / "text_to_video" / "text_to_video.py"
EXPECTED_MD5 = "08e606b9c522fee4b6f30cee8b77db40"
PROMPT = (
    "At sunrise, a glowing paper lantern boat drifts through a narrow canal between mossy stone walls, "
    "soft fog above the water, the camera slowly gliding forward as golden reflections shimmer across "
    "the ripples, cinematic, realistic, highly detailed."
)
NEGATIVE_PROMPT = "worst quality, blurry, jittery motion, distorted, oversaturated, artifacts"

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _get_ltx2_model() -> str:
    return os.environ.get("VLLM_TEST_LTX2_MODEL", "Lightricks/LTX-2")


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.slow
@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_ltx2_cfg_parallel_parity(tmp_path: Path):
    generated_path = tmp_path / "ltx2_refactor_6s.mp4"

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    cmd = [
        sys.executable,
        str(EXAMPLE),
        "--model",
        _get_ltx2_model(),
        "--prompt",
        PROMPT,
        "--negative-prompt",
        NEGATIVE_PROMPT,
        "--height",
        "256",
        "--width",
        "256",
        "--num-frames",
        "145",
        "--num-inference-steps",
        "6",
        "--guidance-scale",
        "4.0",
        "--frame-rate",
        "24",
        "--fps",
        "24",
        "--seed",
        "42",
        "--cfg-parallel-size",
        "2",
        "--enforce-eager",
        "--output",
        str(generated_path),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    generated_md5 = _md5(generated_path)
    assert generated_md5 == EXPECTED_MD5, (
        f"Unexpected output md5: {generated_md5} != {EXPECTED_MD5}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
