# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.diffusion.utils import hf_utils


def test_is_diffusion_model_detects_dreamzero(monkeypatch):
    hf_utils.is_diffusion_model.cache_clear()

    def fake_get_hf_file_to_dict(filename: str, model_name: str):
        assert filename == "model_index.json" or filename == "config.json"
        if filename == "model_index.json":
            raise FileNotFoundError
        return {
            "model_type": "vla",
            "architectures": ["VLA"],
            "action_head_cfg": {
                "_target_": "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead",
            },
        }

    monkeypatch.setattr(hf_utils, "get_hf_file_to_dict", fake_get_hf_file_to_dict)
    monkeypatch.setattr(hf_utils, "load_diffusers_config", lambda model_name: (_ for _ in ()).throw(FileNotFoundError))

    assert hf_utils.is_diffusion_model("dreamzero-local") is True
    hf_utils.is_diffusion_model.cache_clear()
