# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc


def test_enrich_config_detects_dreamzero_from_namespace(monkeypatch):
    def fake_get_hf_file_to_dict(filename: str, model_name: str):
        assert model_name == "dreamzero-local"
        if filename == "model_index.json":
            raise FileNotFoundError
        if filename == "config.json":
            return {
                "model_type": "vla",
                "architectures": ["VLA"],
                "action_head_cfg": {
                    "_target_": "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead",
                },
            }
        raise AssertionError(f"Unexpected filename: {filename}")

    monkeypatch.setattr(
        "vllm_omni.diffusion.stage_diffusion_proc.get_hf_file_to_dict",
        fake_get_hf_file_to_dict,
    )

    od_config = SimpleNamespace(
        model="dreamzero-local",
        model_class_name=None,
        tf_model_config=None,
        multimodal_support_updated=False,
    )
    od_config.update_multimodal_support = lambda: setattr(
        od_config, "multimodal_support_updated", True
    )

    proc = StageDiffusionProc(model="dreamzero-local", od_config=od_config)
    proc._enrich_config()

    assert od_config.model_class_name == "DreamZeroPipeline"
    assert od_config.tf_model_config is not None
    assert od_config.multimodal_support_updated is True
