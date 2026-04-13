import os

from vllm_omni.entrypoints.utils import resolve_model_config_path


def test_lingbot_world_fast_config_resolution(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.file_or_path_exists",
        lambda model, filename, revision=None: filename == "config.json",
    )

    def raise_value_error(*args, **kwargs):
        raise ValueError("not a transformers config")

    monkeypatch.setattr("vllm_omni.entrypoints.utils.get_config", raise_value_error)
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.get_hf_file_to_dict",
        lambda filename, model, revision=None: {"_class_name": "WanModel", "model_type": "i2v"},
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.current_omni_platform.get_default_stage_config_path",
        lambda: "vllm_omni/model_executor/stage_configs",
    )

    result = resolve_model_config_path("robbyant/lingbot-world-fast")

    assert result is not None
    assert os.path.basename(result) == "lingbot_world_fast.yaml"
