import numpy as np

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def test_step_splits_dict_output_into_images_and_multimodal_output():
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.pre_process_func = None
    engine.post_process_func = None
    engine.od_config = type("Cfg", (), {"enable_cpu_offload": False, "model_class_name": "DreamZeroPipeline"})()
    engine.add_req_and_wait_for_response = lambda request: DiffusionOutput(
        output={
            "video": np.zeros((1, 4, 16, 8, 8), dtype=np.float32),
            "actions": np.zeros((24, 8), dtype=np.float32),
        },
    )

    request = OmniDiffusionRequest(
        prompts=["robot prompt"],
        request_ids=["req-1"],
        sampling_params=OmniDiffusionSamplingParams(),
    )

    results = DiffusionEngine.step(engine, request)

    assert len(results) == 1
    assert len(results[0].images) == 1
    assert "actions" in results[0].multimodal_output
    assert results[0].multimodal_output["actions"].shape == (24, 8)


def test_step_slices_actions_per_request_for_multi_request_batch():
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.pre_process_func = None
    engine.post_process_func = None
    engine.od_config = type("Cfg", (), {"enable_cpu_offload": False, "model_class_name": "DreamZeroPipeline"})()
    engine.add_req_and_wait_for_response = lambda request: DiffusionOutput(
        output={
            "video": [
                np.zeros((4, 16, 8, 8), dtype=np.float32),
                np.ones((4, 16, 8, 8), dtype=np.float32),
            ],
            "actions": np.stack(
                [
                    np.zeros((24, 8), dtype=np.float32),
                    np.ones((24, 8), dtype=np.float32),
                ],
                axis=0,
            ),
        },
    )

    request = OmniDiffusionRequest(
        prompts=["robot prompt 0", "robot prompt 1"],
        request_ids=["req-0", "req-1"],
        sampling_params=OmniDiffusionSamplingParams(num_outputs_per_prompt=1),
    )

    results = DiffusionEngine.step(engine, request)

    assert len(results) == 2
    assert len(results[0].images) == 1
    assert len(results[1].images) == 1
    np.testing.assert_array_equal(results[0].multimodal_output["actions"], np.zeros((24, 8), dtype=np.float32))
    np.testing.assert_array_equal(results[1].multimodal_output["actions"], np.ones((24, 8), dtype=np.float32))
