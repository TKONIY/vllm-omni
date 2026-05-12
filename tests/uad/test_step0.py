from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3_uad import (
    HunyuanImage3UADForConditionalGeneration,
)
from vllm_omni.uad.engine import AsyncUADEngine, UADEngine
from vllm_omni.uad.omni.adapter.hunyuan_image3 import HunyuanImage3UADAdapter
from vllm_omni.uad.runner import UADRunner

pytestmark = pytest.mark.cpu


def test_uad_engine_step0_routes_through_hunyuan_uad_entrypoint() -> None:
    model = HunyuanImage3UADForConditionalGeneration(vocab_size=100)
    engine = UADEngine(runner=UADRunner(adapter=HunyuanImage3UADAdapter(model=model)))

    request = engine.add_request("req-0", [10, 11])
    output = engine.step()

    assert output.request_ids == ["req-0"]
    assert model.forward_calls == 1
    assert [token.token_id for token in output.outputs[0].new_engine_tokens] == [12]
    assert [token.token_id for token in request.engine_tokens] == [10, 11, 12]
    assert [token.token_id for token in request.materialized_tokens] == [12]
    assert request.num_computed_tokens == 2
    assert request.phase == "ar_decode"


def test_uad_engine_step0_decode_computes_previous_sampled_token() -> None:
    model = HunyuanImage3UADForConditionalGeneration(vocab_size=100)
    engine = UADEngine(runner=UADRunner(adapter=HunyuanImage3UADAdapter(model=model)))

    request = engine.add_request("req-0", [7])
    engine.step()
    engine.step()

    assert [token.token_id for token in request.engine_tokens] == [7, 8, 9]
    assert [token.token_id for token in request.materialized_tokens] == [8, 9]
    assert request.num_computed_tokens == 2
    assert model.forward_calls == 2


@pytest.mark.asyncio
async def test_async_uad_engine_step0_wrapper() -> None:
    async_engine = AsyncUADEngine()

    request = await async_engine.add_request("req-async", [1])
    output = await async_engine.step()

    assert output.request_ids == ["req-async"]
    assert [token.token_id for token in request.engine_tokens] == [1, 2]
    assert [token.token_id for token in request.materialized_tokens] == [2]
