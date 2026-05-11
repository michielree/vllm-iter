import weakref

import pytest

from vllm import SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from vllm_iter import IterableLLM

MODEL_NAME = "distilbert/distilgpt2"

pytestmark = [pytest.mark.integration, pytest.mark.slow]

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = IterableLLM(
        model=MODEL_NAME,
        max_num_batched_tokens=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.10,
        enforce_eager=True,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


def test_generate_iter_matches_generate_prompt_order(llm: IterableLLM):
    sampling_params = SamplingParams(max_tokens=8)

    def prompt_iter():
        yield from PROMPTS

    iter_outputs = list(
        llm.generate_iter(
            prompt_iter(),
            sampling_params=sampling_params,
            use_tqdm=False,
            max_inflight=2,
        )
    )

    assert len(iter_outputs) == len(PROMPTS)
    assert [output.prompt for output in iter_outputs] == PROMPTS
    assert all(output.finished for output in iter_outputs)
    assert all(output.outputs for output in iter_outputs)
    assert all(output.outputs[0].token_ids for output in iter_outputs)
    assert all(isinstance(output.outputs[0].text, str) for output in iter_outputs)
