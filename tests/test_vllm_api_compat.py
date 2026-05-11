from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind


def test_vllm_symbols_required_by_iterable_llm_are_available():
    assert LLM is not None
    assert SamplingParams is not None
    assert PromptType is not None
    assert LoRARequest is not None
    assert RequestOutput is not None
    assert RequestOutputKind is not None
    assert cleanup_dist_env_and_memory is not None
    assert callable(getattr(LLM, "_preprocess_cmpl_one", None))
