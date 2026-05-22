from __future__ import annotations

from itertools import count
from types import SimpleNamespace

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.outputs import CompletionOutput, RequestOutput

from vllm_iter import IterableLLM


class FakeEngine:
    def __init__(self):
        self.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=2)
        )
        self.requests = []

    def get_num_unfinished_requests(self):
        return len(self.requests)

    def add_request(
        self,
        request_id,
        engine_input,
        params,
        *,
        lora_request=None,
        priority=0,
    ):
        if engine_input == "too long":
            raise VLLMValidationError(
                "maximum context length exceeded",
                parameter="input_tokens",
                value=16385,
            )
        self.requests.append((request_id, engine_input))

    def step(self):
        request_id, prompt = self.requests.pop(0)
        return [
            RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=[1],
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text=f"ok: {prompt}",
                        token_ids=[2],
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason="stop",
                    )
                ],
                finished=True,
            )
        ]

    def abort_request(self, request_ids, internal=False):
        self.requests = [
            request for request in self.requests if request[0] not in request_ids
        ]


class FakeIterableLLM(IterableLLM):
    def __init__(self):
        self.model_config = SimpleNamespace(runner_type="generate")
        self.llm_engine = FakeEngine()
        self.request_counter = count()

    def _preprocess_cmpl_one(
        self,
        prompt,
        *,
        tokenization_kwargs=None,
        mm_processor_kwargs=None,
    ):
        return prompt

    def _resolve_mm_lora(self, engine_input, lora_request):
        return lora_request


def test_generate_iter_yields_validation_error_and_continues_in_input_order():
    llm = FakeIterableLLM()

    outputs = list(
        llm.generate_iter(
            ["first", "too long", "third"],
            sampling_params=SamplingParams(max_tokens=1),
            use_tqdm=False,
            max_inflight=2,
        )
    )

    assert [output.outputs[0].text for output in outputs] == [
        "ok: first",
        "maximum context length exceeded (parameter=input_tokens, value=16385)",
        "ok: third",
    ]
    assert [output.outputs[0].finish_reason for output in outputs] == [
        "stop",
        "error",
        "stop",
    ]
    assert all(output.finished for output in outputs)
