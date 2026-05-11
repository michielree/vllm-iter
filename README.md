# vllm-iter

`vllm-iter` provides iterable-friendly offline generation utilities on top of
`vllm.LLM`.

`IterableLLM.generate_iter()` keeps a bounded number of unfinished requests in
flight while yielding final completions incrementally in input order. The
`max_inflight` setting controls wrapper-managed unfinished requests, not the
model-side microbatch size. By default it follows vLLM's scheduler capacity
(`max_num_seqs`), and you can set a smaller or larger value to tune host-side
backpressure. When the input iterable length is known ahead of time, pass
`total=` to keep the tqdm progress bar stable instead of growing it dynamically
as prompts are submitted.

## Motivation

This is useful when your prompts come from a large streaming source and you do
not want to materialize either the full input set or the full output set in
memory.

For example, imagine a huge JSON Lines file. You can iterate over the input
lines, convert each one into a `PromptType`, and feed that iterable directly
into `generate_iter()`. As results come back, you can immediately append them
to an output JSON Lines file and forget about them. That keeps memory bounded and
also means a crash or interruption does not lose everything that was already
written.

## Installation

```bash
pip install git+https://github.com/michielree/vllm-iter.git
```

## Example

```python
from vllm import SamplingParams

from vllm_iter import IterableLLM


def prompt_source():
    for topic in ["redis", "postgres", "kafka"]:
        yield f"Give me a one-line summary of {topic}."


llm = IterableLLM(
    model="distilbert/distilgpt2",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.10,
    enforce_eager=True,
)

for output in llm.generate_iter(
    prompt_source(),
    sampling_params=SamplingParams(max_tokens=32),
    use_tqdm=False,
):
    print(output.prompt)
    print(output.outputs[0].text)
```
