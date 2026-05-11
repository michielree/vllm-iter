from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Literal

from tqdm.auto import tqdm

from vllm import LLM, SamplingParams
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind

Ordering = Literal["input"]

logger = logging.getLogger(__name__)


class IterableLLM(LLM):
    def generate_iter(
        self,
        prompts: Iterable[PromptType],
        sampling_params: SamplingParams | None = None,
        *,
        total: int | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: LoRARequest | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        max_inflight: int | None = None,
        ordering: Ordering = "input",
    ) -> Iterator[RequestOutput]:
        """Generate final outputs incrementally from an iterable of prompts.

        `max_inflight` limits the total unfinished requests managed by this
        wrapper. When left as `None`, it follows the engine scheduler's
        `max_num_seqs` capacity rather than using a fixed host-side cap.
        """
        runner_type = self.model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.generate() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        self._validate_generate_iter_args(
            sampling_params=sampling_params,
            lora_request=lora_request,
            priority=priority,
            total=total,
            max_inflight=max_inflight,
            ordering=ordering,
        )

        if self.llm_engine.get_num_unfinished_requests() > 0:
            raise RuntimeError(
                "IterableLLM.generate_iter() requires an idle engine with no "
                "unfinished requests."
            )

        max_inflight = self._resolve_max_inflight(max_inflight)
        logger.info(
            "Starting iterable generation: max_inflight=%s ordering=%s total=%s",
            max_inflight,
            ordering,
            total,
        )
        params = sampling_params
        prompt_iter = iter(prompts)
        request_id_to_index: dict[str, int] = {}
        buffered_outputs: dict[int, RequestOutput] = {}
        next_input_index = 0
        next_output_index = 0
        yielded_count = 0
        exhausted = False
        pbar = self._create_progress_bar(use_tqdm, total)

        try:
            while True:
                while not exhausted and len(request_id_to_index) < max_inflight:
                    try:
                        prompt = next(prompt_iter)
                    except StopIteration:
                        exhausted = True
                        break

                    request_id = self._submit_iter_request(
                        prompt=prompt,
                        params=params,
                        lora_request=lora_request,
                        priority=priority,
                        tokenization_kwargs=tokenization_kwargs,
                        mm_processor_kwargs=mm_processor_kwargs,
                    )
                    request_id_to_index[request_id] = next_input_index
                    logger.debug(
                        "Submitted iterable request: request_id=%s input_index=%s",
                        request_id,
                        next_input_index,
                    )
                    next_input_index += 1

                    if pbar is not None and total is None:
                        pbar.total = next_input_index
                        pbar.refresh()

                if exhausted and not request_id_to_index:
                    break

                step_outputs = self.llm_engine.step()
                for output in step_outputs:
                    if not isinstance(output, RequestOutput) or not output.finished:
                        continue

                    input_index = request_id_to_index.pop(output.request_id, None)
                    if input_index is None:
                        logger.debug(
                            "Ignoring finished output for unknown request: request_id=%s",
                            output.request_id,
                        )
                        continue

                    logger.debug(
                        "Finished iterable request: request_id=%s input_index=%s",
                        output.request_id,
                        input_index,
                    )
                    if pbar is not None:
                        self._update_progress_bar(pbar, output)

                    buffered_outputs[input_index] = output
                    while next_output_index in buffered_outputs:
                        buffered_output = buffered_outputs.pop(next_output_index)
                        logger.debug(
                            "Yielding final output in input order: "
                            "request_id=%s input_index=%s",
                            buffered_output.request_id,
                            next_output_index,
                        )
                        yielded_count += 1
                        yield buffered_output
                        next_output_index += 1
        except Exception:
            if request_id_to_index:
                logger.exception(
                    "Aborting %s unfinished iterable requests after error",
                    len(request_id_to_index),
                )
                self.llm_engine.abort_request(
                    list(request_id_to_index.keys()),
                    internal=False,
                )
            raise
        finally:
            if pbar is not None:
                pbar.close()
            logger.info(
                "Finished iterable generation: submitted=%s yielded=%s",
                next_input_index,
                yielded_count,
            )

    def _submit_iter_request(
        self,
        *,
        prompt: PromptType,
        params: SamplingParams,
        lora_request: LoRARequest | None,
        priority: int,
        tokenization_kwargs: dict[str, Any] | None,
        mm_processor_kwargs: dict[str, Any] | None,
    ) -> str:
        engine_input = self._preprocess_cmpl_one(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        if isinstance(params, SamplingParams):
            params.output_kind = RequestOutputKind.FINAL_ONLY

        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(
            request_id,
            engine_input,
            params,
            lora_request=self._resolve_mm_lora(engine_input, lora_request),
            priority=priority,
        )
        return request_id

    def _validate_generate_iter_args(
        self,
        *,
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None,
        priority: int,
        total: int | None,
        max_inflight: int | None,
        ordering: Ordering,
    ) -> None:
        if isinstance(sampling_params, Sequence):
            raise ValueError(
                "IterableLLM.generate_iter() only accepts a single "
                "sampling_params value."
            )
        if isinstance(lora_request, Sequence):
            raise ValueError(
                "IterableLLM.generate_iter() only accepts a single lora_request value."
            )
        if isinstance(priority, Sequence) or not isinstance(priority, int):
            raise ValueError(
                "IterableLLM.generate_iter() only accepts a single integer priority."
            )
        if total is not None and total < 0:
            raise ValueError("total must be greater than or equal to 0.")
        if max_inflight is not None and max_inflight <= 0:
            raise ValueError("max_inflight must be greater than 0.")
        if ordering != "input":
            raise ValueError("ordering must be 'input'.")

    def _resolve_max_inflight(self, max_inflight: int | None) -> int:
        if max_inflight is not None:
            return max_inflight

        return self.llm_engine.vllm_config.scheduler_config.max_num_seqs

    def _create_progress_bar(
        self,
        use_tqdm: bool | Callable[..., tqdm],
        total: int | None,
    ) -> tqdm | None:
        if not use_tqdm:
            return None

        tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
        return tqdm_func(
            total=0 if total is None else total,
            desc="Processed prompts",
            dynamic_ncols=True,
            postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),
        )

    def _update_progress_bar(self, pbar: tqdm, output: RequestOutput) -> None:
        pbar.update(1)
        if output.prompt_token_ids is None:
            return

        total_in_toks = getattr(pbar, "_vllm_iter_total_in_toks", 0)
        total_out_toks = getattr(pbar, "_vllm_iter_total_out_toks", 0)
        n = len(output.outputs)
        total_in_toks += len(output.prompt_token_ids) * n
        total_out_toks += sum(len(stp.token_ids) for stp in output.outputs)
        setattr(pbar, "_vllm_iter_total_in_toks", total_in_toks)
        setattr(pbar, "_vllm_iter_total_out_toks", total_out_toks)

        elapsed = pbar.format_dict.get("elapsed", 0) or 0
        if elapsed <= 0:
            return

        in_spd = total_in_toks / elapsed
        out_spd = total_out_toks / elapsed
        pbar.postfix = (
            f"est. speed input: {in_spd:.2f} toks/s, output: {out_spd:.2f} toks/s"
        )
        pbar.refresh()
