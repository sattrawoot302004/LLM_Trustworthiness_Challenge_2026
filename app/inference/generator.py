from __future__ import annotations

from typing import Iterable


class MainGenerator:
    def __init__(self, config: dict) -> None:
        from vllm import LLM

        self.config = config
        model_path = config["models"]["generator"]
        generation = config["generation"]

        llm_kwargs = {
            "model": model_path,
            "tokenizer": model_path,
            "trust_remote_code": True,
            "dtype": "auto",
            "quantization": None,
            "max_model_len": int(generation["max_model_len"]),
            "gpu_memory_utilization": float(generation["gpu_memory_utilization"]),
            "max_num_seqs": int(generation["max_num_seqs"]),
            "seed": int(generation["seed"]),
            "disable_log_stats": True,
        }
        for optional_key in (
            "max_num_batched_tokens",
            "enforce_eager",
            "enable_prefix_caching",
            "enable_chunked_prefill",
        ):
            if optional_key in generation:
                llm_kwargs[optional_key] = generation[optional_key]
        if generation.get("language_model_only", False):
            llm_kwargs["language_model_only"] = True
        if generation.get("speculative_config"):
            llm_kwargs["speculative_config"] = dict(
                generation["speculative_config"]
            )

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.chat_template_kwargs = dict(
            generation.get("chat_template_kwargs") or {}
        )
        self.last_finish_reasons: list[str] = []

    def _format_messages(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    def generate(
        self,
        messages_list: list[list[dict]],
        max_tokens: Iterable[int],
    ) -> list[str]:
        from vllm import SamplingParams

        prompts = [self._format_messages(messages) for messages in messages_list]
        token_budgets = [int(value) for value in max_tokens]
        if len(prompts) != len(token_budgets):
            raise ValueError(
                f"Prompt/token-budget count mismatch: "
                f"{len(prompts)} != {len(token_budgets)}"
            )
        if not prompts:
            self.last_finish_reasons = []
            return []

        outputs: list[str] = [""] * len(prompts)
        finish_reasons: list[str] = ["no_output"] * len(prompts)

        generation = self.config["generation"]
        # vLLM accepts one SamplingParams object per request.  Sending the
        # mixed-length requests together lets its continuous-batching
        # scheduler keep the H100 occupied instead of draining one token-
        # budget group before starting the next.
        sampling_params = [
            SamplingParams(
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_tokens=token_budget,
                seed=int(generation["seed"]),
            )
            for token_budget in token_budgets
        ]
        batch_outputs = self.llm.generate(prompts, sampling_params)
        if len(batch_outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM output count mismatch: {len(batch_outputs)} != {len(prompts)}"
            )
        for index, output in enumerate(batch_outputs):
            if output.outputs:
                outputs[index] = output.outputs[0].text
                finish_reasons[index] = str(
                    getattr(output.outputs[0], "finish_reason", None) or "unknown"
                )

        self.last_finish_reasons = finish_reasons
        return outputs
