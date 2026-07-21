from __future__ import annotations

from collections import defaultdict
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
        }
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

    def generate(self, messages_list: list[list[dict]], max_tokens: Iterable[int]) -> list[str]:
        from vllm import SamplingParams

        prompts = [self._format_messages(messages) for messages in messages_list]
        token_budgets = [int(value) for value in max_tokens]
        outputs: list[str] = [""] * len(prompts)
        finish_reasons: list[str] = ["no_output"] * len(prompts)

        grouped: dict[int, list[int]] = defaultdict(list)
        for index, token_budget in enumerate(token_budgets):
            grouped[token_budget].append(index)

        generation = self.config["generation"]
        for token_budget, indices in grouped.items():
            sampling = SamplingParams(
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_tokens=token_budget,
                seed=int(generation["seed"]),
            )
            batch_prompts = [prompts[index] for index in indices]
            batch_outputs = self.llm.generate(batch_prompts, sampling)
            for index, output in zip(indices, batch_outputs, strict=True):
                if output.outputs:
                    outputs[index] = output.outputs[0].text
                    finish_reasons[index] = str(
                        getattr(output.outputs[0], "finish_reason", None) or "unknown"
                    )

        self.last_finish_reasons = finish_reasons
        return outputs
