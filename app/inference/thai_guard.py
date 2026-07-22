from __future__ import annotations

from typing import Iterable


def _head_tail(token_ids: list[int], limit: int) -> list[int]:
    """Keep both the instruction opening and any important suffix."""
    if limit <= 0:
        return []
    if len(token_ids) <= limit:
        return list(token_ids)
    if limit == 1:
        return token_ids[:1]

    tail = max(1, limit // 4)
    head = limit - tail
    return token_ids[:head] + token_ids[-tail:]


def allocate_pair_token_budget(
    query_length: int,
    response_length: int,
    content_budget: int,
    response_reserve: int,
) -> tuple[int, int]:
    """Allocate a fixed context window without allowing the query to hide output."""
    if content_budget < 0:
        raise ValueError("content_budget must be non-negative")

    response_budget = min(response_length, response_reserve, content_budget)
    query_budget = min(query_length, content_budget - response_budget)
    remaining = content_budget - response_budget - query_budget

    response_extra = min(max(0, response_length - response_budget), remaining)
    response_budget += response_extra
    remaining -= response_extra
    query_budget += min(max(0, query_length - query_budget), remaining)
    return query_budget, response_budget


class ThaiSafetyGuard:
    def __init__(self, config: dict) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.config = config
        self.torch = torch
        self.device = config["guards"].get("thai_device", "cpu")
        self.batch_size = int(config["guards"]["thai_batch_size"])
        self.max_length = int(config["guards"].get("thai_max_length", 128))
        self.response_reserve = int(
            config["guards"].get("thai_response_token_reserve", 64)
        )
        model_path = config["models"]["thai_guard"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _prepare_pair(self, query: str, response: str) -> tuple[dict, dict]:
        prefix_ids = self.tokenizer.encode("input: ", add_special_tokens=False)
        separator_ids = self.tokenizer.encode(" output: ", add_special_tokens=False)
        query_ids = self.tokenizer.encode(query, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        content_budget = (
            self.max_length - special_tokens - len(prefix_ids) - len(separator_ids)
        )
        if content_budget < 1:
            raise ValueError(
                "thai_max_length is too small for guard serialization markers"
            )

        query_budget, response_budget = allocate_pair_token_budget(
            query_length=len(query_ids),
            response_length=len(response_ids),
            content_budget=content_budget,
            response_reserve=self.response_reserve,
        )
        visible_query = _head_tail(query_ids, query_budget)
        visible_response = _head_tail(response_ids, response_budget)
        content_ids = prefix_ids + visible_query + separator_ids + visible_response
        encoded = self.tokenizer.prepare_for_model(
            content_ids,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
        )
        if len(encoded["input_ids"]) > self.max_length:
            raise ValueError("guard serialization exceeded thai_max_length")

        visibility = {
            "guard_query_total_tokens": len(query_ids),
            "guard_query_visible_tokens": len(visible_query),
            "guard_response_total_tokens": len(response_ids),
            "guard_response_token_budget": response_budget,
            "estimated_visible_tokens": len(visible_response),
        }
        return encoded, visibility

    def classify_batch(
        self,
        queries: Iterable[str],
        responses: Iterable[str],
    ) -> list[dict]:
        pairs = list(zip(queries, responses, strict=True))
        if not pairs:
            return []

        results: list[dict] = []
        threshold = float(self.config["guards"]["thai_harmful_threshold"])

        try:
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                prepared = [
                    self._prepare_pair(query, response)[0]
                    for query, response in batch
                ]
                encoded = self.tokenizer.pad(
                    prepared,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with self.torch.no_grad():
                    logits = self.model(**encoded).logits
                    probabilities = self.torch.softmax(logits, dim=-1)

                for probs in probabilities:
                    harmful_probability = infer_harmful_probability(
                        probs.detach().cpu().tolist(),
                        getattr(self.model.config, "id2label", None),
                    )
                    results.append(
                        {
                            "label": (
                                "harmful"
                                if harmful_probability >= threshold
                                else "safe"
                            ),
                            "harmful_probability": harmful_probability,
                        }
                    )
        except Exception as exc:
            results = [
                {
                    "label": "unknown",
                    "harmful_probability": 1.0,
                    "raw_error": str(exc),
                }
                for _ in pairs
            ]

        return results

    def response_visibility_batch(
        self,
        queries: Iterable[str],
        responses: Iterable[str],
    ) -> list[dict]:
        pairs = list(zip(queries, responses, strict=True))
        if not pairs:
            return []

        return [
            self._prepare_pair(query, response)[1]
            for query, response in pairs
        ]


def infer_harmful_probability(
    probabilities: list[float],
    id2label: dict | None,
) -> float:
    if not probabilities:
        return 0.0

    if id2label:
        for index, label in id2label.items():
            if "harm" in str(label).lower() or "unsafe" in str(label).lower():
                return float(probabilities[int(index)])

    if len(probabilities) == 1:
        return float(probabilities[0])
    return float(probabilities[-1])
