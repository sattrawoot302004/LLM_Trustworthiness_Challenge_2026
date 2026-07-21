from __future__ import annotations

from typing import Iterable


class ThaiSafetyGuard:
    def __init__(self, config: dict) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.config = config
        self.torch = torch
        self.device = config["guards"].get("thai_device", "cpu")
        self.batch_size = int(config["guards"]["thai_batch_size"])
        self.max_length = int(config["guards"].get("thai_max_length", 128))
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

    def classify_batch(self, queries: Iterable[str], responses: Iterable[str]) -> list[dict]:
        pairs = [
            f"input: {query} output: {response}"
            for query, response in zip(queries, responses, strict=True)
        ]
        if not pairs:
            return []

        results: list[dict] = []
        threshold = float(self.config["guards"]["thai_harmful_threshold"])

        try:
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
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
