from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import run
from app.config import load_config
from app.inference.generator import MainGenerator
from app.io_csv import find_input_csv, read_queries, validate_submission, write_submission
from app.progress import report_progress


class CsvContractTests(unittest.TestCase):
    def test_dataset_csv_is_preferred_and_output_order_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "other.csv").write_text("id,query\nwrong,no\n", encoding="utf-8")
            (root / "dataset.csv").write_text(
                "id,question\nb,second\na,first\n", encoding="utf-8"
            )

            input_path = find_input_csv(root)
            self.assertEqual(input_path.name, "dataset.csv")
            records = read_queries(input_path)
            self.assertEqual([record["query"] for record in records], ["second", "first"])
            output_path = root / "submission.csv"
            write_submission(records, ["two", "one"], output_path)

            self.assertEqual(validate_submission(records, output_path), 2)
            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["id"] for row in rows], ["b", "a"])

    def test_validator_rejects_reordered_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            path.write_text("id,response\na,one\nb,two\n", encoding="utf-8")
            records = [{"id": "b"}, {"id": "a"}]
            with self.assertRaisesRegex(ValueError, "ids/order"):
                validate_submission(records, path)

    def test_official_250_row_question_format_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "dataset.csv"
            rows = ["id,question"] + [
                f"q{index:03d},question {index}" for index in range(250)
            ]
            input_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            records = read_queries(input_path)
            output_path = root / "submission.csv"
            write_submission(
                records,
                [f"response {index}" for index in range(250)],
                output_path,
            )

            self.assertEqual(len(records), 250)
            self.assertEqual(records[0]["id"], "q000")
            self.assertEqual(records[-1]["id"], "q249")
            self.assertEqual(validate_submission(records, output_path), 250)


class ConfigTests(unittest.TestCase):
    def test_runtime_gpu_overrides_do_not_leak_between_loads(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "VLLM_GPU_MEMORY_UTILIZATION": "0.75",
                "VLLM_MAX_NUM_SEQS": "12",
                "THAI_GUARD_BATCH_SIZE": "16",
            },
            clear=True,
        ):
            overridden = load_config("missing.yaml")
        with patch.dict("os.environ", {}, clear=True):
            defaults = load_config("missing.yaml")

        self.assertEqual(overridden["generation"]["gpu_memory_utilization"], 0.75)
        self.assertEqual(overridden["generation"]["max_num_seqs"], 12)
        self.assertEqual(overridden["guards"]["thai_batch_size"], 16)
        self.assertEqual(defaults["generation"]["gpu_memory_utilization"], 0.84)
        self.assertEqual(defaults["generation"]["max_num_seqs"], 24)


class GeneratorBatchingTests(unittest.TestCase):
    def test_mixed_token_budgets_use_one_vllm_batch(self) -> None:
        calls: list[tuple[list[str], list[object]]] = []

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[-1]["content"]

        class FakeLLM:
            init_kwargs: dict = {}

            def __init__(self, **kwargs):
                self.init_kwargs = kwargs

            def get_tokenizer(self):
                return FakeTokenizer()

            def generate(self, prompts, sampling_params):
                calls.append((prompts, sampling_params))
                return [
                    types.SimpleNamespace(
                        outputs=[
                            types.SimpleNamespace(text=f"answer:{prompt}", finish_reason="stop")
                        ]
                    )
                    for prompt in prompts
                ]

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams
        config = {
            "models": {"generator": "/model"},
            "generation": {
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.8,
                "max_num_seqs": 24,
                "max_num_batched_tokens": 16384,
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "seed": 42,
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }

        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            generator = MainGenerator(config)
            outputs = generator.generate(
                [[{"role": "user", "content": value}] for value in ("a", "b", "c")],
                [128, 384, 224],
            )

        self.assertEqual(outputs, ["answer:a", "answer:b", "answer:c"])
        self.assertEqual(len(calls), 1)
        self.assertEqual([item.max_tokens for item in calls[0][1]], [128, 384, 224])
        self.assertEqual(generator.last_finish_reasons, ["stop", "stop", "stop"])


class ProgressTests(unittest.TestCase):
    def test_progress_retries_transient_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "progress"
            executable.touch()
            failures = [
                subprocess.CalledProcessError(1, "progress"),
                subprocess.TimeoutExpired("progress", 1),
                None,
            ]
            with (
                patch("app.progress.subprocess.run", side_effect=failures) as mocked_run,
                patch("app.progress.time.sleep"),
            ):
                report_progress(str(executable), 7, attempts=3)
            self.assertEqual(mocked_run.call_count, 3)


class EndToEndContractTests(unittest.TestCase):
    def test_n_is_reported_only_after_valid_submission_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "model" / "test"
            input_dir.mkdir(parents=True)
            (input_dir / "dataset.csv").write_text(
                "id,question\n1,q1\n2,q2\n3,q3\n", encoding="utf-8"
            )
            output_path = root / "result" / "submission.csv"
            status_path = root / "result" / "run_status.json"
            config = {
                "paths": {
                    "input_dir": str(input_dir),
                    "output_file": str(output_path),
                    "status_file": str(status_path),
                    "progress_program": str(root / "benchmark_lib" / "progress"),
                }
            }
            progress_values: list[int] = []

            class FakePipeline:
                def __init__(self, config):
                    self.diagnostics = {"fake": True}

                def process(self, records):
                    return [f"response-{record['id']}" for record in records]

            fake_pipeline = types.ModuleType("app.pipeline")
            fake_pipeline.TrustworthinessPipeline = FakePipeline

            def capture_progress(*, executable, completed, **kwargs):
                if completed == 3:
                    self.assertEqual(validate_submission(read_queries(input_dir / "dataset.csv"), output_path), 3)
                progress_values.append(completed)

            with (
                patch.dict(sys.modules, {"app.pipeline": fake_pipeline}),
                patch.object(run, "load_config", return_value=config),
                patch.object(run, "report_progress", side_effect=capture_progress),
                patch.dict("os.environ", {"STARTUP_SLEEP_SECONDS": "0"}, clear=False),
            ):
                run.main()

            self.assertEqual(progress_values, [3])
            self.assertTrue(status_path.is_file())

    def test_model_failure_still_produces_a_valid_submission_and_reports_n(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "model" / "test"
            input_dir.mkdir(parents=True)
            (input_dir / "dataset.csv").write_text(
                "id,question\n1,hello\n", encoding="utf-8"
            )
            output_path = root / "result" / "submission.csv"
            status_path = root / "result" / "run_status.json"
            config = {
                "paths": {
                    "input_dir": str(input_dir),
                    "output_file": str(output_path),
                    "status_file": str(status_path),
                    "progress_program": str(root / "benchmark_lib" / "progress"),
                }
            }
            progress_values: list[int] = []

            class FailingPipeline:
                def __init__(self, config):
                    pass

                def process(self, records):
                    raise RuntimeError("simulated model OOM")

            fake_pipeline = types.ModuleType("app.pipeline")
            fake_pipeline.TrustworthinessPipeline = FailingPipeline

            with (
                patch.dict(sys.modules, {"app.pipeline": fake_pipeline}),
                patch.object(run, "load_config", return_value=config),
                patch.object(
                    run,
                    "report_progress",
                    side_effect=lambda *, completed, **kwargs: progress_values.append(
                        completed
                    ),
                ),
                patch.dict(
                    "os.environ",
                    {
                        "STARTUP_SLEEP_SECONDS": "0",
                        "FAIL_ON_EMERGENCY_FALLBACK": "0",
                    },
                    clear=False,
                ),
            ):
                run.main()

            records = read_queries(input_dir / "dataset.csv")
            self.assertEqual(validate_submission(records, output_path), 1)
            self.assertEqual(progress_values, [1])
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "emergency_fallback")
            self.assertEqual(status["progress_completed"], 1)


if __name__ == "__main__":
    unittest.main()
