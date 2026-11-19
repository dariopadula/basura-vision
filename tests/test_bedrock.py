import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from basura_vision.bedrock import CATEGORIES, label_batch, label_one_photo
from basura_vision.manifests import write_jsonl
from basura_vision.storage import LocalStorage


class FakeBedrockClient:
    def __init__(self, labels, raw_text=None, usage=None):
        self.labels = labels
        self.raw_text = raw_text
        self.usage = usage or {"inputTokens": 100, "outputTokens": 50}
        self.meta = SimpleNamespace(region_name="us-east-1")
        self.request = None
        self.request_count = 0

    def converse(self, **request):
        self.request = request
        self.request_count += 1
        text = self.raw_text
        if text is None:
            text = f"Resultado:\n```json\n{json.dumps(self.labels)}\n```"
        return {
            "output": {
                "message": {
                    "content": [{"text": text}],
                },
            },
            "usage": self.usage,
            "metrics": {"latencyMs": 25},
            "stopReason": "end_turn",
            "ResponseMetadata": {"RequestId": "request-123"},
        }


class BedrockTests(unittest.TestCase):
    def test_labels_one_original_and_saves_validated_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photos = root / "photos"
            photos.mkdir()
            Image.new("RGB", (20, 10), "white").save(photos / "one.png")
            run_dir = root / "runs" / "reference_30_smoke"
            run_dir.mkdir(parents=True)
            write_jsonl(run_dir / "manifest.jsonl", [{"id": "photo_1", "path": "one.png"}])
            settings = SimpleNamespace(runs_root=root / "runs")
            labels = {
                category: {"presente": False, "severidad": None}
                for category in CATEGORIES
            }
            client = FakeBedrockClient(labels)

            destination = label_one_photo(
                settings,
                LocalStorage(photos),
                "reference_30_smoke",
                "photo_1",
                "amazon.nova-lite-v1:0",
                client=client,
            )

            record = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(record["labels"], labels)
            self.assertEqual(record["image_source"], "original")
            self.assertEqual(record["region"], "us-east-1")
            self.assertEqual(client.request["modelId"], "amazon.nova-lite-v1:0")
            image = client.request["messages"][0]["content"][0]["image"]
            self.assertEqual(image["format"], "jpeg")
            self.assertTrue(image["source"]["bytes"].startswith(b"\xff\xd8"))

    def test_batch_writes_jsonl_summary_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photos = root / "photos"
            photos.mkdir()
            manifest = []
            detections = []
            for index in range(3):
                name = f"photo_{index}.jpg"
                Image.new("RGB", (20, 10), "white").save(photos / name)
                manifest.append({"id": f"photo_{index}", "path": name})
                detections.append({
                    "id": f"photo_{index}",
                    "path": name,
                    "status": "agrupable" if index < 2 else "sin_residuos",
                })
            run_dir = root / "runs" / "reference_30_smoke"
            run_dir.mkdir(parents=True)
            write_jsonl(run_dir / "manifest.jsonl", manifest)
            write_jsonl(run_dir / "detections.jsonl", detections)
            settings = SimpleNamespace(runs_root=root / "runs")
            labels = {
                category: {"presente": False, "severidad": None}
                for category in CATEGORIES
            }
            client = FakeBedrockClient(labels)

            label_dir = label_batch(
                settings,
                LocalStorage(photos),
                "reference_30_smoke",
                "nova_lite_labels_v1",
                "amazon.nova-lite-v1:0",
                checkpoint_every=2,
                client=client,
            )

            results = (label_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            summary = json.loads((label_dir / "cluster_summary.json").read_text(encoding="utf-8"))
            run = json.loads((label_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(len(results), 3)
            self.assertEqual(summary["completed"], 3)
            self.assertEqual(run["status"], "complete")
            self.assertEqual(client.request_count, 3)

            label_batch(
                settings,
                LocalStorage(photos),
                "reference_30_smoke",
                "nova_lite_labels_v1",
                "amazon.nova-lite-v1:0",
                checkpoint_every=2,
                client=client,
            )
            self.assertEqual(client.request_count, 3)

    def test_invalid_response_keeps_usage_and_estimated_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photos = root / "photos"
            photos.mkdir()
            Image.new("RGB", (20, 10), "white").save(photos / "one.jpg")
            run_dir = root / "runs" / "reference"
            run_dir.mkdir(parents=True)
            write_jsonl(run_dir / "manifest.jsonl", [{"id": "photo_1", "path": "one.jpg"}])
            write_jsonl(
                run_dir / "detections.jsonl",
                [{"id": "photo_1", "path": "one.jpg", "status": "sin_residuos"}],
            )
            settings = SimpleNamespace(runs_root=root / "runs")
            client = FakeBedrockClient({}, raw_text="respuesta no JSON")

            label_dir = label_batch(
                settings,
                LocalStorage(photos),
                "reference",
                "invalid_response",
                "amazon.nova-lite-v1:0",
                client=client,
            )

            invocation = json.loads(
                (label_dir / "invocations.jsonl").read_text(encoding="utf-8").strip()
            )
            run = json.loads((label_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(invocation["status"], "invalid_response")
            self.assertEqual(invocation["aws_request_id"], "request-123")
            self.assertEqual(invocation["tokens_and_cost"]["input_tokens"], 100)
            self.assertEqual(invocation["tokens_and_cost"]["output_tokens"], 50)
            self.assertEqual(
                invocation["tokens_and_cost"]["estimated_total_cost"],
                "0.000018000000",
            )
            self.assertEqual(run["completed"], 0)
            self.assertEqual(run["total_tokens"], 150)
            self.assertEqual(run["estimated_total_cost"], "0.000018000000")
            self.assertFalse((label_dir / "results.jsonl").exists())

if __name__ == "__main__":
    unittest.main()
