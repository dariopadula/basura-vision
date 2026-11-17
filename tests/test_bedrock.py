import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from basura_vision.bedrock import CATEGORIES, label_one_photo
from basura_vision.manifests import write_jsonl
from basura_vision.storage import LocalStorage


class FakeBedrockClient:
    def __init__(self, labels):
        self.labels = labels
        self.meta = SimpleNamespace(region_name="us-east-1")
        self.request = None

    def converse(self, **request):
        self.request = request
        return {
            "output": {"message": {"content": [{"text": json.dumps(self.labels)}]}},
            "usage": {"inputTokens": 100, "outputTokens": 50},
            "metrics": {"latencyMs": 25},
            "stopReason": "end_turn",
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


if __name__ == "__main__":
    unittest.main()
