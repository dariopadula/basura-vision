import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from basura_vision.bedrock import CATEGORIES
from basura_vision.bedrock_review import save_blind_reviews
from basura_vision.bedrock_review_report import build_review_report
from basura_vision.cli import main
from basura_vision.manifests import write_jsonl


def labels():
    return {
        category: {"presente": False, "severidad": None}
        for category in CATEGORIES
    }


def review(photo_id: str, category_values: dict) -> dict:
    categories = {
        category: {
            "presente": False,
            "severidad": None,
            "evidencia_dudosa": False,
        }
        for category in CATEGORIES
    }
    categories.update(category_values)
    return {
        "id": photo_id,
        "path": f"photos/{photo_id}.jpg",
        "reviewed": True,
        "categories": categories,
        "notes": "revisión humana",
    }


class BedrockReviewReportTests(unittest.TestCase):
    def test_builds_comparison_and_flags_false_severe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label_dir = root / "runs" / "reference" / "bedrock" / "pilot40"
            label_dir.mkdir(parents=True)
            selection = [
                {"id": "photo_1", "path": "photos/photo_1.jpg", "cluster": 1},
                {"id": "photo_2", "path": "photos/photo_2.jpg", "cluster": 2},
            ]
            first = labels()
            first["contenedor_desbordado"] = {"presente": True, "severidad": "severo"}
            first["contenedor_roto"] = {"presente": True, "severidad": "severo"}
            second = labels()
            second["contenedor_desbordado"] = {"presente": True, "severidad": "severo"}
            write_jsonl(label_dir / "selection.jsonl", selection)
            write_jsonl(label_dir / "results.jsonl", [
                {"id": "photo_1", "path": "photos/photo_1.jpg", "labels": first},
                {"id": "photo_2", "path": "photos/photo_2.jpg", "labels": second},
            ])
            save_blind_reviews(label_dir / "blind_reviews.json", {
                "photo_1": review("photo_1", {
                    "contenedor_roto": {
                        "presente": False,
                        "severidad": None,
                        "evidencia_dudosa": True,
                    },
                }),
                "photo_2": review("photo_2", {
                    "contenedor_desbordado": {
                        "presente": True,
                        "severidad": "medio",
                        "evidencia_dudosa": False,
                    },
                }),
            })
            settings = SimpleNamespace(runs_root=root / "runs")

            destination = build_review_report(settings, "reference", "pilot40")

            report = json.loads((label_dir / "review_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(report["coverage"]["comparable"], 2)
            self.assertEqual(
                report["priority"]["false_severe_conclusive"]["contenedor_desbordado"],
                ["photo_1"],
            )
            self.assertEqual(
                report["priority"]["false_severe_conclusive"]["contenedor_roto"],
                [],
            )
            desborde = report["categories"]["contenedor_desbordado"]
            self.assertEqual(desborde["presence"]["false_positive"], 1)
            self.assertEqual(desborde["severity"]["model_overestimated"], 1)
            with (label_dir / "review_comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 12)
            self.assertEqual(destination, label_dir / "review_comparison.md")

    def test_report_cli_does_not_open_storage(self):
        settings = SimpleNamespace(runs_root=Path("runs"))
        with (
            patch("basura_vision.cli.load_settings", return_value=settings),
            patch("basura_vision.cli.photo_storage", side_effect=AssertionError("no debe abrir S3")),
            patch(
                "basura_vision.bedrock_review_report.build_review_report",
                return_value=Path("report.md"),
            ) as build,
            patch("builtins.print"),
        ):
            main([
                "bedrock-review-report",
                "--run-id", "reference",
                "--label-run-id", "pilot40",
            ])
        build.assert_called_once_with(settings, "reference", "pilot40")


if __name__ == "__main__":
    unittest.main()
