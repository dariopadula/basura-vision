import csv
import json
import tempfile
import unittest
from pathlib import Path

from basura_vision.bedrock import CATEGORIES
from basura_vision.bedrock_summary import (
    build_photo_labels,
    build_summary,
    list_label_runs,
    write_summary,
)
from basura_vision.manifests import write_jsonl


def labels(**present):
    return {
        category: {
            "presente": category in present,
            "severidad": present.get(category),
        }
        for category in CATEGORIES
    }


class BedrockSummaryTests(unittest.TestCase):
    def _build(self, root: Path) -> tuple[Path, Path]:
        run_dir = root / "runs" / "reference"
        label_dir = run_dir / "bedrock" / "labels1000"
        label_dir.mkdir(parents=True)
        write_jsonl(label_dir / "selection.jsonl", [
            {"id": "a", "path": "photos/a.jpg", "cluster": "1"},
            {"id": "b", "path": "photos/b.jpg", "cluster": "1"},
            {"id": "c", "path": "photos/c.jpg", "cluster": "1"},
            {"id": "d", "path": "photos/d.jpg", "cluster": "sin_residuos"},
        ])
        write_jsonl(label_dir / "results.jsonl", [
            {"id": "a", "path": "photos/a.jpg", "labels": labels(basura_fuera="leve", poda="severo")},
            {"id": "b", "path": "photos/b.jpg", "labels": labels(basura_fuera="medio")},
            {"id": "c", "path": "photos/c.jpg", "labels": labels()},
        ])
        write_jsonl(label_dir / "invocations.jsonl", [
            {
                "id": "inv-1", "path": "photos/a.jpg", "observed_latency_ms": 100,
                "tokens_and_cost": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "estimated_total_cost": "0.001"},
            },
            {
                "id": "inv-2", "path": "photos/b.jpg", "metrics": {"latencyMs": 200},
                "tokens_and_cost": {"input_tokens": 20, "output_tokens": 3, "total_tokens": 23, "estimated_total_cost": "0.002"},
            },
        ])
        (label_dir / "run.json").write_text(json.dumps({
            "status": "in_progress", "model_id": "amazon.nova-lite-v1:0",
            "prompt_version": "bedrock_labels_v1", "cost_currency": "USD",
            "selected": 4, "completed": 3,
        }), encoding="utf-8")
        return run_dir, label_dir

    def test_builds_likert_heatmap_and_cooccurrence_data(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, _ = self._build(Path(directory))
            summary = build_summary(run_dir, "labels1000")

            self.assertEqual(summary["run_status"], "in_progress")
            self.assertEqual(summary["totals"]["selected"], 4)
            self.assertEqual(summary["totals"]["valid"], 3)
            self.assertEqual(summary["totals"]["total_tokens"], 35)
            self.assertEqual(summary["totals"]["estimated_total_cost"], "0.003")
            self.assertEqual(summary["totals"]["median_latency_ms"], 150.0)

            group = summary["groups"][0]
            self.assertEqual(group["group"], "1")
            self.assertEqual(group["coverage_percentage"], 100.0)
            waste = group["labels"]["basura_fuera"]
            self.assertEqual(waste["present"], {"count": 2, "percentage": 66.67})
            self.assertEqual(waste["ausente"], {"count": 1, "percentage": 33.33})
            self.assertEqual(waste["leve"]["percentage"], 33.33)
            self.assertEqual(waste["medio"]["percentage"], 33.33)
            pair = next(
                row for row in group["cooccurrences"]
                if row["category_a"] == "basura_fuera" and row["category_b"] == "poda"
            )
            self.assertEqual(pair, {
                "category_a": "basura_fuera", "category_b": "poda",
                "count": 1, "percentage": 33.33,
            })
            special = summary["groups"][1]
            self.assertEqual((special["group"], special["valid"], special["pending"]), ("sin_residuos", 0, 1))
            self.assertEqual(special["labels"]["poda"]["present"]["percentage"], 0.0)

    def test_writes_reusable_json_and_csv_and_lists_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, _ = self._build(Path(directory))
            output = write_summary(run_dir, "labels1000")

            self.assertTrue((output / "summary.json").is_file())
            with (output / "groups.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual(rows[0]["group"], "1")
            self.assertEqual(rows[0]["present_percentage"], "66.67")
            with (output / "cooccurrences.csv").open(encoding="utf-8-sig", newline="") as handle:
                pairs = list(csv.DictReader(handle))
            self.assertEqual(len(pairs), 30)
            self.assertEqual(list_label_runs(run_dir), [{
                "label_run_id": "labels1000", "status": "in_progress",
                "selected": 4, "completed": 3, "model_id": "amazon.nova-lite-v1:0",
            }])

    def test_returns_compact_photo_labels_for_viewer(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, _ = self._build(Path(directory))
            details = build_photo_labels(run_dir, "labels1000")

            self.assertEqual(len(details["selected"]), 4)
            self.assertEqual(len(details["results"]), 3)
            self.assertEqual(details["selected"][-1], {"id": "d", "cluster": "sin_residuos"})
            result = next(row for row in details["results"] if row["id"] == "a")
            self.assertEqual(result["cluster"], "1")
            self.assertEqual(result["labels"]["basura_fuera"]["severidad"], "leve")

    def test_rejects_result_outside_frozen_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, label_dir = self._build(Path(directory))
            with (label_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "id": "outside", "path": "photos/outside.jpg", "labels": labels(),
                }) + "\n")
            with self.assertRaisesRegex(ValueError, "ajenos a la selección"):
                build_summary(run_dir, "labels1000")


if __name__ == "__main__":
    unittest.main()
