import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from basura_vision.bedrock_sampling import (
    GROUPS,
    bounded_proportional_quotas,
    prepare_stratified_sample,
)
from basura_vision.cli import main
from basura_vision.manifests import read_jsonl, write_jsonl


class BedrockSamplingTests(unittest.TestCase):
    def _build_run(self, root: Path) -> tuple[SimpleNamespace, Path]:
        run_dir = root / "runs" / "reference"
        run_dir.mkdir(parents=True)
        manifest = []
        detections = []
        assignments = []
        for group in GROUPS:
            for number in range(12):
                photo_id = f"{group}_{number:02d}"
                path = f"photos/{photo_id}.jpg"
                manifest.append({
                    "id": photo_id,
                    "path": path,
                    "event_id": f"event_{photo_id}",
                })
                status = group if group in {"sin_contenedor", "sin_residuos"} else "agrupable"
                detections.append({"id": photo_id, "path": path, "status": status})
                if group not in {"sin_contenedor", "sin_residuos"}:
                    assignments.append({
                        "id": photo_id,
                        "cluster": int(group),
                        "pc1": number,
                        "pc2": -number,
                        "ratio": number / 10,
                    })
        write_jsonl(run_dir / "manifest.jsonl", manifest)
        write_jsonl(run_dir / "detections.jsonl", detections)
        write_jsonl(run_dir / "assignments.jsonl", assignments)
        for label_run_id, suffix in (("pilot", 0), ("validation", 1)):
            label_dir = run_dir / "bedrock" / label_run_id
            label_dir.mkdir(parents=True)
            write_jsonl(label_dir / "selection.jsonl", [
                {
                    "id": f"{group}_{suffix:02d}",
                    "path": f"photos/{group}_{suffix:02d}.jpg",
                    "event_id": f"event_{group}_{suffix:02d}",
                    "cluster": group,
                }
                for group in GROUPS
            ])
        return SimpleNamespace(runs_root=root / "runs"), run_dir

    def test_real_reference_quotas_are_exact_and_bounded(self):
        counts = {
            "1": 90, "2": 211, "3": 98, "4": 120, "5": 177,
            "6": 180, "7": 230, "8": 57, "9": 256, "10": 191,
            "11": 196, "12": 70, "13": 148, "14": 160, "15": 139,
            "sin_contenedor": 57, "sin_residuos": 619,
        }
        quotas = bounded_proportional_quotas(counts, 1000, 20, 100)
        self.assertEqual(sum(quotas.values()), 1000)
        self.assertEqual(quotas["sin_residuos"], 100)
        self.assertEqual(quotas["8"], 22)
        self.assertEqual(quotas["9"], 97)
        self.assertTrue(all(20 <= value <= 100 for value in quotas.values()))

    def test_freezes_random_stratified_sample_without_reserved_events(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, run_dir = self._build_run(Path(directory))
            arguments = (
                settings,
                "reference",
                "characterization85",
                "labels85",
                "amazon.nova-lite-v1:0",
                "us-east-1",
                ["pilot", "validation"],
            )
            plan_dir = prepare_stratified_sample(
                *arguments,
                total=85,
                minimum_per_group=3,
                maximum_per_group=8,
                seed=1234,
            )

            selection = read_jsonl(plan_dir / "selection.jsonl")
            plan = json.loads((plan_dir / "sample_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(len(selection), 85)
            self.assertEqual(
                Counter(str(row["cluster"]) for row in selection),
                Counter({group: 5 for group in GROUPS}),
            )
            self.assertEqual(len({row["event_id"] for row in selection}), 85)
            self.assertFalse(any(row["id"].endswith(("_00", "_01")) for row in selection))
            self.assertEqual(sum(plan["quotas"].values()), 85)
            label_dir = run_dir / "bedrock" / "labels85"
            run = json.loads((label_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((run["selected"], run["status"]), (85, "prepared"))
            self.assertFalse((label_dir / "execution_control.json").exists())
            self.assertFalse((label_dir / "invocations.jsonl").exists())
            frozen = (plan_dir / "selection.jsonl").read_bytes()
            repeated = prepare_stratified_sample(
                *arguments,
                total=85,
                minimum_per_group=3,
                maximum_per_group=8,
                seed=1234,
            )
            self.assertEqual(repeated, plan_dir)
            self.assertEqual((plan_dir / "selection.jsonl").read_bytes(), frozen)

    def test_stratified_cli_does_not_open_storage(self):
        settings = SimpleNamespace(
            runs_root=Path("runs"),
            bedrock_model_id="amazon.nova-lite-v1:0",
            bedrock_region=None,
        )
        with (
            patch("basura_vision.cli.load_settings", return_value=settings),
            patch("basura_vision.cli.photo_storage", side_effect=AssertionError("no debe abrir S3")),
            patch(
                "basura_vision.bedrock_sampling.prepare_stratified_sample",
                return_value=Path("plan"),
            ) as prepare,
            patch("builtins.print"),
        ):
            main([
                "bedrock-stratified-sample",
                "--run-id", "reference",
                "--sample-plan-id", "characterization1000",
                "--label-run-id", "labels1000",
                "--exclude-label-run-id", "pilot40",
                "--exclude-label-run-id", "validation34",
                "--region", "us-east-1",
            ])
        prepare.assert_called_once_with(
            settings,
            "reference",
            "characterization1000",
            "labels1000",
            "amazon.nova-lite-v1:0",
            "us-east-1",
            ["pilot40", "validation34"],
            total=1000,
            minimum_per_group=20,
            maximum_per_group=100,
            seed=20261120,
        )


if __name__ == "__main__":
    unittest.main()
