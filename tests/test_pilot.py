import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from basura_vision.cli import main
from basura_vision.manifests import read_jsonl, write_jsonl
from basura_vision.pilot import prepare_pilot_samples


class PilotSamplesTests(unittest.TestCase):
    def _build_run(self, root: Path) -> tuple[SimpleNamespace, Path]:
        run_dir = root / "runs" / "reference"
        run_dir.mkdir(parents=True)
        manifest = []
        detections = []
        assignments = []

        for group in range(1, 16):
            for number in range(6):
                photo_id = f"cluster_{group:02d}_{number:02d}"
                path = f"photos/{photo_id}.jpg"
                manifest.append({
                    "id": photo_id,
                    "path": path,
                    "event_id": f"event_{photo_id}",
                })
                detections.append({"id": photo_id, "path": path, "status": "agrupable"})
                assignments.append({
                    "id": photo_id,
                    "cluster": group,
                    "pc1": group * 10 + number,
                    "pc2": group * -5 + number * 2,
                    "ratio": number / 10,
                })

        for group, count in (("sin_contenedor", 8), ("sin_residuos", 8)):
            for number in range(count):
                photo_id = f"{group}_{number:02d}"
                path = f"photos/{photo_id}.jpg"
                manifest.append({
                    "id": photo_id,
                    "path": path,
                    "event_id": f"event_{photo_id}",
                })
                detections.append({"id": photo_id, "path": path, "status": group})

        write_jsonl(run_dir / "manifest.jsonl", manifest)
        write_jsonl(run_dir / "detections.jsonl", detections)
        write_jsonl(run_dir / "assignments.jsonl", assignments)
        return SimpleNamespace(runs_root=root / "runs"), run_dir

    def test_freezes_disjoint_pilot_and_validation_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, run_dir = self._build_run(Path(directory))

            plan_dir = prepare_pilot_samples(
                settings,
                "reference",
                "v1_seed",
                "pilot40",
                "validation34",
                "amazon.nova-lite-v1:0",
                "us-east-1",
                seed=1234,
            )

            pilot = read_jsonl(plan_dir / "pilot_selection.jsonl")
            validation = read_jsonl(plan_dir / "validation_selection.jsonl")
            self.assertEqual(len(pilot), 40)
            self.assertEqual(len(validation), 34)
            self.assertEqual(
                Counter(str(row["cluster"]) for row in pilot),
                Counter({
                    **{str(group): 2 for group in range(1, 16)},
                    "sin_contenedor": 5,
                    "sin_residuos": 5,
                }),
            )
            self.assertEqual(
                Counter(str(row["cluster"]) for row in validation),
                Counter({
                    **{str(group): 2 for group in range(1, 16)},
                    "sin_contenedor": 2,
                    "sin_residuos": 2,
                }),
            )
            self.assertFalse({row["id"] for row in pilot} & {row["id"] for row in validation})
            self.assertFalse(
                {row["event_id"] for row in pilot}
                & {row["event_id"] for row in validation}
            )
            for label_run_id, expected in (("pilot40", 40), ("validation34", 34)):
                label_dir = run_dir / "bedrock" / label_run_id
                frozen = read_jsonl(label_dir / "selection.jsonl")
                metadata = json.loads((label_dir / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(len(frozen), expected)
                self.assertEqual(metadata["selected"], expected)
                self.assertEqual(metadata["status"], "prepared")
                self.assertFalse((label_dir / "execution_control.json").exists())
                self.assertFalse((label_dir / "invocations.jsonl").exists())

    def test_repeating_same_plan_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _ = self._build_run(Path(directory))
            arguments = (
                settings,
                "reference",
                "v1_seed",
                "pilot40",
                "validation34",
                "amazon.nova-lite-v1:0",
                "us-east-1",
            )
            plan_dir = prepare_pilot_samples(*arguments, seed=42)
            before = {
                name: (plan_dir / name).read_bytes()
                for name in ("pilot_selection.jsonl", "validation_selection.jsonl", "sample_plan.json")
            }

            repeated = prepare_pilot_samples(*arguments, seed=42)

            self.assertEqual(repeated, plan_dir)
            self.assertEqual(
                before,
                {name: (plan_dir / name).read_bytes() for name in before},
            )

    def test_samples_cli_does_not_open_storage(self):
        settings = SimpleNamespace(
            runs_root=Path("runs"),
            bedrock_model_id="amazon.nova-lite-v1:0",
            bedrock_region=None,
        )
        with (
            patch("basura_vision.cli.load_settings", return_value=settings),
            patch("basura_vision.cli.photo_storage", side_effect=AssertionError("no debe abrir S3")),
            patch("basura_vision.pilot.prepare_pilot_samples", return_value=Path("plan")) as prepare,
            patch("builtins.print"),
        ):
            main([
                "bedrock-samples",
                "--run-id", "reference",
                "--sample-plan-id", "v1_seed",
                "--pilot-label-run-id", "pilot40",
                "--validation-label-run-id", "validation34",
                "--region", "us-east-1",
                "--seed", "1234",
            ])
        prepare.assert_called_once_with(
            settings,
            "reference",
            "v1_seed",
            "pilot40",
            "validation34",
            "amazon.nova-lite-v1:0",
            "us-east-1",
            seed=1234,
        )


if __name__ == "__main__":
    unittest.main()
