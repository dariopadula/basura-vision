import os
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from basura_vision.annotator import load_annotations, save_annotations, validate_boxes
from basura_vision.cli import main
from basura_vision.clustering import cluster_run
from basura_vision.compute import resolve_device
from basura_vision.config import ConfigError, load_settings
from basura_vision.detection import _resume_index
from basura_vision.manifests import read_jsonl, write_jsonl
from basura_vision.storage import (
    S3Storage,
    LocalStorage,
    canonical_run_files,
    download_run,
    upload_run,
)
from basura_vision.training import yolo_label
from scripts.run_viewer_cloudera import assigned_port


class CoreTests(unittest.TestCase):
    def test_default_configuration_is_local(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            settings = load_settings(Path(directory) / "missing.env")
        self.assertEqual((settings.app_env, settings.storage, settings.device), ("local", "local", "auto"))

    def test_compute_device_auto_uses_available_hardware(self):
        class FakeCuda:
            def __init__(self, available): self.available = available
            def is_available(self): return self.available
            def device_count(self): return 1 if self.available else 0

        self.assertEqual(resolve_device("auto", SimpleNamespace(cuda=FakeCuda(False))), "cpu")
        self.assertEqual(resolve_device("auto", SimpleNamespace(cuda=FakeCuda(True))), "cuda:0")
        with self.assertRaisesRegex(RuntimeError, "no la detecta"):
            resolve_device("cuda", SimpleNamespace(cuda=FakeCuda(False)))

    def test_configuration_rejects_invalid_device(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"BV_DEVICE": "gpu"},
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigError, "BV_DEVICE"):
                load_settings(Path(directory) / "missing.env")

    def test_detection_resume_requires_manifest_prefix_and_crops(self):
        manifest = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "crops").mkdir()
            (run_dir / "crops/a.jpg").write_bytes(b"crop")
            detections = [{"id": "a", "crop_path": "crops/a.jpg"}]
            self.assertEqual(_resume_index(manifest, detections, run_dir), 1)
            with self.assertRaisesRegex(ValueError, "prefijo"):
                _resume_index(manifest, [{"id": "b", "crop_path": None}], run_dir)
            with self.assertRaisesRegex(FileNotFoundError, "Falta el recorte"):
                _resume_index(
                    manifest,
                    [{"id": "a", "crop_path": "crops/missing.jpg"}],
                    run_dir,
                )

    def test_cloudera_requires_s3_configuration(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"APP_ENV": "cloudera"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "BV_S3_CONNECTION"):
                load_settings(Path(directory) / "missing.env")

    def test_storage_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalStorage(Path(directory))
            with self.assertRaises(ValueError):
                store.read_bytes("../secret")
            store.write_bytes("a/b.txt", b"ok")
            self.assertEqual(store.read_bytes("a/b.txt"), b"ok")

    def test_s3_run_roundtrip_with_common_interface(self):
        class FakeS3:
            def __init__(self): self.objects = {}
            def put_object(self, Bucket, Key, Body): self.objects[(Bucket, Key)] = bytes(Body)
            def get_object(self, Bucket, Key): return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}
            def upload_file(self, source, bucket, key): self.objects[(bucket, key)] = Path(source).read_bytes()
            def download_file(self, bucket, key, destination): Path(destination).write_bytes(self.objects[(bucket, key)])
            def list_objects_v2(self, **args):
                keys = [key for bucket, key in self.objects if bucket == args["Bucket"] and key.startswith(args["Prefix"])]
                return {"Contents": [{"Key": key} for key in keys], "IsTruncated": False}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); source = base / "source"; source.mkdir(); (source / "run.json").write_text("{}", encoding="utf-8")
            client = FakeS3(); storage = S3Storage(client, "bucket", "base")
            uploaded = upload_run(storage, source, "runs/r1")
            self.assertEqual(uploaded, ["base/runs/r1/run.json"])
            storage.write_bytes("runs/r10/unrelated.json", b"no")
            restored = base / "restored"; download_run(storage, "runs/r1", restored)
            self.assertEqual((restored / "run.json").read_text(encoding="utf-8"), "{}")
            self.assertFalse((restored / "unrelated.json").exists())

    def test_canonical_run_excludes_diagnostic_variants(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            run_dir = Path(directory)
            (run_dir / "model").mkdir()
            (run_dir / "crops").mkdir()
            (run_dir / "run.json").write_text(
                json.dumps({"state": "completed", "model": {"path": "model/best.pt"}}),
                encoding="utf-8",
            )
            (run_dir / "detections.jsonl").write_text(
                json.dumps({"id": "a", "crop_path": "crops/a.jpg"}) + "\n",
                encoding="utf-8",
            )
            for relative in {
                "manifest.jsonl",
                "embeddings.npy",
                "embedding_index.jsonl",
                "embeddings.json",
                "assignments.jsonl",
                "clustering.json",
                "model/best.pt",
                "crops/a.jpg",
            }:
                path = run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"data")
            (run_dir / "assignments_gpu.jsonl").write_bytes(b"diagnostic")

            selected = {
                path.relative_to(run_dir).as_posix()
                for path in canonical_run_files(run_dir)
            }

            self.assertEqual(
                selected,
                {
                    "run.json",
                    "manifest.jsonl",
                    "detections.jsonl",
                    "embeddings.npy",
                    "embedding_index.jsonl",
                    "embeddings.json",
                    "assignments.jsonl",
                    "clustering.json",
                    "model/best.pt",
                    "crops/a.jpg",
                },
            )
            self.assertNotIn("assignments_gpu.jsonl", selected)

            relative_run_dir = Path(os.path.relpath(run_dir, Path.cwd()))
            selected_from_relative = {
                path.relative_to(run_dir).as_posix()
                for path in canonical_run_files(relative_run_dir)
            }
            self.assertEqual(selected_from_relative, selected)

    def test_canonical_upload_dry_run_does_not_open_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "r1"
            run_dir.mkdir(parents=True)
            settings = SimpleNamespace(runs_root=run_dir.parent)
            with (
                patch("basura_vision.cli.load_settings", return_value=settings),
                patch("basura_vision.cli.canonical_run_files", return_value=[run_dir / "run.json"]),
                patch("basura_vision.cli.photo_storage", side_effect=AssertionError("no debe abrir S3")),
                patch("builtins.print") as output,
            ):
                main(["upload-run", "--run-id", "r1", "--canonical", "--dry-run"])
            output.assert_any_call("Archivos seleccionados: 1")

    def test_annotation_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            box = {"class": "residuos", "box": [0.1, 0.2, 0.3, 0.4]}
            self.assertEqual(validate_boxes([box]), [box])
            data = {"x": {"reviewed": True, "negative_confirmed": False, "boxes": [box]}}
            save_annotations(path, data)
            self.assertEqual(load_annotations(path), data)

    def test_normalized_box_matches_yolo_contract(self):
        line = yolo_label({"class": "residuos", "box": [0.1, 0.2, 0.3, 0.4]})
        self.assertEqual(line, "1 0.20000000 0.30000000 0.20000000 0.20000000")

    def test_cloudera_launcher_uses_assigned_port(self):
        self.assertEqual(assigned_port({"CDSW_READONLY_PORT": "8100", "CDSW_APP_PORT": "8090"}), "8100")
        self.assertEqual(assigned_port({"CDSW_APP_PORT": "8090"}), "8090")
        with self.assertRaisesRegex(RuntimeError, "no asignó"):
            assigned_port({})

    def test_viewer_run_can_be_configured_from_env_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "BV_RUN_ID=my_application_run\nBV_RUN_SOURCE=runs/source_run\n",
                encoding="utf-8",
            )
            settings = load_settings(env_file)
        self.assertEqual(settings.viewer_run_id, "my_application_run")
        self.assertEqual(settings.viewer_run_source, "runs/source_run")

    def test_cluster_cli_accepts_smaller_pca_for_smoke_test(self):
        settings = SimpleNamespace(runs_root=Path("runs"))
        with (
            patch("basura_vision.cli.load_settings", return_value=settings),
            patch("basura_vision.cli.photo_storage", side_effect=AssertionError("cluster no debe abrir S3")),
            patch("basura_vision.clustering.cluster_run") as cluster_run,
        ):
            main([
                "cluster",
                "--run-id", "reference_30_smoke",
                "--clusters", "3",
                "--pca-components", "10",
            ])
        cluster_run.assert_called_once_with(
            Path("runs/reference_30_smoke"),
            clusters=3,
            pca_components=10,
        )

    def test_embed_cli_uses_configured_storage(self):
        settings = SimpleNamespace(runs_root=Path("runs"), device="auto")
        storage = object()
        with (
            patch("basura_vision.cli.load_settings", return_value=settings),
            patch("basura_vision.cli.photo_storage", return_value=storage) as photo_storage,
            patch("basura_vision.clustering.extract_embeddings") as extract_embeddings,
        ):
            main(["embed", "--run-id", "reference_30_smoke"])
        photo_storage.assert_called_once_with(settings)
        extract_embeddings.assert_called_once_with(
            Path("runs/reference_30_smoke"), storage, "auto"
        )

    def test_bedrock_prepare_does_not_open_storage_or_client(self):
        settings = SimpleNamespace(
            runs_root=Path("runs"),
            bedrock_model_id="amazon.nova-lite-v1:0",
            bedrock_region=None,
        )
        with (
            patch("basura_vision.cli.load_settings", return_value=settings),
            patch("basura_vision.cli.photo_storage", side_effect=AssertionError("no debe abrir S3")),
            patch("basura_vision.bedrock.prepare_label_run", return_value=Path("prepared")) as prepare,
            patch("builtins.print"),
        ):
            main([
                "bedrock-prepare",
                "--run-id", "reference",
                "--label-run-id", "pilot",
                "--region", "us-east-1",
            ])
        prepare.assert_called_once_with(
            settings,
            "reference",
            "pilot",
            "amazon.nova-lite-v1:0",
            "us-east-1",
            max_estimated_cost_usd=None,
            approved=False,
        )

    def test_cluster_reads_embedding_index_without_photo_path(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            rows = [
                {"id": f"photo_{index:02d}", "ratio": index / 100, "status": "agrupable"}
                for index in range(1, 16)
            ]
            write_jsonl(run_dir / "embedding_index.jsonl", rows)
            vectors = np.random.default_rng(42).normal(size=(15, 512)).astype(np.float32)
            np.save(run_dir / "embeddings.npy", vectors)

            result = cluster_run(run_dir, clusters=3, pca_components=10)

            assignments = read_jsonl(
                result,
                required_fields={"id", "cluster", "pc1", "pc2", "ratio"},
            )
            self.assertEqual(len(assignments), 15)
            self.assertEqual({row["cluster"] for row in assignments}, {1, 2, 3})
            parameters = json.loads((run_dir / "clustering.json").read_text(encoding="utf-8"))
            self.assertEqual((parameters["pca_components"], parameters["clusters"]), (10, 3))
            self.assertEqual(parameters["numeric_dtype"], "float64")


if __name__ == "__main__":
    unittest.main()

