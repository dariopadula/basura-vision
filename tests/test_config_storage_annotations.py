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
from basura_vision.config import ConfigError, load_settings
from basura_vision.manifests import read_jsonl, write_jsonl
from basura_vision.storage import S3Storage, LocalStorage, download_run, upload_run
from basura_vision.training import yolo_label
from scripts.run_viewer_cloudera import assigned_port


class CoreTests(unittest.TestCase):
    def test_default_configuration_is_local(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            settings = load_settings(Path(directory) / "missing.env")
        self.assertEqual((settings.app_env, settings.storage), ("local", "local"))

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

    def test_embed_cli_does_not_open_storage(self):
        settings = SimpleNamespace(runs_root=Path("runs"))
        with (
            patch("basura_vision.cli.load_settings", return_value=settings),
            patch("basura_vision.cli.photo_storage", side_effect=AssertionError("embed no debe abrir S3")),
            patch("basura_vision.clustering.extract_embeddings") as extract_embeddings,
        ):
            main(["embed", "--run-id", "reference_30_smoke"])
        extract_embeddings.assert_called_once_with(Path("runs/reference_30_smoke"))

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


if __name__ == "__main__":
    unittest.main()

