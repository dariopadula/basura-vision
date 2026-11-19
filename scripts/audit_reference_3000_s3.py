from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from basura_vision.config import load_settings
from basura_vision.storage import canonical_run_files, photo_storage


EXPECTED_STATUSES = {
    "agrupable": 2323,
    "sin_contenedor": 57,
    "sin_residuos": 619,
    "error_entrada": 1,
}
EXPECTED_ERROR_ID = "176073_3029927_5189472"
REQUIRED_FILES = {
    "run.json",
    "manifest.jsonl",
    "detections.jsonl",
    "embeddings.npy",
    "embedding_index.jsonl",
    "embeddings.json",
    "assignments.jsonl",
    "clustering.json",
    "model/best.pt",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_lines(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audita sin escrituras la referencia amplia local y su prefijo S3."
    )
    parser.add_argument("--run-id", default="reference_3000_gpu")
    parser.add_argument("--source", default="runs/reference_3000_gpu")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()

    settings = load_settings(args.env_file)
    storage = photo_storage(settings)
    source = args.source.strip("/")
    marker = source + "/"
    keys = sorted(
        key
        for key in storage.list(source)
        if key == source or key.startswith(marker)
    )

    remote_inventory = []
    selected_content: dict[str, bytes] = {}
    for key in keys:
        relative = key[len(marker) :] if key.startswith(marker) else ""
        data = storage.read_bytes(key)
        remote_inventory.append(
            {"path": relative, "bytes": len(data), "sha256": sha256_bytes(data)}
        )
        if relative in REQUIRED_FILES:
            selected_content[relative] = data

    local_dir = settings.runs_root / args.run_id
    local_inventory = []
    ignored_local = []
    if local_dir.is_dir():
        all_local = sorted(item for item in local_dir.rglob("*") if item.is_file())
        canonical_local = canonical_run_files(local_dir)
        canonical_set = set(canonical_local)
        ignored_local = [
            path.relative_to(local_dir).as_posix()
            for path in all_local
            if path not in canonical_set
        ]
        for path in canonical_local:
            local_inventory.append(
                {
                    "path": path.relative_to(local_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    remote_by_path = {item["path"]: item for item in remote_inventory}
    local_by_path = {item["path"]: item for item in local_inventory}
    common = sorted(remote_by_path.keys() & local_by_path.keys())
    mismatched = [
        path
        for path in common
        if remote_by_path[path]["sha256"] != local_by_path[path]["sha256"]
    ]

    allowed = REQUIRED_FILES | {
        path for path in remote_by_path if path.startswith("crops/")
    }
    report: dict = {
        "mode": "read_only",
        "run_id": args.run_id,
        "source": source,
        "storage": settings.storage,
        "s3_bucket": settings.s3_bucket,
        "configured_s3_prefix": settings.s3_prefix,
        "local_dir": str(local_dir),
        "remote_object_count": len(remote_inventory),
        "local_file_count": len(local_inventory),
        "ignored_local_diagnostics": ignored_local,
        "required_missing_remote": sorted(REQUIRED_FILES - remote_by_path.keys()),
        "unexpected_remote": sorted(remote_by_path.keys() - allowed),
        "missing_remote_vs_local": sorted(local_by_path.keys() - remote_by_path.keys()),
        "missing_local_vs_remote": sorted(remote_by_path.keys() - local_by_path.keys()),
        "hash_mismatches": mismatched,
        "remote_inventory": remote_inventory,
        "checks": {},
    }

    checks = report["checks"]
    if not report["required_missing_remote"]:
        manifest = json_lines(selected_content["manifest.jsonl"])
        detections = json_lines(selected_content["detections.jsonl"])
        embedding_index = json_lines(selected_content["embedding_index.jsonl"])
        assignments = json_lines(selected_content["assignments.jsonl"])
        run = json.loads(selected_content["run.json"])
        embedding_meta = json.loads(selected_content["embeddings.json"])
        clustering = json.loads(selected_content["clustering.json"])
        vectors = np.load(io.BytesIO(selected_content["embeddings.npy"]), allow_pickle=False)

        manifest_ids = [row["id"] for row in manifest]
        detection_ids = [row["id"] for row in detections]
        embedding_ids = [row["id"] for row in embedding_index]
        assignment_ids = [row["id"] for row in assignments]
        crop_paths = [row["crop_path"] for row in detections if row.get("crop_path")]
        status_counts = Counter(row["status"] for row in detections)
        cluster_counts = Counter(str(row["cluster"]) for row in assignments)
        error_ids = [row["id"] for row in detections if row["status"] == "error_entrada"]
        model_sha = sha256_bytes(selected_content["model/best.pt"])
        active_model_sha = (
            sha256_file(settings.model_path) if settings.model_path.is_file() else None
        )

        checks.update(
            {
                "manifest_records": len(manifest),
                "manifest_unique_ids": len(set(manifest_ids)),
                "detections_records": len(detections),
                "detection_status_counts": dict(sorted(status_counts.items())),
                "error_input_ids": error_ids,
                "embedding_index_records": len(embedding_index),
                "embedding_index_unique_ids": len(set(embedding_ids)),
                "embeddings_shape": list(vectors.shape),
                "embedding_metadata_count": embedding_meta.get("count"),
                "embedding_metadata_dimensions": embedding_meta.get("dimensions"),
                "assignments_records": len(assignments),
                "cluster_counts": dict(sorted(cluster_counts.items())),
                "numeric_cluster_total": sum(
                    count for cluster, count in cluster_counts.items() if cluster.isdigit()
                ),
                "referenced_crops": len(crop_paths),
                "missing_referenced_crops": sorted(set(crop_paths) - remote_by_path.keys()),
                "manifest_detection_ids_equal": set(manifest_ids) == set(detection_ids),
                "embedding_assignment_ids_equal": set(embedding_ids) == set(assignment_ids),
                "crop_embedding_ids_equal": {
                    row["id"] for row in detections if row.get("crop_path")
                }
                == set(embedding_ids),
                "clustering": clustering,
                "run_model_sha256": run.get("model", {}).get("sha256"),
                "remote_model_sha256": model_sha,
                "active_model_sha256": active_model_sha,
            }
        )

        expected_clusters = {str(number) for number in range(1, 16)} | {
            "sin_contenedor"
        }
        checks_ok = all(
            [
                len(manifest) == 3000,
                len(set(manifest_ids)) == 3000,
                len(detections) == 3000,
                status_counts == Counter(EXPECTED_STATUSES),
                error_ids == [EXPECTED_ERROR_ID],
                len(embedding_index) == 2380,
                len(set(embedding_ids)) == 2380,
                list(vectors.shape) == [2380, 512],
                embedding_meta.get("count") == 2380,
                embedding_meta.get("dimensions") == 512,
                len(assignments) == 2380,
                set(cluster_counts) == expected_clusters,
                cluster_counts["sin_contenedor"] == 57,
                sum(
                    count for cluster, count in cluster_counts.items() if cluster.isdigit()
                )
                == 2323,
                len(crop_paths) == 2380,
                not checks["missing_referenced_crops"],
                checks["manifest_detection_ids_equal"],
                checks["embedding_assignment_ids_equal"],
                checks["crop_embedding_ids_equal"],
                clustering.get("clusters") == 15,
                clustering.get("pca_components") == 30,
                clustering.get("numeric_dtype") == "float64",
                run.get("model", {}).get("sha256") == model_sha,
                active_model_sha == model_sha,
            ]
        )
    else:
        checks_ok = False

    report["passed"] = all(
        [
            bool(remote_inventory),
            bool(local_inventory),
            not report["required_missing_remote"],
            not report["unexpected_remote"],
            not report["missing_remote_vs_local"],
            not report["missing_local_vs_remote"],
            not mismatched,
            checks_ok,
        ]
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
