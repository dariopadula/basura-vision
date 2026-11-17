from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from .manifests import read_jsonl, write_jsonl


def extract_embeddings(run_dir: Path) -> Path:
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    detections = read_jsonl(run_dir / "detections.jsonl")
    weights = ResNet18_Weights.DEFAULT
    network = resnet18(weights=weights)
    network.fc = torch.nn.Identity()
    network.eval()
    transform = weights.transforms()
    rows, vectors = [], []
    with torch.inference_mode():
        for row in detections:
            if not row.get("crop_path"):
                continue
            with Image.open(run_dir / row["crop_path"]) as image:
                vector = network(transform(image.convert("RGB")).unsqueeze(0))[0].numpy()
            rows.append({"id": row["id"], "ratio": row["ratio"], "status": row["status"]})
            vectors.append(vector)
    np.save(run_dir / "embeddings.npy", np.asarray(vectors, dtype=np.float32))
    write_jsonl(run_dir / "embedding_index.jsonl", rows)
    return run_dir / "embeddings.npy"


def cluster_run(run_dir: Path, clusters: int = 15, pca_components: int = 30) -> Path:
    rows = read_jsonl(
        run_dir / "embedding_index.jsonl",
        required_fields={"id", "ratio", "status"},
    )
    vectors = np.load(run_dir / "embeddings.npy")
    valid = np.asarray([row["ratio"] is not None for row in rows])
    if valid.sum() < max(clusters, pca_components + 1):
        raise ValueError(f"Se requieren al menos {max(clusters, pca_components + 1)} casos agrupables; hay {valid.sum()}")
    space = PCA(n_components=pca_components, random_state=42).fit_transform(normalize(vectors))
    ratios = np.log1p([row["ratio"] for row in rows if row["ratio"] is not None])
    z_score = (ratios - ratios.mean()) / (ratios.std() or 1)
    pca = space[valid]
    features = np.column_stack((pca, 0.5 * np.sqrt(pca.var(axis=0).sum()) * z_score))
    labels = KMeans(n_clusters=clusters, n_init=30, random_state=42).fit_predict(features)
    label_iter = iter(labels)
    assignments = []
    for row, (pc1, pc2) in zip(rows, space[:, :2]):
        cluster = int(next(label_iter)) + 1 if row["ratio"] is not None else "sin_contenedor"
        assignments.append({"id": row["id"], "cluster": cluster, "pc1": float(pc1), "pc2": float(pc2), "ratio": row["ratio"]})
    write_jsonl(run_dir / "assignments.jsonl", assignments)
    (run_dir / "clustering.json").write_text(json.dumps({"pca_components": pca_components, "clusters": clusters, "ratio_weight": 0.5, "random_state": 42, "kmeans_n_init": 30}, indent=2) + "\n", encoding="utf-8")
    return run_dir / "assignments.jsonl"

