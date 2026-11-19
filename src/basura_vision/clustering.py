from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from .compute import resolve_device
from .manifests import read_jsonl, write_jsonl


def extract_embeddings(run_dir: Path, requested_device: str = "auto") -> Path:
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    detections = read_jsonl(run_dir / "detections.jsonl")
    device_name = resolve_device(requested_device, torch)
    device = torch.device(device_name)
    print(f"Dispositivo de embeddings: {device_name}", flush=True)
    weights = ResNet18_Weights.DEFAULT
    network = resnet18(weights=weights)
    network.fc = torch.nn.Identity()
    network.eval().to(device)
    transform = weights.transforms()
    rows, vectors = [], []
    with torch.inference_mode():
        for row in detections:
            if not row.get("crop_path"):
                continue
            with Image.open(run_dir / row["crop_path"]) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
                vector = network(tensor)[0].detach().cpu().numpy()
            rows.append({"id": row["id"], "ratio": row["ratio"], "status": row["status"]})
            vectors.append(vector)
    np.save(run_dir / "embeddings.npy", np.asarray(vectors, dtype=np.float32))
    write_jsonl(run_dir / "embedding_index.jsonl", rows)
    (run_dir / "embeddings.json").write_text(
        json.dumps(
            {
                "model": "resnet18",
                "weights": "ResNet18_Weights.DEFAULT",
                "requested_device": requested_device,
                "device": device_name,
                "count": len(rows),
                "dimensions": 512,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir / "embeddings.npy"


def cluster_run(run_dir: Path, clusters: int = 15, pca_components: int = 30) -> Path:
    rows = read_jsonl(
        run_dir / "embedding_index.jsonl",
        required_fields={"id", "ratio", "status"},
    )
    # La corrida histórica convertía las listas de PyTorch mediante np.array,
    # lo que producía float64. Mantener esa precisión antes de PCA evita que
    # pequeñas diferencias numéricas cambien las inicializaciones de K-means.
    vectors = np.asarray(np.load(run_dir / "embeddings.npy"), dtype=np.float64)
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
    (run_dir / "clustering.json").write_text(json.dumps({"pca_components": pca_components, "clusters": clusters, "ratio_weight": 0.5, "random_state": 42, "kmeans_n_init": 30, "numeric_dtype": "float64"}, indent=2) + "\n", encoding="utf-8")
    return run_dir / "assignments.jsonl"

