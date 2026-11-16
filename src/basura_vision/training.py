from __future__ import annotations

import io
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from .config import Settings
from .storage import Storage

CLASSES = {"contenedor": 0, "residuos": 1}


def yolo_label(entry: dict) -> str:
    x1, y1, x2, y2 = entry["box"]
    return f"{CLASSES[entry['class']]} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}"


def prepare_dataset(settings: Settings, storage: Storage, annotations_path: Path, version: str, output_root: Path) -> Path:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8")); reviewed = [row for row in annotations.values() if row.get("reviewed") is True]
    if len(reviewed) < 10: raise ValueError("Se necesitan al menos 10 fotos revisadas")
    events = sorted({str(row.get("event_id") or "").strip() for row in reviewed})
    if "" in events or len(events) < 5: raise ValueError("Cada foto debe tener intervención y deben existir al menos cinco")
    random.Random(42).shuffle(events); test_count = max(1, round(len(events) * 0.15)); val_count = max(1, round(len(events) * 0.20))
    test_events = set(events[:test_count]); val_events = set(events[test_count:test_count + val_count]); root = output_root / version
    if root.exists(): raise FileExistsError(f"La versión ya existe: {root}")
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True); (root / "labels" / split).mkdir(parents=True)
    manifest, counts = [], Counter()
    for index, row in enumerate(sorted(reviewed, key=lambda value: value["id"])):
        event = str(row["event_id"]); split = "test" if event in test_events else "val" if event in val_events else "train"; name = f"photo_{index:06d}"
        with Image.open(io.BytesIO(storage.read_bytes(row["path"]))) as source: image = ImageOps.exif_transpose(source).convert("RGB")
        image.save(root / "images" / split / f"{name}.jpg", quality=95); lines = []
        if row.get("negative_confirmed") and row.get("boxes"): raise ValueError(f"Negativo con cajas: {row['id']}")
        if not row.get("boxes") and not row.get("negative_confirmed"): raise ValueError(f"Foto revisada vacía sin negativo confirmado: {row['id']}")
        for entry in row.get("boxes", []):
            lines.append(yolo_label(entry))
            counts[f"{split}_{entry['class']}"] += 1
        (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[f"{split}_photos"] += 1; counts[f"{split}_negative"] += int(row.get("negative_confirmed", False))
        manifest.append({"id": row["id"], "path": row["path"], "event_id": event, "split": split, "file": f"{name}.jpg", "negative_confirmed": row.get("negative_confirmed", False)})
    (root / "dataset.json").write_text(json.dumps({"version": version, "seed": 42, "counts": dict(counts), "photos": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "data.yaml").write_text(f'path: "{root.resolve().as_posix()}"\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: contenedor\n  1: residuos\n', encoding="utf-8")
    return root


def train(dataset_dir: Path, initial_weight: Path, output: Path, epochs: int = 50) -> None:
    from ultralytics import YOLO
    YOLO(str(initial_weight)).train(data=str(dataset_dir / "data.yaml"), epochs=epochs, imgsz=640, device="cpu", project=str(output), name="candidate", exist_ok=False, seed=42, workers=0)

