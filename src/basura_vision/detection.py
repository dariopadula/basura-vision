from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .config import Settings
from .compute import resolve_device
from .manifests import read_jsonl, write_jsonl
from .storage import Storage

WASTE_CLASS = 1
CONTAINER_CLASS = 0
WASTE_CONFIDENCE = 0.10
CONTAINER_CONFIDENCE = 0.25
PREDICT_CONFIDENCE = 0.05
IOU = 0.70
MAX_DETECTIONS = 300
CHECKPOINT_EVERY = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _area(box: list[int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _box(values, width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = values
    box = [max(0, math.floor(x1)), max(0, math.floor(y1)), min(width, math.ceil(x2)), min(height, math.ceil(y2))]
    return box if box[0] < box[2] and box[1] < box[3] else None


def analyze_prediction(image: Image.Image, prediction) -> tuple[dict, Image.Image | None]:
    waste, containers, all_boxes = [], [], []
    for values, class_id, confidence in zip(
        prediction.boxes.xyxy.tolist(), prediction.boxes.cls.tolist(), prediction.boxes.conf.tolist()
    ):
        box = _box(values, image.width, image.height)
        if box is None:
            continue
        record = {"class_id": int(class_id), "class_name": "residuos" if int(class_id) == 1 else "contenedor", "confidence": float(confidence), "box": box}
        if int(class_id) == WASTE_CLASS and confidence >= WASTE_CONFIDENCE:
            waste.append(record)
            all_boxes.append(record)
        elif int(class_id) == CONTAINER_CLASS and confidence >= CONTAINER_CONFIDENCE:
            containers.append(record)
            all_boxes.append(record)
    if not waste:
        return {"status": "sin_residuos", "reason": "no_waste_above_threshold", "boxes": all_boxes, "crop_box": None, "union_area": 0, "container_box": None, "ratio": None}, None
    mask = Image.new("1", image.size)
    draw = ImageDraw.Draw(mask)
    for item in waste:
        x1, y1, x2, y2 = item["box"]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=1)
    crop_box = list(mask.getbbox())
    union_area = int(np.count_nonzero(np.asarray(mask.crop(tuple(crop_box)))))
    container = max(containers, key=lambda item: _area(item["box"]), default=None)
    return {
        "status": "agrupable" if container else "sin_contenedor",
        "reason": None if container else "waste_found_but_no_container_above_threshold",
        "boxes": all_boxes,
        "crop_box": crop_box,
        "union_area": union_area,
        "container_box": container["box"] if container else None,
        "ratio": union_area / _area(container["box"]) if container else None,
    }, image.crop(tuple(crop_box))


def resolve_model(settings: Settings, storage: Storage) -> Path:
    if settings.model_path.is_file():
        return settings.model_path
    if settings.model_object:
        return storage.materialize(settings.model_object, settings.temp_root / "models" / Path(settings.model_object).name)
    raise FileNotFoundError(f"No existe el peso activo: {settings.model_path}. Configure BV_MODEL_PATH o BV_MODEL_OBJECT.")


def _write_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resume_index(manifest: list[dict], detections: list[dict], run_dir: Path) -> int:
    expected = [row["id"] for row in manifest[: len(detections)]]
    actual = [row["id"] for row in detections]
    if actual != expected:
        raise ValueError("La corrida parcial no coincide con el prefijo del manifiesto")
    for row in detections:
        crop_path = row.get("crop_path")
        if crop_path and not (run_dir / crop_path).is_file():
            raise FileNotFoundError(f"Falta el recorte parcial: {crop_path}")
    return len(detections)


def _metadata(
    settings: Settings,
    run_id: str,
    run_model: Path,
    model_hash: str,
    model_names: dict,
    device: str,
    detections: list[dict],
    total: int,
    created_at: str,
    state: str,
) -> dict:
    statuses = ("agrupable", "sin_residuos", "sin_contenedor", "error_entrada")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "processed": len(detections),
        "total": total,
        "app_env": settings.app_env,
        "model": {
            "path": run_model.relative_to(run_model.parents[1]).as_posix(),
            "filename": run_model.name,
            "sha256": model_hash,
            "classes": model_names,
        },
        "parameters": {
            "predict_confidence": PREDICT_CONFIDENCE,
            "waste_confidence": WASTE_CONFIDENCE,
            "container_confidence": CONTAINER_CONFIDENCE,
            "iou": IOU,
            "max_detections": MAX_DETECTIONS,
            "requested_device": settings.device,
            "device": device,
        },
        "counts": {
            status: sum(row["status"] == status for row in detections)
            for status in statuses
        },
    }


def run_detection(settings: Settings, storage: Storage, manifest_path: Path, run_id: str) -> Path:
    from ultralytics import YOLO

    manifest = read_jsonl(manifest_path)
    model_path = resolve_model(settings, storage)
    model = YOLO(str(model_path))
    device = resolve_device(settings.device)
    print(f"Dispositivo de detección: {device}", flush=True)
    if model.names != {0: "contenedor", 1: "residuos"}:
        raise ValueError(f"Mapa de clases inesperado: {model.names!r}")
    run_dir = settings.runs_root / run_id
    (run_dir / "crops").mkdir(parents=True, exist_ok=True)
    (run_dir / "model").mkdir(exist_ok=True)
    run_model = run_dir / "model" / model_path.name
    model_hash = sha256(model_path)
    if run_model.is_file():
        if sha256(run_model) != model_hash:
            raise ValueError("El peso de la corrida parcial no coincide con el peso activo")
    else:
        shutil.copy2(model_path, run_model)

    saved_manifest = run_dir / "manifest.jsonl"
    if saved_manifest.is_file():
        if read_jsonl(saved_manifest) != manifest:
            raise ValueError("La corrida parcial usa otro manifiesto")
    else:
        write_jsonl(saved_manifest, manifest)

    detections_path = run_dir / "detections.jsonl"
    detections = (
        read_jsonl(detections_path)
        if detections_path.is_file() and detections_path.stat().st_size
        else []
    )
    start = _resume_index(manifest, detections, run_dir)
    metadata_path = run_dir / "run.json"
    previous = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    previous_device = previous.get("parameters", {}).get("device")
    if detections and previous_device and previous_device != device:
        raise ValueError(
            f"La corrida parcial usó {previous_device}; no puede retomarse con {device}"
        )
    created_at = previous.get("created_at") or datetime.now(timezone.utc).isoformat()
    if start:
        print(f"Retomando detección: {start}/{len(manifest)} ya procesadas", flush=True)
    if start == len(manifest):
        _write_metadata(
            metadata_path,
            _metadata(settings, run_id, run_model, model_hash, model.names, device, detections, len(manifest), created_at, "completed"),
        )
        print("La corrida ya estaba completa.", flush=True)
        return run_dir

    _write_metadata(
        metadata_path,
        _metadata(settings, run_id, run_model, model_hash, model.names, device, detections, len(manifest), created_at, "running"),
    )
    try:
        with detections_path.open("a", encoding="utf-8", newline="\n") as handle:
            for number, item in enumerate(manifest[start:], start + 1):
                try:
                    content = storage.read_bytes(item["path"])
                    with Image.open(io.BytesIO(content)) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                except Exception as exc:
                    result = {
                        "status": "error_entrada",
                        "reason": "source_read_failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "boxes": [],
                        "crop_box": None,
                        "union_area": 0,
                        "container_box": None,
                        "ratio": None,
                        "id": item["id"],
                        "path": item["path"],
                        "event_id": item.get("event_id"),
                        "width": None,
                        "height": None,
                        "crop_path": None,
                    }
                else:
                    prediction = model.predict(image, conf=PREDICT_CONFIDENCE, iou=IOU, max_det=MAX_DETECTIONS, device=device, verbose=False)[0]
                    result, crop = analyze_prediction(image, prediction)
                    crop_path = None
                    if crop is not None:
                        crop_path = f"crops/{item['id']}.jpg"
                        crop.save(run_dir / crop_path, quality=95)
                    result.update(id=item["id"], path=item["path"], event_id=item.get("event_id"), width=image.width, height=image.height, crop_path=crop_path)

                handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                detections.append(result)
                print(f"{number}/{len(manifest)} {item['id']} {result['status']}", flush=True)
                if number % CHECKPOINT_EVERY == 0:
                    os.fsync(handle.fileno())
                    _write_metadata(
                        metadata_path,
                        _metadata(settings, run_id, run_model, model_hash, model.names, device, detections, len(manifest), created_at, "running"),
                    )
            os.fsync(handle.fileno())
    except BaseException:
        _write_metadata(
            metadata_path,
            _metadata(settings, run_id, run_model, model_hash, model.names, device, detections, len(manifest), created_at, "interrupted"),
        )
        raise

    _write_metadata(
        metadata_path,
        _metadata(settings, run_id, run_model, model_hash, model.names, device, detections, len(manifest), created_at, "completed"),
    )
    return run_dir

