from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from .config import ROOT, Settings
from .manifests import read_jsonl
from .storage import Storage

PROMPT_VERSION = "bedrock_labels_v1"
PROMPT_PATH = ROOT / "prompts" / f"{PROMPT_VERSION}.txt"
CATEGORIES = (
    "basura_fuera",
    "residuos_esparcidos",
    "poda",
    "contenedor_desbordado",
    "contenedor_roto",
    "voluminoso",
)
SEVERITIES = {"leve", "medio", "severo"}


def prepare_image(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        output = io.BytesIO()
        image.save(output, "JPEG", quality=95)
    return output.getvalue()


def parse_labels(text: str) -> dict:
    try:
        labels = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Bedrock no devolvió un JSON válido") from exc
    if not isinstance(labels, dict) or set(labels) != set(CATEGORIES):
        raise ValueError(f"La respuesta debe contener exactamente: {', '.join(CATEGORIES)}")
    for category in CATEGORIES:
        value = labels[category]
        if not isinstance(value, dict) or set(value) != {"presente", "severidad"}:
            raise ValueError(f"Estructura inválida para {category}")
        present = value["presente"]
        severity = value["severidad"]
        if not isinstance(present, bool):
            raise ValueError(f"{category}.presente debe ser booleano")
        if present and severity not in SEVERITIES:
            raise ValueError(f"{category}.severidad debe ser leve, medio o severo")
        if not present and severity is not None:
            raise ValueError(f"{category}.severidad debe ser null cuando no está presente")
    return labels


def bedrock_client(region: str | None = None):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 no está instalado; instalar el extra [bedrock]") from exc
    session = boto3.Session(region_name=region)
    selected_region = session.region_name
    if not selected_region:
        raise RuntimeError("No hay región AWS configurada")
    return session.client("bedrock-runtime", region_name=selected_region)


def label_one_photo(
    settings: Settings,
    storage: Storage,
    run_id: str,
    photo_id: str,
    model_id: str,
    region: str | None = None,
    client=None,
) -> Path:
    run_dir = (settings.runs_root / run_id).resolve()
    if not run_dir.is_dir() or not run_dir.is_relative_to(settings.runs_root.resolve()):
        raise ValueError(f"Corrida inexistente o inválida: {run_id}")
    manifest = {row["id"]: row for row in read_jsonl(run_dir / "manifest.jsonl")}
    if photo_id not in manifest:
        raise ValueError(f"ID inexistente en la corrida: {photo_id}")

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    photo = manifest[photo_id]
    image = prepare_image(storage.read_bytes(photo["path"]))
    client = client or bedrock_client(region)
    selected_region = getattr(getattr(client, "meta", None), "region_name", None) or region
    response = client.converse(
        modelId=model_id,
        messages=[{
            "role": "user",
            "content": [
                {"image": {"format": "jpeg", "source": {"bytes": image}}},
                {"text": prompt},
            ],
        }],
        inferenceConfig={"maxTokens": 300, "temperature": 0},
    )
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    raw_text = "\n".join(block["text"] for block in blocks if "text" in block).strip()
    if not raw_text:
        raise ValueError("Bedrock no devolvió contenido de texto")
    labels = parse_labels(raw_text)

    record = {
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "photo_id": photo_id,
        "path": photo["path"],
        "image_source": "original",
        "model_id": model_id,
        "region": selected_region,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "labels": labels,
        "raw_text": raw_text,
        "usage": response.get("usage"),
        "metrics": response.get("metrics"),
        "stop_reason": response.get("stopReason"),
    }
    name = hashlib.sha256(photo_id.encode("utf-8")).hexdigest()[:16] + ".json"
    destination = run_dir / "bedrock" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
