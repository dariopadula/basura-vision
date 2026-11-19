from __future__ import annotations

import hashlib
import io
import json
import os
import time
import uuid
from contextlib import contextmanager
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from .config import ROOT, Settings
from .manifests import read_jsonl, write_jsonl
from .pricing import PRICING_CATALOG, estimate_cost, freeze_price, pricing_sha256
from .storage import Storage, download_run, join_key, upload_run

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
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        labels = json.loads(candidate)
    except json.JSONDecodeError as exc:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Bedrock no devolvió un JSON válido") from exc
        try:
            labels = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError as nested_exc:
            raise ValueError("Bedrock no devolvió un JSON válido") from nested_exc
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


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _invoke_photo(
    storage: Storage,
    photo: dict,
    run_id: str,
    model_id: str,
    prompt: str,
    client,
    region: str | None,
) -> dict:
    record, invocation = _invoke_photo_audited(
        storage,
        photo,
        run_id,
        model_id,
        prompt,
        client,
        region,
        attempt=1,
    )
    if record is None:
        raise ValueError(invocation["error"]["message"])
    return record


def _invoke_photo_audited(
    storage: Storage,
    photo: dict,
    run_id: str,
    model_id: str,
    prompt: str,
    client,
    region: str | None,
    attempt: int,
    frozen_pricing: dict | None = None,
    pricing_reference: str | None = None,
) -> tuple[dict | None, dict]:
    selected_region = getattr(getattr(client, "meta", None), "region_name", None) or region
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    invocation = {
        "schema_version": "1",
        "invocation_id": str(uuid.uuid4()),
        "source_run_id": run_id,
        "photo_id": photo["id"],
        "path": photo["path"],
        "attempt": attempt,
        "model_id": model_id,
        "region": selected_region,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "pricing_sha256": pricing_reference,
        "pricing_catalog_sha256": (
            frozen_pricing.get("catalog", {}).get("sha256") if frozen_pricing else None
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "invoked": False,
    }
    try:
        image = prepare_image(storage.read_bytes(photo["path"]))
    except Exception as exc:
        invocation.update({
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "error_before_invocation",
            "aws_request_id": None,
            "stop_reason": None,
            "usage": None,
            "tokens_and_cost": estimate_cost(None, frozen_pricing) if frozen_pricing else None,
            "metrics": None,
            "raw_text": None,
            "observed_latency_ms": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        return None, invocation

    request_started = time.perf_counter()
    invocation["invoked"] = True
    invocation["request_started_at"] = datetime.now(timezone.utc).isoformat()
    try:
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
    except Exception as exc:
        invocation.update({
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "invocation_error",
            "aws_request_id": None,
            "stop_reason": None,
            "usage": None,
            "tokens_and_cost": estimate_cost(None, frozen_pricing) if frozen_pricing else None,
            "metrics": None,
            "raw_text": None,
            "observed_latency_ms": round((time.perf_counter() - request_started) * 1000, 3),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        return None, invocation

    usage = response.get("usage")
    if frozen_pricing:
        tokens_and_cost = estimate_cost(usage, frozen_pricing)
    else:
        tokens_and_cost = {
            "input_tokens": (usage or {}).get("inputTokens"),
            "output_tokens": (usage or {}).get("outputTokens"),
            "total_tokens": (usage or {}).get("totalTokens"),
            "estimated_input_cost": None,
            "estimated_output_cost": None,
            "estimated_total_cost": None,
            "currency": None,
            "cost_kind": None,
        }
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    raw_text = "\n".join(block["text"] for block in blocks if "text" in block).strip()
    invocation.update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "aws_request_id": response.get("ResponseMetadata", {}).get("RequestId"),
        "stop_reason": response.get("stopReason"),
        "usage": usage,
        "tokens_and_cost": tokens_and_cost,
        "metrics": response.get("metrics"),
        "raw_text": raw_text,
        "observed_latency_ms": round((time.perf_counter() - request_started) * 1000, 3),
    })
    try:
        if not raw_text:
            raise ValueError("Bedrock no devolvió contenido de texto")
        labels = parse_labels(raw_text)
    except Exception as exc:
        invocation.update({
            "status": "invalid_response",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        return None, invocation

    invocation.update({"status": "valid", "error": None})
    record = {
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "photo_id": photo["id"],
        "path": photo["path"],
        "image_source": "original",
        "model_id": model_id,
        "region": selected_region,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "invocation_id": invocation["invocation_id"],
        "pricing_sha256": pricing_reference,
        "labels": labels,
        "raw_text": raw_text,
        "usage": usage,
        "tokens_and_cost": tokens_and_cost,
        "metrics": response.get("metrics"),
        "stop_reason": response.get("stopReason"),
    }
    return record, invocation


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
    client = client or bedrock_client(region)
    record = _invoke_photo(storage, photo, run_id, model_id, prompt, client, region)
    name = hashlib.sha256(photo_id.encode("utf-8")).hexdigest()[:16] + ".json"
    destination = run_dir / "bedrock" / name
    _write_json(destination, record)
    return destination


def _selection(run_dir: Path) -> tuple[list[dict], list[dict]]:
    manifest = read_jsonl(run_dir / "manifest.jsonl")
    detections = {
        row["id"]: row
        for row in read_jsonl(run_dir / "detections.jsonl")
    }
    assignment_path = run_dir / "assignments.jsonl"
    assignments = {
        row["id"]: row
        for row in read_jsonl(
            assignment_path,
            required_fields={"id", "cluster", "pc1", "pc2", "ratio"},
        )
    } if assignment_path.is_file() else {}
    rows = []
    exclusions = []
    for photo in manifest:
        detection = detections[photo["id"]]
        if detection["status"] == "error_entrada":
            exclusions.append({
                "id": photo["id"],
                "path": photo["path"],
                "detection_status": detection["status"],
                "reason": detection.get("reason") or detection.get("error") or "error_entrada",
            })
            continue
        assignment = assignments.get(photo["id"])
        group = assignment["cluster"] if assignment else detection["status"]
        rows.append({
            "id": photo["id"],
            "path": photo["path"],
            "cluster": group,
            "detection_status": detection["status"],
        })
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("La selección contiene IDs duplicados")
    if {row["id"] for row in rows} & {row["id"] for row in exclusions}:
        raise ValueError("Un ID no puede estar seleccionado y excluido")
    return rows, exclusions


def _label_run_dir(settings: Settings, run_id: str, label_run_id: str) -> tuple[Path, Path]:
    if Path(label_run_id).name != label_run_id:
        raise ValueError("label_run_id debe ser un nombre simple, sin carpetas")
    run_dir = (settings.runs_root / run_id).resolve()
    if not run_dir.is_dir() or not run_dir.is_relative_to(settings.runs_root.resolve()):
        raise ValueError(f"Corrida inexistente o inválida: {run_id}")
    return run_dir, run_dir / "bedrock" / label_run_id


def prepare_label_run(
    settings: Settings,
    run_id: str,
    label_run_id: str,
    model_id: str,
    region: str,
    pricing_catalog: Path = PRICING_CATALOG,
    max_estimated_cost_usd: str | None = None,
    approved: bool = False,
) -> Path:
    if not region:
        raise ValueError("La región es obligatoria para congelar la tarifa")
    run_dir, label_dir = _label_run_dir(settings, run_id, label_run_id)
    label_dir.mkdir(parents=True, exist_ok=True)

    selection_path = label_dir / "selection.jsonl"
    exclusions_path = label_dir / "exclusions.jsonl"
    if selection_path.is_file() or exclusions_path.is_file():
        if not selection_path.is_file() or not exclusions_path.is_file():
            raise ValueError("La preparación existente está incompleta")
        selection = read_jsonl(selection_path)
        exclusions = (
            read_jsonl(exclusions_path)
            if exclusions_path.read_text(encoding="utf-8").strip()
            else []
        )
    else:
        selection, exclusions = _selection(run_dir)
        write_jsonl(selection_path, selection)
        write_jsonl(exclusions_path, exclusions)

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_path = label_dir / "prompt.txt"
    if prompt_path.is_file():
        if prompt_path.read_text(encoding="utf-8") != prompt:
            raise ValueError("La corrida existente conserva otro prompt")
    else:
        prompt_path.write_text(prompt, encoding="utf-8", newline="\n")

    pricing_path = label_dir / "pricing.json"
    frozen_pricing = freeze_price(
        pricing_path,
        {
            "model_id": model_id,
            "region": region,
            "routing": "direct",
            "service_tier": "standard",
            "inference_mode": "on_demand",
            "currency": "USD",
        },
        pricing_catalog,
    )
    identity = {
        "schema_version": "2",
        "label_run_id": label_run_id,
        "source_run_id": run_id,
        "image_source": "original",
        "model_id": model_id,
        "region": region,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": _sha256_path(prompt_path),
        "pricing_sha256": pricing_sha256(pricing_path),
        "selection_sha256": _sha256_path(selection_path),
        "exclusions_sha256": _sha256_path(exclusions_path),
        "selection": "accessible_manifest_photos",
        "persistence_version": "confirmed_attempts_v1",
        "selected": len(selection),
        "excluded": len(exclusions),
    }
    run_path = label_dir / "run.json"
    if run_path.is_file():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if previous.get(key) != value:
                raise ValueError(f"La corrida existente usa otro valor para {key}")
        started_at = previous["started_at"]
        status = previous.get("status", "prepared")
        progress = {
            key: previous.get(key, default)
            for key, default in {
                "completed": 0,
                "pending": len(selection),
                "error_attempts": 0,
                "invocation_attempts": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_total_cost": "0",
                "cost_currency": frozen_pricing["rate"]["currency"],
                "attempts_without_known_cost": 0,
                "cost_kind": frozen_pricing["cost_kind"],
            }.items()
        }
    else:
        started_at = datetime.now(timezone.utc).isoformat()
        status = "prepared"
        progress = {
            "completed": 0,
            "pending": len(selection),
            "error_attempts": 0,
            "invocation_attempts": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_total_cost": "0",
            "cost_currency": frozen_pricing["rate"]["currency"],
            "attempts_without_known_cost": 0,
            "cost_kind": frozen_pricing["cost_kind"],
        }
    _write_json(run_path, {
        **identity,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        **progress,
    })

    if max_estimated_cost_usd is not None:
        maximum = Decimal(str(max_estimated_cost_usd))
        if maximum <= 0:
            raise ValueError("El tope de costo debe ser positivo")
        control_path = label_dir / "execution_control.json"
        control = {
            "schema_version": "1",
            "approved": bool(approved),
            "approved_at": datetime.now(timezone.utc).isoformat() if approved else None,
            "max_estimated_cost_usd": format(maximum, "f"),
            **{
                key: identity[key]
                for key in (
                    "model_id",
                    "region",
                    "prompt_sha256",
                    "pricing_sha256",
                    "selection_sha256",
                    "exclusions_sha256",
                )
            },
        }
        if control_path.is_file():
            previous_control = json.loads(control_path.read_text(encoding="utf-8"))
            comparable = dict(previous_control)
            comparable.pop("approved_at", None)
            comparable.pop("approved", None)
            expected = dict(control)
            expected.pop("approved_at", None)
            expected.pop("approved", None)
            if comparable != expected:
                raise ValueError("El control de ejecución existente es incompatible")
            if previous_control.get("approved"):
                control = previous_control
            elif approved:
                _write_json(control_path, control)
        else:
            _write_json(control_path, control)
    return label_dir


def _write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_jsonl_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _attempt_filename(photo_id: str, attempt: int) -> str:
    digest = hashlib.sha256(photo_id.encode("utf-8")).hexdigest()[:20]
    return f"{digest}__{attempt:04d}.json"


def _persist_verified(
    storage: Storage,
    label_dir: Path,
    path: Path,
    upload_destination: str | None,
) -> str | None:
    if not upload_destination:
        return None
    relative = path.relative_to(label_dir).as_posix()
    key = join_key(upload_destination, relative)
    storage.upload_file(path, key)
    remote = storage.read_bytes(key)
    local = path.read_bytes()
    if hashlib.sha256(remote).digest() != hashlib.sha256(local).digest():
        raise IOError(f"La confirmación remota no coincide: {key}")
    return key


def _recover_confirmations(label_dir: Path) -> dict[str, list[dict]]:
    attempts_dir = label_dir / "attempts"
    confirmations_dir = label_dir / "confirmations"
    attempts = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(attempts_dir.glob("*.json"))
    } if attempts_dir.is_dir() else {}
    confirmations = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(confirmations_dir.glob("*.json"))
    } if confirmations_dir.is_dir() else {}
    if not attempts and not confirmations:
        return {
            "invocations": _read_jsonl_rows(label_dir / "invocations.jsonl"),
            "results": _read_jsonl_rows(label_dir / "results.jsonl"),
            "errors": _read_jsonl_rows(label_dir / "errors.jsonl"),
            "uncertain": [],
        }
    uncertain_names = sorted(attempts.keys() - confirmations.keys())
    uncertain = [attempts[name] for name in uncertain_names]
    uncertain_path = label_dir / "uncertain_attempts.jsonl"
    if uncertain:
        _write_jsonl_rows(uncertain_path, uncertain)
    elif uncertain_path.exists():
        uncertain_path.unlink()

    ordered = [confirmations[name] for name in sorted(confirmations)]
    invocations = [row["invocation"] for row in ordered]
    results = [row["result"] for row in ordered if row.get("result") is not None]
    errors = [row["error"] for row in ordered if row.get("error") is not None]
    for path, rows in (
        (label_dir / "invocations.jsonl", invocations),
        (label_dir / "results.jsonl", results),
        (label_dir / "errors.jsonl", errors),
    ):
        if rows:
            _write_jsonl_rows(path, rows)
        elif path.exists():
            path.unlink()
    return {
        "invocations": invocations,
        "results": results,
        "errors": errors,
        "uncertain": uncertain,
    }


def _checkpoint_inventory(label_dir: Path) -> dict:
    excluded = {"checkpoint.json", ".writer.lock"}
    files = [
        path
        for path in sorted(label_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]
    return {
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "path": path.relative_to(label_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
            for path in files
        ],
    }


def _validate_checkpoint(label_dir: Path) -> None:
    checkpoint_path = label_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        return
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    for item in checkpoint.get("files", []):
        path = label_dir / item["path"]
        if not path.is_file():
            raise ValueError(f"Checkpoint incompleto: falta {item['path']}")
        if path.stat().st_size != item["bytes"] or _sha256_path(path) != item["sha256"]:
            raise ValueError(f"Checkpoint inválido: {item['path']}")


def restore_label_run(
    storage: Storage,
    source: str,
    label_dir: Path,
) -> Path:
    if label_dir.exists() and any(label_dir.iterdir()):
        raise FileExistsError(f"El destino de restauración no está vacío: {label_dir}")
    download_run(storage, source, label_dir)
    _validate_checkpoint(label_dir)
    recovered = _recover_confirmations(label_dir)
    if recovered["uncertain"]:
        raise RuntimeError(
            f"La restauración contiene {len(recovered['uncertain'])} intentos inciertos"
        )
    return label_dir


@contextmanager
def _writer_lock(label_dir: Path):
    path = label_dir / ".writer.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Ya existe un escritor para {label_dir.name}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "pid": os.getpid(),
                "writer_id": str(uuid.uuid4()),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }))
        yield
    finally:
        path.unlink(missing_ok=True)


def _summary(
    selection: list[dict],
    results: list[dict],
    invocations: list[dict],
    error_attempts: int,
) -> dict:
    by_group: dict[str, dict] = {}
    result_by_id = {row["id"]: row for row in results}
    for item in selection:
        group = str(item["cluster"])
        summary = by_group.setdefault(group, {
            "selected": 0,
            "completed": 0,
            "labels": {
                category: {
                    "present": 0,
                    "percentage": 0.0,
                    "severity": {"leve": 0, "medio": 0, "severo": 0},
                }
                for category in CATEGORIES
            },
        })
        summary["selected"] += 1
        result = result_by_id.get(item["id"])
        if not result:
            continue
        summary["completed"] += 1
        for category, value in result["labels"].items():
            if value["presente"]:
                summary["labels"][category]["present"] += 1
                summary["labels"][category]["severity"][value["severidad"]] += 1
    for summary in by_group.values():
        completed = summary["completed"]
        for label in summary["labels"].values():
            label["percentage"] = round(100 * label["present"] / completed, 2) if completed else 0.0
    token_rows = [row.get("tokens_and_cost") or {} for row in invocations]
    input_tokens = sum(row.get("input_tokens") or 0 for row in token_rows)
    output_tokens = sum(row.get("output_tokens") or 0 for row in token_rows)
    total_tokens = sum(row.get("total_tokens") or 0 for row in token_rows)
    known_costs = [
        Decimal(row["estimated_total_cost"])
        for row in token_rows
        if row.get("estimated_total_cost") is not None
    ]
    return {
        "selected": len(selection),
        "completed": len(results),
        "pending": len(selection) - len(results),
        "error_attempts": error_attempts,
        "invocation_attempts": len(invocations),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_total_cost": format(sum(known_costs, Decimal("0")), "f"),
        "cost_currency": "USD",
        "attempts_without_known_cost": len(token_rows) - len(known_costs),
        "cost_kind": "estimated_from_reported_tokens",
        "groups": by_group,
    }


def label_batch(
    settings: Settings,
    storage: Storage,
    run_id: str,
    label_run_id: str,
    model_id: str,
    region: str | None = None,
    checkpoint_every: int = 5,
    upload_destination: str | None = None,
    client=None,
    pricing_catalog: Path = PRICING_CATALOG,
    require_approval: bool = False,
    max_attempts_per_photo: int = 3,
) -> Path:
    if max_attempts_per_photo < 1:
        raise ValueError("max_attempts_per_photo debe ser positivo")
    client = client or bedrock_client(region)
    selected_region = getattr(getattr(client, "meta", None), "region_name", None) or region
    if not selected_region:
        raise ValueError("No se pudo determinar la región de Bedrock")
    label_dir = prepare_label_run(
        settings,
        run_id,
        label_run_id,
        model_id,
        selected_region,
        pricing_catalog,
    )
    selection_path = label_dir / "selection.jsonl"
    selection = read_jsonl(selection_path)
    prompt = (label_dir / "prompt.txt").read_text(encoding="utf-8")
    pricing_path = label_dir / "pricing.json"
    frozen_pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
    pricing_reference = pricing_sha256(pricing_path)
    run_path = label_dir / "run.json"
    configuration = json.loads(run_path.read_text(encoding="utf-8"))
    started_at = configuration["started_at"]
    control = None
    if require_approval:
        control_path = label_dir / "execution_control.json"
        if not control_path.is_file():
            raise RuntimeError("Falta execution_control.json aprobado")
        control = json.loads(control_path.read_text(encoding="utf-8"))
        if not control.get("approved"):
            raise RuntimeError("La ejecución no está aprobada")
        for key in (
            "model_id",
            "region",
            "prompt_sha256",
            "pricing_sha256",
            "selection_sha256",
            "exclusions_sha256",
        ):
            if control.get(key) != configuration.get(key):
                raise RuntimeError(f"El control aprobado no coincide en {key}")

    with _writer_lock(label_dir):
        recovered = _recover_confirmations(label_dir)
        if recovered["uncertain"]:
            raise RuntimeError(
                f"Hay {len(recovered['uncertain'])} intentos inciertos; revisarlos antes de reintentar"
            )
        completed = {row["id"] for row in recovered["results"]}
        attempts_by_id: dict[str, int] = {}
        for invocation in recovered["invocations"]:
            photo_id = invocation["photo_id"]
            attempts_by_id[photo_id] = max(
                attempts_by_id.get(photo_id, 0),
                int(invocation["attempt"]),
            )
        processed_since_checkpoint = 0

        def checkpoint(status: str) -> dict:
            current = _recover_confirmations(label_dir)
            if current["uncertain"]:
                raise RuntimeError("No se puede consolidar con intentos inciertos")
            summary = _summary(
                selection,
                current["results"],
                current["invocations"],
                len(current["errors"]),
            )
            _write_json(label_dir / "cluster_summary.json", summary)
            _write_json(run_path, {
                **configuration,
                "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                **{
                    key: summary[key]
                    for key in (
                        "selected",
                        "completed",
                        "pending",
                        "error_attempts",
                        "invocation_attempts",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "estimated_total_cost",
                        "cost_currency",
                        "attempts_without_known_cost",
                        "cost_kind",
                    )
                },
            })
            _write_json(label_dir / "checkpoint.json", _checkpoint_inventory(label_dir))
            if upload_destination:
                sources = [
                    path
                    for path in label_dir.rglob("*")
                    if path.is_file() and path.name != ".writer.lock"
                ]
                upload_run(storage, label_dir, upload_destination, sources)
                _persist_verified(
                    storage,
                    label_dir,
                    label_dir / "checkpoint.json",
                    upload_destination,
                )
                print(
                    f"Checkpoint confirmado: {summary['completed']}/{summary['selected']}",
                    flush=True,
                )
            return summary

        checkpoint("in_progress")
        for position, item in enumerate(selection, 1):
            if item["id"] in completed:
                print(f"[{position}/{len(selection)}] {item['id']}: ya completada", flush=True)
                continue
            current = _summary(
                selection,
                recovered["results"],
                recovered["invocations"],
                len(recovered["errors"]),
            )
            if control:
                if current["attempts_without_known_cost"]:
                    checkpoint("paused_unknown_cost")
                    raise RuntimeError("Hay intentos con costo desconocido; ejecución pausada")
                if Decimal(current["estimated_total_cost"]) >= Decimal(
                    control["max_estimated_cost_usd"]
                ):
                    checkpoint("paused_budget")
                    raise RuntimeError("Se alcanzó el tope de costo aprobado")

            attempt = attempts_by_id.get(item["id"], 0) + 1
            if attempt > max_attempts_per_photo:
                print(
                    f"[{position}/{len(selection)}] {item['id']}: límite de intentos alcanzado",
                    flush=True,
                )
                continue
            filename = _attempt_filename(item["id"], attempt)
            attempt_path = label_dir / "attempts" / filename
            attempt_record = {
                "id": item["id"],
                "path": item["path"],
                "cluster": item["cluster"],
                "attempt": attempt,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "started",
                "model_id": model_id,
                "prompt_sha256": configuration["prompt_sha256"],
                "pricing_sha256": configuration["pricing_sha256"],
                "selection_sha256": configuration["selection_sha256"],
            }
            _write_json(attempt_path, attempt_record)
            _persist_verified(storage, label_dir, attempt_path, upload_destination)

            record, invocation = _invoke_photo_audited(
                storage,
                item,
                run_id,
                model_id,
                prompt,
                client,
                selected_region,
                attempt,
                frozen_pricing,
                pricing_reference,
            )
            error_record = None
            if record is not None:
                record["id"] = record.pop("photo_id")
                record["cluster"] = item["cluster"]
                record["detection_status"] = item["detection_status"]
            else:
                invocation_error = invocation["error"]
                error_record = {
                    "id": item["id"],
                    "path": item["path"],
                    "cluster": item["cluster"],
                    "attempted_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": invocation_error["type"],
                    "error": invocation_error["message"],
                    "invocation_id": invocation["invocation_id"],
                }
            confirmation_path = label_dir / "confirmations" / filename
            _write_json(confirmation_path, {
                "schema_version": "1",
                "confirmed_at": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt_record,
                "invocation": invocation,
                "result": record,
                "error": error_record,
            })
            _persist_verified(storage, label_dir, confirmation_path, upload_destination)
            recovered = _recover_confirmations(label_dir)
            attempts_by_id[item["id"]] = attempt
            if record is not None:
                completed.add(item["id"])
                print(f"[{position}/{len(selection)}] {item['id']}: OK confirmado", flush=True)
            else:
                print(
                    f"[{position}/{len(selection)}] {item['id']}: ERROR confirmado {invocation['error']['type']}",
                    flush=True,
                )
            processed_since_checkpoint += 1
            if processed_since_checkpoint >= checkpoint_every:
                checkpoint("in_progress")
                processed_since_checkpoint = 0
        checkpoint("complete" if len(completed) == len(selection) else "partial")
    return label_dir
