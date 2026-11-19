from __future__ import annotations

import hashlib
import io
import json
import os
import time
import uuid
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from .config import ROOT, Settings
from .manifests import read_jsonl, write_jsonl
from .pricing import PRICING_CATALOG, estimate_cost, freeze_price, pricing_sha256
from .storage import Storage, upload_run

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


def _selection(run_dir: Path) -> list[dict]:
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
    for photo in manifest:
        detection = detections[photo["id"]]
        assignment = assignments.get(photo["id"])
        group = assignment["cluster"] if assignment else detection["status"]
        rows.append({
            "id": photo["id"],
            "path": photo["path"],
            "cluster": group,
            "detection_status": detection["status"],
        })
    return rows


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
) -> Path:
    if Path(label_run_id).name != label_run_id:
        raise ValueError("label_run_id debe ser un nombre simple, sin carpetas")
    run_dir = (settings.runs_root / run_id).resolve()
    if not run_dir.is_dir() or not run_dir.is_relative_to(settings.runs_root.resolve()):
        raise ValueError(f"Corrida inexistente o inválida: {run_id}")
    label_dir = run_dir / "bedrock" / label_run_id
    label_dir.mkdir(parents=True, exist_ok=True)
    selection_path = label_dir / "selection.jsonl"
    selection = read_jsonl(selection_path) if selection_path.is_file() else _selection(run_dir)
    if not selection_path.is_file():
        write_jsonl(selection_path, selection)

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    client = client or bedrock_client(region)
    selected_region = getattr(getattr(client, "meta", None), "region_name", None) or region
    pricing_path = label_dir / "pricing.json"
    frozen_pricing = freeze_price(
        pricing_path,
        {
            "model_id": model_id,
            "region": selected_region,
            "routing": "direct",
            "service_tier": "standard",
            "inference_mode": "on_demand",
            "currency": "USD",
        },
        pricing_catalog,
    )
    pricing_reference = pricing_sha256(pricing_path)
    configuration = {
        "schema_version": "1",
        "label_run_id": label_run_id,
        "source_run_id": run_id,
        "image_source": "original",
        "model_id": model_id,
        "region": selected_region,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "pricing_sha256": pricing_reference,
        "selection": "all_manifest_photos",
    }
    run_path = label_dir / "run.json"
    if run_path.is_file():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        for key, value in configuration.items():
            if previous.get(key) != value:
                raise ValueError(f"La corrida existente usa otro valor para {key}")
        started_at = previous["started_at"]
    else:
        started_at = datetime.now(timezone.utc).isoformat()
        (label_dir / "prompt.txt").write_text(prompt, encoding="utf-8", newline="\n")

    results_path = label_dir / "results.jsonl"
    errors_path = label_dir / "errors.jsonl"
    invocations_path = label_dir / "invocations.jsonl"
    results = read_jsonl(results_path) if results_path.is_file() else []
    invocation_fields = {"invocation_id", "photo_id", "attempt"}
    invocations = (
        read_jsonl(invocations_path, required_fields=invocation_fields)
        if invocations_path.is_file()
        else []
    )
    completed = {row["id"] for row in results}
    attempts_by_id: dict[str, int] = {}
    for invocation in invocations:
        photo_id = invocation["photo_id"]
        attempts_by_id[photo_id] = max(
            attempts_by_id.get(photo_id, 0),
            int(invocation["attempt"]),
        )
    error_attempts = len(errors_path.read_text(encoding="utf-8").splitlines()) if errors_path.is_file() else 0
    processed_since_checkpoint = 0

    def checkpoint(status: str) -> None:
        current_results = read_jsonl(results_path) if results_path.is_file() else []
        current_invocations = (
            read_jsonl(invocations_path, required_fields=invocation_fields)
            if invocations_path.is_file()
            else []
        )
        summary = _summary(selection, current_results, current_invocations, error_attempts)
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
        if upload_destination:
            upload_run(storage, label_dir, upload_destination)
            print(f"Checkpoint S3: {summary['completed']}/{summary['selected']}", flush=True)

    checkpoint("in_progress")
    for position, item in enumerate(selection, 1):
        if item["id"] in completed:
            print(f"[{position}/{len(selection)}] {item['id']}: ya completada", flush=True)
            continue
        invocation = None
        try:
            attempt = attempts_by_id.get(item["id"], 0) + 1
            record, invocation = _invoke_photo_audited(
                storage,
                item,
                run_id,
                model_id,
                prompt,
                client,
                region,
                attempt,
                frozen_pricing,
                pricing_reference,
            )
            _append_jsonl(invocations_path, invocation)
            attempts_by_id[item["id"]] = attempt
            if record is None:
                error = invocation["error"]
                raise ValueError(f"{error['type']}: {error['message']}")
            record["id"] = record.pop("photo_id")
            record["cluster"] = item["cluster"]
            record["detection_status"] = item["detection_status"]
            _append_jsonl(results_path, record)
            completed.add(item["id"])
            print(f"[{position}/{len(selection)}] {item['id']}: OK", flush=True)
        except Exception as exc:
            error_attempts += 1
            invocation_error = (invocation or {}).get("error") or {}
            _append_jsonl(errors_path, {
                "id": item["id"],
                "path": item["path"],
                "cluster": item["cluster"],
                "attempted_at": datetime.now(timezone.utc).isoformat(),
                "error_type": invocation_error.get("type", type(exc).__name__),
                "error": invocation_error.get("message", str(exc)),
                "invocation_id": (invocation or {}).get("invocation_id"),
            })
            print(f"[{position}/{len(selection)}] {item['id']}: ERROR {type(exc).__name__}", flush=True)
        processed_since_checkpoint += 1
        if processed_since_checkpoint >= checkpoint_every:
            checkpoint("in_progress")
            processed_since_checkpoint = 0
    checkpoint("complete" if len(completed) == len(selection) else "partial")
    return label_dir
