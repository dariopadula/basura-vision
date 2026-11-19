from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .config import ROOT


PRICING_CATALOG = ROOT / "config" / "pricing.csv"
QUERY_FIELDS = (
    "model_id",
    "region",
    "routing",
    "service_tier",
    "inference_mode",
    "currency",
)
REQUIRED_COLUMNS = {
    "provider",
    "model_name",
    *QUERY_FIELDS,
    "input_usd_per_million",
    "output_usd_per_million",
    "effective_from",
    "checked_at",
    "source_url",
}
COST_QUANTUM = Decimal("0.000000000001")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_query(query: Mapping[str, object]) -> dict[str, str]:
    missing = [field for field in QUERY_FIELDS if not query.get(field)]
    if missing:
        raise ValueError("Consulta de tarifa incompleta: " + ", ".join(missing))
    normalized = {field: str(query[field]).strip() for field in QUERY_FIELDS}
    for field in ("routing", "service_tier", "inference_mode"):
        normalized[field] = normalized[field].lower()
    normalized["currency"] = normalized["currency"].upper()
    return normalized


def resolve_price(catalog_path: Path, query: Mapping[str, object]) -> dict[str, object]:
    normalized = normalize_query(query)
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        if missing_columns:
            raise ValueError("Columnas faltantes en pricing.csv: " + ", ".join(missing_columns))
        matches = [
            row
            for row in reader
            if all(str(row.get(field, "")).strip().lower() == value.lower() for field, value in normalized.items())
        ]
    if len(matches) != 1:
        raise ValueError(
            f"La consulta de tarifa debe resolver una fila; coincidencias: {len(matches)}"
        )
    row = matches[0]
    try:
        input_rate = Decimal(row["input_usd_per_million"])
        output_rate = Decimal(row["output_usd_per_million"])
    except (InvalidOperation, KeyError) as exc:
        raise ValueError("Tarifa inválida en pricing.csv") from exc
    if input_rate < 0 or output_rate < 0:
        raise ValueError("Las tarifas no pueden ser negativas")
    return {
        "provider": row["provider"].strip(),
        "model_name": row["model_name"].strip(),
        **normalized,
        "input_usd_per_million": str(input_rate),
        "output_usd_per_million": str(output_rate),
        "effective_from": row["effective_from"].strip() or None,
        "checked_at": row["checked_at"].strip(),
        "source_url": row["source_url"].strip(),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def freeze_price(
    destination: Path,
    query: Mapping[str, object],
    catalog_path: Path = PRICING_CATALOG,
) -> dict:
    normalized = normalize_query(query)
    if destination.is_file():
        frozen = json.loads(destination.read_text(encoding="utf-8"))
        if frozen.get("query") != normalized:
            raise ValueError("pricing.json pertenece a otra consulta de tarifa")
        return frozen
    rate = resolve_price(catalog_path, normalized)
    frozen = {
        "schema_version": "1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "query": normalized,
        "rate": rate,
        "catalog": {
            "type": "csv",
            "filename": catalog_path.name,
            "sha256": sha256_file(catalog_path),
        },
        "cost_kind": "estimated_from_reported_tokens",
        "billing_note": "Estimación; no sustituye la facturación de AWS.",
    }
    _write_json(destination, frozen)
    return frozen


def pricing_sha256(pricing_path: Path) -> str:
    return sha256_file(pricing_path)


def estimate_cost(usage: Mapping[str, object] | None, frozen: Mapping[str, object]) -> dict:
    input_tokens = usage.get("inputTokens") if usage else None
    output_tokens = usage.get("outputTokens") if usage else None
    reported_total = usage.get("totalTokens") if usage else None
    if reported_total is None and input_tokens is not None and output_tokens is not None:
        reported_total = int(input_tokens) + int(output_tokens)

    rate = frozen["rate"]

    def component(tokens: object | None, field: str) -> str | None:
        if tokens is None:
            return None
        value = Decimal(int(tokens)) * Decimal(str(rate[field])) / Decimal(1_000_000)
        return format(value.quantize(COST_QUANTUM), "f")

    input_cost = component(input_tokens, "input_usd_per_million")
    output_cost = component(output_tokens, "output_usd_per_million")
    total_cost = None
    if input_cost is not None and output_cost is not None:
        total_cost = format(
            (Decimal(input_cost) + Decimal(output_cost)).quantize(COST_QUANTUM),
            "f",
        )
    return {
        "input_tokens": int(input_tokens) if input_tokens is not None else None,
        "output_tokens": int(output_tokens) if output_tokens is not None else None,
        "total_tokens": int(reported_total) if reported_total is not None else None,
        "estimated_input_cost": input_cost,
        "estimated_output_cost": output_cost,
        "estimated_total_cost": total_cost,
        "currency": rate["currency"],
        "cost_kind": frozen["cost_kind"],
    }
