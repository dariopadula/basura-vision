from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from decimal import Decimal
from itertools import combinations
from pathlib import Path

from .bedrock import CATEGORIES, SEVERITIES

CATEGORY_LABELS = {
    "basura_fuera": "Basura fuera",
    "residuos_esparcidos": "Residuos esparcidos",
    "poda": "Poda",
    "contenedor_desbordado": "Contenedor desbordado",
    "contenedor_roto": "Contenedor roto",
    "voluminoso": "Voluminoso",
}
SEVERITY_ORDER = ("leve", "medio", "severo")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: JSON inválido") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: se esperaba un objeto JSON")
        rows.append(row)
    return rows


def _by_id(rows: list[dict], source: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        photo_id = str(row.get("id", "")).strip()
        if not photo_id or photo_id in indexed:
            raise ValueError(f"{source}: ID vacío o duplicado: {photo_id!r}")
        indexed[photo_id] = row
    return indexed


def _group_key(value: object) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        order = {"sin_contenedor": 0, "sin_residuos": 1}
        return (1, order.get(text, 2), text)


def _percentage(count: int, total: int) -> float:
    return round(100 * count / total, 2) if total else 0.0


def _validate_labels(result: dict) -> dict:
    labels = result.get("labels")
    if not isinstance(labels, dict) or set(labels) != set(CATEGORIES):
        raise ValueError(f"Resultado inválido para {result.get('id')}: etiquetas incompletas")
    for category in CATEGORIES:
        value = labels[category]
        if not isinstance(value, dict):
            raise ValueError(f"Resultado inválido para {result.get('id')}: {category}")
        present = value.get("presente")
        severity = value.get("severidad")
        if not isinstance(present, bool):
            raise ValueError(f"Resultado inválido para {result.get('id')}: presencia")
        if present and severity not in SEVERITIES:
            raise ValueError(f"Resultado inválido para {result.get('id')}: severidad")
        if not present and severity is not None:
            raise ValueError(f"Resultado inválido para {result.get('id')}: ausencia")
    return labels


def _latency(invocation: dict) -> float | None:
    metrics = invocation.get("metrics") or {}
    value = metrics.get("latencyMs")
    if value is None:
        value = invocation.get("observed_latency_ms")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)


def build_summary(run_dir: Path, label_run_id: str) -> dict:
    if Path(label_run_id).name != label_run_id:
        raise ValueError("label_run_id debe ser un nombre simple")
    run_dir = run_dir.resolve()
    label_dir = run_dir / "bedrock" / label_run_id
    if not label_dir.is_dir():
        raise ValueError(f"Corrida Bedrock inexistente: {label_run_id}")

    selection = _read_jsonl(label_dir / "selection.jsonl")
    if not selection:
        raise ValueError(f"Selección inexistente o vacía: {label_run_id}")
    selected_by_id = _by_id(selection, "selection.jsonl")
    results = _read_jsonl(label_dir / "results.jsonl")
    result_by_id = _by_id(results, "results.jsonl") if results else {}
    unknown = sorted(set(result_by_id) - set(selected_by_id))
    if unknown:
        raise ValueError(f"Hay resultados ajenos a la selección: {unknown[:3]}")

    grouped: dict[str, list[dict]] = {}
    for row in selection:
        group = str(row.get("cluster", "")).strip()
        if not group:
            raise ValueError(f"La selección no indica grupo para {row.get('id')}")
        grouped.setdefault(group, []).append(row)

    groups = []
    for group_name in sorted(grouped, key=_group_key):
        selected_rows = grouped[group_name]
        valid_results = [
            result_by_id[row["id"]]
            for row in selected_rows
            if row["id"] in result_by_id
        ]
        valid = len(valid_results)
        label_counts = {
            category: {"ausente": 0, "leve": 0, "medio": 0, "severo": 0}
            for category in CATEGORIES
        }
        pair_counts = Counter()
        combination_counts = Counter()
        for result in valid_results:
            labels = _validate_labels(result)
            present = []
            for category in CATEGORIES:
                value = labels[category]
                state = value["severidad"] if value["presente"] else "ausente"
                label_counts[category][state] += 1
                if value["presente"]:
                    present.append(category)
            combination_counts[tuple(present)] += 1
            for pair in combinations(present, 2):
                pair_counts[pair] += 1

        label_summary = {}
        for category in CATEGORIES:
            counts = label_counts[category]
            present_count = sum(counts[level] for level in SEVERITY_ORDER)
            label_summary[category] = {
                "label": CATEGORY_LABELS[category],
                "present": {"count": present_count, "percentage": _percentage(present_count, valid)},
                **{
                    state: {"count": counts[state], "percentage": _percentage(counts[state], valid)}
                    for state in ("ausente", *SEVERITY_ORDER)
                },
            }

        cooccurrences = [
            {
                "category_a": left,
                "category_b": right,
                "count": pair_counts[(left, right)],
                "percentage": _percentage(pair_counts[(left, right)], valid),
            }
            for left, right in combinations(CATEGORIES, 2)
        ]
        exact_combinations = [
            {
                "categories": list(combo),
                "count": count,
                "percentage": _percentage(count, valid),
            }
            for combo, count in sorted(
                combination_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        maximum = max((value["present"]["count"] for value in label_summary.values()), default=0)
        dominant = [
            category
            for category in CATEGORIES
            if maximum and label_summary[category]["present"]["count"] == maximum
        ]
        groups.append({
            "group": group_name,
            "selected": len(selected_rows),
            "valid": valid,
            "pending": len(selected_rows) - valid,
            "coverage_percentage": _percentage(valid, len(selected_rows)),
            "labels": label_summary,
            "dominant_labels": dominant,
            "cooccurrences": cooccurrences,
            "exact_combinations": exact_combinations,
        })

    invocations = _read_jsonl(label_dir / "invocations.jsonl")
    errors = _read_jsonl(label_dir / "errors.jsonl")
    uncertain = _read_jsonl(label_dir / "uncertain_attempts.jsonl")
    token_rows = [row.get("tokens_and_cost") or {} for row in invocations]
    known_costs = [
        Decimal(str(row["estimated_total_cost"]))
        for row in token_rows
        if row.get("estimated_total_cost") is not None
    ]
    latencies = [value for row in invocations if (value := _latency(row)) is not None]
    run_metadata = {}
    run_path = label_dir / "run.json"
    if run_path.is_file():
        run_metadata = json.loads(run_path.read_text(encoding="utf-8"))

    return {
        "schema_version": "1",
        "source_run_id": run_dir.name,
        "label_run_id": label_run_id,
        "model_id": run_metadata.get("model_id"),
        "prompt_version": run_metadata.get("prompt_version"),
        "run_status": run_metadata.get("status", "unknown"),
        "categories": [
            {"id": category, "label": CATEGORY_LABELS[category]}
            for category in CATEGORIES
        ],
        "totals": {
            "selected": len(selection),
            "valid": len(results),
            "pending": len(selection) - len(results),
            "coverage_percentage": _percentage(len(results), len(selection)),
            "invocation_attempts": len(invocations),
            "error_attempts": len(errors),
            "uncertain_attempts": len(uncertain),
            "input_tokens": sum(row.get("input_tokens") or 0 for row in token_rows),
            "output_tokens": sum(row.get("output_tokens") or 0 for row in token_rows),
            "total_tokens": sum(row.get("total_tokens") or 0 for row in token_rows),
            "estimated_total_cost": format(sum(known_costs, Decimal("0")), "f"),
            "cost_currency": run_metadata.get("cost_currency", "USD"),
            "attempts_without_known_cost": len(token_rows) - len(known_costs),
            "median_latency_ms": round(statistics.median(latencies), 3) if latencies else None,
            "p95_latency_ms": _percentile(latencies, 0.95),
            "attempts_without_latency": len(invocations) - len(latencies),
        },
        "groups": groups,
    }


def write_summary(run_dir: Path, label_run_id: str) -> Path:
    summary = build_summary(run_dir, label_run_id)
    output_dir = run_dir.resolve() / "bedrock" / label_run_id / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "groups.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "group", "selected", "valid", "category", "present_count", "present_percentage",
            "absent_count", "absent_percentage", "leve_count", "leve_percentage",
            "medio_count", "medio_percentage", "severo_count", "severo_percentage",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in summary["groups"]:
            for category in CATEGORIES:
                label = group["labels"][category]
                writer.writerow({
                    "group": group["group"], "selected": group["selected"], "valid": group["valid"],
                    "category": category,
                    "present_count": label["present"]["count"],
                    "present_percentage": label["present"]["percentage"],
                    **{
                        f"{prefix}_{field}": label[state][field]
                        for state, prefix in (("ausente", "absent"), ("leve", "leve"), ("medio", "medio"), ("severo", "severo"))
                        for field in ("count", "percentage")
                    },
                })
    with (output_dir / "cooccurrences.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("group", "valid", "category_a", "category_b", "count", "percentage"),
        )
        writer.writeheader()
        for group in summary["groups"]:
            for row in group["cooccurrences"]:
                writer.writerow({"group": group["group"], "valid": group["valid"], **row})
    return output_dir


def list_label_runs(run_dir: Path) -> list[dict]:
    bedrock_dir = run_dir.resolve() / "bedrock"
    if not bedrock_dir.is_dir():
        return []
    runs = []
    for label_dir in sorted(path for path in bedrock_dir.iterdir() if path.is_dir()):
        if not (label_dir / "selection.jsonl").is_file():
            continue
        metadata = {}
        if (label_dir / "run.json").is_file():
            metadata = json.loads((label_dir / "run.json").read_text(encoding="utf-8"))
        runs.append({
            "label_run_id": label_dir.name,
            "status": metadata.get("status", "unknown"),
            "selected": metadata.get("selected"),
            "completed": metadata.get("completed"),
            "model_id": metadata.get("model_id"),
        })
    return runs


def build_photo_labels(run_dir: Path, label_run_id: str) -> dict:
    """Return the frozen sample and compact valid labels for the photo viewer."""
    if Path(label_run_id).name != label_run_id:
        raise ValueError("label_run_id debe ser un nombre simple")
    label_dir = run_dir.resolve() / "bedrock" / label_run_id
    if not label_dir.is_dir():
        raise ValueError(f"Corrida Bedrock inexistente: {label_run_id}")
    selection = _read_jsonl(label_dir / "selection.jsonl")
    if not selection:
        raise ValueError(f"Selección inexistente o vacía: {label_run_id}")
    selected_by_id = _by_id(selection, "selection.jsonl")
    results = _read_jsonl(label_dir / "results.jsonl")
    result_by_id = _by_id(results, "results.jsonl") if results else {}
    unknown = sorted(set(result_by_id) - set(selected_by_id))
    if unknown:
        raise ValueError(f"Hay resultados ajenos a la selección: {unknown[:3]}")
    compact_results = []
    for photo_id, result in result_by_id.items():
        compact_results.append({
            "id": photo_id,
            "cluster": str(selected_by_id[photo_id]["cluster"]),
            "labels": _validate_labels(result),
        })
    return {
        "label_run_id": label_run_id,
        "selected": [
            {"id": row["id"], "cluster": str(row["cluster"])}
            for row in selection
        ],
        "results": compact_results,
    }
