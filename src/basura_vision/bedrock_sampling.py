from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

from .bedrock import _selection, prepare_label_run
from .config import Settings
from .manifests import read_jsonl
from .pilot import (
    NUMERIC_GROUPS,
    SPECIAL_GROUPS,
    _event_key,
    _freeze_jsonl,
    _sha256,
)
from .pricing import PRICING_CATALOG


GROUPS = (*NUMERIC_GROUPS, *SPECIAL_GROUPS)


def bounded_proportional_quotas(
    counts: dict[str, int],
    total: int,
    minimum: int,
    maximum: int,
) -> dict[str, int]:
    if set(counts) != set(GROUPS):
        raise ValueError("Las cuotas requieren exactamente los 17 grupos")
    if minimum < 1 or maximum < minimum:
        raise ValueError("Mínimo y máximo inválidos")
    if any(value < 1 for value in counts.values()):
        raise ValueError("Todos los grupos deben contener fotos")
    lower = {group: min(minimum, counts[group]) for group in GROUPS}
    upper = {group: min(maximum, counts[group]) for group in GROUPS}
    if not sum(lower.values()) <= total <= sum(upper.values()):
        raise ValueError(
            f"Total {total} incompatible con límites "
            f"[{sum(lower.values())}, {sum(upper.values())}]"
        )

    low = 0.0
    high = max(total / min(counts.values()), 1.0)
    while sum(min(upper[g], max(lower[g], high * counts[g])) for g in GROUPS) < total:
        high *= 2
    for _ in range(100):
        middle = (low + high) / 2
        assigned = sum(
            min(upper[group], max(lower[group], middle * counts[group]))
            for group in GROUPS
        )
        if assigned < total:
            low = middle
        else:
            high = middle

    raw = {
        group: min(upper[group], max(lower[group], high * counts[group]))
        for group in GROUPS
    }
    quotas = {group: math.floor(raw[group]) for group in GROUPS}
    remaining = total - sum(quotas.values())
    candidates = sorted(
        (group for group in GROUPS if quotas[group] < upper[group]),
        key=lambda group: (-(raw[group] - quotas[group]), GROUPS.index(group)),
    )
    for group in candidates[:remaining]:
        quotas[group] += 1
    if sum(quotas.values()) != total:
        raise AssertionError("No se pudo ajustar el total de cuotas")
    return quotas


def _freeze_json(path: Path, value: dict) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("created_at", None)
        expected = dict(value)
        expected.pop("created_at", None)
        if comparable != expected:
            raise ValueError(f"El plan congelado difiere: {path}")
        return
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_stratified_sample(
    settings: Settings,
    run_id: str,
    sample_plan_id: str,
    label_run_id: str,
    model_id: str,
    region: str,
    excluded_label_run_ids: list[str],
    total: int = 1000,
    minimum_per_group: int = 20,
    maximum_per_group: int = 100,
    seed: int = 20261120,
    pricing_catalog: Path = PRICING_CATALOG,
) -> Path:
    if Path(sample_plan_id).name != sample_plan_id:
        raise ValueError("sample_plan_id debe ser un nombre simple")
    if len(set(excluded_label_run_ids)) != len(excluded_label_run_ids):
        raise ValueError("Hay label_run_id excluidos repetidos")
    run_dir = (settings.runs_root / run_id).resolve()
    if not run_dir.is_dir() or not run_dir.is_relative_to(settings.runs_root.resolve()):
        raise ValueError(f"Corrida inexistente o inválida: {run_id}")

    source, _ = _selection(run_dir)
    manifest = {row["id"]: row for row in read_jsonl(run_dir / "manifest.jsonl")}
    enriched = [
        {**row, "event_id": manifest[row["id"]].get("event_id")}
        for row in source
    ]
    by_group: dict[str, list[dict]] = {group: [] for group in GROUPS}
    for row in enriched:
        group = str(row["cluster"])
        if group not in by_group:
            raise ValueError(f"Grupo inesperado: {group}")
        by_group[group].append(row)

    excluded_ids: set[str] = set()
    excluded_events: set[str] = set()
    excluded_sources = []
    for excluded_label_run_id in excluded_label_run_ids:
        if Path(excluded_label_run_id).name != excluded_label_run_id:
            raise ValueError("label_run_id excluido inválido")
        path = run_dir / "bedrock" / excluded_label_run_id / "selection.jsonl"
        rows = read_jsonl(path)
        excluded_ids.update(row["id"] for row in rows)
        excluded_events.update(_event_key(row) for row in rows)
        excluded_sources.append({
            "label_run_id": excluded_label_run_id,
            "selection_sha256": _sha256(path),
            "records": len(rows),
        })

    available = {
        group: [
            row for row in rows
            if row["id"] not in excluded_ids and _event_key(row) not in excluded_events
        ]
        for group, rows in by_group.items()
    }
    counts = {group: len(rows) for group, rows in available.items()}
    unique_event_counts = {
        group: len({_event_key(row) for row in rows})
        for group, rows in available.items()
    }
    quotas = bounded_proportional_quotas(
        counts,
        total,
        minimum_per_group,
        maximum_per_group,
    )
    insufficient = {
        group: {"quota": quotas[group], "unique_events": unique_event_counts[group]}
        for group in GROUPS
        if unique_event_counts[group] < quotas[group]
    }
    if insufficient:
        raise ValueError(f"No hay eventos únicos suficientes: {insufficient}")

    rng = random.Random(seed)
    shuffled = {}
    for group in GROUPS:
        candidates = sorted(available[group], key=lambda row: row["id"])
        rng.shuffle(candidates)
        shuffled[group] = candidates
    selected: list[dict] = []
    selected_by_group = Counter()
    used_events = set(excluded_events)
    positions = {group: 0 for group in GROUPS}
    while len(selected) < total:
        progressed = False
        for group in GROUPS:
            if selected_by_group[group] >= quotas[group]:
                continue
            candidates = shuffled[group]
            while positions[group] < len(candidates):
                candidate = candidates[positions[group]]
                positions[group] += 1
                event = _event_key(candidate)
                if event in used_events:
                    continue
                selected.append({
                    **candidate,
                    "selection_role": "random_stratified_group",
                })
                selected_by_group[group] += 1
                used_events.add(event)
                progressed = True
                break
        if not progressed:
            raise ValueError("No se pudo completar la muestra sin repetir eventos")

    if len({row["id"] for row in selected}) != total:
        raise AssertionError("La muestra contiene IDs duplicados")
    selected_events = {_event_key(row) for row in selected}
    if len(selected_events) != total or selected_events & excluded_events:
        raise AssertionError("La muestra contiene eventos duplicados o reservados")
    actual_counts = Counter(str(row["cluster"]) for row in selected)
    if dict(actual_counts) != quotas:
        raise AssertionError("La muestra no coincide con las cuotas calculadas")

    plan_dir = run_dir / "bedrock" / "sample_plans" / sample_plan_id
    plan_dir.mkdir(parents=True, exist_ok=True)
    selection_path = plan_dir / "selection.jsonl"
    _freeze_jsonl(selection_path, selected)
    plan = {
        "schema_version": "1",
        "sample_plan_id": sample_plan_id,
        "source_run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "target": total,
        "minimum_per_group": minimum_per_group,
        "maximum_per_group": maximum_per_group,
        "method": "random within group; quotas proportional with iterative min/max redistribution",
        "event_unique": True,
        "excluded_label_runs": excluded_sources,
        "source_group_counts": {
            group: len(by_group[group]) for group in GROUPS
        },
        "available_group_counts": counts,
        "available_unique_event_counts": unique_event_counts,
        "quotas": quotas,
        "selected_group_counts": {
            group: actual_counts[group] for group in GROUPS
        },
        "selection_sha256": _sha256(selection_path),
    }
    _freeze_json(plan_dir / "sample_plan.json", plan)

    label_dir = run_dir / "bedrock" / label_run_id
    label_dir.mkdir(parents=True, exist_ok=True)
    _freeze_jsonl(label_dir / "selection.jsonl", selected)
    _freeze_jsonl(label_dir / "exclusions.jsonl", [])
    prepare_label_run(
        settings,
        run_id,
        label_run_id,
        model_id,
        region,
        pricing_catalog,
    )
    return plan_dir
