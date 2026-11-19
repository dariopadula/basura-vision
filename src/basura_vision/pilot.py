from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

from .bedrock import _selection, prepare_label_run
from .config import Settings
from .manifests import read_jsonl, write_jsonl
from .pricing import PRICING_CATALOG


NUMERIC_GROUPS = tuple(str(number) for number in range(1, 16))
SPECIAL_GROUPS = ("sin_contenedor", "sin_residuos")
PILOT_COUNTS = {**{group: 2 for group in NUMERIC_GROUPS}, "sin_contenedor": 5, "sin_residuos": 5}
VALIDATION_COUNTS = {**{group: 2 for group in NUMERIC_GROUPS}, "sin_contenedor": 2, "sin_residuos": 2}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event_key(row: dict) -> str:
    return str(row.get("event_id") or row["id"])


def _available(rows: list[dict], used_events: set[str]) -> list[dict]:
    return [row for row in sorted(rows, key=lambda item: item["id"]) if _event_key(row) not in used_events]


def _choose_random(
    rows: list[dict],
    count: int,
    rng: random.Random,
    used_events: set[str],
    role: str,
) -> list[dict]:
    candidates = _available(rows, used_events)
    rng.shuffle(candidates)
    chosen = []
    for candidate in candidates:
        event = _event_key(candidate)
        if event in used_events:
            continue
        selected = {**candidate, "selection_role": role}
        chosen.append(selected)
        used_events.add(event)
        if len(chosen) == count:
            return chosen
    raise ValueError(f"No hay {count} eventos disponibles para {role}")


def _contrast(random_row: dict, rows: list[dict], used_events: set[str]) -> dict:
    candidates = _available(rows, used_events)
    if not candidates:
        raise ValueError(f"No hay contraste disponible para el grupo {random_row['cluster']}")
    all_rows = [random_row, *candidates]
    features = [
        (
            float(row.get("pc1") or 0.0),
            float(row.get("pc2") or 0.0),
            math.log1p(float(row.get("ratio") or 0.0)),
        )
        for row in all_rows
    ]
    means = [sum(values) / len(values) for values in zip(*features)]
    deviations = []
    for position, mean in enumerate(means):
        variance = sum((values[position] - mean) ** 2 for values in features) / len(features)
        deviations.append(math.sqrt(variance) or 1.0)
    origin = features[0]

    def distance(index_and_row):
        index, row = index_and_row
        values = features[index + 1]
        squared = sum(
            ((values[position] - origin[position]) / deviations[position]) ** 2
            for position in range(3)
        )
        return squared, row["id"]

    _, selected = max(enumerate(candidates), key=distance)
    used_events.add(_event_key(selected))
    return {**selected, "selection_role": "contrast_pc1_pc2_ratio"}


def _freeze_jsonl(path: Path, rows: list[dict]) -> None:
    if path.is_file():
        existing = (
            read_jsonl(path)
            if path.read_text(encoding="utf-8").strip()
            else []
        )
        if existing != rows:
            raise ValueError(f"La muestra congelada difiere: {path}")
        return
    write_jsonl(path, rows)


def prepare_pilot_samples(
    settings: Settings,
    run_id: str,
    sample_plan_id: str,
    pilot_label_run_id: str,
    validation_label_run_id: str,
    model_id: str,
    region: str,
    seed: int = 20261119,
    pricing_catalog: Path = PRICING_CATALOG,
) -> Path:
    if Path(sample_plan_id).name != sample_plan_id:
        raise ValueError("sample_plan_id debe ser un nombre simple")
    run_dir = (settings.runs_root / run_id).resolve()
    if not run_dir.is_dir() or not run_dir.is_relative_to(settings.runs_root.resolve()):
        raise ValueError(f"Corrida inexistente o inválida: {run_id}")

    selection, exclusions = _selection(run_dir)
    if exclusions:
        excluded_ids = {row["id"] for row in exclusions}
    else:
        excluded_ids = set()
    manifest = {row["id"]: row for row in read_jsonl(run_dir / "manifest.jsonl")}
    assignments = {
        row["id"]: row
        for row in read_jsonl(
            run_dir / "assignments.jsonl",
            required_fields={"id", "cluster", "pc1", "pc2", "ratio"},
        )
    }
    enriched = []
    for row in selection:
        manifest_row = manifest[row["id"]]
        assignment = assignments.get(row["id"], {})
        enriched.append({
            **row,
            "event_id": manifest_row.get("event_id"),
            "pc1": assignment.get("pc1"),
            "pc2": assignment.get("pc2"),
            "ratio": assignment.get("ratio"),
        })
    by_group: dict[str, list[dict]] = {}
    for row in enriched:
        by_group.setdefault(str(row["cluster"]), []).append(row)
    expected_groups = set(NUMERIC_GROUPS) | set(SPECIAL_GROUPS)
    if set(by_group) != expected_groups:
        raise ValueError(f"Grupos inesperados: {sorted(set(by_group) ^ expected_groups)}")

    rng = random.Random(seed)
    pilot: list[dict] = []
    pilot_events: set[str] = set()
    for group in NUMERIC_GROUPS:
        first = _choose_random(by_group[group], 1, rng, pilot_events, "random")[0]
        pilot.extend([first, _contrast(first, by_group[group], pilot_events)])
    for group in SPECIAL_GROUPS:
        pilot.extend(
            _choose_random(
                by_group[group],
                PILOT_COUNTS[group],
                rng,
                pilot_events,
                "random",
            )
        )

    validation: list[dict] = []
    validation_events: set[str] = set(pilot_events)
    for group in (*NUMERIC_GROUPS, *SPECIAL_GROUPS):
        validation.extend(
            _choose_random(
                by_group[group],
                VALIDATION_COUNTS[group],
                rng,
                validation_events,
                "independent_random",
            )
        )
    if len(pilot) != 40 or len(validation) != 34:
        raise AssertionError("Tamaños de muestra inesperados")
    if {row["id"] for row in pilot} & {row["id"] for row in validation}:
        raise AssertionError("Las muestras comparten IDs")
    if {_event_key(row) for row in pilot} & {_event_key(row) for row in validation}:
        raise AssertionError("Las muestras comparten event_id")
    if excluded_ids & ({row["id"] for row in pilot} | {row["id"] for row in validation}):
        raise AssertionError("Una exclusión apareció en las muestras")

    plan_dir = run_dir / "bedrock" / "sample_plans" / sample_plan_id
    plan_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = plan_dir / "pilot_selection.jsonl"
    validation_path = plan_dir / "validation_selection.jsonl"
    _freeze_jsonl(pilot_path, pilot)
    _freeze_jsonl(validation_path, validation)
    plan_path = plan_dir / "sample_plan.json"
    plan = {
        "schema_version": "1",
        "sample_plan_id": sample_plan_id,
        "source_run_id": run_id,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "criteria": {
            "pilot_numeric": "one random and one farthest standardized pc1/pc2/log1p(ratio) per cluster",
            "pilot_special": "five random unique events per group",
            "validation": "two random events per group, excluding every pilot event",
        },
        "pilot_count": len(pilot),
        "validation_count": len(validation),
        "pilot_group_counts": dict(sorted(Counter(str(row["cluster"]) for row in pilot).items())),
        "validation_group_counts": dict(sorted(Counter(str(row["cluster"]) for row in validation).items())),
        "event_overlap": [],
        "pilot_sha256": _sha256(pilot_path),
        "validation_sha256": _sha256(validation_path),
    }
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("created_at", None)
        expected = dict(plan)
        expected.pop("created_at", None)
        if comparable != expected:
            raise ValueError("El plan de muestras existente difiere")
    else:
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for label_run_id, rows in (
        (pilot_label_run_id, pilot),
        (validation_label_run_id, validation),
    ):
        label_dir = run_dir / "bedrock" / label_run_id
        label_dir.mkdir(parents=True, exist_ok=True)
        _freeze_jsonl(label_dir / "selection.jsonl", rows)
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
