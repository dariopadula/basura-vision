from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .bedrock import CATEGORIES, _label_run_dir
from .bedrock_review import load_blind_reviews, validate_blind_review
from .config import Settings
from .manifests import read_jsonl


SEVERITY_RANK = {None: 0, "leve": 1, "medio": 2, "severo": 3}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "photo_id",
        "path",
        "event_id",
        "cluster",
        "selection_role",
        "category",
        "human_present",
        "human_severity",
        "human_doubtful",
        "model_present",
        "model_severity",
        "presence_agreement",
        "severity_agreement",
        "severity_difference",
        "human_notes",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown(report: dict) -> str:
    lines = [
        "# Comparación ciega del piloto Bedrock",
        "",
        f"Fotos comparadas: **{report['coverage']['comparable']}** de "
        f"{report['coverage']['selected']} seleccionadas.",
        "",
        "| Categoría | Acuerdo presencia | FP | FN | Dudas humanas | Severidad exacta | Sobreestimada |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        values = report["categories"][category]
        agreement = values["presence"]["agreement_rate"]
        severity = values["severity"]["exact_agreement_rate"]
        lines.append(
            f"| {category} | "
            f"{agreement if agreement is not None else '—'} | "
            f"{values['presence']['false_positive']} | "
            f"{values['presence']['false_negative']} | "
            f"{values['human_doubtful']} | "
            f"{severity if severity is not None else '—'} | "
            f"{values['severity']['model_overestimated']} |"
        )
    lines.extend([
        "",
        "## Controles prioritarios",
        "",
        f"Falsos `severo` concluyentes en desborde: "
        f"**{len(report['priority']['false_severe_conclusive']['contenedor_desbordado'])}**.",
        "",
        f"Falsos `severo` concluyentes en rotura: "
        f"**{len(report['priority']['false_severe_conclusive']['contenedor_roto'])}**.",
        "",
        "Los desacuerdos detallados están en `review_comparison.csv`.",
        "",
    ])
    return "\n".join(lines)


def build_review_report(
    settings: Settings,
    run_id: str,
    label_run_id: str,
) -> Path:
    _, label_dir = _label_run_dir(settings, run_id, label_run_id)
    selection_path = label_dir / "selection.jsonl"
    results_path = label_dir / "results.jsonl"
    reviews_path = label_dir / "blind_reviews.json"
    for path in (selection_path, results_path, reviews_path):
        if not path.is_file():
            raise FileNotFoundError(f"Falta {path}")

    selection = read_jsonl(selection_path)
    results = read_jsonl(
        results_path,
        required_fields={"photo_id", "labels"},
    )
    reviews = load_blind_reviews(reviews_path)
    selected = {row["id"]: row for row in selection}
    predicted = {str(row["photo_id"]): row for row in results}
    if len(predicted) != len(results):
        raise ValueError("results.jsonl contiene photo_id duplicados")
    expected_ids = set(selected)
    if set(predicted) != expected_ids:
        raise ValueError("Los resultados no cubren exactamente la selección")
    if set(reviews) != expected_ids:
        missing = sorted(expected_ids - set(reviews))
        extra = sorted(set(reviews) - expected_ids)
        raise ValueError(
            f"Las revisiones no cubren la selección; faltan={missing}, sobran={extra}"
        )

    categories = {}
    detail_rows = []
    false_severe = {
        "contenedor_desbordado": [],
        "contenedor_roto": [],
    }
    disagreement_ids: set[str] = set()
    for category in CATEGORIES:
        presence = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
        conclusive = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
        severity_comparable = 0
        severity_exact = 0
        overestimated = 0
        underestimated = 0
        doubtful = 0
        for photo_id, photo in selected.items():
            review = validate_blind_review(reviews[photo_id])
            human = review["categories"][category]
            model = predicted[photo_id]["labels"][category]
            human_present = human["presente"]
            model_present = model["presente"]
            if human["evidencia_dudosa"]:
                doubtful += 1
            if human_present and model_present:
                key = "true_positive"
            elif not human_present and not model_present:
                key = "true_negative"
            elif not human_present and model_present:
                key = "false_positive"
            else:
                key = "false_negative"
            presence[key] += 1
            if not human["evidencia_dudosa"]:
                conclusive[key] += 1
            severity_difference = None
            severity_agreement = None
            if human_present and model_present:
                severity_comparable += 1
                severity_difference = (
                    SEVERITY_RANK[model["severidad"]]
                    - SEVERITY_RANK[human["severidad"]]
                )
                severity_agreement = severity_difference == 0
                severity_exact += int(severity_agreement)
                overestimated += int(severity_difference > 0)
                underestimated += int(severity_difference < 0)
            if key in {"false_positive", "false_negative"} or severity_difference not in {None, 0}:
                disagreement_ids.add(photo_id)
            if (
                category in false_severe
                and not human["evidencia_dudosa"]
                and not human_present
                and model_present
                and model["severidad"] == "severo"
            ):
                false_severe[category].append(photo_id)
            detail_rows.append({
                "photo_id": photo_id,
                "path": photo["path"],
                "event_id": photo.get("event_id"),
                "cluster": photo.get("cluster"),
                "selection_role": photo.get("selection_role"),
                "category": category,
                "human_present": human_present,
                "human_severity": human["severidad"],
                "human_doubtful": human["evidencia_dudosa"],
                "model_present": model_present,
                "model_severity": model["severidad"],
                "presence_agreement": human_present == model_present,
                "severity_agreement": severity_agreement,
                "severity_difference": severity_difference,
                "human_notes": review["notes"],
            })
        presence_total = sum(presence.values())
        conclusive_total = sum(conclusive.values())
        categories[category] = {
            "human_doubtful": doubtful,
            "presence": {
                **presence,
                "agreement_rate": _ratio(
                    presence["true_positive"] + presence["true_negative"],
                    presence_total,
                ),
                "precision": _ratio(
                    presence["true_positive"],
                    presence["true_positive"] + presence["false_positive"],
                ),
                "recall": _ratio(
                    presence["true_positive"],
                    presence["true_positive"] + presence["false_negative"],
                ),
            },
            "conclusive_presence": {
                **conclusive,
                "agreement_rate": _ratio(
                    conclusive["true_positive"] + conclusive["true_negative"],
                    conclusive_total,
                ),
                "count": conclusive_total,
            },
            "severity": {
                "comparable_positive": severity_comparable,
                "exact_agreement": severity_exact,
                "exact_agreement_rate": _ratio(severity_exact, severity_comparable),
                "model_overestimated": overestimated,
                "model_underestimated": underestimated,
            },
        }

    report = {
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": run_id,
        "label_run_id": label_run_id,
        "sources": {
            "selection_sha256": _sha256(selection_path),
            "results_sha256": _sha256(results_path),
            "blind_reviews_sha256": _sha256(reviews_path),
        },
        "coverage": {
            "selected": len(selection),
            "model_results": len(results),
            "human_reviews": len(reviews),
            "comparable": len(expected_ids),
        },
        "categories": categories,
        "priority": {
            "disagreement_photo_ids": sorted(disagreement_ids),
            "disagreement_photos": len(disagreement_ids),
            "false_severe_conclusive": false_severe,
        },
    }
    json_path = label_dir / "review_comparison.json"
    csv_path = label_dir / "review_comparison.csv"
    markdown_path = label_dir / "review_comparison.md"
    _write_json(json_path, report)
    _write_csv(csv_path, detail_rows)
    markdown_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    return markdown_path
