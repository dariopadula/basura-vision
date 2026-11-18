"""Construye el manifiesto autónomo de 3.000 desde la corrida experimental.

El proyecto experimental se indica explícitamente y solo se usa para generar
el archivo versionado. La operación normal de Basura Visión no depende de él.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_COUNTS = {
    **{
        str(number): count
        for number, count in enumerate(
            [183, 151, 225, 162, 71, 243, 202, 160, 112, 125, 199, 181, 164, 87, 59],
            start=1,
        )
    },
    "sin_contenedor": 57,
    "sin_residuos": 619,
}


def optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_root",
        type=Path,
        help="raíz explícita del proyecto experimental",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="destino; por defecto manifests/reference_3000.jsonl",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source_dir = (
        args.source_root.resolve()
        / "resultados/clusters/yolo_ratio_3000_k15"
    )
    csv_path = source_dir / "clusters.csv"
    crops_path = source_dir / "recortes_yolo.jsonl"
    output = args.output or project_root / "manifests/reference_3000.jsonl"

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    crop_rows = [
        json.loads(line)
        for line in crops_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if len(csv_rows) != 3000 or len(crop_rows) != 3000:
        raise ValueError(
            f"Se esperaban 3.000 filas; CSV={len(csv_rows)}, recortes={len(crop_rows)}"
        )

    crops_by_id = {str(row["id_foto"]): row for row in crop_rows}
    if len(crops_by_id) != 3000:
        raise ValueError("Hay IDs duplicados en recortes_yolo.jsonl")

    result = []
    ids: set[str] = set()
    paths: set[str] = set()
    counts: dict[str, int] = {}
    for source in csv_rows:
        photo_id = str(source["id_foto"]).strip()
        path = source["imagen_manifest"].strip().replace("\\", "/")
        status = source["cluster"].strip()
        crop = crops_by_id.get(photo_id)
        if not photo_id or photo_id in ids:
            raise ValueError(f"ID vacío o duplicado: {photo_id!r}")
        if not path or path in paths:
            raise ValueError(f"Ruta vacía o duplicada: {path!r}")
        if crop is None:
            raise ValueError(f"Falta el recorte de referencia para {photo_id}")
        if crop["original"].replace("\\", "/") != path or str(crop["grupo"]) != status:
            raise ValueError(f"CSV y recortes no coinciden para {photo_id}")

        ratio = optional_float(source["ratio"])
        reference_cluster = int(status) if status.isdigit() else None
        rectangle = crop.get("rectangulo")
        if status == "sin_residuos" and (ratio is not None or rectangle is not None):
            raise ValueError(f"Referencia sin residuos inconsistente: {photo_id}")
        if reference_cluster is not None and (ratio is None or rectangle is None):
            raise ValueError(f"Referencia agrupable incompleta: {photo_id}")

        ids.add(photo_id)
        paths.add(path)
        counts[status] = counts.get(status, 0) + 1
        result.append(
            {
                "id": photo_id,
                "path": path,
                "event_id": str(source["evento_id"]).strip(),
                "type": int(source["tipo"]),
                "source_split": source["split_origen"].strip(),
                "reference_status": status,
                "reference_cluster": reference_cluster,
                "reference_ratio": ratio,
                "reference_crop_box": rectangle,
                "reference_pc1": float(source["pc1"]),
                "reference_pc2": float(source["pc2"]),
            }
        )

    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Distribución inesperada: {counts}")
    if any(row["type"] != 1 for row in result):
        raise ValueError("El manifiesto contiene fotos que no son tipo 1")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in result:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)
    print(f"Escritas {len(result)} referencias en {output}")
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
