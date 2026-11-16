"""Copia los 30 originales y el peso activo desde el proyecto experimental.

Solo se usa durante la migración local. El repositorio autónomo no importa código
ni resuelve rutas hacia el padre durante su operación normal.
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path, help="raíz explícita del proyecto experimental")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source_root.resolve()
    manifest = root / "manifests/reference_30.jsonl"
    data_source = source / "data/dataset-20261026T155725Z-1-001/dataset"
    for line in (value for value in manifest.read_text(encoding="utf-8").splitlines() if value.strip()):
        row = json.loads(line)
        origin = data_source / row["path"]
        destination = root / "data/photos" / row["path"]
        if not origin.is_file():
            raise FileNotFoundError(origin)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
    weight_source = source / "yolo_identificacion/boxes_multiples_embeddings/entrenamientos/multiples_cajas/weights/best.pt"
    weight_destination = root / "models/active/best.pt"
    weight_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weight_source, weight_destination)
    print("Copiados 30 originales y el peso activo.")


if __name__ == "__main__":
    main()

