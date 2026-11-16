"""Lanzador del visor como Cloudera Application.

Variables opcionales:
    BV_RUN_ID: nombre local de la corrida restaurada.
    BV_RUN_SOURCE: prefijo relativo de la corrida guardada en S3.
"""
from __future__ import annotations

import os
from pathlib import Path


def assigned_port(environment=os.environ) -> str:
    port = environment.get("CDSW_READONLY_PORT") or environment.get("CDSW_APP_PORT")
    if not port or not port.isdigit():
        raise RuntimeError("Cloudera no asignó CDSW_READONLY_PORT ni CDSW_APP_PORT")
    return port


def main() -> None:
    os.environ["BV_HOST"] = "127.0.0.1"
    os.environ["BV_VIEWER_PORT"] = assigned_port()

    from basura_vision.config import load_settings
    from basura_vision.storage import download_run, photo_storage
    from basura_vision.viewer import serve_viewer

    settings = load_settings()
    storage = photo_storage(settings)
    run_id = os.environ.get("BV_RUN_ID", "reference_30_application")
    if Path(run_id).name != run_id:
        raise ValueError("BV_RUN_ID debe ser un nombre simple, sin carpetas")
    source = os.environ.get("BV_RUN_SOURCE", "runs/reference_30_cloudera")
    run_dir = settings.runs_root / run_id
    required = [run_dir / "run.json", run_dir / "manifest.jsonl", run_dir / "detections.jsonl"]
    if not all(path.is_file() for path in required):
        print(f"Restaurando {run_id} desde S3...", flush=True)
        downloaded = download_run(storage, source, run_dir)
        print(f"Artefactos restaurados: {len(downloaded)}", flush=True)
    serve_viewer(settings, storage, run_id, open_browser=False)


if __name__ == "__main__":
    main()

