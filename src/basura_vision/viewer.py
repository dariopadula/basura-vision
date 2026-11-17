from __future__ import annotations

import io
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

from .config import ROOT, Settings
from .manifests import read_jsonl
from .storage import Storage


def build_items(run_dir: Path) -> list[dict]:
    manifest = {row["id"]: row for row in read_jsonl(run_dir / "manifest.jsonl")}
    detections = {row["id"]: row for row in read_jsonl(run_dir / "detections.jsonl")}
    assignment_path = run_dir / "assignments.jsonl"
    assignments = {
        row["id"]: row
        for row in read_jsonl(
            assignment_path,
            required_fields={"id", "cluster", "pc1", "pc2", "ratio"},
        )
    } if assignment_path.is_file() else {}
    items = []
    for photo_id, item in manifest.items():
        detection = detections[photo_id]
        assignment = assignments.get(photo_id, {})
        items.append({
            "id": photo_id, "path": item["path"], "event_id": item.get("event_id"),
            "status": detection["status"], "reason": detection.get("reason"), "ratio": detection.get("ratio"),
            "crop_path": detection.get("crop_path"), "crop_box": detection.get("crop_box"),
            "boxes": detection.get("boxes", []), "cluster": assignment.get("cluster", item.get("reference_cluster")),
            "reference_status": item.get("reference_status"), "reference_ratio": item.get("reference_ratio"),
        })
    return items


def serve_viewer(settings: Settings, storage: Storage, run_id: str, open_browser: bool = True) -> None:
    run_dir = (settings.runs_root / run_id).resolve()
    if not run_dir.is_dir() or not run_dir.is_relative_to(settings.runs_root.resolve()):
        raise ValueError(f"Corrida inexistente o inválida: {run_id}")
    items = build_items(run_dir)
    by_id = {item["id"]: item for item in items}

    class Handler(BaseHTTPRequestHandler):
        def send_content(self, content: bytes, content_type: str, status: int = 200):
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(content)))
            self.end_headers(); self.wfile.write(content)

        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path == "/":
                    return self.send_content((ROOT / "web/viewer.html").read_bytes(), "text/html; charset=utf-8")
                if request.path == "/api/items":
                    return self.send_content(json.dumps(items, ensure_ascii=False).encode(), "application/json")
                if request.path in {"/original", "/crop"}:
                    photo_id = parse_qs(request.query)["id"][0]; item = by_id[photo_id]
                    if request.path == "/original":
                        with Image.open(io.BytesIO(storage.read_bytes(item["path"]))) as source:
                            image = ImageOps.exif_transpose(source).convert("RGB")
                    else:
                        if not item["crop_path"]: raise FileNotFoundError("Este caso no tiene recorte")
                        with Image.open(run_dir / item["crop_path"]) as source: image = source.convert("RGB")
                    output = io.BytesIO(); image.save(output, "JPEG", quality=90)
                    return self.send_content(output.getvalue(), "image/jpeg")
            except (KeyError, OSError, ValueError) as exc:
                return self.send_error(404, str(exc))
            self.send_error(404)

    server = ThreadingHTTPServer((settings.host, settings.viewer_port), Handler)
    url = f"http://127.0.0.1:{settings.viewer_port}"
    print(f"Visor de {run_id}: {url} (Ctrl+C para cerrar)", flush=True)
    if open_browser and settings.app_env == "local": webbrowser.open(url)
    server.serve_forever()

