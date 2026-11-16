from __future__ import annotations

import io
import json
import os
import secrets
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

from .config import ROOT, Settings
from .manifests import read_jsonl
from .storage import Storage


def load_annotations(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def validate_boxes(boxes) -> list[dict]:
    if not isinstance(boxes, list) or len(boxes) > 500: raise ValueError("Lista de cajas inválida")
    for entry in boxes:
        box = entry.get("box")
        if entry.get("class") not in {"contenedor", "residuos"} or not isinstance(box, list) or len(box) != 4:
            raise ValueError("Caja inválida")
        if not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in box): raise ValueError("Coordenadas fuera de [0,1]")
        x1, y1, x2, y2 = box
        if x2 - x1 < 0.005 or y2 - y1 < 0.005: raise ValueError("Caja demasiado pequeña")
    return boxes


def save_annotations(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, indent=2); output.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def serve_annotator(settings: Settings, storage: Storage, manifest_path: Path, open_browser: bool = True) -> None:
    photos = read_jsonl(manifest_path); by_id = {item["id"]: item for item in photos}; token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def send_content(self, content: bytes, content_type: str):
            self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content))); self.end_headers(); self.wfile.write(content)

        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path == "/": return self.send_content((ROOT / "web/annotator.html").read_bytes(), "text/html; charset=utf-8")
                if request.path == "/api/start":
                    body = {"photos": photos, "annotations": load_annotations(settings.annotations_path), "token": token}
                    return self.send_content(json.dumps(body, ensure_ascii=False).encode(), "application/json")
                if request.path == "/photo":
                    photo_id = parse_qs(request.query)["id"][0]
                    with Image.open(io.BytesIO(storage.read_bytes(by_id[photo_id]["path"]))) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                    output = io.BytesIO(); image.save(output, "JPEG", quality=92)
                    return self.send_content(output.getvalue(), "image/jpeg")
            except (KeyError, OSError, ValueError) as exc: return self.send_error(404, str(exc))
            self.send_error(404)

        def do_POST(self):
            if self.path != "/api/save" or self.headers.get("X-Token") != token: return self.send_error(403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 2_000_000: raise ValueError("Solicitud demasiado grande")
                item = json.loads(self.rfile.read(length)); photo = by_id[str(item["id"])]; boxes = validate_boxes(item["boxes"])
                data = load_annotations(settings.annotations_path)
                data[photo["id"]] = {"id": photo["id"], "path": photo["path"], "event_id": photo.get("event_id"), "reviewed": True, "negative_confirmed": not boxes, "boxes": boxes}
                save_annotations(settings.annotations_path, data)
                return self.send_content(b'{"ok":true}', "application/json")
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc: return self.send_error(400, str(exc))

    server = ThreadingHTTPServer((settings.host, settings.annotator_port), Handler); url = f"http://127.0.0.1:{settings.annotator_port}"
    print(f"Anotador: {url} (Ctrl+C para cerrar)", flush=True)
    if open_browser and settings.app_env == "local": webbrowser.open(url)
    server.serve_forever()

