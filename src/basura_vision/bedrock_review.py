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

from .bedrock import CATEGORIES, SEVERITIES, _label_run_dir
from .config import ROOT, Settings
from .manifests import read_jsonl
from .storage import Storage, join_key


def load_blind_reviews(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("El archivo de revisión debe ser un objeto JSON")
    return value


def validate_blind_review(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Revisión inválida")
    categories = value.get("categories")
    if not isinstance(categories, dict) or set(categories) != set(CATEGORIES):
        raise ValueError("La revisión debe incluir exactamente las seis categorías")
    normalized = {}
    for category in CATEGORIES:
        item = categories[category]
        expected = {"presente", "severidad", "evidencia_dudosa"}
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError(f"Estructura inválida para {category}")
        present = item["presente"]
        severity = item["severidad"]
        doubtful = item["evidencia_dudosa"]
        if not isinstance(present, bool) or not isinstance(doubtful, bool):
            raise ValueError(f"Presencia y duda deben ser booleanas para {category}")
        if present and severity not in SEVERITIES:
            raise ValueError(f"Severidad inválida para {category}")
        if not present and severity is not None:
            raise ValueError(f"La severidad debe ser null si {category} está ausente")
        normalized[category] = {
            "presente": present,
            "severidad": severity,
            "evidencia_dudosa": doubtful,
        }
    notes = value.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 2000:
        raise ValueError("Las notas deben ser texto de hasta 2.000 caracteres")
    return {"categories": normalized, "notes": notes.strip()}


def save_blind_reviews(path: Path, value: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def persist_blind_reviews(
    path: Path,
    value: dict[str, dict],
    storage: Storage | None = None,
    upload_destination: str | None = None,
) -> None:
    save_blind_reviews(path, value)
    if upload_destination:
        if storage is None:
            raise ValueError("Se requiere almacenamiento para respaldar la revisión")
        storage.upload_file(
            path,
            join_key(upload_destination, "blind_reviews.json"),
        )


def serve_blind_review(
    settings: Settings,
    storage: Storage,
    run_id: str,
    label_run_id: str,
    open_browser: bool = True,
    upload_destination: str | None = None,
) -> None:
    _, label_dir = _label_run_dir(settings, run_id, label_run_id)
    selection_path = label_dir / "selection.jsonl"
    if not selection_path.is_file():
        raise ValueError(f"La corrida de etiquetas no está preparada: {label_run_id}")
    photos = read_jsonl(selection_path)
    public_photos = [
        {
            key: photo.get(key)
            for key in ("id", "path", "event_id", "cluster", "detection_status", "selection_role")
        }
        for photo in photos
    ]
    by_id = {photo["id"]: photo for photo in photos}
    reviews_path = label_dir / "blind_reviews.json"
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def send_content(self, content: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path == "/":
                    return self.send_content(
                        (ROOT / "web/bedrock_review.html").read_bytes(),
                        "text/html; charset=utf-8",
                    )
                if request.path == "/api/start":
                    body = {
                        "photos": public_photos,
                        "reviews": load_blind_reviews(reviews_path),
                        "categories": list(CATEGORIES),
                        "token": token,
                    }
                    return self.send_content(
                        json.dumps(body, ensure_ascii=False).encode(),
                        "application/json",
                    )
                if request.path == "/photo":
                    photo_id = parse_qs(request.query)["id"][0]
                    with Image.open(io.BytesIO(storage.read_bytes(by_id[photo_id]["path"]))) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                    output = io.BytesIO()
                    image.save(output, "JPEG", quality=92)
                    return self.send_content(output.getvalue(), "image/jpeg")
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return self.send_error(404, str(exc))
            self.send_error(404)

        def do_POST(self):
            if self.path != "/api/save" or self.headers.get("X-Token") != token:
                return self.send_error(403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 100_000:
                    raise ValueError("Solicitud demasiado grande")
                payload = json.loads(self.rfile.read(length))
                photo = by_id[str(payload["id"])]
                review = validate_blind_review(payload)
                reviews = load_blind_reviews(reviews_path)
                reviews[photo["id"]] = {
                    "id": photo["id"],
                    "path": photo["path"],
                    "event_id": photo.get("event_id"),
                    "cluster": photo.get("cluster"),
                    "reviewed": True,
                    **review,
                }
                persist_blind_reviews(
                    reviews_path,
                    reviews,
                    storage,
                    upload_destination,
                )
                return self.send_content(b'{"ok":true}', "application/json")
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return self.send_error(400, str(exc))

    server = ThreadingHTTPServer((settings.host, settings.viewer_port), Handler)
    url = f"http://127.0.0.1:{settings.viewer_port}"
    print(
        f"Revisión ciega de {label_run_id}: {url} (Ctrl+C para cerrar)",
        flush=True,
    )
    if upload_destination:
        print(f"Respaldo de revisiones: {upload_destination}", flush=True)
    if open_browser and settings.app_env == "local":
        webbrowser.open(url)
    server.serve_forever()
