from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Protocol

from .config import Settings


def safe_key(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not value or ".." in path.parts:
        raise ValueError(f"Ruta relativa inválida: {value!r}")
    return path.as_posix()


def join_key(*parts: str) -> str:
    return "/".join(safe_key(part).strip("/") for part in parts if part and part.strip("/"))


class Storage(Protocol):
    def read_bytes(self, key: str) -> bytes: ...
    def write_bytes(self, key: str, data: bytes) -> str: ...
    def materialize(self, key: str, destination: Path) -> Path: ...
    def upload_file(self, source: Path, key: str) -> str: ...
    def list(self, prefix: str = "") -> list[str]: ...


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def path(self, key: str) -> Path:
        result = (self.root / safe_key(key)).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError("Ruta fuera del almacenamiento local")
        return result

    def read_bytes(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> str:
        destination = self.path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, destination)
        return key

    def materialize(self, key: str, destination: Path) -> Path:
        source = self.path(key)
        if source == destination.resolve():
            return source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination

    def upload_file(self, source: Path, key: str) -> str:
        return self.write_bytes(key, source.read_bytes())

    def list(self, prefix: str = "") -> list[str]:
        base = self.path(prefix) if prefix else self.root
        if not base.exists():
            return []
        return [path.relative_to(self.root).as_posix() for path in base.rglob("*") if path.is_file()]


class S3Storage:
    def __init__(self, client, bucket: str, prefix: str = ""):
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return join_key(self.prefix, key)

    def read_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()

    def write_bytes(self, key: str, data: bytes) -> str:
        absolute = self._key(key)
        self.client.put_object(Bucket=self.bucket, Key=absolute, Body=data)
        return absolute

    def materialize(self, key: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._key(key), str(destination))
        return destination

    def upload_file(self, source: Path, key: str) -> str:
        absolute = self._key(key)
        self.client.upload_file(str(source), self.bucket, absolute)
        return absolute

    def list(self, prefix: str = "") -> list[str]:
        absolute = self._key(prefix) if prefix else self.prefix
        keys: list[str] = []
        token = None
        while True:
            args = {"Bucket": self.bucket, "Prefix": absolute}
            if token:
                args["ContinuationToken"] = token
            response = self.client.list_objects_v2(**args)
            base = self.prefix + "/" if self.prefix else ""
            keys.extend(item["Key"][len(base):] if item["Key"].startswith(base) else item["Key"] for item in response.get("Contents", []))
            if not response.get("IsTruncated"):
                return keys
            token = response["NextContinuationToken"]


def cml_s3_client(connection_name: str):
    try:
        import cml.data_v1 as cmldata
    except ImportError as exc:
        raise RuntimeError("cml.data_v1 no está instalado; ejecutar dentro de CML") from exc
    return cmldata.get_connection(connection_name).get_base_connection()


def photo_storage(settings: Settings, client=None) -> Storage:
    if settings.storage == "local":
        return LocalStorage(settings.data_root)
    client = client or cml_s3_client(settings.s3_connection or "")
    return S3Storage(client, settings.s3_bucket or "", settings.s3_prefix)


CANONICAL_RUN_FILES = {
    "run.json",
    "manifest.jsonl",
    "detections.jsonl",
    "embeddings.npy",
    "embedding_index.jsonl",
    "embeddings.json",
    "assignments.jsonl",
    "clustering.json",
}


def _run_file(run_dir: Path, relative: str) -> Path:
    relative_path = PurePosixPath(safe_key(relative))
    path = (run_dir / Path(*relative_path.parts)).resolve()
    root = run_dir.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artefacto fuera de la corrida: {relative!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Falta el artefacto canónico {relative}")
    return path


def canonical_run_files(run_dir: Path) -> list[Path]:
    """Return the final pipeline artifacts, excluding diagnostic variants."""
    run_dir = run_dir.resolve()
    run_path = _run_file(run_dir, "run.json")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("state") != "completed":
        raise ValueError("La corrida debe estar completada antes de respaldarla")

    files = {_run_file(run_dir, relative) for relative in CANONICAL_RUN_FILES}
    model_path = run.get("model", {}).get("path")
    if not model_path:
        raise ValueError("run.json no declara model.path")
    files.add(_run_file(run_dir, model_path))

    detections_path = _run_file(run_dir, "detections.jsonl")
    for number, raw in enumerate(detections_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            detection = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{detections_path}:{number}: JSON inválido") from exc
        crop_path = detection.get("crop_path")
        if crop_path:
            files.add(_run_file(run_dir, crop_path))

    return sorted(files, key=lambda path: path.relative_to(run_dir).as_posix())


def upload_run(
    storage: Storage,
    run_dir: Path,
    run_key: str,
    sources: Iterable[Path] | None = None,
) -> list[str]:
    uploaded = []
    selected = sources if sources is not None else (
        path for path in run_dir.rglob("*") if path.is_file()
    )
    for source in sorted(selected, key=lambda path: path.relative_to(run_dir).as_posix()):
        uploaded.append(storage.upload_file(source, join_key(run_key, source.relative_to(run_dir).as_posix())))
    return uploaded


def download_run(storage: Storage, run_key: str, run_dir: Path) -> list[Path]:
    prefix = safe_key(run_key).rstrip("/")
    keys = storage.list(prefix)
    if not keys:
        raise FileNotFoundError(f"No hay objetos bajo {prefix}")
    downloaded = []
    for key in keys:
        marker = prefix + "/"
        if not key.startswith(marker):
            continue
        relative = PurePosixPath(key[len(marker):])
        destination = run_dir / Path(*relative.parts)
        downloaded.append(storage.materialize(key, destination))
    return downloaded

