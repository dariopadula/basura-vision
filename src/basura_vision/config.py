from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALID_ENVS = {"local", "cloudera"}


class ConfigError(ValueError):
    pass


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"{path}:{number}: se esperaba NOMBRE=valor")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _integer(values: dict[str, str], key: str, default: int | str) -> int:
    try:
        value = int(values.get(key, str(default)))
    except ValueError as exc:
        raise ConfigError(f"{key} debe ser un entero") from exc
    if not 1 <= value <= 65535:
        raise ConfigError(f"{key} debe estar entre 1 y 65535")
    return value


@dataclass(frozen=True)
class Settings:
    app_env: str
    storage: str
    data_root: Path
    model_path: Path
    model_object: str | None
    runs_root: Path
    annotations_path: Path
    temp_root: Path
    host: str
    viewer_port: int
    annotator_port: int
    viewer_run_id: str
    viewer_run_source: str
    bedrock_model_id: str
    bedrock_region: str | None
    device: str
    s3_connection: str | None = None
    s3_bucket: str | None = None
    s3_prefix: str = ""


def load_settings(env_file: Path | None = None) -> Settings:
    values = _read_env(env_file or ROOT / ".env")
    values.update({key: value for key, value in os.environ.items() if key.startswith("BV_") or key == "APP_ENV"})
    app_env = values.get("APP_ENV", "local").lower()
    if app_env not in VALID_ENVS:
        raise ConfigError(f"APP_ENV inválido: {app_env!r}; usar local o cloudera")
    storage = values.get("BV_STORAGE", "s3" if app_env == "cloudera" else "local").lower()
    if storage not in {"local", "s3"}:
        raise ConfigError("BV_STORAGE debe ser local o s3")
    required = []
    if storage == "s3":
        required = [key for key in ("BV_S3_CONNECTION", "BV_S3_BUCKET") if not values.get(key)]
    if required:
        raise ConfigError("Faltan variables requeridas: " + ", ".join(required))
    default_temp = os.environ.get("TMP", os.environ.get("TEMP", "/tmp"))
    shared_port = values.get("BV_PORT")
    device = values.get("BV_DEVICE", "auto").strip().lower()
    if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", device):
        raise ConfigError("BV_DEVICE debe ser auto, cpu, cuda o cuda:N")
    return Settings(
        app_env=app_env,
        storage=storage,
        data_root=_path(values.get("BV_DATA_ROOT", "data/photos")),
        model_path=_path(values.get("BV_MODEL_PATH", "models/active/best.pt")),
        model_object=values.get("BV_MODEL_OBJECT") or None,
        runs_root=_path(values.get("BV_RUNS_ROOT", "runs")),
        annotations_path=_path(values.get("BV_ANNOTATIONS_PATH", "data/annotations/annotations.json")),
        temp_root=_path(values.get("BV_TEMP_ROOT", str(Path(default_temp) / "basura_vision"))),
        host=values.get("BV_HOST", "0.0.0.0" if app_env == "cloudera" else "127.0.0.1"),
        viewer_port=_integer(values, "BV_VIEWER_PORT", shared_port or 8765),
        annotator_port=_integer(values, "BV_ANNOTATOR_PORT", shared_port or 8768),
        viewer_run_id=values.get("BV_RUN_ID", "reference_30_application"),
        viewer_run_source=values.get("BV_RUN_SOURCE", "runs/reference_30_cloudera"),
        bedrock_model_id=values.get("BV_BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0"),
        bedrock_region=values.get("BV_BEDROCK_REGION") or None,
        device=device,
        s3_connection=values.get("BV_S3_CONNECTION"),
        s3_bucket=values.get("BV_S3_BUCKET"),
        s3_prefix=values.get("BV_S3_PREFIX", "").strip("/"),
    )

