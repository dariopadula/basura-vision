from __future__ import annotations

import json
from pathlib import Path

from .storage import safe_key

REQUIRED = frozenset({"id", "path"})


def read_jsonl(path: Path, required_fields=REQUIRED) -> list[dict]:
    rows = []
    ids = set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: JSON inválido") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: se esperaba un objeto JSON")
        missing = set(required_fields) - row.keys()
        if missing:
            raise ValueError(f"{path}:{number}: faltan {sorted(missing)}")
        if "id" in row:
            row["id"] = str(row["id"]).strip()
            if not row["id"] or row["id"] in ids:
                raise ValueError(f"{path}:{number}: ID vacío o duplicado: {row['id']!r}")
            ids.add(row["id"])
        if "path" in row:
            row["path"] = safe_key(str(row["path"]))
        rows.append(row)
    if not rows:
        raise ValueError(f"JSONL vacío: {path}")
    return rows


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)

