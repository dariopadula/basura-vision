from __future__ import annotations

import json
from pathlib import Path

from .storage import safe_key

REQUIRED = {"id", "path"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    ids = set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: JSON inválido") from exc
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"{path}:{number}: faltan {sorted(missing)}")
        row["id"] = str(row["id"]).strip()
        if not row["id"] or row["id"] in ids:
            raise ValueError(f"{path}:{number}: ID vacío o duplicado: {row['id']!r}")
        row["path"] = safe_key(str(row["path"]))
        ids.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError(f"Manifiesto vacío: {path}")
    return rows


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)

