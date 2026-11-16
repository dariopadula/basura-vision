# Inventario propuesto para el primer commit

La auditoría `git ls-files --others --exclude-standard` del 2026-11-16 produjo estos archivos versionables:

```text
.env.example
.gitignore
README.md
docs/deployment.md
docs/first_commit.md
docs/reference.md
manifests/reference_30.jsonl
pyproject.toml
scripts/compare_reference.py
scripts/import_reference_assets.py
src/basura_vision/__init__.py
src/basura_vision/annotator.py
src/basura_vision/cli.py
src/basura_vision/clustering.py
src/basura_vision/config.py
src/basura_vision/detection.py
src/basura_vision/manifests.py
src/basura_vision/storage.py
src/basura_vision/training.py
src/basura_vision/viewer.py
tests/test_config_storage_annotations.py
web/annotator.html
web/viewer.html
```

Git ignora `.env`, `data/photos/`, `data/annotations/`, `models/`, `runs/`, entornos, cachés, credenciales y formatos habituales de pesos. El repositorio fue inicializado en esta carpeta; no hay commit ni remoto configurado.

