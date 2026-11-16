# Basura Visión

Proyecto autónomo para detectar contenedores y residuos, generar recortes, agrupar casos, revisar resultados, anotar cajas y preparar nuevas versiones de entrenamiento. Todo se ejecuta desde esta carpeta; no hay imports ni rutas implícitas al proyecto experimental.

## Preparar un clon limpio en Windows

Requiere Python 3.10 o 3.11 y Git. En PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Colocar el peso activo en `models/active/best.pt` y los originales bajo `data/photos/`, conservando las rutas relativas del manifiesto. Esos activos no se versionan. Durante esta migración, si se dispone del proyecto experimental, se pueden importar explícitamente:

```powershell
python scripts/import_reference_assets.py D:\Git\proyecto_basura_vision_baseline
```

En Cloudera AI Workbench no se crea otro `.venv`: el Runtime instala paquetes del proyecto bajo `/home/cdsw/.local`. La guía de despliegue registra las versiones comprobadas y la configuración S3. Sin `.env` y sin `APP_ENV`, el entorno es `local`. Rutas relativas de configuración se resuelven siempre desde esta raíz. Valores inválidos o variables requeridas ausentes producen un error directo.

## Ejecutar y verificar la muestra fija

```powershell
python -m basura_vision.cli detect --manifest manifests/reference_30.jsonl --run-id reference_30
python scripts/compare_reference.py runs/reference_30
python -m basura_vision.cli viewer --run-id reference_30
python -m basura_vision.cli annotator --manifest manifests/reference_30.jsonl
```

El detector guarda `run.json`, el manifiesto exacto, `detections.jsonl` con cajas/confianzas/estado/motivo/ratio y los recortes. El visor abre la corrida sin recalcular YOLO, muestra original y recorte, filtra por grupo o exclusión y exporta IDs/rutas. El anotador guarda cajas normalizadas sobre la imagen con orientación EXIF corregida, permite confirmar negativos y retoma `data/annotations/annotations.json`.

La muestra conserva los grupos calculados sobre 3.000. No se usa un ajuste hecho sobre 30 como resultado comparable. Sí puede hacerse una prueba técnica del flujo con los 22 recortes disponibles: PCA de 10 componentes y K-means de 3 grupos sobre los 15 casos agrupables.

```powershell
python -m basura_vision.cli embed --run-id reference_30_smoke
python -m basura_vision.cli cluster --run-id reference_30_smoke --clusters 3 --pca-components 10
```

Esta prueba solo confirma que embeddings, PCA, K-means y el visor funcionan de punta a punta. Los grupos definitivos se calculan sobre el lote de 3.000:

```powershell
python -m basura_vision.cli embed --run-id lote_3000
python -m basura_vision.cli cluster --run-id lote_3000 --clusters 15 --pca-components 30
```

## Reentrenamiento

Las fotos no revisadas no aparecen en anotaciones. Una foto guardada sin cajas lleva `reviewed=true` y `negative_confirmed=true`. Para preparar una versión con particiones por intervención e independiente de prueba:

```powershell
python -m basura_vision.cli prepare-dataset --version v001
python -m basura_vision.cli train --dataset training_data/v001 --initial-weight models/active/best.pt --output runs/training_v001 --epochs 50
```

El candidato no se activa automáticamente. Evaluarlo con el `test` reservado y cambiar `BV_MODEL_PATH` solo después de decidir su promoción.

## Pruebas y documentación

```powershell
python -m unittest discover -s tests -v
```

- [Referencia congelada](docs/reference.md)
- [Cloudera, S3, Git y Bedrock](docs/deployment.md)
- [Abrir visor y anotador en Cloudera](docs/visores_cloudera.md)
- [Inventario del primer commit](docs/first_commit.md)

