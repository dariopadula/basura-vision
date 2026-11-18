# Referencia congelada

## Detector activo

- Archivo experimental: `yolo_identificacion/boxes_multiples_embeddings/entrenamientos/multiples_cajas/weights/best.pt`.
- SHA-256: `86259dfefdf1a416488776b46937713f71f66dc153e6ab4ad57b5b58df013a3b`.
- Clases: `0 = contenedor`, `1 = residuos`.
- Umbrales: predicción inicial 0,05; residuos 0,10; contenedor 0,25; IoU 0,70; máximo 300 detecciones; CPU.
- Entrenamiento de origen: YOLO11n, 50 épocas, tamaño 640, semilla 42. La validación experimental informó AP50 0,881 para contenedor y 0,531 para residuos.

## Agrupación de 3.000

ResNet18 con pesos predeterminados, `fc=Identity`, transformaciones oficiales, normalización L2, PCA-30 y K-means de 15 grupos (`n_init=30`, semilla 42). La variable de ratio es `log1p(ratio)`, estandarizada y ponderada por `0,5 * sqrt(suma(varianza PCA))`. La corrida de referencia contiene 2.324 agrupables, 619 sin residuos y 57 sin contenedor.

`manifests/reference_3000.jsonl` congela las 3.000 filas de la corrida experimental. Cada fila conserva ID, ruta relativa, evento, partición de origen, estado/grupo, ratio, rectángulo de recorte y las dos coordenadas PCA históricas. Se generó de forma verificable desde `clusters.csv` y `recortes_yolo.jsonl` mediante `scripts/build_reference_3000.py`; el proyecto autónomo no necesita esos archivos experimentales para operar. Su SHA-256 es `ff5fc09924af8de8006d655524a38ae2736c552a618f7ace21f19390b857f86f`.

`manifests/reference_30.jsonl` congela 30 IDs de esa corrida: uno por cada grupo, ocho sin residuos y siete sin contenedor. Los grupos son siempre los del ajuste sobre 3.000, no un ajuste sobre 30. Cada fila conserva estado/grupo, ratio y rectángulo de recorte disponibles en la referencia. Las cajas y confianzas no se guardaron en la corrida histórica y se obtendrán al verificar el pipeline migrado.

## Dependencias observadas

La corrida experimental usó Python 3.11.9 y, en el entorno observado el 2026-11-16, torch 2.14.1, torchvision 0.29.1, numpy 2.4.6, Pillow 13.0.0, scikit-learn 1.9.1 y ultralytics 8.4.208. El proyecto declara rangos compatibles en `pyproject.toml`; cada corrida registra el hash del modelo y sus parámetros.

