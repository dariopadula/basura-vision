# Cloudera, S3, Git y Bedrock

## S3 en Cloudera

La implementación usa `cml.data_v1.get_connection(BV_S3_CONNECTION).get_base_connection()`, según los ejemplos recibidos. Configurar una conexión CML con permiso sobre el bucket y luego definir `APP_ENV=cloudera`, `BV_STORAGE=s3`, `BV_S3_CONNECTION`, `BV_S3_BUCKET`, `BV_S3_PREFIX`, `BV_MODEL_OBJECT` y `BV_TEMP_ROOT`. Las credenciales permanecen en la conexión de CML y no se guardan en `.env`.

Las rutas del manifiesto son keys relativas a `BV_S3_PREFIX`. El detector materializa el peso en `BV_TEMP_ROOT`; los originales se leen objeto a objeto. `upload-run` guarda manifiesto, copia del modelo, metadatos, detecciones, recortes, embeddings y asignaciones bajo una key de corrida común. `download-run` restaura esos artefactos para abrirlos en el visor sin ejecutar YOLO.

La prueba real en CML se completó con Python 3.10.20, torch 2.14.1 CPU, torchvision 0.29.1 CPU, scikit-learn 1.7.2 y ultralytics 8.4.208. Las 30 fotos y el modelo se leyeron desde S3; los 30 estados, recortes y ratios coincidieron con Windows. Se guardaron 26 artefactos de corrida en S3, se restauraron en otro directorio y se abrió el visor restaurado sin ejecutar YOLO. La diferencia de scikit-learn respecto de Windows no intervino todavía porque esta prueba no ejecutó PCA/K-means.

Para abrir las interfaces en una sesión consultar [visores en Cloudera](visores_cloudera.md). El visor se desplegó y abrió correctamente como Application con la corrida restaurada de 30. La corrida de 3.000 sigue pendiente.

## Prueba técnica de agrupación con las 30 fotos

Antes de cargar las 3.000 se puede comprobar el encadenamiento completo sobre una copia de la corrida de referencia. Hay 22 recortes, pero solo 15 tienen contenedor y ratio; por eso esta prueba usa PCA-10 y K-means con 3 grupos. No permite evaluar ni comparar los grupos definitivos.

```bash
python3 -m basura_vision.cli download-run \
  --run-id reference_30_smoke \
  --source runs/reference_30_cloudera

python3 -m basura_vision.cli embed --run-id reference_30_smoke

python3 -m basura_vision.cli cluster \
  --run-id reference_30_smoke \
  --clusters 3 \
  --pca-components 10
```

La primera extracción puede descargar los pesos ImageNet de ResNet18. El resultado esperado es `embeddings.npy` con 22 vectores, `embedding_index.jsonl`, `assignments.jsonl` y `clustering.json`. Después puede abrirse `reference_30_smoke` en el visor. La validación comparable sigue siendo PCA-30 y K-means-15 sobre las 3.000.

## Git

El repositorio independiente fue publicado en GitHub. `.env`, fotos, anotaciones, pesos y corridas permanecen excluidos. Antes de cada publicación revisar `git status --short` y `git check-ignore -v` para los activos locales.

## Bedrock opcional

Queda desactivado. La futura integración recibirá un JSONL exportado por el visor, leerá la foto original por `Storage`, enviará la guía vigente de seis etiquetas y guardará respuesta estructurada, ID/versión del modelo y hash/versión de la guía en una corrida separada. Faltan región, modelo habilitado, permisos y guía definitiva; no se simula una validación sin esos datos.

