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

Validación realizada en CML el 2026-11-17: ResNet18 produjo una matriz 22 × 512; PCA-10/K-means-3 generó 22 asignaciones —grupos de 12, 2 y 1 casos, más siete `sin_contenedor`— y el visor mostró los grupos, originales y recortes. La distribución no se interpreta como calidad porque solo hay 15 casos agrupables.

Una vez descargada la corrida, `embed` y `cluster` trabajan únicamente con sus archivos locales y no abren la conexión S3. Esto permite continuar aunque la API de conexiones de CML esté temporalmente indisponible. Los comandos que leen o escriben objetos —detección, descarga, subida, visor, anotador y preparación del dataset— sí inicializan el almacenamiento configurado.

## Git

El repositorio independiente fue publicado en GitHub. `.env`, fotos, anotaciones, pesos y corridas permanecen excluidos. Antes de cada publicación revisar `git status --short` y `git check-ignore -v` para los activos locales.

## Bedrock opcional

No se ejecuta como parte de detección ni clustering. La conexión de Bedrock fue comprobada en CML sin exponer credenciales. El comando `bedrock-one` implementa una prueba explícita sobre una única foto original: lee el objeto mediante `Storage`, corrige su orientación, invoca `Converse` y valida las seis categorías y severidades antes de guardar el resultado en la corrida.

Configurar las credenciales mediante el mecanismo privado de CML y, opcionalmente, estas variables no sensibles:

```text
BV_BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
BV_BEDROCK_REGION=<region-habilitada>
```

Prueba preparada, todavía no ejecutada contra el modelo real:

```bash
python3 -m basura_vision.cli bedrock-one \
  --run-id reference_30_smoke \
  --photo-id 180072_3848356_6204427
```

La salida incluye el JSON validado, ID y ruta de foto, fuente `original`, modelo, región, versión y SHA-256 del prompt, uso, latencia y motivo de finalización. El prompt `bedrock_labels_v1` conserva exactamente las seis categorías facilitadas. El lector acepta JSON puro o el mismo objeto envuelto en un bloque Markdown `json`, manteniendo el texto original para auditoría. Doce pruebas pasan; una usa un cliente simulado. La primera llamada real alcanzó Nova Lite, pero su respuesta envuelta fue rechazada antes de esta corrección; falta repetirla para acreditar el ciclo completo. El procesamiento por lote y el resumen por clúster siguen pendientes.

