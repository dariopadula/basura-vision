# Cloudera, S3, Git y Bedrock

## S3 en Cloudera

La implementación usa `cml.data_v1.get_connection(BV_S3_CONNECTION).get_base_connection()`, según los ejemplos recibidos. Configurar una conexión CML con permiso sobre el bucket y luego definir `APP_ENV=cloudera`, `BV_STORAGE=s3`, `BV_S3_CONNECTION`, `BV_S3_BUCKET`, `BV_S3_PREFIX`, `BV_MODEL_OBJECT` y `BV_TEMP_ROOT`. Las credenciales permanecen en la conexión de CML y no se guardan en `.env`.

Las rutas del manifiesto son keys relativas a `BV_S3_PREFIX`. El detector materializa el peso en `BV_TEMP_ROOT`; los originales se leen objeto a objeto. `upload-run` guarda manifiesto, copia del modelo, metadatos, detecciones, recortes, embeddings y asignaciones bajo una key de corrida común. `download-run` restaura esos artefactos para abrirlos en el visor sin ejecutar YOLO.

Cuando una corrida conserva variantes diagnósticas, revisar primero el inventario final sin abrir S3:

```bash
python3 -m basura_vision.cli upload-run \
  --run-id reference_3000_gpu \
  --destination runs/reference_3000_gpu \
  --canonical \
  --dry-run
```

`--canonical` selecciona los ocho archivos finales del pipeline, el modelo declarado por `run.json` y solo los recortes referenciados por las detecciones. Después de comprobar el conteo, repetir sin `--dry-run` para publicar exactamente esa selección. La opción no elimina ni modifica los diagnósticos locales.

Detección escribe cada resultado inmediatamente y actualiza `run.json` cada 25 fotos. Repetir el comando con el mismo manifiesto, modelo, dispositivo y `run-id` retoma el prefijo completo; combinaciones incompatibles se rechazan. Un objeto S3 ausente o una imagen ilegible se registra como `error_entrada` y el lote continúa, permitiendo revisar o recuperar esos originales por separado.

La validación amplia en CML usó Python 3.10.20, torch/torchvision 2.14.1/0.29.1, scikit-learn 1.7.2 y ultralytics 8.4.208. La rueda `torch-2.14.1+cu130` reconoció una Tesla T4 de 15 GiB. YOLO reprodujo sin diferencias las 30 referencias y luego generó 3.000 registros: 2.323 agrupables, 619 sin residuos, 57 sin contenedor y un `error_entrada` por un objeto S3 ausente. ResNet18 produjo 2.380 embeddings; CPU y GPU dieron ARI 1,0 entre sí. GPU tardó 55 segundos y 4 vCPU, 2 minutos 8 segundos. PCA-30/K-means-15 terminó en unos cuatro segundos sobre CPU.

La corrida histórica calculaba ResNet18 directamente sobre el recorte en memoria. Leer el JPEG del visor reducía el coseno promedio a 0,99847 en la muestra; recortar nuevamente el original mediante `crop_box` produjo diferencia máxima 0,0 en los 22 embeddings comparables. Por eso `embed` requiere acceso al almacenamiento de originales. La corrida final de 2.999 originales obtuvo ARI 0,7293, NMI 0,7991 y 86,31 % de coincidencia óptima frente al agrupamiento histórico. Replicar el código y dependencias históricos quitando únicamente la foto ausente produjo ARI 0,7462 y 86,40 %, evidencia de que esa ausencia explica la mayor parte del cambio global de K-means.

Para lotes grandes puede iniciarse una sesión con GPU. `BV_DEVICE=auto` selecciona `cuda:0` cuando la compilación instalada de PyTorch reconoce CUDA y mantiene CPU en Windows o en sesiones CML sin GPU. Detección guarda `requested_device` y `device` en `run.json`; embeddings hace lo mismo en `embeddings.json`. Una GPU visible en `nvidia-smi` no basta si PyTorch lleva el sufijo `+cpu`: debe instalarse la rueda CUDA compatible antes de ejecutar el lote.

Para abrir las interfaces en una sesión consultar [visores en Cloudera](visores_cloudera.md). El visor se desplegó y abrió correctamente como Application con la corrida restaurada de 30. La corrida amplia debe guardarse en S3 y abrirse todavía en el visor.

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

`embed` vuelve a leer los originales mediante el almacenamiento configurado y los recorta con las coordenadas guardadas por detección. Esto evita usar los JPEG con pérdida destinados al visor y reproduce los embeddings históricos calculados directamente sobre píxeles originales. Una vez generado `embeddings.npy`, `cluster` trabaja únicamente con archivos locales y puede repetirse sin conexión S3. Los comandos que leen o escriben objetos —detección, embeddings, descarga, subida, visor, anotador y preparación del dataset— inicializan el almacenamiento configurado.

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

La salida incluye el JSON validado, ID y ruta de foto, fuente `original`, modelo, región, versión y SHA-256 del prompt, uso, latencia y motivo de finalización. El prompt `bedrock_labels_v1` conserva exactamente las seis categorías facilitadas. El lector acepta JSON puro o un objeto rodeado por texto/bloque Markdown, manteniendo el texto original para auditoría. Doce pruebas pasan.

Validación real realizada en CML el 2026-11-17 sobre `180072_3848356_6204427`: Nova Lite devolvió y el sistema guardó las seis categorías; consumió 2.754 tokens de entrada y 175 de salida, informó 1.433 ms y `end_turn`. La revisión visual considera plausibles basura fuera, residuos esparcidos y poda; `voluminoso=false` también es razonable. Las severidades `severo` para contenedor desbordado y roto no están claramente sustentadas por la imagen y requieren calibración. `bedrock-one` reemplaza el archivo del mismo ID; el lote separado conserva su selección y reanuda sin repetir resultados completos.

La prueba de lote quedó implementada, pero todavía no ejecutada contra Bedrock real:

```bash
python3 -m basura_vision.cli bedrock-batch \
  --run-id reference_30_smoke \
  --label-run-id nova_lite__bedrock_labels_v1__reference30 \
  --checkpoint-every 5
```

La corrida usa las 30 fotos originales y una selección congelada. Cada resultado se agrega localmente a `results.jsonl`; una repetición omite IDs completados. Cada cinco intentos actualiza `run.json` y `cluster_summary.json` y sube el directorio a `runs/reference_30_smoke/bedrock/<label-run-id>` dentro del prefijo S3 configurado. Los errores se registran aparte y se reintentan al reanudar. El modelo o hash de prompt no pueden cambiar dentro del mismo `label-run-id`. Trece pruebas pasan, incluida una interrupción simulada mediante una segunda ejecución que no repite llamadas completadas.

