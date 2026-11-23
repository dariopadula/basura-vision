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

`BV_DEVICE=auto` es el valor predeterminado: YOLO y ResNet18 usan `cuda:0` cuando PyTorch detecta una GPU y CPU en caso contrario. Puede fijarse `cpu`, `cuda` o `cuda:N`; cada etapa registra el dispositivo resuelto en sus metadatos.

## Ejecutar y verificar la muestra fija

```powershell
python -m basura_vision.cli detect --manifest manifests/reference_30.jsonl --run-id reference_30
python scripts/compare_reference.py runs/reference_30
python -m basura_vision.cli viewer --run-id reference_30
python -m basura_vision.cli annotator --manifest manifests/reference_30.jsonl
```

El detector guarda incrementalmente `run.json`, el manifiesto exacto, `detections.jsonl` con cajas/confianzas/estado/motivo/ratio y los recortes JPEG del visor. Si se interrumpe, el mismo comando y `run-id` retoman el prefijo ya guardado. Un original ausente o ilegible queda como `error_entrada` y no detiene el resto del lote. Para reproducir el cálculo histórico sin pérdida JPEG, `embed` vuelve a leer cada original mediante el almacenamiento configurado y aplica el `crop_box` guardado; después `cluster` reutiliza localmente esos embeddings. El visor abre la corrida completa sin recalcular YOLO, muestra original y recorte, filtra por grupo o exclusión y exporta IDs/rutas. El anotador guarda cajas normalizadas sobre la imagen con orientación EXIF corregida, permite confirmar negativos y retoma `data/annotations/annotations.json`.

La muestra conserva los grupos calculados sobre 3.000. No se usa un ajuste hecho sobre 30 como resultado comparable. Sí puede hacerse una prueba técnica del flujo con los 22 recortes disponibles: PCA de 10 componentes y K-means de 3 grupos sobre los 15 casos agrupables.

```powershell
python -m basura_vision.cli embed --run-id reference_30_smoke
python -m basura_vision.cli cluster --run-id reference_30_smoke --clusters 3 --pca-components 10
```

Esta prueba solo confirma que embeddings, PCA, K-means y el visor funcionan de punta a punta. Los grupos definitivos se calculan sobre el lote de 3.000:

```powershell
python -m basura_vision.cli detect --manifest manifests/reference_3000.jsonl --run-id reference_3000_cloudera
python -m basura_vision.cli embed --run-id reference_3000_cloudera
python -m basura_vision.cli cluster --run-id reference_3000_cloudera --clusters 15 --pca-components 30
```

`manifests/reference_3000.jsonl` está versionado y selecciona exactamente 3.000 de los originales disponibles en el almacenamiento. Además conserva estado, grupo, ratio, rectángulo y PCA históricos para comparar la nueva ejecución.

## Prueba opcional de Bedrock con una foto

Las credenciales AWS se configuran fuera del repositorio. `BV_BEDROCK_MODEL_ID` y `BV_BEDROCK_REGION` son opcionales; el modelo predeterminado es `amazon.nova-lite-v1:0`. Para etiquetar una única foto original de una corrida:

```bash
python3 -m basura_vision.cli bedrock-one \
  --run-id reference_30_smoke \
  --photo-id 180072_3848356_6204427
```

El comando no procesa el resto del lote. Corrige la orientación, normaliza la imagen a JPEG, exige el JSON de seis categorías y guarda el resultado bajo `runs/<run-id>/bedrock/`, junto con modelo, región, versión y hash del prompt, uso y latencia informados por Bedrock. Los resultados y credenciales permanecen fuera de Git. El prompt versionado está en `prompts/bedrock_labels_v1.txt`.

Preparar una identidad de lote no invoca Bedrock ni abre S3:

```bash
python3 -m basura_vision.cli bedrock-prepare \
  --run-id reference_3000_s3_restore \
  --label-run-id nova_lite__bedrock_labels_v1__reference2999 \
  --region us-east-1
```

La preparación cruza manifiesto, detecciones y asignaciones, congela los originales accesibles en `selection.jsonl` y registra los `error_entrada` en `exclusions.jsonl`. También conserva `run.json`, `prompt.txt` y `pricing.json`. Para autorizar una ejecución se repite la preparación con el mismo ID, el tope aprobado y `--approve-execution`. El comando `bedrock-batch` exige además `--execute-approved`; sin ambos controles no llama al modelo.

Cada intento crea un marcador previo y una confirmación inmutable. Solo avanza después de verificar la confirmación en S3. `invocations.jsonl`, `results.jsonl` y `errors.jsonl` se regeneran desde esas confirmaciones; una restauración aplica las posteriores al último checkpoint. Una repetición omite resultados válidos, limita reintentos y detiene los intentos inciertos para revisión. Un bloqueo local impide dos escritores simultáneos. El modelo, región, selección, exclusiones, prompt, tarifa y tope deben coincidir con el control aprobado.

Para el piloto se congelan primero dos muestras independientes, sin abrir S3 ni
invocar Bedrock:

```bash
python3 -m basura_vision.cli bedrock-samples \
  --run-id reference_3000_s3_restore \
  --sample-plan-id bedrock_labels_v1__seed_20261119 \
  --pilot-label-run-id nova_lite__bedrock_labels_v1__pilot40 \
  --validation-label-run-id nova_lite__bedrock_labels_v1__validation34 \
  --region us-east-1 \
  --seed 20261119
```

El piloto contiene dos originales por clúster —uno aleatorio y uno de
contraste— y cinco de cada grupo especial, para un total de 40. La validación
reserva otros dos por cada uno de los 17 grupos, total 34. No se comparten IDs
ni `event_id` entre ambas muestras. El comando es idempotente: una repetición
acepta solamente los mismos archivos congelados. La autorización y el tope se
agregan después de revisar el plan y antes de llamar al modelo.

Después de ejecutar el piloto, la revisión humana se hace sin exponer las
predicciones del modelo:

```bash
python3 -m basura_vision.cli bedrock-review \
  --run-id reference_3000_s3_restore \
  --label-run-id nova_lite__bedrock_labels_v1__pilot40 \
  --destination runs/reference_3000_gpu/bedrock/nova_lite__bedrock_labels_v1__pilot40 \
  --no-browser
```

La interfaz muestra solamente el original, el grupo y el criterio de muestreo.
Permite guardar y retomar presencia, severidad, evidencia dudosa y notas para
las seis categorías. `blind_reviews.json` queda en el label run local y, cuando
se indica `--destination` con almacenamiento S3, se respalda en ese prefijo tras
cada guardado.

Cuando todas las revisiones están completas, generar la comparación sin abrir
S3 ni llamar al modelo:

```bash
python3 -m basura_vision.cli bedrock-review-report \
  --run-id reference_3000_s3_restore \
  --label-run-id nova_lite__bedrock_labels_v1__pilot40
```

El comando exige cobertura exacta y crea `review_comparison.json`,
`review_comparison.csv` y `review_comparison.md`. Reporta acuerdos de presencia,
falsos positivos y negativos, diferencias de severidad y controles específicos
para falsos `severo` de desborde y rotura. Las métricas concluyentes excluyen
las decisiones humanas marcadas como dudosas.

Para caracterizar los grupos con una muestra más amplia, congelar 1.000 fotos
aleatorias estratificadas, sin reutilizar el piloto ni la validación:

```bash
python3 -m basura_vision.cli bedrock-stratified-sample \
  --run-id reference_3000_s3_restore \
  --sample-plan-id cluster_characterization_1000_seed_20261120 \
  --label-run-id nova_lite__bedrock_labels_v1__cluster1000 \
  --exclude-label-run-id nova_lite__bedrock_labels_v1__pilot40 \
  --exclude-label-run-id nova_lite__bedrock_labels_v1__validation34 \
  --total 1000 \
  --minimum-per-group 20 \
  --maximum-per-group 100 \
  --region us-east-1 \
  --seed 20261120
```

Las cuotas mantienen la proporcionalidad mediante redistribución iterativa al
aplicar el mínimo y el máximo. La selección es aleatoria dentro de cada grupo,
usa eventos únicos y deja el label run en `prepared`, sin aprobación ni llamadas
a Bedrock.

Para congelar la caracterización después de ejecutar el lote:

```bash
python3 -m basura_vision.cli bedrock-summary \
  --run-id reference_3000_s3_restore \
  --label-run-id nova_lite__bedrock_labels_v1__cluster1000
```

El comando no abre S3 ni invoca Bedrock. Genera `summary/summary.json`,
`summary/groups.csv` y `summary/cooccurrences.csv` dentro del label run. Todos
los porcentajes de ausencia y severidad usan como denominador las respuestas
válidas del grupo; la cobertura se informa por separado.

El visor de resultados descubre los label runs disponibles y ofrece
**Resumen de grupos** dentro del mismo servidor. La vista incluye un mapa de
calor de presencia para los 15 clústeres y los grupos especiales, un gráfico
Likert divergente —ausencia a la izquierda y severidades a la derecha— y una
matriz de coocurrencias. Los gráficos se identifican explícitamente como una
caracterización exploratoria según Bedrock.

Desde cada grupo, **Ver sus fotos** abre solamente los IDs de la muestra
Bedrock correspondiente. Los grupos `sin_residuos` y `sin_contenedor` se
resuelven por estado, no por número de clúster. En el visor se puede elegir la
corrida de etiquetas y filtrar todas las fotos, la muestra seleccionada, las
respuestas válidas o las fotos sin etiqueta. Cada tarjeta conserva un indicador
compacto; al pulsarlo se abre el detalle de situaciones presentes y severidades,
con las ausentes plegadas.

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

