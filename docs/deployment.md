# Cloudera, S3, Git y Bedrock

## S3 en Cloudera

La implementación usa `cml.data_v1.get_connection(BV_S3_CONNECTION).get_base_connection()`, según los ejemplos recibidos. Configurar una conexión CML con permiso sobre el bucket y luego definir `APP_ENV=cloudera`, `BV_STORAGE=s3`, `BV_S3_CONNECTION`, `BV_S3_BUCKET`, `BV_S3_PREFIX`, `BV_MODEL_OBJECT`, `BV_TEMP_ROOT`, `BV_HOST=0.0.0.0` y el puerto asignado en `BV_PORT`.

Las rutas del manifiesto son keys relativas a `BV_S3_PREFIX`. El detector materializa el peso en `BV_TEMP_ROOT`; los originales se leen objeto a objeto. `upload-run` guarda manifiesto, copia del modelo, metadatos, detecciones, recortes, embeddings y asignaciones bajo una key de corrida común. `download-run` restaura esos artefactos para abrirlos en el visor sin ejecutar YOLO. Está pendiente ejecutar la prueba contra el bucket real porque no hay conexión ni credenciales disponibles en este entorno.

Para una sesión CML: instalar el paquete, configurar variables, ejecutar `basura-vision detect ...` y `basura-vision viewer ...`. Como Application, usar el mismo comando `viewer`, `BV_HOST=0.0.0.0` y el puerto inyectado por CML. La prueba requerida es comparar por ID la corrida de 30 con `scripts/compare_reference.py`; después correr el manifiesto de 3.000.

## Git

Desde `basura_vision/`: `git init`, revisar `git status --short`, `git check-ignore -v` para `.env`, fotos, pesos y corridas, agregar solo fuentes/documentación/manifiestos, crear el primer commit y luego configurar el remoto. No se inicializó ni publicó un repositorio automáticamente: faltan la URL y la decisión explícita de publicación.

## Bedrock opcional

Queda desactivado. La futura integración recibirá un JSONL exportado por el visor, leerá la foto original por `Storage`, enviará la guía vigente de seis etiquetas y guardará respuesta estructurada, ID/versión del modelo y hash/versión de la guía en una corrida separada. Faltan región, modelo habilitado, permisos y guía definitiva; no se simula una validación sin esos datos.

