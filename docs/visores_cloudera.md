# Visores en Cloudera AI Workbench

Guía corta para abrir las dos interfaces de `basura-vision` dentro de una sesión activa de Cloudera. Los procesos duran mientras la sesión permanezca activa. Para acceso permanente todavía falta validar una Cloudera Application.

## Antes de iniciar

El proyecto debe estar instalado y `.env` debe contener `APP_ENV=cloudera`, la conexión S3, bucket, prefijo y `BV_MODEL_OBJECT`. Nunca guardar credenciales en `.env`; CML las obtiene de la conexión configurada.

Comprobar los puertos asignados:

```bash
echo "CDSW_APP_PORT=$CDSW_APP_PORT"
echo "CDSW_READONLY_PORT=$CDSW_READONLY_PORT"
```

El visor de resultados usa `CDSW_READONLY_PORT`. El anotador usa `CDSW_APP_PORT` porque guarda cambios. Cloudera recomienda enlazar ambos servicios a `127.0.0.1`.

## Visor de resultados

El visor abre una corrida local ya existente. Si la corrida está solo en S3, restaurarla sin ejecutar YOLO:

```bash
python3 -m basura_vision.cli download-run \
  --run-id reference_30_restored \
  --source runs/reference_30_cloudera
```

Iniciar en segundo plano:

```bash
nohup env \
  BV_HOST=127.0.0.1 \
  BV_VIEWER_PORT="$CDSW_READONLY_PORT" \
  python3 -m basura_vision.cli viewer \
    --run-id reference_30_restored \
    --no-browser \
  > /tmp/basura_vision_viewer.log 2>&1 &

echo $! > /tmp/basura_vision_viewer.pid
sleep 3
```

Comprobar que responde:

```bash
curl -s -o /dev/null \
  -w "Respuesta local: HTTP %{http_code}\n" \
  "http://127.0.0.1:${CDSW_READONLY_PORT}/"
```

Debe devolver `HTTP 200`. Abrirlo desde la opción **Web UI / Read Only** de la sesión. Si esa opción no aparece, construir la URL dentro de Cloudera y abrirla en otra pestaña, sin compartirla:

```bash
printf 'https://read-only-%s.%s\n' \
  "$CDSW_ENGINE_ID" \
  "$CDSW_DOMAIN"
```

Las marcas del visor viven en el navegador. Exportar la selección JSONL antes de recargar o cerrar la página.

Detener el visor:

```bash
kill "$(cat /tmp/basura_vision_viewer.pid)"
```

## Anotador de cajas

El anotador lee originales desde S3 y guarda las anotaciones en la ruta local configurada por `BV_ANNOTATIONS_PATH`. No las sube automáticamente a S3.

Iniciar en segundo plano:

```bash
nohup env \
  BV_HOST=127.0.0.1 \
  BV_ANNOTATOR_PORT="$CDSW_APP_PORT" \
  python3 -m basura_vision.cli annotator \
    --manifest manifests/reference_30.jsonl \
    --no-browser \
  > /tmp/basura_vision_annotator.log 2>&1 &

echo $! > /tmp/basura_vision_annotator.pid
sleep 3
```

Comprobar que responde:

```bash
curl -s -o /dev/null \
  -w "Respuesta local: HTTP %{http_code}\n" \
  "http://127.0.0.1:${CDSW_APP_PORT}/"
```

Abrirlo desde la opción **Web UI / Application** de la sesión. Si hace falta construir la URL:

```bash
printf 'https://%s.%s\n' \
  "$CDSW_ENGINE_ID" \
  "$CDSW_DOMAIN"
```

Detener el anotador:

```bash
kill "$(cat /tmp/basura_vision_annotator.pid)"
```

## Diagnóstico rápido

Un número como `[1] 645` al iniciar con `&` es el PID, no un error HTTP. Las líneas `GET / ... 200` son accesos normales de Cloudera.

Si la interfaz no abre, comprobar primero el proceso y el log correspondiente:

```bash
ps -p "$(cat /tmp/basura_vision_viewer.pid)" -o pid,stat,cmd
sed -n '1,120p' /tmp/basura_vision_viewer.log

ps -p "$(cat /tmp/basura_vision_annotator.pid)" -o pid,stat,cmd
sed -n '1,120p' /tmp/basura_vision_annotator.log
```

No iniciar dos visores en el mismo puerto. Si el log informa `Address already in use`, detener primero el proceso anterior. La sesión y su Web UI deben permanecer activas.

