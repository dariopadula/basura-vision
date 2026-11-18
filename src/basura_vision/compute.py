from __future__ import annotations


def resolve_device(requested: str, torch_module=None) -> str:
    """Resuelve auto/cpu/cuda a un dispositivo disponible y explícito."""
    if torch_module is None:
        import torch as torch_module

    value = requested.strip().lower()
    if value == "auto":
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda:"):
        if not torch_module.cuda.is_available():
            raise RuntimeError(
                f"BV_DEVICE={requested} requiere CUDA, pero PyTorch no la detecta"
            )
        try:
            index = int(value.split(":", 1)[1])
        except ValueError as exc:
            raise RuntimeError(f"Dispositivo CUDA inválido: {requested!r}") from exc
        if index < 0 or index >= torch_module.cuda.device_count():
            raise RuntimeError(
                f"BV_DEVICE={requested} no existe; GPUs disponibles: "
                f"{torch_module.cuda.device_count()}"
            )
        return f"cuda:{index}"
    raise RuntimeError(f"Dispositivo no soportado: {requested!r}")
