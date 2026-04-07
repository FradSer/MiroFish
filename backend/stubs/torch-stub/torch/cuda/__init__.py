"""Stub for torch.cuda -- always reports no CUDA available."""


def is_available() -> bool:
    return False


def device_count() -> int:
    return 0
