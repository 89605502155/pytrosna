"""CRC-32C (Castagnoli) checksums.

The ``crc32c`` package provides a hardware-accelerated implementation; a
table-driven pure-Python fallback keeps the library usable without it.
"""

from __future__ import annotations

from collections.abc import Callable

_POLY = 0x82F63B78


def _make_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ _POLY if crc & 1 else crc >> 1
        table.append(crc)
    return table


_TABLE = _make_table()


def crc32c_python(data: bytes | bytearray | memoryview, value: int = 0) -> int:
    """Pure-Python CRC-32C of ``data``, continuing from ``value``."""
    crc = value ^ 0xFFFFFFFF
    table = _TABLE
    for byte in bytes(data):
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _load() -> Callable[[bytes, int], int]:
    try:
        import crc32c as accelerated
    except ImportError:  # pragma: no cover - the dependency is normally installed
        return crc32c_python

    def accelerated_crc(data: bytes, value: int = 0) -> int:
        return int(accelerated.crc32c(data, value))

    return accelerated_crc


_impl = _load()


def crc32c(data: bytes | bytearray | memoryview, value: int = 0) -> int:
    """CRC-32C of ``data``; pass a previous result as ``value`` to continue it."""
    return _impl(bytes(data) if not isinstance(data, bytes) else data, value)
