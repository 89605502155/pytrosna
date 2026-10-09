"""General-purpose codecs (SPEC §7), the second stage of compression."""

from __future__ import annotations

import enum
import sys
from typing import Any

import lz4.block

from .errors import CorruptedError, InvalidArgumentError, UnsupportedError

__all__ = ["Codec"]

# An LZ4 block cannot expand more than 255:1 (each extra length byte adds 255).
_LZ4_MAX_RATIO = 255

if sys.version_info >= (3, 14):
    from compression import zstd as _zstd

    def _zstd_compress(raw: bytes, level: int) -> bytes:
        return _zstd.compress(raw, level=level)

    def _zstd_decompress(stored: bytes, raw_len: int) -> bytes:
        decompressor = _zstd.ZstdDecompressor()
        try:
            out = decompressor.decompress(stored, max_length=raw_len + 1)
        except _zstd.ZstdError as e:
            msg = f"zstd: {e}"
            raise CorruptedError(msg) from None
        if len(out) == raw_len and not decompressor.eof and not decompressor.needs_input:
            # The output is complete; the frame epilogue (e.g. a checksum) is pending.
            try:
                out += decompressor.decompress(b"", max_length=1)
            except _zstd.ZstdError as e:
                msg = f"zstd: {e}"
                raise CorruptedError(msg) from None
        if not decompressor.eof:
            if len(out) > raw_len:
                return out  # reported as a size mismatch by the caller
            msg = "zstd: the frame is incomplete"
            raise CorruptedError(msg)
        if decompressor.unused_data:
            msg = "trailing bytes after zstd frame"
            raise CorruptedError(msg)
        return out

else:  # pragma: no cover - exercised on Python < 3.14 only
    import zstandard as _zstd

    def _zstd_compress(raw: bytes, level: int) -> bytes:
        return _zstd.ZstdCompressor(level=level).compress(raw)

    def _zstd_decompress(stored: bytes, raw_len: int) -> bytes:
        try:
            # The first read is bounded, so a decompression bomb cannot exhaust
            # memory; the second pass checks the end of the frame.
            reader = _zstd.ZstdDecompressor().stream_reader(stored, read_across_frames=False)
            out = reader.read(raw_len + 1)
            if len(out) != raw_len:
                return out
            obj = _zstd.ZstdDecompressor().decompressobj()
            obj.decompress(stored)
        except _zstd.ZstdError as e:
            msg = f"zstd: {e}"
            raise CorruptedError(msg) from None
        if not obj.eof:
            msg = "zstd: the frame is incomplete"
            raise CorruptedError(msg)
        if obj.unused_data:
            msg = "trailing bytes after zstd frame"
            raise CorruptedError(msg)
        return out


class Codec(enum.IntEnum):
    """General-purpose compression applied to encoded segments."""

    NONE = 0
    """No compression."""
    LZ4 = 1
    """LZ4 block compression (fast)."""
    ZSTD = 2
    """Zstandard (better ratio; the default)."""

    @property
    def label(self) -> str:
        """Human-readable name: ``"none"``, ``"lz4"`` or ``"zstd"``."""
        return self.name.lower()

    @classmethod
    def from_code(cls, code: int) -> Codec:
        try:
            return cls(code)
        except ValueError:
            msg = f"codec code {code}"
            raise UnsupportedError(msg) from None

    @classmethod
    def parse(cls, value: Any) -> Codec:
        """Accepts a :class:`Codec`, its code or a name such as ``"zstd"``."""
        if isinstance(value, Codec):
            return value
        if isinstance(value, str):
            names = {
                "none": cls.NONE,
                "uncompressed": cls.NONE,
                "lz4": cls.LZ4,
                "zstd": cls.ZSTD,
                "zstandard": cls.ZSTD,
            }
            try:
                return names[value.lower()]
            except KeyError:
                msg = f"unknown codec {value!r}"
                raise InvalidArgumentError(msg) from None
        if value is None:
            return cls.NONE
        msg = f"unknown codec {value!r}"
        raise InvalidArgumentError(msg)

    def compress(self, raw: bytes, level: int = 3) -> bytes | None:
        """Compresses ``raw``; returns ``None`` if that would not make it smaller."""
        if self is Codec.NONE:
            return None
        if self is Codec.LZ4:
            out = lz4.block.compress(raw, store_size=False) if raw else b""
        else:
            out = _zstd_compress(raw, level)
        return out if len(out) < len(raw) else None

    def decompress(self, stored: bytes, raw_len: int) -> bytes:
        """Decompresses ``stored``, which must expand to exactly ``raw_len`` bytes."""
        if self is Codec.NONE:
            out = stored
        elif self is Codec.LZ4:
            # A larger claimed size is corruption; refusing it avoids reserving that memory.
            if raw_len > len(stored) * _LZ4_MAX_RATIO + 16:
                msg = f"LZ4 block of {len(stored)} bytes cannot expand to {raw_len} bytes"
                raise CorruptedError(msg)
            if raw_len == 0 and not stored:
                out = b""
            else:
                try:
                    out = lz4.block.decompress(stored, uncompressed_size=raw_len)
                except lz4.block.LZ4BlockError as e:
                    msg = f"LZ4: {e}"
                    raise CorruptedError(msg) from None
        else:
            out = _zstd_decompress(stored, raw_len)
        if len(out) != raw_len:
            msg = f"segment decompressed to {len(out)} bytes, expected {raw_len}"
            raise CorruptedError(msg)
        return out
