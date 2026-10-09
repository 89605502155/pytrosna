"""Primitive types of the format (SPEC §1): little-endian integers, LEB128
varints, strings and key-value maps, read through a bounds-checked cursor."""

from __future__ import annotations

import struct

from .errors import CorruptedError

U64_MASK = (1 << 64) - 1
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_I64 = struct.Struct("<q")


def zigzag(n: int) -> int:
    """Maps a signed 64-bit integer to an unsigned one (small magnitudes stay small)."""
    return ((n << 1) ^ (n >> 63)) & U64_MASK


def unzigzag(n: int) -> int:
    """Inverse of :func:`zigzag`."""
    return (n >> 1) ^ -(n & 1)


def to_i64(n: int) -> int:
    """Interprets the low 64 bits of ``n`` as a two's-complement signed integer."""
    n &= U64_MASK
    return n - (1 << 64) if n >> 63 else n


def ceil8(n: int) -> int:
    """``ceil(n / 8)``."""
    return (n + 7) >> 3


def bit_width(v: int) -> int:
    """Number of bits needed to represent the unsigned value ``v`` (0 for 0)."""
    return v.bit_length()


class ByteReader:
    """A cursor over bytes whose every read is bounds-checked."""

    __slots__ = ("_buf", "_pos")

    def __init__(self, buf: bytes | bytearray | memoryview) -> None:
        self._buf = bytes(buf)
        self._pos = 0

    @property
    def position(self) -> int:
        return self._pos

    @property
    def remaining(self) -> int:
        return len(self._buf) - self._pos

    def finish(self, what: str) -> None:
        """Fails unless the whole input has been consumed."""
        if self.remaining:
            msg = f"{self.remaining} unexpected trailing bytes after {what}"
            raise CorruptedError(msg)

    def take(self, n: int) -> bytes:
        if n < 0 or n > self.remaining:
            msg = f"unexpected end of data: need {n} bytes, {self.remaining} left"
            raise CorruptedError(msg)
        out = self._buf[self._pos : self._pos + n]
        self._pos += n
        return out

    def rest(self) -> bytes:
        return self.take(self.remaining)

    def u8(self) -> int:
        if self._pos >= len(self._buf):
            msg = "unexpected end of data: need 1 bytes, 0 left"
            raise CorruptedError(msg)
        b = self._buf[self._pos]
        self._pos += 1
        return b

    def u32(self) -> int:
        return int(_U32.unpack(self.take(4))[0])

    def u64(self) -> int:
        return int(_U64.unpack(self.take(8))[0])

    def i64(self) -> int:
        return int(_I64.unpack(self.take(8))[0])

    def uvarint(self) -> int:
        """Reads an unsigned LEB128 integer (at most 10 bytes, no overflow)."""
        value = 0
        for i in range(10):
            byte = self.u8()
            part = byte & 0x7F
            if i == 9 and part > 1:
                msg = "varint overflows 64 bits"
                raise CorruptedError(msg)
            value |= part << (7 * i)
            if not byte & 0x80:
                return value
        msg = "varint longer than 10 bytes"
        raise CorruptedError(msg)

    def svarint(self) -> int:
        return unzigzag(self.uvarint())

    def uvarint_u32(self) -> int:
        """Reads a varint that must fit into 32 bits."""
        v = self.uvarint()
        if v > 0xFFFF_FFFF:
            msg = f"value {v} does not fit into 32 bits"
            raise CorruptedError(msg)
        return v

    def length(self, maximum: int, what: str) -> int:
        """Reads a varint used as a length or count, rejecting values above ``maximum``."""
        v = self.uvarint()
        if v > maximum:
            msg = f"{what} {v} exceeds the limit {maximum}"
            raise CorruptedError(msg)
        return v

    def string(self) -> str:
        n = self.length(self.remaining, "string length")
        raw = self.take(n)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            msg = "string is not valid UTF-8"
            raise CorruptedError(msg) from None

    def opt_string(self) -> str | None:
        flag = self.u8()
        if flag == 0:
            return None
        if flag == 1:
            return self.string()
        msg = f"invalid optional-string flag {flag}"
        raise CorruptedError(msg)

    def kv_map(self) -> dict[str, str]:
        # Every pair needs at least two bytes, which bounds the count.
        count = self.length(self.remaining // 2, "metadata entry count")
        out: dict[str, str] = {}
        for _ in range(count):
            key = self.string()
            value = self.string()
            if key in out:
                msg = "duplicate metadata key"
                raise CorruptedError(msg)
            out[key] = value
        return dict(sorted(out.items()))


class ByteWriter:
    """Append-only writing of the primitive types of the format."""

    __slots__ = ("buf",)

    def __init__(self) -> None:
        self.buf = bytearray()

    def __len__(self) -> int:
        return len(self.buf)

    def getvalue(self) -> bytes:
        return bytes(self.buf)

    def raw(self, data: bytes | bytearray | memoryview) -> None:
        self.buf += data

    def u8(self, v: int) -> None:
        self.buf.append(v)

    def u32(self, v: int) -> None:
        self.buf += _U32.pack(v)

    def u64(self, v: int) -> None:
        self.buf += _U64.pack(v)

    def i64(self, v: int) -> None:
        self.buf += _I64.pack(v)

    def uvarint(self, v: int) -> None:
        self.buf += encode_uvarint(v)

    def svarint(self, v: int) -> None:
        self.buf += encode_uvarint(zigzag(v))

    def string(self, s: str) -> None:
        data = s.encode("utf-8")
        self.buf += encode_uvarint(len(data))
        self.buf += data

    def opt_string(self, s: str | None) -> None:
        if s is None:
            self.buf.append(0)
        else:
            self.buf.append(1)
            self.string(s)

    def kv_map(self, mapping: dict[str, str]) -> None:
        # Keys are written in the byte order of their UTF-8 encoding, like a
        # Rust BTreeMap<String, _>, so that the bytes are deterministic.
        items = sorted(mapping.items(), key=lambda kv: kv[0].encode("utf-8"))
        self.uvarint(len(items))
        for key, value in items:
            self.string(key)
            self.string(value)


def encode_uvarint(v: int) -> bytes:
    """The shortest LEB128 encoding of an unsigned 64-bit integer."""
    if v < 0x80:
        return bytes((v,))
    out = bytearray()
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)
    return bytes(out)
