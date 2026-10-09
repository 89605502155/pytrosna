"""Least-significant-bit-first bit streams (SPEC §1).

Two implementations are provided: :class:`BitWriter` / :class:`BitReader`
work field by field (used by decoders whose fields depend on earlier ones),
while :func:`pack_fields` and :func:`unpack_fixed` process whole arrays of
fields at once with NumPy.
"""

from __future__ import annotations

import numpy as np

from ._bytes import U64_MASK, ceil8
from .errors import CorruptedError

_U64 = np.uint64


class BitWriter:
    """Writes bit fields into bytes, least significant bit first."""

    __slots__ = ("_acc", "_nbits", "_out")

    def __init__(self) -> None:
        self._out = bytearray()
        self._acc = 0
        self._nbits = 0

    def write(self, value: int, n: int) -> None:
        """Writes the ``n`` low bits of ``value`` (``0 <= n <= 64``)."""
        if n == 0:
            return
        self._acc |= (value & ((1 << n) - 1)) << self._nbits
        self._nbits += n
        if self._nbits >= 64:
            self._out += (self._acc & U64_MASK).to_bytes(8, "little")
            self._acc >>= 64
            self._nbits -= 64

    def write_bit(self, bit: bool) -> None:
        self.write(int(bit), 1)

    def finish(self) -> bytes:
        """Pads the last byte with zero bits and returns the bytes."""
        tail = ceil8(self._nbits)
        return bytes(self._out) + self._acc.to_bytes(8, "little")[:tail]


class BitReader:
    """Reads bit fields written by :class:`BitWriter`."""

    __slots__ = ("_data", "_len_bits", "pos")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.pos = 0
        self._len_bits = len(data) * 8

    @property
    def remaining_bits(self) -> int:
        return self._len_bits - self.pos

    def read(self, n: int) -> int:
        """Reads ``n <= 64`` bits."""
        if n == 0:
            return 0
        pos = self.pos
        if self._len_bits - pos < n:
            msg = "unexpected end of bit stream"
            raise CorruptedError(msg)
        byte = pos >> 3
        word = int.from_bytes(self._data[byte : byte + 9], "little") >> (pos & 7)
        self.pos = pos + n
        return word & ((1 << n) - 1)

    def read_bit(self) -> bool:
        pos = self.pos
        if pos >= self._len_bits:
            msg = "unexpected end of bit stream"
            raise CorruptedError(msg)
        self.pos = pos + 1
        return bool((self._data[pos >> 3] >> (pos & 7)) & 1)

    def read_unary(self, maximum: int) -> int:
        """Counts consecutive one bits, up to ``maximum``, consuming them and the
        terminating zero bit (if fewer than ``maximum`` ones were found)."""
        ones = 0
        while ones < maximum:
            if not self.read_bit():
                return ones
            ones += 1
        return ones

    def finish(self) -> None:
        """Checks that only zero padding bits of the final byte remain."""
        check_padding(self._data, self.pos)


def check_padding(data: bytes, used_bits: int) -> None:
    """Checks that ``data`` holds exactly ``used_bits`` bits plus zero padding."""
    if ceil8(used_bits) != len(data):
        msg = "trailing bytes after bit stream"
        raise CorruptedError(msg)
    if used_bits % 8 and data[-1] >> (used_bits % 8):
        msg = "non-zero padding bits in bit stream"
        raise CorruptedError(msg)


_ALL_ONES = np.uint64(0xFFFF_FFFF_FFFF_FFFF)


def _low_mask(widths: np.ndarray) -> np.ndarray:
    """``2**w - 1`` for each width ``w`` (0…64) as uint64."""
    w = widths.astype(_U64)
    return np.where(widths >= 64, _ALL_ONES, (_U64(1) << np.minimum(w, _U64(63))) - _U64(1))


def pack_fields(values: np.ndarray, widths: np.ndarray) -> bytes:
    """Packs fields ``(values[i], widths[i])`` (widths 0…64) into an LSB-first
    bit stream padded with zero bits to whole bytes.

    Each field lands in one 64-bit word or straddles two; the fields of a
    word are combined with one vectorized OR per word.
    """
    values = np.asarray(values, dtype=_U64)
    widths = np.asarray(widths, dtype=np.int64)
    total = int(widths.sum())
    if total == 0:
        return b""
    keep = widths > 0
    if not bool(keep.all()):
        values, widths = values[keep], widths[keep]
    v = values & _low_mask(widths)
    ends = np.cumsum(widths)
    starts = ends - widths
    word = starts >> 6
    offset = (starts & 63).astype(_U64)
    words = np.zeros(((total + 63) >> 6) + 1, dtype=_U64)
    first = np.concatenate(([0], np.flatnonzero(word[1:] != word[:-1]) + 1))
    words[word[first]] |= np.bitwise_or.reduceat(v << offset, first)
    spill = (offset + widths.astype(_U64)) > _U64(64)
    if bool(spill.any()):
        # only the last field of a word can spill into the next one
        words[word[spill] + 1] |= v[spill] >> (_U64(64) - offset[spill])
    return words.astype("<u8").tobytes()[: ceil8(total)]


def pack_fixed(values: np.ndarray, width: int) -> bytes:
    """Packs ``values`` with ``width`` bits each."""
    values = np.asarray(values, dtype=_U64)
    return pack_fields(values, np.full(values.size, width, dtype=np.int64))


def _words64(data: bytes) -> np.ndarray:
    """The bytes as little-endian 64-bit words, with one zero word of padding."""
    size = (len(data) + 7) // 8 + 1
    padded = np.zeros(size * 8, dtype=np.uint8)
    padded[: len(data)] = np.frombuffer(data, dtype=np.uint8)
    return padded.view("<u8").astype(_U64)


def unpack_fixed(data: bytes, n: int, width: int) -> np.ndarray:
    """Reads ``n`` fields of ``width`` bits; ``data`` must hold exactly
    ``ceil8(n * width)`` bytes with zero padding bits."""
    check_padding(data, n * width)
    if width == 0 or n == 0:
        return np.zeros(n, dtype=_U64)
    words = _words64(data)
    starts = np.arange(n, dtype=np.int64) * width
    word = starts >> 6
    offset = (starts & 63).astype(_U64)
    out = words[word] >> offset
    spill = (offset + _U64(width)) > _U64(64)
    if bool(spill.any()):
        out[spill] |= words[word[spill] + 1] << (_U64(64) - offset[spill])
    if width < 64:
        out &= _U64((1 << width) - 1)
    return out
