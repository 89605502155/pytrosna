"""Column encodings (SPEC §6), the first stage of the two-stage compression.

Every encoder takes the *dense* (non-null) values of a segment as a NumPy
array and returns bytes; every decoder takes bytes and the number of values
and returns a NumPy array, validating its input exactly as the reference
implementation does (trailing bytes, padding bits, ranges).
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import numpy as np

from ._bits import check_padding, pack_fields, pack_fixed, unpack_fixed
from ._bytes import ByteReader, ByteWriter, ceil8, encode_uvarint, to_i64, zigzag
from .errors import CorruptedError, InvalidArgumentError, UnsupportedError
from .types import DataType

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["Encoding", "EncodingPolicy"]

GROUP = 128
"""Deltas of ``DeltaBitPack`` are bit-packed in groups of this many values."""

MAX_RAW_SEGMENT = 1 << 30
"""Upper bound on the decompressed size of a segment (SPEC §5.3)."""

_U64 = np.uint64
_I64 = np.int64


class Encoding(enum.IntEnum):
    """Encoding of the values of a segment. The value is the code stored in files."""

    PLAIN = 0
    """Fixed-width little-endian values; bitmaps for booleans."""
    DELTA_BIT_PACK = 1
    """Deltas with frame-of-reference bit packing in groups of 128 (TS_2DIFF-like)."""
    DELTA_OF_DELTA = 2
    """Gorilla-style prefix codes for the delta of deltas (time stamps)."""
    RLE = 3
    """Run-length encoding."""
    XOR = 4
    """Gorilla XOR encoding of IEEE 754 bit patterns."""
    DICTIONARY = 5
    """Dictionary of distinct strings plus bit-packed indices."""

    @property
    def label(self) -> str:
        """Human-readable name such as ``"delta-of-delta"``."""
        return self.name.lower().replace("_", "-").replace("delta-bit-pack", "delta-bitpack")

    @classmethod
    def from_code(cls, code: int) -> Encoding:
        try:
            return cls(code)
        except ValueError:
            msg = f"encoding code {code}"
            raise UnsupportedError(msg) from None

    @staticmethod
    def candidates(data_type: DataType | None) -> tuple[Encoding, ...]:
        """Encodings applicable to a column of ``data_type``, or to the time
        column if ``data_type`` is ``None``, the classic one first."""
        return _CANDIDATES[data_type]

    @staticmethod
    def classic(data_type: DataType | None) -> Encoding:
        """The classic encoding of the literature for each kind of column."""
        return _CANDIDATES[data_type][0]

    def check_allowed(self, data_type: DataType | None) -> None:
        if self not in _CANDIDATES[data_type]:
            what = "the time column" if data_type is None else data_type.label
            msg = f"encoding {self.label} is not allowed for {what}"
            raise CorruptedError(msg)


_CANDIDATES: dict[DataType | None, tuple[Encoding, ...]] = {
    None: (Encoding.DELTA_OF_DELTA, Encoding.DELTA_BIT_PACK, Encoding.PLAIN),
    DataType.BOOL: (Encoding.RLE, Encoding.PLAIN),
    DataType.INT32: (Encoding.DELTA_BIT_PACK, Encoding.RLE, Encoding.PLAIN),
    DataType.INT64: (Encoding.DELTA_BIT_PACK, Encoding.RLE, Encoding.PLAIN),
    DataType.FLOAT32: (Encoding.XOR, Encoding.PLAIN),
    DataType.FLOAT64: (Encoding.XOR, Encoding.PLAIN),
    DataType.STRING: (Encoding.DICTIONARY, Encoding.PLAIN),
}


class EncodingPolicy(enum.Enum):
    """How the writer chooses the encoding of each segment."""

    ADAPTIVE = "adaptive"
    """Encode with every applicable encoding and keep the smallest result (default)."""
    CLASSIC = "classic"
    """Always use :meth:`Encoding.classic`."""
    PLAIN = "plain"
    """Always use :attr:`Encoding.PLAIN` (a baseline for experiments)."""

    @classmethod
    def parse(cls, value: EncodingPolicy | str) -> EncodingPolicy:
        if isinstance(value, EncodingPolicy):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            msg = f"unknown encoding policy {value!r}"
            raise InvalidArgumentError(msg) from None

    def candidates(self, data_type: DataType | None) -> tuple[Encoding, ...]:
        everything = _CANDIDATES[data_type]
        if self is EncodingPolicy.ADAPTIVE:
            return everything
        if self is EncodingPolicy.CLASSIC:
            return everything[:1]
        return (Encoding.PLAIN,)


def smallest(
    candidates: Sequence[Encoding], encode: Callable[[Encoding], bytes]
) -> tuple[Encoding, bytes]:
    """Encodes with each candidate and returns the smallest output; ties are
    resolved in favour of the earlier candidate."""
    best: tuple[Encoding, bytes] | None = None
    for encoding in candidates:
        out = encode(encoding)
        if best is None or len(out) < len(best[1]):
            best = (encoding, out)
    assert best is not None  # noqa: S101 - there is always a candidate
    return best


# ---------------------------------------------------------------- helpers


def _bit_length(x: np.ndarray) -> np.ndarray:
    """Bit length of each element of a uint64 array."""
    y = x.astype(_U64, copy=True)
    out = np.zeros(y.shape, dtype=np.int64)
    for shift in (32, 16, 8, 4, 2, 1):
        big = y >= (_U64(1) << _U64(shift))
        out += big * shift
        y = np.where(big, y >> _U64(shift), y)
    out += y > 0
    return out


def _zigzag(signed: np.ndarray) -> np.ndarray:
    """Vectorized zigzag of int64 values, as uint64."""
    s = signed.astype(_I64, copy=False)
    return (s.view(_U64) << _U64(1)) ^ (s >> _I64(63)).view(_U64)


def _unzigzag(z: np.ndarray) -> np.ndarray:
    """Vectorized inverse zigzag, as int64."""
    z = z.astype(_U64, copy=False)
    return ((z >> _U64(1)) ^ (_U64(0) - (z & _U64(1)))).view(_I64)


def _varint_lengths(x: np.ndarray) -> np.ndarray:
    """Number of bytes of the LEB128 encoding of each uint64 element."""
    return np.maximum(1, (_bit_length(x) + 6) // 7)


def _varint_matrix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LEB128 bytes of each element (one row each, 10 columns) and a mask of
    the bytes that belong to the encoding."""
    x = x.astype(_U64, copy=False)
    lengths = _varint_lengths(x)
    k = np.arange(10)
    groups = (x[:, None] >> (_U64(7) * k.astype(_U64))[None, :]) & _U64(0x7F)
    more = k[None, :] < (lengths[:, None] - 1)
    matrix = (groups | (more * _U64(0x80))).astype(np.uint8)
    return matrix, k[None, :] < lengths[:, None]


def _interleave_varints(*columns: np.ndarray) -> bytes:
    """For each row ``i``, the LEB128 encodings of ``columns[0][i]``,
    ``columns[1][i]``, … concatenated, all rows in order."""
    parts = [_varint_matrix(c) for c in columns]
    matrix = np.concatenate([p[0] for p in parts], axis=1)
    mask = np.concatenate([p[1] for p in parts], axis=1)
    return matrix[mask].tobytes()


def _wrapping_deltas(values: np.ndarray) -> np.ndarray:
    """``values[i] - values[i-1]`` modulo 2^64, as uint64."""
    u = values.astype(_I64, copy=False).view(_U64)
    return u[1:] - u[:-1]


def _cumulative(first: int, deltas: np.ndarray) -> np.ndarray:
    """``first`` followed by its running sums with ``deltas``, modulo 2^64, as int64."""
    out = np.empty(deltas.size + 1, dtype=_U64)
    out[0] = first & 0xFFFF_FFFF_FFFF_FFFF
    np.cumsum(deltas.astype(_U64, copy=False), out=out[1:])
    out[1:] += out[0]
    return out.view(_I64)


def decode_capacity(n: int, data_len: int) -> int:
    """How many values ``data_len`` bytes can plausibly hold."""
    return min(n, data_len * 8 + 64)


# ---------------------------------------------------------------- time and integers


def int_plain_encode(values: np.ndarray, width: int) -> bytes:
    dtype = "<i4" if width == 4 else "<i8"
    return values.astype(dtype).tobytes()


def int_plain_decode(data: bytes, n: int, width: int) -> np.ndarray:
    if n * width != len(data):
        msg = f"plain segment has {len(data)} bytes for {n} values of {width} bytes"
        raise CorruptedError(msg)
    dtype = "<i4" if width == 4 else "<i8"
    return np.frombuffer(data, dtype=dtype).astype(_I64)


def delta_of_delta_encode(values: np.ndarray) -> bytes:
    """SPEC §6.3: ``svarint v₀``, ``svarint Δ₁`` and prefix codes for ``Dᵢ = Δᵢ - Δᵢ₋₁``."""
    n = values.size
    if n == 0:
        return b""
    out = ByteWriter()
    out.svarint(int(values[0]))
    if n < 2:
        return out.getvalue()
    deltas = _wrapping_deltas(values)
    out.svarint(to_i64(int(deltas[0])))
    if n < 3:
        return out.getvalue()
    z = _zigzag((deltas[1:] - deltas[:-1]).view(_I64))
    prefix = np.full(z.size, 0b1111, dtype=_U64)
    prefix_bits = np.full(z.size, 4, dtype=np.int64)
    payload_bits = np.full(z.size, 64, dtype=np.int64)
    for limit, code, code_bits, bits in (
        (1 << 32, 0b0111, 4, 32),
        (1 << 16, 0b011, 3, 16),
        (1 << 8, 0b01, 2, 8),
        (1, 0b0, 1, 0),
    ):
        hit = z < _U64(limit)
        prefix[hit] = code
        prefix_bits[hit] = code_bits
        payload_bits[hit] = bits
    fields = np.empty(2 * z.size, dtype=_U64)
    widths = np.empty(2 * z.size, dtype=np.int64)
    fields[0::2], fields[1::2] = prefix, z
    widths[0::2], widths[1::2] = prefix_bits, payload_bits
    out.raw(pack_fields(fields, widths))
    return out.getvalue()


def delta_of_delta_decode(data: bytes, n: int) -> np.ndarray:
    r = ByteReader(data)
    if n == 0:
        r.finish("delta-of-delta values")
        return np.empty(0, dtype=_I64)
    first = r.svarint()
    if n == 1:
        r.finish("delta-of-delta values")
        return np.array([first], dtype=_I64)
    delta = r.svarint()
    if n == 2:
        r.finish("delta-of-delta values")
        return _cumulative(first, np.array([delta & 0xFFFF_FFFF_FFFF_FFFF], dtype=_U64))
    stream = r.rest()
    codes = n - 2
    z = _decode_prefix_codes(stream, codes)
    dd = _unzigzag(z).view(_U64)
    deltas = np.empty(n - 1, dtype=_U64)
    deltas[0] = delta & 0xFFFF_FFFF_FFFF_FFFF
    np.cumsum(dd, out=deltas[1:])
    deltas[1:] += deltas[0]
    return _cumulative(first, deltas)


def _words(data: bytes) -> list[int]:
    """For every byte offset ``b`` of ``data``, the 64 bits starting there
    (little-endian, zeros past the end), so that a decoder can read any field
    of up to 57 bits with one shift."""
    size = len(data) + 8
    padded = np.zeros(size + 8, dtype=np.uint8)
    padded[: len(data)] = np.frombuffer(data, dtype=np.uint8)
    words = np.zeros(size, dtype=_U64)
    for k in range(8):
        words |= padded[k : k + size].astype(_U64) << _U64(8 * k)
    return words.tolist()


def _decode_prefix_codes(stream: bytes, count: int) -> np.ndarray:
    """Decodes ``count`` delta-of-delta prefix codes. Runs of ``0`` codes
    (regular sampling) are consumed many at a time."""
    total_bits = len(stream) * 8
    words = _words(stream)
    z = [0] * count
    produced = 0
    pos = 0
    while produced < count:
        if pos >= total_bits:
            msg = "unexpected end of bit stream"
            raise CorruptedError(msg)
        word = words[pos >> 3] >> (pos & 7)
        if not word & 1:
            # a run of zero bits, each a zero code
            run = (word & -word).bit_length() - 1 if word else 57
            run = min(run, count - produced, total_bits - pos)
            produced += run
            pos += run
            continue
        if not word & 2:
            prefix, bits = 2, 8
        elif not word & 4:
            prefix, bits = 3, 16
        elif not word & 8:
            prefix, bits = 4, 32
        else:
            prefix, bits = 4, 64
        pos += prefix
        offset = pos & 7
        value = words[pos >> 3] >> offset
        if offset + bits > 64:
            value |= words[(pos >> 3) + 8] << (64 - offset)
        z[produced] = value & ((1 << bits) - 1)
        produced += 1
        pos += bits
    if pos > total_bits:
        msg = "unexpected end of bit stream"
        raise CorruptedError(msg)
    check_padding(stream, pos)
    return np.array(z, dtype=_U64)


def delta_bitpack_encode(values: np.ndarray) -> bytes:
    """SPEC §6.2: ``svarint v₀``, then deltas in groups of 128 with a minimum,
    a bit width and the bit-packed offsets from the minimum.

    Every group starts on a byte boundary, so the whole output is assembled
    as one bit stream of 8-bit header fields, offsets and padding.
    """
    if values.size == 0:
        return b""
    deltas = _wrapping_deltas(values).view(_I64)
    first = encode_uvarint(zigzag(int(values[0])))
    if deltas.size == 0:
        return first
    starts = np.arange(0, deltas.size, GROUP)
    lows = np.minimum.reduceat(deltas, starts)
    sizes = np.diff(np.append(starts, deltas.size))
    offsets = deltas.view(_U64) - np.repeat(lows.view(_U64), sizes)
    widths = _bit_length(np.maximum.reduceat(offsets, starts))
    value_parts: list[np.ndarray] = [np.frombuffer(first, dtype=np.uint8).astype(_U64)]
    width_parts: list[np.ndarray] = [np.full(len(first), 8, dtype=np.int64)]
    groups = zip(lows.tolist(), widths.tolist(), sizes.tolist(), strict=True)
    for g, (low, width, size) in enumerate(groups):
        header = encode_uvarint(zigzag(low)) + bytes((width,))
        value_parts.append(np.frombuffer(header, dtype=np.uint8).astype(_U64))
        width_parts.append(np.full(len(header), 8, dtype=np.int64))
        value_parts.append(offsets[g * GROUP : g * GROUP + size])
        width_parts.append(np.full(size, width, dtype=np.int64))
        pad = -(size * width) % 8
        if pad:
            value_parts.append(np.zeros(1, dtype=_U64))
            width_parts.append(np.array([pad], dtype=np.int64))
    return pack_fields(np.concatenate(value_parts), np.concatenate(width_parts))


def delta_bitpack_decode(data: bytes, n: int) -> np.ndarray:
    r = ByteReader(data)
    if n == 0:
        r.finish("delta-bitpack values")
        return np.empty(0, dtype=_I64)
    first = r.svarint()
    deltas = np.empty(n - 1, dtype=_U64)
    done = 0
    while done < n - 1:
        size = min(GROUP, n - 1 - done)
        low = r.svarint()
        width = r.u8()
        if width > 64:
            msg = f"bit width {width} exceeds 64"
            raise CorruptedError(msg)
        packed = r.take(ceil8(size * width))
        deltas[done : done + size] = unpack_fixed(packed, size, width) + _U64(
            low & 0xFFFF_FFFF_FFFF_FFFF
        )
        done += size
    r.finish("delta-bitpack values")
    return _cumulative(first, deltas)


def _runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lengths and values of the runs of equal consecutive values."""
    if values.size == 0:
        return np.empty(0, dtype=np.int64), values[:0]
    starts = np.concatenate(([0], np.flatnonzero(values[1:] != values[:-1]) + 1))
    lengths = np.diff(np.concatenate((starts, [values.size])))
    return lengths, values[starts]


def rle_size(values: np.ndarray) -> int:
    """Size of :func:`rle_encode` without producing it."""
    lengths, run_values = _runs(values)
    return int(
        _varint_lengths(lengths.astype(_U64)).sum()
        + _varint_lengths(_zigzag(run_values.astype(_I64))).sum()
    )


def rle_encode(values: np.ndarray) -> bytes:
    """SPEC §6.4 for integers: ``uvarint run_length`` and ``svarint value`` per run."""
    if values.size == 0:
        return b""
    lengths, run_values = _runs(values.astype(_I64, copy=False))
    return _interleave_varints(lengths.astype(_U64), _zigzag(run_values))


def rle_decode(data: bytes, n: int) -> np.ndarray:
    r = ByteReader(data)
    lengths: list[int] = []
    values: list[int] = []
    filled = 0
    while filled < n:
        run = r.length(n - filled, "run length")
        if run == 0:
            msg = "zero-length run"
            raise CorruptedError(msg)
        values.append(r.svarint())
        lengths.append(run)
        filled += run
    r.finish("rle values")
    return np.repeat(np.array(values, dtype=_I64), np.array(lengths, dtype=np.int64))


# ---------------------------------------------------------------- booleans


def bool_plain_encode(values: np.ndarray) -> bytes:
    return np.packbits(values.astype(np.bool_), bitorder="little").tobytes()


def bool_plain_decode(data: bytes, n: int) -> np.ndarray:
    if len(data) != ceil8(n):
        msg = f"boolean bitmap has {len(data)} bytes for {n} values"
        raise CorruptedError(msg)
    if n % 8 and data[-1] >> (n % 8):
        msg = "non-zero padding bits in boolean bitmap"
        raise CorruptedError(msg)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    return bits[:n].astype(np.bool_)


def bool_rle_encode(values: np.ndarray) -> bytes:
    """SPEC §6.4 for booleans: ``uvarint run_length`` and a ``u8`` value per run."""
    if values.size == 0:
        return b""
    lengths, run_values = _runs(values.astype(np.bool_))
    # A value 0 or 1 is a one-byte "varint" equal to the u8 the format requires.
    return _interleave_varints(lengths.astype(_U64), run_values.astype(_U64))


def bool_rle_size(values: np.ndarray) -> int:
    lengths, _ = _runs(values.astype(np.bool_))
    return int(_varint_lengths(lengths.astype(_U64)).sum()) + lengths.size


def bool_rle_decode(data: bytes, n: int) -> np.ndarray:
    r = ByteReader(data)
    lengths: list[int] = []
    values: list[bool] = []
    filled = 0
    while filled < n:
        run = r.length(n - filled, "run length")
        if run == 0:
            msg = "zero-length run"
            raise CorruptedError(msg)
        byte = r.u8()
        if byte > 1:
            msg = f"invalid boolean byte {byte}"
            raise CorruptedError(msg)
        lengths.append(run)
        values.append(bool(byte))
        filled += run
    r.finish("rle booleans")
    return np.repeat(np.array(values, dtype=np.bool_), np.array(lengths, dtype=np.int64))


# ---------------------------------------------------------------- floating point


def float_bits(values: np.ndarray, width: int) -> np.ndarray:
    """IEEE 754 bit patterns as uint64 (float32 zero-extended)."""
    if width == 32:
        return values.astype(np.float32, copy=False).view(np.uint32).astype(_U64)
    return values.astype(np.float64, copy=False).view(_U64)


def bits_to_floats(bits: np.ndarray, width: int) -> np.ndarray:
    if width == 32:
        return bits.astype(np.uint32).view(np.float32)
    return bits.astype(_U64, copy=False).view(np.float64)


def float_plain_encode(bits: np.ndarray, width: int) -> bytes:
    dtype = "<u4" if width == 32 else "<u8"
    return bits.astype(dtype).tobytes()


def float_plain_decode(data: bytes, n: int, width: int) -> np.ndarray:
    size = width // 8
    if n * size != len(data):
        msg = f"plain segment has {len(data)} bytes for {n} values of {size} bytes"
        raise CorruptedError(msg)
    dtype = "<u4" if width == 32 else "<u8"
    return np.frombuffer(data, dtype=dtype).astype(_U64)


def xor_encode(bits: np.ndarray, width: int) -> bytes:
    """SPEC §6.5: Gorilla XOR with leading/trailing-zero windows.

    Only the choice between reusing the current window and opening a new one
    is sequential; it is made in a loop over the changed values, and the bit
    fields are then assembled for all values at once.
    """
    n = bits.size
    if n == 0:
        return b""
    fw = 6 if width == 64 else 5
    x = bits[1:] ^ bits[:-1]
    length = _bit_length(x)
    lead = width - length
    low = x & (_U64(0) - x)
    trail = np.where(x == 0, 0, _bit_length(low) - 1)
    changed = np.flatnonzero(x)
    new_window = np.zeros(x.size, dtype=np.bool_)
    window_lead = np.zeros(x.size, dtype=np.int64)
    window_trail = np.zeros(x.size, dtype=np.int64)
    if changed.size:
        leads = lead[changed].tolist()
        trails = trail[changed].tolist()
        is_new = [False] * len(leads)
        wl_out = [0] * len(leads)
        wt_out = [0] * len(leads)
        wl = wt = -1
        for k, (ld, tr) in enumerate(zip(leads, trails, strict=True)):
            if wl < 0 or ld < wl or tr < wt:
                wl, wt = ld, tr
                is_new[k] = True
            wl_out[k] = wl
            wt_out[k] = wt
        new_window[changed] = is_new
        window_lead[changed] = wl_out
        window_trail[changed] = wt_out
    nonzero = x != 0
    reuse = nonzero & ~new_window
    size = width - window_lead - window_trail
    # four fields per value: control bits, lead, size - 1, meaningful bits
    fields = np.zeros((x.size, 4), dtype=_U64)
    widths = np.zeros((x.size, 4), dtype=np.int64)
    fields[:, 0] = np.where(new_window, 0b11, np.where(reuse, 0b01, 0))
    widths[:, 0] = np.where(nonzero, 2, 1)
    fields[:, 1] = np.where(new_window, window_lead, 0).astype(_U64)
    widths[:, 1] = np.where(new_window, fw, 0)
    fields[:, 2] = np.where(new_window, size - 1, 0).astype(_U64)
    widths[:, 2] = np.where(new_window, fw, 0)
    fields[:, 3] = np.where(nonzero, x >> window_trail.astype(_U64), _U64(0))
    widths[:, 3] = np.where(nonzero, size, 0)
    return pack_fields(
        np.concatenate(([bits[0]], fields.ravel())),
        np.concatenate(([width], widths.ravel())),
    )


def xor_decode(data: bytes, n: int, width: int) -> np.ndarray:
    if n == 0:
        check_padding(data, 0)
        return np.empty(0, dtype=_U64)
    fw = 6 if width == 64 else 5
    field_mask = (1 << fw) - 1
    total_bits = len(data) * 8
    if total_bits < width:
        msg = "unexpected end of bit stream"
        raise CorruptedError(msg)
    words = _words(data)
    first = words[0] & ((1 << width) - 1)
    xs = [0] * n
    xs[0] = first
    pos = width
    i = 1
    window_l = window_t = -1
    window_size = 0
    while i < n:
        if pos >= total_bits:
            msg = "unexpected end of bit stream"
            raise CorruptedError(msg)
        word = words[pos >> 3] >> (pos & 7)
        if not word & 1:
            # unchanged values: a run of zero bits
            run = (word & -word).bit_length() - 1 if word else 57
            run = min(run, n - i, total_bits - pos)
            i += run
            pos += run
            continue
        if word & 2:
            lead = (word >> 2) & field_mask
            size = ((word >> (2 + fw)) & field_mask) + 1
            if lead + size > width:
                msg = f"XOR window of {lead}+{size} bits exceeds {width} bits"
                raise CorruptedError(msg)
            window_l, window_t, window_size = lead, width - lead - size, size
            pos += 2 + 2 * fw
        else:
            if window_l < 0:
                msg = "XOR stream reuses a window before defining one"
                raise CorruptedError(msg)
            pos += 2
        offset = pos & 7
        value = words[pos >> 3] >> offset
        if offset + window_size > 64:
            value |= words[(pos >> 3) + 8] << (64 - offset)
        xs[i] = (value & ((1 << window_size) - 1)) << window_t
        pos += window_size
        i += 1
    if pos > total_bits:
        msg = "unexpected end of bit stream"
        raise CorruptedError(msg)
    check_padding(data, pos)
    return np.bitwise_xor.accumulate(np.array(xs, dtype=_U64))


# ---------------------------------------------------------------- strings


def string_plain_encode(values: Sequence[str]) -> bytes:
    parts = []
    for s in values:
        data = s.encode("utf-8")
        parts.append(encode_uvarint(len(data)))
        parts.append(data)
    return b"".join(parts)


def string_plain_decode(data: bytes, n: int) -> np.ndarray:
    # Each string needs at least one length byte, which bounds n.
    if n > len(data):
        msg = "string segment is too short for its count"
        raise CorruptedError(msg)
    r = ByteReader(data)
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = r.string()
    r.finish("plain strings")
    return out


def dictionary_encode(values: Sequence[str]) -> bytes:
    """SPEC §6.6: distinct strings in first-occurrence order, then bit-packed indices."""
    if not values:
        return b""
    ids: dict[str, int] = {}
    indices = np.fromiter(
        (ids.setdefault(s, len(ids)) for s in values), dtype=_U64, count=len(values)
    )
    out = ByteWriter()
    out.uvarint(len(ids))
    for entry in ids:
        out.string(entry)
    out.raw(pack_fixed(indices, (len(ids) - 1).bit_length()))
    return out.getvalue()


def dictionary_decode(data: bytes, n: int) -> np.ndarray:
    r = ByteReader(data)
    if n == 0:
        r.finish("empty dictionary segment")
        return np.empty(0, dtype=object)
    size = r.length(n, "dictionary size")
    if size == 0:
        msg = "empty dictionary for a non-empty segment"
        raise CorruptedError(msg)
    entries: list[str] = []
    seen: set[str] = set()
    for _ in range(size):
        s = r.string()
        if s in seen:
            msg = "duplicate dictionary entry"
            raise CorruptedError(msg)
        seen.add(s)
        entries.append(s)
    width = (size - 1).bit_length()
    packed = r.take(ceil8(n * width))
    r.finish("dictionary indices")
    indices = unpack_fixed(packed, n, width)
    if indices.size and int(indices.max()) >= size:
        msg = "dictionary index out of range"
        raise CorruptedError(msg)
    # A long entry repeated by a few index bits could expand a tiny segment
    # enormously, so the size of the result is checked before it is built.
    lengths = np.array([len(e.encode("utf-8")) for e in entries], dtype=np.int64)
    if int(lengths[indices.astype(np.int64)].sum()) > MAX_RAW_SEGMENT:
        msg = f"dictionary segment expands beyond {MAX_RAW_SEGMENT} bytes"
        raise CorruptedError(msg)
    table = np.empty(size, dtype=object)
    table[:] = entries
    return table[indices.astype(np.int64)]


def encode_zigzag_scalar(v: int) -> int:
    """Zigzag of a Python integer (exposed for tests)."""
    return zigzag(v)
