"""Segments: one column of a ``Data`` frame (SPEC §5.3). A segment is an
optional validity bitmap followed by the encoded non-null values, the whole
compressed with a codec."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import encodings as enc
from ._bytes import ceil8
from ._checksum import crc32c
from ._format import SegmentHeader, Statistics
from .codecs import Codec
from .column import Column
from .encodings import MAX_RAW_SEGMENT, Encoding, EncodingPolicy
from .errors import CorruptedError, LimitExceededError
from .types import DataType


@dataclass(frozen=True)
class EncodeOptions:
    policy: EncodingPolicy = EncodingPolicy.ADAPTIVE
    codec: Codec = Codec.ZSTD
    level: int = 3


def _finish(
    raw: bytes,
    encoding: Encoding,
    null_count: int,
    stats: Statistics | None,
    options: EncodeOptions,
) -> tuple[SegmentHeader, bytes]:
    if len(raw) > MAX_RAW_SEGMENT:
        msg = f"an encoded segment of {len(raw)} bytes exceeds the 1 GiB limit; use smaller blocks"
        raise LimitExceededError(msg)
    compressed = options.codec.compress(raw, options.level)
    codec, stored = (Codec.NONE, raw) if compressed is None else (options.codec, compressed)
    header = SegmentHeader(
        encoding,
        codec,
        null_count > 0,
        null_count,
        len(stored),
        len(raw),
        crc32c(stored),
        stats,
    )
    return header, stored


def _int_stats(values: np.ndarray) -> Statistics | None:
    if values.size == 0:
        return None
    return Statistics(int(values.min()), int(values.max()))


def _float_stats(values: np.ndarray) -> Statistics | None:
    finite = values[~np.isnan(values)].astype(np.float64)
    if finite.size == 0:
        return None
    return Statistics(float(np.min(finite)), float(np.max(finite)), is_float=True)


def encode_time(times: np.ndarray, options: EncodeOptions) -> tuple[SegmentHeader, bytes]:
    """Encodes the time column of a block."""

    def encode(encoding: Encoding) -> bytes:
        if encoding is Encoding.DELTA_OF_DELTA:
            return enc.delta_of_delta_encode(times)
        if encoding is Encoding.DELTA_BIT_PACK:
            return enc.delta_bitpack_encode(times)
        return enc.int_plain_encode(times, 8)

    encoding, raw = enc.smallest(options.policy.candidates(None), encode)
    return _finish(raw, encoding, 0, _int_stats(times), options)


def _encode_ints(
    candidates: tuple[Encoding, ...], values: np.ndarray, width: int
) -> tuple[Encoding, bytes]:
    best: tuple[Encoding, bytes] | None = None
    for encoding in candidates:
        if encoding is Encoding.RLE:
            # RLE wins only if strictly smaller; its size is known without encoding.
            if best is not None and enc.rle_size(values) >= len(best[1]):
                continue
            out = enc.rle_encode(values)
        elif encoding is Encoding.DELTA_BIT_PACK:
            out = enc.delta_bitpack_encode(values)
        else:
            out = enc.int_plain_encode(values, width)
        if best is None or len(out) < len(best[1]):
            best = (encoding, out)
    assert best is not None  # noqa: S101
    return best


def _encode_bools(candidates: tuple[Encoding, ...], values: np.ndarray) -> tuple[Encoding, bytes]:
    if candidates == (Encoding.RLE, Encoding.PLAIN) and enc.bool_rle_size(values) > ceil8(
        values.size
    ):
        # Plain is strictly smaller, so RLE need not be produced.
        return Encoding.PLAIN, enc.bool_plain_encode(values)

    def encode(encoding: Encoding) -> bytes:
        if encoding is Encoding.RLE:
            return enc.bool_rle_encode(values)
        return enc.bool_plain_encode(values)

    return enc.smallest(candidates, encode)


def encode_column(column: Column, options: EncodeOptions) -> tuple[SegmentHeader, bytes]:
    """Encodes a value column of a block."""
    data_type = column.data_type
    null_count = column.null_count
    parts: list[bytes] = []
    if column.validity is not None:
        parts.append(np.packbits(column.validity, bitorder="little").tobytes())
    if null_count == len(column):
        candidates: tuple[Encoding, ...] = (Encoding.PLAIN,)
    else:
        candidates = options.policy.candidates(data_type)
    dense = column.dense()
    stats: Statistics | None
    if data_type is DataType.BOOL:
        stats = None
        if dense.size:
            ones = int(dense.sum())
            stats = Statistics(int(ones == dense.size), int(ones > 0))
        encoding, encoded = _encode_bools(candidates, dense)
    elif data_type in (DataType.INT32, DataType.INT64):
        values = dense.astype(np.int64)
        width = 4 if data_type is DataType.INT32 else 8
        encoding, encoded = _encode_ints(candidates, values, width)
        stats = _int_stats(values)
    elif data_type in (DataType.FLOAT32, DataType.FLOAT64):
        width = 32 if data_type is DataType.FLOAT32 else 64
        bits = enc.float_bits(dense, width)

        def encode_float(encoding: Encoding) -> bytes:
            if encoding is Encoding.XOR:
                return enc.xor_encode(bits, width)
            return enc.float_plain_encode(bits, width)

        encoding, encoded = enc.smallest(candidates, encode_float)
        stats = _float_stats(dense)
    else:
        strings = dense.tolist()

        def encode_string(encoding: Encoding) -> bytes:
            if encoding is Encoding.DICTIONARY:
                return enc.dictionary_encode(strings)
            return enc.string_plain_encode(strings)

        encoding, encoded = enc.smallest(candidates, encode_string)
        stats = None
    parts.append(encoded)
    return _finish(b"".join(parts), encoding, null_count, stats, options)


def unpack(
    header: SegmentHeader,
    stored: bytes,
    rows: int,
    data_type: DataType | None,
    *,
    verify: bool,
) -> bytes:
    """Verifies the checksum and decompresses the stored bytes of a segment."""
    if len(stored) != header.stored_len:
        msg = "segment length does not match its header"
        raise CorruptedError(msg)
    if verify and crc32c(stored) != header.crc:
        msg = "segment checksum mismatch"
        raise CorruptedError(msg)
    # Every value encoding of a non-string column needs at most 11 bytes per
    # value, which bounds the decompressed size and defeats decompression bombs.
    if data_type is not DataType.STRING and header.raw_len > ceil8(rows) + rows * 11 + 32:
        msg = "implausible decompressed segment size"
        raise CorruptedError(msg)
    return header.codec.decompress(stored, header.raw_len)


def decode_time(header: SegmentHeader, stored: bytes, rows: int, *, verify: bool) -> np.ndarray:
    """Decodes the time column of a block of ``rows`` rows."""
    raw = unpack(header, stored, rows, None, verify=verify)
    if header.encoding is Encoding.DELTA_OF_DELTA:
        return enc.delta_of_delta_decode(raw, rows)
    if header.encoding is Encoding.DELTA_BIT_PACK:
        return enc.delta_bitpack_decode(raw, rows)
    return enc.int_plain_decode(raw, rows, 8)


def decode_column(
    header: SegmentHeader, stored: bytes, rows: int, data_type: DataType, *, verify: bool
) -> Column:
    """Decodes a value column of a block of ``rows`` rows."""
    raw = unpack(header, stored, rows, data_type, verify=verify)
    null_count = header.null_count
    if null_count > rows:
        msg = "null count exceeds the number of rows"
        raise CorruptedError(msg)
    validity = None
    data = raw
    if header.has_validity:
        size = ceil8(rows)
        if len(raw) < size:
            msg = "segment too short for its validity bitmap"
            raise CorruptedError(msg)
        validity = enc.bool_plain_decode(raw[:size], rows)
        if int(validity.sum()) != rows - null_count:
            msg = "validity bitmap does not match the null count"
            raise CorruptedError(msg)
        data = raw[size:]
    n = rows - null_count
    encoding = header.encoding
    if data_type is DataType.BOOL:
        dense = (
            enc.bool_rle_decode(data, n)
            if encoding is Encoding.RLE
            else enc.bool_plain_decode(data, n)
        )
    elif data_type in (DataType.INT32, DataType.INT64):
        width = 4 if data_type is DataType.INT32 else 8
        if encoding is Encoding.DELTA_BIT_PACK:
            wide = enc.delta_bitpack_decode(data, n)
        elif encoding is Encoding.RLE:
            wide = enc.rle_decode(data, n)
        else:
            wide = enc.int_plain_decode(data, n, width)
        if data_type is DataType.INT32:
            if wide.size and (wide.min() < -(1 << 31) or wide.max() > (1 << 31) - 1):
                msg = "int32 value out of range"
                raise CorruptedError(msg)
            dense = wide.astype(np.int32)
        else:
            dense = wide
    elif data_type in (DataType.FLOAT32, DataType.FLOAT64):
        width = 32 if data_type is DataType.FLOAT32 else 64
        bits = (
            enc.xor_decode(data, n, width)
            if encoding is Encoding.XOR
            else enc.float_plain_decode(data, n, width)
        )
        dense = enc.bits_to_floats(bits, width)
    else:
        dense = (
            enc.dictionary_decode(data, n)
            if encoding is Encoding.DICTIONARY
            else enc.string_plain_decode(data, n)
        )
    if validity is None:
        return Column(data_type, dense)
    if data_type is DataType.STRING:
        values = np.full(rows, "", dtype=object)
    else:
        values = np.zeros(rows, dtype=data_type.numpy_dtype)
    values[validity] = dense
    return Column(data_type, values, validity)
