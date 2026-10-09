"""Segments: encoding and decoding of whole columns (SPEC §5.3)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pytrosna import Codec, Column, DataType, Encoding, EncodingPolicy
from pytrosna._segment import (
    EncodeOptions,
    decode_column,
    decode_time,
    encode_column,
    encode_time,
    unpack,
)
from pytrosna.errors import CorruptedError, LimitExceededError

OPTIONS = [
    EncodeOptions(EncodingPolicy.ADAPTIVE, Codec.ZSTD),
    EncodeOptions(EncodingPolicy.CLASSIC, Codec.LZ4),
    EncodeOptions(EncodingPolicy.PLAIN, Codec.NONE),
]

VALUES = {
    DataType.BOOL: st.booleans(),
    DataType.INT32: st.integers(-(2**31), 2**31 - 1),
    DataType.INT64: st.integers(-(2**63), 2**63 - 1),
    DataType.FLOAT32: st.floats(width=32),
    DataType.FLOAT64: st.floats(),
    DataType.STRING: st.text(max_size=8),
}


def round_trip(column: Column, options: EncodeOptions) -> Column:
    header, stored = encode_column(column, options)
    return decode_column(header, stored, len(column), column.data_type, verify=True)


@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize("data_type", list(DataType))
def test_columns_with_nulls(data_type: DataType, options: EncodeOptions) -> None:
    samples = {
        DataType.BOOL: [True, None, False, True],
        DataType.INT32: [5, None, -7, 5],
        DataType.INT64: [2**62, None, -(2**63), 0],
        DataType.FLOAT32: [1.5, None, math.inf, -0.0],
        DataType.FLOAT64: [math.nan, None, 0.1, 1e300],
        DataType.STRING: ["a", None, "", "ä"],
    }
    column = Column.from_values(data_type, samples[data_type])
    assert round_trip(column, options) == column


@settings(max_examples=40, deadline=None)
@given(st.data())
def test_property_round_trip(data: st.DataObject) -> None:
    data_type = data.draw(st.sampled_from(list(DataType)))
    values = data.draw(st.lists(st.none() | VALUES[data_type], max_size=60))
    options = data.draw(st.sampled_from(OPTIONS))
    column = Column.from_values(data_type, values)
    assert round_trip(column, options) == column


@pytest.mark.parametrize("data_type", list(DataType))
def test_all_null_column_uses_plain(data_type: DataType) -> None:
    column = Column.nulls(data_type, 5)
    header, stored = encode_column(column, OPTIONS[0])
    assert header.encoding is Encoding.PLAIN
    assert header.stats is None
    assert header.null_count == 5
    assert decode_column(header, stored, 5, data_type, verify=True) == column


def test_statistics() -> None:
    header, _ = encode_column(
        Column.from_values("float64", [math.nan, 2.0, -1.0, None]), OPTIONS[0]
    )
    assert header.stats is not None
    assert (header.stats.min, header.stats.max, header.stats.is_float) == (-1.0, 2.0, True)
    header, _ = encode_column(Column.from_values("float64", [math.nan]), OPTIONS[0])
    assert header.stats is None
    header, _ = encode_column(Column.from_values("bool", [True, True]), OPTIONS[0])
    assert header.stats is not None
    assert (header.stats.min, header.stats.max) == (1, 1)
    header, _ = encode_column(Column.from_values("bool", [False, True]), OPTIONS[0])
    assert header.stats is not None
    assert (header.stats.min, header.stats.max) == (0, 1)
    header, _ = encode_column(Column.from_values("int32", [3, -4]), OPTIONS[0])
    assert header.stats is not None
    assert (header.stats.min, header.stats.max) == (-4, 3)
    header, _ = encode_column(Column.from_values("string", ["a"]), OPTIONS[0])
    assert header.stats is None


def test_adaptive_choice() -> None:
    constant = Column("int64", np.full(1000, 42, dtype=np.int64))
    assert encode_column(constant, OPTIONS[0])[0].encoding is Encoding.RLE
    ramp = Column("int64", np.arange(1000, dtype=np.int64))
    assert encode_column(ramp, OPTIONS[0])[0].encoding is Encoding.DELTA_BIT_PACK
    noisy = Column("int64", np.random.default_rng(0).integers(-(2**62), 2**62, 1000))
    assert encode_column(noisy, OPTIONS[0])[0].encoding is Encoding.PLAIN
    flags = Column("bool", np.random.default_rng(0).random(1000) < 0.5)
    assert encode_column(flags, OPTIONS[0])[0].encoding is Encoding.PLAIN
    few = Column("bool", np.repeat([True, False], 500))
    assert encode_column(few, OPTIONS[0])[0].encoding is Encoding.RLE
    labels = Column.from_values("string", ["on", "off"] * 50)
    assert encode_column(labels, OPTIONS[0])[0].encoding is Encoding.DICTIONARY


def test_time_segments() -> None:
    times = np.arange(0, 10_000_000, 1000, dtype=np.int64)
    for options in OPTIONS:
        header, stored = encode_time(times, options)
        assert header.stats is not None
        assert (header.stats.min, header.stats.max) == (0, 9_999_000)
        assert np.array_equal(decode_time(header, stored, times.size, verify=True), times)
    # The adaptive policy keeps the smallest encoding. For a perfectly regular
    # series bit packing with width 0 (3 bytes per 128 points) beats one bit
    # per point; with rare jitter delta-of-delta wins.
    from pytrosna import encodings as enc

    rng = np.random.default_rng(0)
    jitter = times + (rng.random(times.size) < 0.01) * rng.integers(1, 5, times.size)
    for series in (times, jitter):
        header, _ = encode_time(series, OPTIONS[0])
        sizes = {
            Encoding.DELTA_OF_DELTA: len(enc.delta_of_delta_encode(series)),
            Encoding.DELTA_BIT_PACK: len(enc.delta_bitpack_encode(series)),
            Encoding.PLAIN: series.size * 8,
        }
        assert header.raw_len == min(sizes.values())
        assert sizes[header.encoding] == header.raw_len
    assert encode_time(times, OPTIONS[0])[0].encoding is Encoding.DELTA_BIT_PACK
    assert encode_time(jitter, OPTIONS[0])[0].encoding is Encoding.DELTA_OF_DELTA


def test_codec_falls_back_to_none_when_useless() -> None:
    header, stored = encode_column(Column.from_values("int64", [1]), OPTIONS[0])
    assert header.codec is Codec.NONE
    assert stored == stored  # noqa: PLR0124 - stored is plain


def test_damage_is_detected() -> None:
    header, stored = encode_column(Column("int64", np.arange(100, dtype=np.int64) ** 2), OPTIONS[2])
    damaged = bytes([stored[0] ^ 1]) + stored[1:]
    with pytest.raises(CorruptedError, match="checksum"):
        decode_column(header, damaged, 100, DataType.INT64, verify=True)
    # Without checksum verification the damage goes unnoticed.
    decode_column(header, damaged, 100, DataType.INT64, verify=False)
    with pytest.raises(CorruptedError, match="length"):
        unpack(header, stored[:-1], 100, DataType.INT64, verify=True)


def test_implausible_sizes_are_rejected() -> None:
    header, stored = encode_column(Column("int64", np.arange(10, dtype=np.int64)), OPTIONS[2])
    with pytest.raises(CorruptedError, match="implausible"):
        unpack(header, stored, 1, DataType.INT64, verify=True)


def test_validity_errors() -> None:
    column = Column.from_values("int64", [1, None, 3])
    header, stored = encode_column(column, OPTIONS[2])
    with pytest.raises(CorruptedError, match="null count exceeds"):
        decode_column(header, stored, 0, DataType.INT64, verify=True)
    from dataclasses import replace

    wrong = replace(header, null_count=2)
    with pytest.raises(CorruptedError, match="does not match the null count"):
        decode_column(wrong, stored, 3, DataType.INT64, verify=True)
    short = replace(header, stored_len=0, raw_len=0, crc=0)
    with pytest.raises(CorruptedError, match="too short"):
        decode_column(short, b"", 3, DataType.INT64, verify=True)


def test_int32_range_is_checked_on_decode() -> None:
    header, stored = encode_column(Column("int64", np.array([2**40])), OPTIONS[2])
    from dataclasses import replace

    # an int64 payload read as int32 plain is the wrong size; use RLE instead
    header, stored = encode_column(Column("int64", np.full(10, 2**40)), OPTIONS[0])
    assert header.encoding is Encoding.RLE
    with pytest.raises(CorruptedError, match="int32 value out of range"):
        decode_column(replace(header), stored, 10, DataType.INT32, verify=True)


def test_segment_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import pytrosna._segment as segment_module

    monkeypatch.setattr(segment_module, "MAX_RAW_SEGMENT", 10)
    with pytest.raises(LimitExceededError):
        encode_column(Column("int64", np.arange(100, dtype=np.int64) ** 3), OPTIONS[2])
