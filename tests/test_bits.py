"""LSB-first bit streams."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pytrosna._bits import (
    BitReader,
    BitWriter,
    check_padding,
    pack_fields,
    pack_fixed,
    unpack_fixed,
)
from pytrosna._bytes import ceil8
from pytrosna.errors import CorruptedError

fields_strategy = st.lists(
    st.tuples(st.integers(0, (1 << 64) - 1), st.integers(0, 64)), max_size=200
)


def test_lsb_first_layout() -> None:
    w = BitWriter()
    w.write_bit(True)
    w.write_bit(False)
    w.write(0b11, 2)
    assert w.finish() == bytes([0b0000_1101])


def test_wide_fields_cross_words() -> None:
    w = BitWriter()
    w.write(0b101, 3)
    w.write((1 << 64) - 1, 64)
    w.write(0x1234_5678_9ABC_DEF0, 64)
    w.write(1, 1)
    data = w.finish()
    assert len(data) == ceil8(3 + 64 + 64 + 1)
    r = BitReader(data)
    assert r.read(3) == 0b101
    assert r.read(64) == (1 << 64) - 1
    assert r.read(64) == 0x1234_5678_9ABC_DEF0
    assert r.read(1) == 1
    assert r.read(0) == 0
    r.finish()


def test_reading_past_the_end_fails() -> None:
    r = BitReader(b"\xff")
    assert r.read(8) == 0xFF
    assert r.remaining_bits == 0
    with pytest.raises(CorruptedError):
        r.read(1)
    with pytest.raises(CorruptedError):
        r.read_bit()


def test_padding_must_be_zero() -> None:
    r = BitReader(bytes([0b1000_0001]))
    assert r.read_bit()
    with pytest.raises(CorruptedError, match="padding"):
        r.finish()
    with pytest.raises(CorruptedError, match="trailing"):
        check_padding(b"\x00\x00", 3)


def test_unary_codes() -> None:
    w = BitWriter()
    for ones in (0, 1, 3, 4):
        for _ in range(ones):
            w.write_bit(True)
        if ones < 4:
            w.write_bit(False)
    r = BitReader(w.finish())
    assert [r.read_unary(4) for _ in range(4)] == [0, 1, 3, 4]


@given(fields_strategy)
def test_fields_round_trip(fields: list[tuple[int, int]]) -> None:
    w = BitWriter()
    for v, n in fields:
        w.write(v, n)
    data = w.finish()
    assert len(data) == ceil8(sum(n for _, n in fields))
    r = BitReader(data)
    for v, n in fields:
        assert r.read(n) == v & ((1 << n) - 1)
    r.finish()


@given(fields_strategy)
def test_vectorized_packing_matches_the_bit_writer(fields: list[tuple[int, int]]) -> None:
    w = BitWriter()
    for v, n in fields:
        w.write(v, n)
    values = np.array([v for v, _ in fields], dtype=np.uint64)
    widths = np.array([n for _, n in fields], dtype=np.int64)
    assert pack_fields(values, widths) == w.finish()


def test_packing_many_fields_uses_chunks() -> None:
    n = 20_000  # more than one chunk
    values = np.arange(n, dtype=np.uint64)
    widths = np.full(n, 15, dtype=np.int64)
    data = pack_fields(values, widths)
    assert np.array_equal(unpack_fixed(data, n, 15), values)


@given(st.integers(0, 64), st.lists(st.integers(0, (1 << 64) - 1), max_size=100))
def test_fixed_width_round_trip(width: int, raw: list[int]) -> None:
    mask = (1 << width) - 1
    values = np.array([v & mask for v in raw], dtype=np.uint64)
    data = pack_fixed(values, width)
    assert len(data) == ceil8(len(raw) * width)
    assert np.array_equal(unpack_fixed(data, len(raw), width), values)


def test_unpack_fixed_validates_its_input() -> None:
    with pytest.raises(CorruptedError):
        unpack_fixed(b"\x00\x00", 1, 3)
    with pytest.raises(CorruptedError, match="padding"):
        unpack_fixed(b"\xff", 1, 3)
    assert unpack_fixed(b"", 5, 0).tolist() == [0] * 5
    big = np.arange(70_000, dtype=np.uint64) % 7
    assert np.array_equal(unpack_fixed(pack_fixed(big, 3), big.size, 3), big)
