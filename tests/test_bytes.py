"""Primitive types of the format (SPEC §1)."""

from __future__ import annotations

import contextlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pytrosna._bytes import (
    I64_MAX,
    I64_MIN,
    U64_MASK,
    ByteReader,
    ByteWriter,
    bit_width,
    ceil8,
    encode_uvarint,
    to_i64,
    unzigzag,
    zigzag,
)
from pytrosna.errors import CorruptedError


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        (0, b"\x00"),
        (1, b"\x01"),
        (127, b"\x7f"),
        (128, b"\x80\x01"),
        (300, b"\xac\x02"),
        (U64_MASK, b"\xff" * 9 + b"\x01"),
    ],
)
def test_varint_known_values(value: int, encoded: bytes) -> None:
    assert encode_uvarint(value) == encoded
    r = ByteReader(encoded)
    assert r.uvarint() == value
    assert r.remaining == 0


def test_varint_rejects_overflow_and_overlong() -> None:
    with pytest.raises(CorruptedError, match="overflows"):
        ByteReader(b"\xff" * 9 + b"\x02").uvarint()
    with pytest.raises(CorruptedError, match="longer than 10"):
        ByteReader(b"\x80" * 11).uvarint()
    with pytest.raises(CorruptedError, match="end of data"):
        ByteReader(b"\x80").uvarint()


def test_zigzag_known_values() -> None:
    assert zigzag(0) == 0
    assert zigzag(-1) == 1
    assert zigzag(1) == 2
    assert zigzag(I64_MAX) == U64_MASK - 1
    assert zigzag(I64_MIN) == U64_MASK


@given(st.integers(I64_MIN, I64_MAX))
def test_zigzag_round_trip(v: int) -> None:
    assert unzigzag(zigzag(v)) == v


@given(st.integers(0, U64_MASK))
def test_uvarint_round_trip(v: int) -> None:
    data = encode_uvarint(v)
    assert len(data) <= 10
    r = ByteReader(data)
    assert r.uvarint() == v
    r.finish("test")


@given(st.integers(I64_MIN, I64_MAX))
def test_svarint_round_trip(v: int) -> None:
    w = ByteWriter()
    w.svarint(v)
    assert ByteReader(w.getvalue()).svarint() == v


def test_fixed_width_integers() -> None:
    w = ByteWriter()
    w.u8(7)
    w.u32(0xDEADBEEF)
    w.u64(U64_MASK)
    w.i64(-2)
    assert len(w) == 1 + 4 + 8 + 8
    r = ByteReader(w.getvalue())
    assert (r.u8(), r.u32(), r.u64(), r.i64()) == (7, 0xDEADBEEF, U64_MASK, -2)
    with pytest.raises(CorruptedError):
        r.u8()


def test_strings_and_maps_round_trip() -> None:
    w = ByteWriter()
    w.string("устройство")
    w.opt_string(None)
    w.opt_string("x")
    w.kv_map({"единица": "МБ", "source": "", "a": "1"})
    r = ByteReader(w.getvalue())
    assert r.string() == "устройство"
    assert r.opt_string() is None
    assert r.opt_string() == "x"
    assert r.kv_map() == {"a": "1", "source": "", "единица": "МБ"}
    r.finish("test")


def test_maps_are_written_in_key_order() -> None:
    a, b = ByteWriter(), ByteWriter()
    a.kv_map({"b": "2", "a": "1"})
    b.kv_map({"a": "1", "b": "2"})
    assert a.getvalue() == b.getvalue()


def test_invalid_input_is_rejected() -> None:
    with pytest.raises(CorruptedError, match="UTF-8"):
        ByteReader(b"\x02\xc3\x28").string()
    with pytest.raises(CorruptedError, match="optional-string flag"):
        ByteReader(b"\x02").opt_string()
    w = ByteWriter()
    w.uvarint(2)
    for _ in range(2):
        w.string("k")
        w.string("v")
    with pytest.raises(CorruptedError, match="duplicate metadata key"):
        ByteReader(w.getvalue()).kv_map()
    with pytest.raises(CorruptedError, match="exceeds the limit"):
        ByteReader(b"\x05ab").string()
    with pytest.raises(CorruptedError, match="trailing"):
        ByteReader(b"\x00\x00").finish("x")
    with pytest.raises(CorruptedError, match="32 bits"):
        ByteReader(encode_uvarint(1 << 32)).uvarint_u32()
    with pytest.raises(CorruptedError, match="end of data"):
        ByteReader(b"ab").take(3)


def test_reader_position_and_rest() -> None:
    r = ByteReader(b"abcdef")
    assert r.take(2) == b"ab"
    assert r.position == 2
    assert r.rest() == b"cdef"
    assert r.remaining == 0


@given(st.binary(max_size=64))
def test_reader_never_crashes_on_garbage(data: bytes) -> None:
    for method in ("uvarint", "string", "kv_map", "opt_string", "i64", "u32"):
        with contextlib.suppress(CorruptedError):
            getattr(ByteReader(data), method)()


def test_helpers() -> None:
    assert [ceil8(n) for n in (0, 1, 8, 9)] == [0, 1, 1, 2]
    assert [bit_width(v) for v in (0, 1, 2, 255, 256)] == [0, 1, 2, 8, 9]
    assert to_i64(U64_MASK) == -1
    assert to_i64(5) == 5
