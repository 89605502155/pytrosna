"""General-purpose codecs (SPEC §7)."""

from __future__ import annotations

import contextlib

import lz4.block
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pytrosna import Codec
from pytrosna.codecs import _zstd_compress
from pytrosna.errors import CorruptedError, InvalidArgumentError, UnsupportedError

SAMPLE = b"".join((i % 17).to_bytes(4, "little") for i in range(10_000))


@pytest.mark.parametrize("codec", [Codec.LZ4, Codec.ZSTD])
def test_round_trip(codec: Codec) -> None:
    stored = codec.compress(SAMPLE)
    assert stored is not None
    assert len(stored) < len(SAMPLE) // 4
    assert codec.decompress(stored, len(SAMPLE)) == SAMPLE


def test_none_is_identity() -> None:
    assert Codec.NONE.compress(SAMPLE) is None
    assert Codec.NONE.decompress(SAMPLE, len(SAMPLE)) == SAMPLE
    with pytest.raises(CorruptedError):
        Codec.NONE.decompress(SAMPLE, 3)


@pytest.mark.parametrize("codec", [Codec.LZ4, Codec.ZSTD])
def test_wrong_length_is_rejected(codec: Codec) -> None:
    stored = codec.compress(SAMPLE)
    assert stored is not None
    with pytest.raises(CorruptedError):
        codec.decompress(stored, len(SAMPLE) - 1)
    with pytest.raises(CorruptedError):
        codec.decompress(stored, len(SAMPLE) + 1)


def test_lz4_refuses_impossible_sizes_before_decompressing() -> None:
    stored = Codec.LZ4.compress(SAMPLE)
    assert stored is not None
    with pytest.raises(CorruptedError, match="cannot expand"):
        Codec.LZ4.decompress(stored, 1 << 30)


def test_lz4_maximum_ratio_is_accepted() -> None:
    raw = bytes(4 << 20)
    stored = Codec.LZ4.compress(raw)
    assert stored is not None
    assert len(raw) <= len(stored) * 255 + 16
    assert Codec.LZ4.decompress(stored, len(raw)) == raw


def test_zstd_rejects_trailing_bytes_and_truncation() -> None:
    stored = Codec.ZSTD.compress(SAMPLE)
    assert stored is not None
    with pytest.raises(CorruptedError, match="trailing"):
        Codec.ZSTD.decompress(stored + b"x", len(SAMPLE))
    with pytest.raises(CorruptedError):
        Codec.ZSTD.decompress(stored[:-5], len(SAMPLE))
    with pytest.raises(CorruptedError):
        Codec.ZSTD.decompress(b"not a zstd frame", 10)


def test_zstd_bomb_is_bounded() -> None:
    bomb = _zstd_compress(bytes(50_000_000), 3)
    with pytest.raises(CorruptedError):
        Codec.ZSTD.decompress(bomb, 1000)


def test_incompressible_data_is_not_compressed() -> None:
    assert Codec.ZSTD.compress(b"\x01\x02\x03") is None
    assert Codec.LZ4.compress(b"") is None
    assert Codec.LZ4.decompress(b"", 0) == b""


def test_lz4_garbage() -> None:
    with pytest.raises(CorruptedError, match="LZ4"):
        Codec.LZ4.decompress(b"\xff\xff\xff", 100)
    # A valid block decompressing to fewer bytes than claimed.
    block = lz4.block.compress(b"abc" * 10, store_size=False)
    with pytest.raises(CorruptedError):
        Codec.LZ4.decompress(block, 31)


@given(st.binary(max_size=256), st.integers(0, 4096))
def test_garbage_never_crashes(data: bytes, size: int) -> None:
    for codec in (Codec.LZ4, Codec.ZSTD):
        with contextlib.suppress(CorruptedError):
            codec.decompress(data, size)


@given(st.binary(max_size=2000))
def test_property_round_trip(data: bytes) -> None:
    for codec in (Codec.LZ4, Codec.ZSTD):
        stored = codec.compress(data)
        if stored is not None:
            assert codec.decompress(stored, len(data)) == data


def test_names_and_codes() -> None:
    assert Codec.parse("ZSTD") is Codec.ZSTD
    assert Codec.parse("zstandard") is Codec.ZSTD
    assert Codec.parse("uncompressed") is Codec.NONE
    assert Codec.parse(None) is Codec.NONE
    assert Codec.parse(Codec.LZ4) is Codec.LZ4
    assert Codec.ZSTD.label == "zstd"
    assert Codec.from_code(1) is Codec.LZ4
    with pytest.raises(UnsupportedError):
        Codec.from_code(9)
    with pytest.raises(InvalidArgumentError):
        Codec.parse("gzip")
    with pytest.raises(InvalidArgumentError):
        Codec.parse(3.5)
