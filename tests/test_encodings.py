"""Column encodings (SPEC §6).

The vectorized encoders are checked byte for byte against straightforward
reference encoders written directly from the specification.
"""

from __future__ import annotations

import contextlib
import itertools
import struct

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pytrosna import DataType, Encoding, EncodingPolicy
from pytrosna import encodings as enc
from pytrosna._bits import BitWriter
from pytrosna._bytes import I64_MAX, I64_MIN, ByteWriter, to_i64, zigzag
from pytrosna.errors import CorruptedError, InvalidArgumentError, UnsupportedError

i64s = st.integers(I64_MIN, I64_MAX)
int_lists = st.lists(i64s, max_size=300)
small_steps = st.lists(st.integers(-3, 3), max_size=300)


# ---------------------------------------------------------------- reference encoders


def ref_delta_of_delta(values: list[int]) -> bytes:
    out = ByteWriter()
    if not values:
        return b""
    out.svarint(values[0])
    if len(values) < 2:
        return out.getvalue()
    prev = to_i64(values[1] - values[0])
    out.svarint(prev)
    if len(values) < 3:
        return out.getvalue()
    bits = BitWriter()
    for a, b in itertools.pairwise(values[1:]):
        delta = to_i64(b - a)
        z = zigzag(to_i64(delta - prev))
        prev = delta
        if z == 0:
            bits.write(0, 1)
        elif z < 1 << 8:
            bits.write(0b01, 2)
            bits.write(z, 8)
        elif z < 1 << 16:
            bits.write(0b011, 3)
            bits.write(z, 16)
        elif z < 1 << 32:
            bits.write(0b0111, 4)
            bits.write(z, 32)
        else:
            bits.write(0b1111, 4)
            bits.write(z, 64)
    out.raw(bits.finish())
    return out.getvalue()


def ref_delta_bitpack(values: list[int]) -> bytes:
    if not values:
        return b""
    out = ByteWriter()
    out.svarint(values[0])
    deltas = [to_i64(b - a) for a, b in itertools.pairwise(values)]
    for start in range(0, len(deltas), 128):
        group = deltas[start : start + 128]
        low = min(group)
        offsets = [(d - low) & ((1 << 64) - 1) for d in group]
        width = max(offsets).bit_length()
        out.svarint(low)
        out.u8(width)
        bits = BitWriter()
        for o in offsets:
            bits.write(o, width)
        out.raw(bits.finish())
    return out.getvalue()


def ref_rle(values: list[int]) -> bytes:
    out = ByteWriter()
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and values[j] == values[i]:
            j += 1
        out.uvarint(j - i)
        out.svarint(values[i])
        i = j
    return out.getvalue()


def ref_xor(bits_list: list[int], width: int) -> bytes:
    if not bits_list:
        return b""
    fw = 6 if width == 64 else 5
    w = BitWriter()
    w.write(bits_list[0], width)
    prev = bits_list[0]
    window: tuple[int, int] | None = None
    for v in bits_list[1:]:
        x = v ^ prev
        prev = v
        if x == 0:
            w.write(0, 1)
            continue
        w.write(1, 1)
        lead = width - x.bit_length()
        trail = (x & -x).bit_length() - 1
        if window is not None and lead >= window[0] and trail >= window[1]:
            w.write(0, 1)
            w.write(x >> window[1], width - window[0] - window[1])
        else:
            length = width - lead - trail
            w.write(1, 1)
            w.write(lead, fw)
            w.write(length - 1, fw)
            w.write(x >> trail, length)
            window = (lead, trail)
    return w.finish()


def float64_bits(values: list[float]) -> list[int]:
    return [struct.unpack("<Q", struct.pack("<d", v))[0] for v in values]


# ---------------------------------------------------------------- time and integers


def test_regular_time_stamps_cost_one_bit_each() -> None:
    values = np.arange(1_000, 1_000 + 1000 * 100, 1000, dtype=np.int64)
    data = enc.delta_of_delta_encode(values)
    # svarint(1000) and svarint(1000) take two bytes each, then 98 zero bits
    assert len(data) == 2 + 2 + 13
    assert np.array_equal(enc.delta_of_delta_decode(data, values.size), values)


@pytest.mark.parametrize("values", [[], [5], [5, 7], [I64_MIN, I64_MAX, I64_MIN, 0]])
def test_delta_of_delta_edge_cases(values: list[int]) -> None:
    array = np.array(values, dtype=np.int64)
    data = enc.delta_of_delta_encode(array)
    assert data == ref_delta_of_delta(values)
    assert enc.delta_of_delta_decode(data, len(values)).tolist() == values


@given(int_lists)
def test_delta_of_delta_matches_reference(values: list[int]) -> None:
    data = enc.delta_of_delta_encode(np.array(values, dtype=np.int64))
    assert data == ref_delta_of_delta(values)
    assert enc.delta_of_delta_decode(data, len(values)).tolist() == values


@given(st.integers(-(10**12), 10**12), small_steps)
def test_delta_of_delta_jittery_series(start: int, steps: list[int]) -> None:
    values = list(np.cumsum([start, *[1000 + s for s in steps]]))
    array = np.array(values, dtype=np.int64)
    data = enc.delta_of_delta_encode(array)
    assert data == ref_delta_of_delta([int(v) for v in values])
    assert np.array_equal(enc.delta_of_delta_decode(data, len(values)), array)


@given(int_lists)
def test_delta_bitpack_matches_reference(values: list[int]) -> None:
    data = enc.delta_bitpack_encode(np.array(values, dtype=np.int64))
    assert data == ref_delta_bitpack(values)
    assert enc.delta_bitpack_decode(data, len(values)).tolist() == values


@given(int_lists)
def test_rle_matches_reference(values: list[int]) -> None:
    array = np.array(values, dtype=np.int64)
    data = enc.rle_encode(array)
    assert data == ref_rle(values)
    assert enc.rle_size(array) == len(data)
    assert enc.rle_decode(data, len(values)).tolist() == values


def test_rle_merges_runs() -> None:
    data = enc.rle_encode(np.array([7, 7, 7, -1, -1, 7], dtype=np.int64))
    assert data == bytes([3, 14, 2, 1, 1, 14])


@given(int_lists)
def test_plain_integers(values: list[int]) -> None:
    array = np.array(values, dtype=np.int64)
    assert enc.int_plain_decode(enc.int_plain_encode(array, 8), len(values), 8).tolist() == values
    narrow = np.clip(array, -(2**31), 2**31 - 1)
    assert np.array_equal(
        enc.int_plain_decode(enc.int_plain_encode(narrow, 4), narrow.size, 4), narrow
    )


# ---------------------------------------------------------------- floating point


@given(st.lists(st.floats(allow_nan=True, allow_infinity=True), max_size=200))
def test_xor_float64_matches_reference(values: list[float]) -> None:
    bits = np.array(float64_bits(values), dtype=np.uint64)
    data = enc.xor_encode(bits, 64)
    assert data == ref_xor(float64_bits(values), 64)
    assert np.array_equal(enc.xor_decode(data, len(values), 64), bits)


@given(st.lists(st.floats(width=32, allow_nan=True), max_size=200))
def test_xor_float32_matches_reference(values: list[float]) -> None:
    array = np.array(values, dtype=np.float32)
    bits = enc.float_bits(array, 32)
    data = enc.xor_encode(bits, 32)
    assert data == ref_xor([int(b) for b in bits], 32)
    decoded = enc.bits_to_floats(enc.xor_decode(data, len(values), 32), 32)
    assert decoded.view(np.uint32).tolist() == array.view(np.uint32).tolist()


def test_xor_special_values_keep_their_bits() -> None:
    values = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-308, 5e-324, 21.5], dtype=np.float64)
    bits = enc.float_bits(values, 64)
    decoded = enc.xor_decode(enc.xor_encode(bits, 64), values.size, 64)
    assert decoded.tolist() == bits.tolist()


def test_constant_floats_cost_one_bit_each() -> None:
    bits = enc.float_bits(np.full(800, 21.5), 64)
    assert len(enc.xor_encode(bits, 64)) == (64 + 799 + 7) // 8


@given(st.lists(st.floats(allow_nan=False), max_size=100))
def test_plain_floats(values: list[float]) -> None:
    bits = np.array(float64_bits(values), dtype=np.uint64)
    assert np.array_equal(
        enc.float_plain_decode(enc.float_plain_encode(bits, 64), len(values), 64), bits
    )
    with np.errstate(over="ignore"):
        f32 = enc.float_bits(np.array(values, dtype=np.float32), 32)
    assert np.array_equal(
        enc.float_plain_decode(enc.float_plain_encode(f32, 32), len(values), 32), f32
    )


# ---------------------------------------------------------------- booleans and strings


@given(st.lists(st.booleans(), max_size=300))
def test_booleans_round_trip(values: list[bool]) -> None:
    array = np.array(values, dtype=np.bool_)
    plain = enc.bool_plain_encode(array)
    assert len(plain) == (len(values) + 7) // 8
    assert enc.bool_plain_decode(plain, len(values)).tolist() == values
    rle = enc.bool_rle_encode(array)
    assert enc.bool_rle_size(array) == len(rle)
    assert enc.bool_rle_decode(rle, len(values)).tolist() == values


@given(st.lists(st.text(max_size=10), max_size=100))
def test_strings_round_trip(values: list[str]) -> None:
    plain = enc.string_plain_encode(values)
    assert enc.string_plain_decode(plain, len(values)).tolist() == values
    dictionary = enc.dictionary_encode(values)
    assert enc.dictionary_decode(dictionary, len(values)).tolist() == values


def test_dictionary_layout() -> None:
    data = enc.dictionary_encode(["b", "a", "b", "b"])
    # two entries in first-occurrence order, then four 1-bit indices 0,1,0,0
    assert data == bytes([2, 1, ord("b"), 1, ord("a"), 0b0010])
    assert enc.dictionary_encode(["x", "x"]) == bytes([1, 1, ord("x")])
    assert enc.dictionary_decode(bytes([1, 1, ord("x")]), 3).tolist() == ["x", "x", "x"]


# ---------------------------------------------------------------- malformed input


@pytest.mark.parametrize(
    ("decoder", "data", "n"),
    [
        (enc.delta_of_delta_decode, b"\x02\x02\x00\x00", 3),  # trailing byte after bits
        (enc.delta_of_delta_decode, b"\x02\x02\x01", 4),  # unexpected end of bits
        (enc.delta_of_delta_decode, b"\x02\x02\xfe", 3),  # padding bits set
        (enc.delta_of_delta_decode, b"\x02", 2),
        (enc.delta_of_delta_decode, b"\x02\x00", 1),
        (enc.delta_bitpack_decode, b"\x02\x00\x41", 2),  # width 65
        (enc.delta_bitpack_decode, b"\x02", 2),
        (enc.delta_bitpack_decode, b"\x00", 0),
        (enc.rle_decode, b"\x00\x02", 1),  # zero-length run
        (enc.rle_decode, b"\x05\x02", 3),  # run longer than the values
        (enc.rle_decode, b"\x01\x02\x01", 1),  # trailing byte
        (enc.bool_rle_decode, b"\x01\x02", 1),  # invalid boolean byte
        (enc.bool_rle_decode, b"\x00\x01", 1),
        (enc.bool_rle_decode, b"\x01\x01\x00", 1),
        (enc.bool_plain_decode, b"\x00\x00", 3),
        (enc.bool_plain_decode, b"\xff", 3),
        (lambda d, n: enc.xor_decode(d, n, 64), b"\x00", 1),
        (lambda d, n: enc.xor_decode(d, n, 64), b"\x01", 0),
        (enc.string_plain_decode, b"\x05ab", 1),
        (enc.string_plain_decode, b"\x00", 2),
        (enc.string_plain_decode, b"\x00\x00", 1),
        (enc.dictionary_decode, b"\x00", 1),
        (enc.dictionary_decode, b"\x02\x01a\x01a\x00", 1),  # duplicate entry
        (enc.dictionary_decode, b"\x02\x01a\x01b\x0e", 3),  # padding bits set
        (enc.dictionary_decode, b"\x03\x01a\x01b\x01c\x03", 1),  # index 3 out of range
        (enc.dictionary_decode, b"\x01", 0),
    ],
)
def test_decoders_reject_malformed_input(decoder: object, data: bytes, n: int) -> None:
    with pytest.raises(CorruptedError):
        decoder(data, n)  # type: ignore[operator]


def test_xor_decoder_rejects_bad_windows() -> None:
    w = BitWriter()
    w.write(0, 64)
    w.write(0b01, 2)  # reuse a window before one exists
    with pytest.raises(CorruptedError, match="before defining"):
        enc.xor_decode(w.finish(), 2, 64)
    w = BitWriter()
    w.write(0, 64)
    w.write(0b11, 2)
    w.write(60, 6)
    w.write(10, 6)  # 60 + 11 > 64
    w.write(0, 11)
    with pytest.raises(CorruptedError, match="exceeds"):
        enc.xor_decode(w.finish(), 2, 64)


def test_plain_decoders_check_sizes() -> None:
    with pytest.raises(CorruptedError):
        enc.int_plain_decode(b"\x00" * 7, 1, 8)
    with pytest.raises(CorruptedError):
        enc.float_plain_decode(b"\x00" * 5, 1, 32)


def test_dictionary_expansion_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(enc, "MAX_RAW_SEGMENT", 100)
    data = enc.dictionary_encode(["x" * 30] * 10)
    with pytest.raises(CorruptedError, match="expands"):
        enc.dictionary_decode(data, 10)


@settings(max_examples=200)
@given(st.binary(max_size=64), st.integers(0, 100))
def test_garbage_never_crashes_decoders(data: bytes, n: int) -> None:
    decoders = [
        enc.delta_of_delta_decode,
        enc.delta_bitpack_decode,
        enc.rle_decode,
        enc.bool_rle_decode,
        enc.bool_plain_decode,
        enc.string_plain_decode,
        enc.dictionary_decode,
        lambda d, k: enc.xor_decode(d, k, 64),
        lambda d, k: enc.xor_decode(d, k, 32),
    ]
    for decoder in decoders:
        with contextlib.suppress(CorruptedError):
            decoder(data, n)


# ---------------------------------------------------------------- selection


def test_candidates_and_policies() -> None:
    assert Encoding.candidates(None) == (
        Encoding.DELTA_OF_DELTA,
        Encoding.DELTA_BIT_PACK,
        Encoding.PLAIN,
    )
    assert Encoding.classic(DataType.FLOAT64) is Encoding.XOR
    assert EncodingPolicy.CLASSIC.candidates(DataType.STRING) == (Encoding.DICTIONARY,)
    assert EncodingPolicy.PLAIN.candidates(DataType.BOOL) == (Encoding.PLAIN,)
    assert EncodingPolicy.parse("Adaptive") is EncodingPolicy.ADAPTIVE
    with pytest.raises(InvalidArgumentError):
        EncodingPolicy.parse("best")
    Encoding.XOR.check_allowed(DataType.FLOAT32)
    with pytest.raises(CorruptedError, match="not allowed for the time column"):
        Encoding.XOR.check_allowed(None)
    with pytest.raises(CorruptedError, match="not allowed for string"):
        Encoding.RLE.check_allowed(DataType.STRING)
    assert Encoding.from_code(1) is Encoding.DELTA_BIT_PACK
    assert Encoding.DELTA_BIT_PACK.label == "delta-bitpack"
    assert Encoding.DELTA_OF_DELTA.label == "delta-of-delta"
    with pytest.raises(UnsupportedError):
        Encoding.from_code(6)


def test_smallest_prefers_the_earlier_candidate_on_ties() -> None:
    outputs = {Encoding.RLE: b"ab", Encoding.PLAIN: b"cd"}
    assert enc.smallest([Encoding.RLE, Encoding.PLAIN], outputs.__getitem__) == (
        Encoding.RLE,
        b"ab",
    )
    outputs[Encoding.PLAIN] = b"c"
    assert enc.smallest([Encoding.RLE, Encoding.PLAIN], outputs.__getitem__)[0] is Encoding.PLAIN


def test_helpers() -> None:
    assert enc.decode_capacity(10, 0) == 10
    assert enc.decode_capacity(10**9, 1) == 72
    assert enc.encode_zigzag_scalar(-1) == 1
