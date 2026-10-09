"""Frames and frame payloads (SPEC §2-§8)."""

from __future__ import annotations

import struct

import pytest

from pytrosna import ColumnSchema, DataType, DeviceSchema, Encoding
from pytrosna import _format as fmt
from pytrosna._bytes import ByteReader, ByteWriter
from pytrosna.codecs import Codec
from pytrosna.errors import (
    CorruptedError,
    LimitExceededError,
    NotTrosnaError,
    UnsupportedError,
    UnsupportedVersionError,
)


def test_file_header() -> None:
    assert fmt.file_header() == b"TROSNA\x01\x00"
    assert fmt.check_file_header(b"TROSNA\x01\x07") == (1, 7)
    with pytest.raises(NotTrosnaError):
        fmt.check_file_header(b"TROSN")
    with pytest.raises(NotTrosnaError):
        fmt.check_file_header(b"PARQUET1")
    with pytest.raises(UnsupportedVersionError, match=r"2\.0"):
        fmt.check_file_header(b"TROSNA\x02\x00")


def test_frame_round_trip() -> None:
    frame = fmt.encode_frame(fmt.FrameKind.META, b"payload")
    assert frame[:4] == b"TBLK"
    assert len(frame) == 16 + 7
    header = fmt.parse_frame_header(frame)
    assert (header.kind, header.length, header.frame_len) == (1, 7, 23)
    fmt.check_frame_bytes(frame)
    damaged = frame[:-5] + b"X" + frame[-4:]
    with pytest.raises(CorruptedError, match="checksum"):
        fmt.check_frame_bytes(damaged)


def test_frame_header_validation() -> None:
    good = fmt.encode_frame(fmt.FrameKind.META, b"")
    with pytest.raises(CorruptedError, match="synchronisation"):
        fmt.parse_frame_header(b"XBLK" + good[4:])
    with pytest.raises(CorruptedError, match="reserved"):
        fmt.parse_frame_header(good[:5] + b"\x01" + good[6:])
    too_long = struct.pack("<4sBBHI", b"TBLK", 1, 0, 0, (1 << 30) + 1)
    with pytest.raises(CorruptedError, match="exceeds"):
        fmt.parse_frame_header(too_long)


def test_frame_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fmt, "MAX_PAYLOAD", 4)
    with pytest.raises(LimitExceededError):
        fmt.encode_frame(fmt.FrameKind.META, b"12345")


def test_frame_kinds() -> None:
    assert fmt.frame_kind(0x03) is fmt.FrameKind.DATA
    assert fmt.frame_kind(0x9A) == 0x9A  # ancillary
    with pytest.raises(UnsupportedError, match="critical"):
        fmt.frame_kind(0x09)
    assert fmt.is_derived(fmt.FrameKind.INDEX)
    assert not fmt.is_derived(0x9A)


def test_meta_round_trip() -> None:
    meta = {"title": "Ферубко", "b": ""}
    assert fmt.decode_meta(fmt.encode_meta(meta)) == meta
    with pytest.raises(CorruptedError):
        fmt.decode_meta(fmt.encode_meta(meta) + b"\x00")


def test_device_round_trip() -> None:
    schema = DeviceSchema(
        "vm01",
        "us",
        (ColumnSchema("cpu", DataType.FLOAT32, {"unit": "%"}), ColumnSchema("s", DataType.STRING)),
        "Europe/Moscow",
        "ts",
        {"site": "Брянск"},
    )
    assert fmt.decode_device(fmt.encode_device(3, schema)) == (3, schema)


def test_device_validation_errors_are_corruption() -> None:
    bad = DeviceSchema("x", "s", (("a", "bool"), ("a", "bool")))
    with pytest.raises(CorruptedError, match="duplicate"):
        fmt.decode_device(fmt.encode_device(0, bad))
    payload = bytearray(fmt.encode_device(0, DeviceSchema("x", "s", (("a", "bool"),))))
    payload[3] = 9  # time unit code
    with pytest.raises(UnsupportedError):
        fmt.decode_device(bytes(payload))


def test_tombstone_round_trip() -> None:
    ranges = [(1, 5), (-(2**63), 2**63 - 1)]
    assert fmt.decode_tombstone(fmt.encode_tombstone(2, ranges)) == (2, ranges)
    with pytest.raises(CorruptedError, match="without ranges"):
        fmt.decode_tombstone(fmt.encode_tombstone(0, []))
    with pytest.raises(CorruptedError, match="start after end"):
        fmt.decode_tombstone(fmt.encode_tombstone(0, [(5, 1)]))


def test_annotation_round_trip() -> None:
    ops = [
        fmt.AnnotationOp(1, False, 0, 10, 20, "spike", "заметка"),
        fmt.AnnotationOp(2, False, 1, 5, 5, "event", None),
        fmt.AnnotationOp(1, remove=True),
    ]
    assert fmt.decode_annotation_ops(fmt.encode_annotation_ops(ops)) == ops


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\x00", "without operations"),
        (b"\x01\x01\x00", "identifier 0"),
        (b"\x01\x03\x01", "unknown annotation operation"),
    ],
)
def test_annotation_errors(payload: bytes, message: str) -> None:
    with pytest.raises(CorruptedError, match=message):
        fmt.decode_annotation_ops(payload)


def test_annotation_interval_check() -> None:
    payload = fmt.encode_annotation_ops([fmt.AnnotationOp(1, False, 0, 9, 1, "x")])
    with pytest.raises(CorruptedError, match="start after end"):
        fmt.decode_annotation_ops(payload)


def test_commit_round_trip_and_hash() -> None:
    record = fmt.CommitRecord(
        1, 123, 8, 2, fmt.ZERO_HASH, b"\x01" * 32, "автор", "сообщение"
    ).seal()
    assert record.hash != fmt.ZERO_HASH
    assert fmt.CommitRecord.decode(record.encode()) == record
    tampered = bytearray(record.encode())
    tampered[9] ^= 1
    with pytest.raises(CorruptedError, match="hash"):
        fmt.CommitRecord.decode(bytes(tampered))


def segment(data_type: DataType | None = None, **changes: object) -> fmt.SegmentHeader:
    stats = (
        fmt.Statistics(0.5, 2.5, is_float=True)
        if data_type in (DataType.FLOAT32, DataType.FLOAT64)
        else fmt.Statistics(-1, 7)
    )
    fields = {
        "encoding": Encoding.PLAIN,
        "codec": Codec.NONE,
        "has_validity": False,
        "null_count": 0,
        "stored_len": 4,
        "raw_len": 4,
        "crc": 99,
        "stats": None if data_type is DataType.STRING else stats,
    } | changes
    return fmt.SegmentHeader(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("data_type", [None, *DataType])
def test_segment_header_round_trip(data_type: DataType | None) -> None:
    header = segment(data_type)
    w = ByteWriter()
    header.write(w)
    r = ByteReader(w.getvalue())
    assert fmt.SegmentHeader.read(r, data_type) == header
    r.finish("test")


@pytest.mark.parametrize(
    ("data_type", "changes", "message"),
    [
        (None, {"encoding": Encoding.XOR}, "not allowed"),
        (None, {"has_validity": True, "null_count": 1}, "cannot contain nulls"),
        (DataType.INT64, {"has_validity": True}, "does not match"),
        (DataType.INT64, {"raw_len": (1 << 30) + 1}, "exceeds the limit"),
    ],
)
def test_segment_header_errors(
    data_type: DataType | None, changes: dict[str, object], message: str
) -> None:
    w = ByteWriter()
    segment(data_type, **changes).write(w)
    with pytest.raises(CorruptedError, match=message):
        fmt.SegmentHeader.read(ByteReader(w.getvalue()), data_type)


def test_segment_header_flag_and_string_stats_errors() -> None:
    w = ByteWriter()
    segment(DataType.INT64).write(w)
    data = bytearray(w.getvalue())
    data[2] |= 4
    with pytest.raises(CorruptedError, match="flags"):
        fmt.SegmentHeader.read(ByteReader(bytes(data)), DataType.INT64)
    w = ByteWriter()
    segment(DataType.INT64).write(w)
    with pytest.raises(CorruptedError, match="no statistics"):
        fmt.SegmentHeader.read(ByteReader(w.getvalue()), DataType.STRING)


def data_header(**changes: object) -> fmt.DataHeader:
    fields = {
        "device_id": 0,
        "row_count": 2,
        "t_min": 1,
        "t_max": 2,
        "segments": (segment(), segment(DataType.INT64)),
    } | changes
    return fmt.DataHeader(**fields)  # type: ignore[arg-type]


def test_data_header_round_trip() -> None:
    header = data_header()
    assert fmt.DataHeader.decode(header.encode(), [[DataType.INT64]]) == header
    assert header.segments_len == 8
    payload, header_bytes = fmt.encode_data_payload(header, [b"aaaa", b"bbbb"])
    assert header_bytes == header.encode()
    assert payload.endswith(b"aaaabbbb")


@pytest.mark.parametrize(
    ("changes", "types", "message"),
    [
        ({"device_id": 1}, [[DataType.INT64]], "unknown device"),
        ({"row_count": 0}, [[DataType.INT64]], "invalid row count"),
        ({"row_count": (1 << 24) + 1}, [[DataType.INT64]], "invalid row count"),
        ({"t_min": 3}, [[DataType.INT64]], "inconsistent"),
        ({"t_min": 2}, [[DataType.INT64]], "inconsistent"),
        ({"row_count": 1}, [[DataType.INT64]], "inconsistent"),
        ({}, [[DataType.INT64, DataType.BOOL]], "segments"),
        (
            {"segments": (segment(), segment(DataType.INT64, has_validity=True, null_count=3))},
            [[DataType.INT64]],
            "null count exceeds",
        ),
    ],
)
def test_data_header_errors(
    changes: dict[str, object], types: list[list[DataType]], message: str
) -> None:
    with pytest.raises(CorruptedError, match=message):
        fmt.DataHeader.decode(data_header(**changes).encode(), types)


def test_index_and_footer_round_trip() -> None:
    entries = [fmt.IndexEntry(1, 8, b"abc"), fmt.IndexEntry(0x90, 40, b"")]
    head, head_hash, decoded = fmt.decode_index(fmt.encode_index(2, b"\x05" * 32, entries))
    assert (head, head_hash, decoded) == (2, b"\x05" * 32, entries)
    assert fmt.decode_footer(fmt.encode_footer(1234)) == 1234
    with pytest.raises(CorruptedError):
        fmt.decode_footer(b"\x00" * 9)
