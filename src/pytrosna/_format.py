"""The physical layout: file header, frames and frame payloads (SPEC sections 2 to 8)."""

from __future__ import annotations

import enum
import hashlib
import struct
from dataclasses import dataclass, field, replace

from ._bytes import ByteReader, ByteWriter
from ._checksum import crc32c
from .codecs import Codec
from .encodings import MAX_RAW_SEGMENT, Encoding
from .errors import (
    CorruptedError,
    LimitExceededError,
    NotTrosnaError,
    SchemaError,
    UnsupportedError,
    UnsupportedVersionError,
)
from .types import MAX_COLUMNS, ColumnSchema, DataType, DeviceSchema, TimeUnit

MAGIC = b"TROSNA"
VERSION_MAJOR = 1
VERSION_MINOR = 0
FILE_HEADER_LEN = 8
SYNC = b"TBLK"
FRAME_HEADER_LEN = 12
FRAME_OVERHEAD = 16
MAX_PAYLOAD = 1 << 30
FOOTER_FRAME_LEN = FRAME_OVERHEAD + 8
MAX_ROWS = 1 << 24
HASH_LEN = 32
ZERO_HASH = bytes(HASH_LEN)

FLAG_VALIDITY = 1
FLAG_STATS = 2

_FRAME_HEADER = struct.Struct("<4sBBHI")
_COMMIT_DOMAIN = b"trosna/commit/v1\x00"


class FrameKind(enum.IntEnum):
    """Kind of a frame (SPEC §3)."""

    META = 0x01
    DEVICE = 0x02
    DATA = 0x03
    TOMBSTONE = 0x04
    ANNOTATION = 0x05
    COMMIT = 0x06
    INDEX = 0x07
    FOOTER = 0x08

    @property
    def is_derived(self) -> bool:
        """``Index`` and ``Footer`` frames are not part of commits."""
        return self in (FrameKind.INDEX, FrameKind.FOOTER)


def frame_kind(code: int) -> FrameKind | int:
    """The kind of a frame code: a :class:`FrameKind`, or the code itself for
    an ancillary frame (high bit set). Unknown critical kinds are an error."""
    try:
        return FrameKind(code)
    except ValueError:
        if code & 0x80:
            return code
        msg = f"critical frame kind {code:#04x}"
        raise UnsupportedError(msg) from None


def is_derived(kind: FrameKind | int) -> bool:
    return isinstance(kind, FrameKind) and kind.is_derived


def file_header() -> bytes:
    return MAGIC + bytes((VERSION_MAJOR, VERSION_MINOR))


def check_file_header(data: bytes) -> tuple[int, int]:
    """Validates a file header and returns ``(major, minor)``."""
    if len(data) < FILE_HEADER_LEN or data[:6] != MAGIC:
        raise NotTrosnaError
    major, minor = data[6], data[7]
    if major != VERSION_MAJOR:
        raise UnsupportedVersionError(major, minor)
    return major, minor


def frame_crc(header: bytes, payload: bytes) -> int:
    """CRC32C over the header bytes after the sync marker and the payload."""
    return crc32c(payload, crc32c(header[4:FRAME_HEADER_LEN]))


def encode_frame(kind: int, payload: bytes) -> bytes:
    """Serialises a complete frame (sync … crc)."""
    if len(payload) > MAX_PAYLOAD:
        msg = f"a frame of {len(payload)} bytes exceeds 1 GiB"
        raise LimitExceededError(msg)
    header = _FRAME_HEADER.pack(SYNC, kind, 0, 0, len(payload))
    return header + payload + struct.pack("<I", frame_crc(header, payload))


@dataclass(frozen=True)
class FrameHeader:
    kind: int
    length: int

    @property
    def frame_len(self) -> int:
        return self.length + FRAME_OVERHEAD


def parse_frame_header(data: bytes) -> FrameHeader:
    """Parses and validates a frame header (not the CRC)."""
    sync, kind, flags, reserved, length = _FRAME_HEADER.unpack(data[:FRAME_HEADER_LEN])
    if sync != SYNC:
        msg = "missing frame synchronisation marker"
        raise CorruptedError(msg)
    if flags or reserved:
        msg = "non-zero reserved bytes in frame header"
        raise CorruptedError(msg)
    if length > MAX_PAYLOAD:
        msg = f"frame length {length} exceeds the limit"
        raise CorruptedError(msg)
    return FrameHeader(kind, length)


def check_frame_bytes(frame: bytes) -> None:
    """Checks the CRC of complete frame bytes."""
    stored = struct.unpack("<I", frame[-4:])[0]
    if frame_crc(frame[:FRAME_HEADER_LEN], frame[FRAME_HEADER_LEN:-4]) != stored:
        msg = "frame checksum mismatch"
        raise CorruptedError(msg)


# ---------------------------------------------------------------- Meta


def encode_meta(metadata: dict[str, str]) -> bytes:
    w = ByteWriter()
    w.kv_map(metadata)
    return w.getvalue()


def decode_meta(payload: bytes) -> dict[str, str]:
    r = ByteReader(payload)
    out = r.kv_map()
    r.finish("Meta frame")
    return out


# ---------------------------------------------------------------- Device


def encode_device(device_id: int, schema: DeviceSchema) -> bytes:
    w = ByteWriter()
    w.uvarint(device_id)
    w.string(schema.name)
    w.u8(schema.time_unit)
    w.opt_string(schema.timezone)
    w.string(schema.time_name)
    w.uvarint(len(schema.columns))
    for column in schema.columns:
        w.string(column.name)
        w.u8(column.data_type)
        w.kv_map(column.metadata)
    w.kv_map(schema.metadata)
    return w.getvalue()


def decode_device(payload: bytes) -> tuple[int, DeviceSchema]:
    r = ByteReader(payload)
    device_id = r.uvarint_u32()
    name = r.string()
    time_unit = TimeUnit.from_code(r.u8())
    timezone = r.opt_string()
    time_name = r.string()
    count = r.length(MAX_COLUMNS, "column count")
    columns = []
    for _ in range(count):
        column_name = r.string()
        data_type = DataType.from_code(r.u8())
        columns.append(ColumnSchema(column_name, data_type, r.kv_map()))
    metadata = r.kv_map()
    r.finish("Device frame")
    schema = DeviceSchema(name, time_unit, tuple(columns), timezone, time_name, metadata)
    try:
        schema.validate()
    except SchemaError as e:
        raise CorruptedError(str(e)) from None
    return device_id, schema


# ---------------------------------------------------------------- segments


@dataclass(frozen=True)
class Statistics:
    """Minimum and maximum of the non-null values of a segment.

    ``min`` and ``max`` are integers for time, bool (0/1) and integer columns
    and floats for floating-point columns (NaN values are ignored).
    """

    min: int | float
    max: int | float
    is_float: bool = False


@dataclass(frozen=True)
class SegmentHeader:
    """Header of a segment as stored in a ``Data`` frame (SPEC §5.3)."""

    encoding: Encoding
    codec: Codec
    has_validity: bool
    null_count: int
    stored_len: int
    raw_len: int
    crc: int
    stats: Statistics | None = None

    def write(self, w: ByteWriter) -> None:
        w.u8(self.encoding)
        w.u8(self.codec)
        flags = (FLAG_VALIDITY if self.has_validity else 0) | (
            FLAG_STATS if self.stats is not None else 0
        )
        w.u8(flags)
        w.uvarint(self.null_count)
        w.uvarint(self.stored_len)
        w.uvarint(self.raw_len)
        w.u32(self.crc)
        if self.stats is not None:
            if self.stats.is_float:
                w.raw(struct.pack("<dd", self.stats.min, self.stats.max))
            else:
                w.i64(int(self.stats.min))
                w.i64(int(self.stats.max))

    @classmethod
    def read(cls, r: ByteReader, data_type: DataType | None) -> SegmentHeader:
        """Parses a header; ``data_type`` is ``None`` for the time column."""
        encoding = Encoding.from_code(r.u8())
        encoding.check_allowed(data_type)
        codec = Codec.from_code(r.u8())
        flags = r.u8()
        if flags & ~(FLAG_VALIDITY | FLAG_STATS):
            msg = f"unknown segment flags {flags:#x}"
            raise CorruptedError(msg)
        null_count = r.uvarint_u32()
        stored_len = r.uvarint_u32()
        raw_len = r.uvarint_u32()
        if raw_len > MAX_RAW_SEGMENT:
            msg = f"segment of {raw_len} bytes exceeds the limit"
            raise CorruptedError(msg)
        crc = r.u32()
        stats = None
        if flags & FLAG_STATS:
            if data_type in (DataType.FLOAT32, DataType.FLOAT64):
                low, high = struct.unpack("<dd", r.take(16))
                stats = Statistics(low, high, is_float=True)
            elif data_type is DataType.STRING:
                msg = "string segments have no statistics"
                raise CorruptedError(msg)
            else:
                stats = Statistics(r.i64(), r.i64())
        header = cls(
            encoding,
            codec,
            bool(flags & FLAG_VALIDITY),
            null_count,
            stored_len,
            raw_len,
            crc,
            stats,
        )
        if data_type is None and (header.null_count or header.has_validity):
            msg = "the time column cannot contain nulls"
            raise CorruptedError(msg)
        if header.has_validity != (header.null_count > 0):
            msg = "validity flag does not match the null count"
            raise CorruptedError(msg)
        return header


# ---------------------------------------------------------------- Data


@dataclass(frozen=True)
class DataHeader:
    """Header of a ``Data`` frame: the time segment first, then the value columns."""

    device_id: int
    row_count: int
    t_min: int
    t_max: int
    segments: tuple[SegmentHeader, ...]

    def encode(self) -> bytes:
        w = ByteWriter()
        w.uvarint(self.device_id)
        w.uvarint(self.row_count)
        w.i64(self.t_min)
        w.i64(self.t_max)
        w.uvarint(len(self.segments))
        for segment in self.segments:
            segment.write(w)
        return w.getvalue()

    @classmethod
    def decode(cls, data: bytes, column_types: list[list[DataType]]) -> DataHeader:
        """Parses a header; ``column_types[device_id]`` are the value column types."""
        r = ByteReader(data)
        device_id = r.uvarint_u32()
        if device_id >= len(column_types):
            msg = f"Data frame for unknown device {device_id}"
            raise CorruptedError(msg)
        types = column_types[device_id]
        row_count = r.uvarint_u32()
        if row_count == 0 or row_count > MAX_ROWS:
            msg = f"invalid row count {row_count}"
            raise CorruptedError(msg)
        t_min = r.i64()
        t_max = r.i64()
        if t_min > t_max or (row_count == 1) != (t_min == t_max):
            msg = "inconsistent time range of a Data frame"
            raise CorruptedError(msg)
        count = r.length(MAX_COLUMNS + 1, "segment count")
        if count != len(types) + 1:
            msg = f"Data frame has {count} segments, the device has {len(types)} columns"
            raise CorruptedError(msg)
        segments = [SegmentHeader.read(r, None)]
        for data_type in types:
            segment = SegmentHeader.read(r, data_type)
            if segment.null_count > row_count:
                msg = "null count exceeds the row count"
                raise CorruptedError(msg)
            segments.append(segment)
        r.finish("Data frame header")
        return cls(device_id, row_count, t_min, t_max, tuple(segments))

    @property
    def segments_len(self) -> int:
        return sum(s.stored_len for s in self.segments)


def encode_data_payload(header: DataHeader, segments: list[bytes]) -> tuple[bytes, bytes]:
    """The payload of a ``Data`` frame and its header bytes."""
    header_bytes = header.encode()
    return struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(segments), header_bytes


# ---------------------------------------------------------------- Tombstone


def encode_tombstone(device_id: int, ranges: list[tuple[int, int]]) -> bytes:
    w = ByteWriter()
    w.uvarint(device_id)
    w.uvarint(len(ranges))
    for start, end in ranges:
        w.i64(start)
        w.i64(end)
    return w.getvalue()


def decode_tombstone(payload: bytes) -> tuple[int, list[tuple[int, int]]]:
    r = ByteReader(payload)
    device_id = r.uvarint_u32()
    count = r.length(r.remaining // 16, "range count")
    if count == 0:
        msg = "tombstone without ranges"
        raise CorruptedError(msg)
    ranges = []
    for _ in range(count):
        start, end = r.i64(), r.i64()
        if start > end:
            msg = "tombstone range with start after end"
            raise CorruptedError(msg)
        ranges.append((start, end))
    r.finish("Tombstone frame")
    return device_id, ranges


# ---------------------------------------------------------------- Annotation


@dataclass(frozen=True)
class AnnotationOp:
    """An annotation operation: an upsert (``remove`` false) or a removal."""

    id: int
    remove: bool = False
    device_id: int = 0
    start: int = 0
    end: int = 0
    label: str = ""
    note: str | None = None


def encode_annotation_ops(ops: list[AnnotationOp]) -> bytes:
    w = ByteWriter()
    w.uvarint(len(ops))
    for op in ops:
        if op.remove:
            w.u8(2)
            w.uvarint(op.id)
        else:
            w.u8(1)
            w.uvarint(op.id)
            w.uvarint(op.device_id)
            w.i64(op.start)
            w.i64(op.end)
            w.string(op.label)
            w.opt_string(op.note)
    return w.getvalue()


def decode_annotation_ops(payload: bytes) -> list[AnnotationOp]:
    r = ByteReader(payload)
    count = r.length(r.remaining // 2, "annotation operation count")
    if count == 0:
        msg = "annotation frame without operations"
        raise CorruptedError(msg)
    ops = []
    for _ in range(count):
        tag = r.u8()
        annotation_id = r.uvarint()
        if annotation_id == 0:
            msg = "annotation identifier 0"
            raise CorruptedError(msg)
        if tag == 1:
            device_id = r.uvarint_u32()
            start, end = r.i64(), r.i64()
            if start > end:
                msg = "annotation with start after end"
                raise CorruptedError(msg)
            label = r.string()
            note = r.opt_string()
            ops.append(AnnotationOp(annotation_id, False, device_id, start, end, label, note))
        elif tag == 2:
            ops.append(AnnotationOp(annotation_id, remove=True))
        else:
            msg = f"unknown annotation operation {tag}"
            raise CorruptedError(msg)
    r.finish("Annotation frame")
    return ops


# ---------------------------------------------------------------- Commit


@dataclass(frozen=True)
class CommitRecord:
    """The payload of a ``Commit`` frame (SPEC §4)."""

    number: int
    time: int
    first_offset: int
    frame_count: int
    prev_hash: bytes
    content_hash: bytes
    author: str | None = None
    message: str | None = None
    hash: bytes = field(default=ZERO_HASH)

    def body(self) -> bytes:
        w = ByteWriter()
        w.u64(self.number)
        w.i64(self.time)
        w.u64(self.first_offset)
        w.u32(self.frame_count)
        w.raw(self.prev_hash)
        w.raw(self.content_hash)
        w.opt_string(self.author)
        w.opt_string(self.message)
        return w.getvalue()

    @staticmethod
    def compute_hash(body: bytes) -> bytes:
        return hashlib.sha256(_COMMIT_DOMAIN + body).digest()

    def seal(self) -> CommitRecord:
        """A copy with ``hash`` computed from the other fields."""
        return replace(self, hash=self.compute_hash(self.body()))

    def encode(self) -> bytes:
        return self.body() + self.hash

    @classmethod
    def decode(cls, payload: bytes) -> CommitRecord:
        """Parses a commit and checks its ``commit_hash``."""
        r = ByteReader(payload)
        number = r.u64()
        time = r.i64()
        first_offset = r.u64()
        frame_count = r.u32()
        prev_hash = r.take(HASH_LEN)
        content_hash = r.take(HASH_LEN)
        author = r.opt_string()
        message = r.opt_string()
        body_len = r.position
        commit_hash = r.take(HASH_LEN)
        r.finish("Commit frame")
        if cls.compute_hash(payload[:body_len]) != commit_hash:
            msg = "commit hash does not match its content"
            raise CorruptedError(msg)
        return cls(
            number,
            time,
            first_offset,
            frame_count,
            prev_hash,
            content_hash,
            author,
            message,
            commit_hash,
        )


# ---------------------------------------------------------------- Index and Footer


@dataclass(frozen=True)
class IndexEntry:
    kind: int
    offset: int
    body: bytes


def encode_index(head_commit: int, head_hash: bytes, entries: list[IndexEntry]) -> bytes:
    w = ByteWriter()
    w.u64(head_commit)
    w.raw(head_hash)
    w.uvarint(len(entries))
    for entry in entries:
        w.u8(entry.kind)
        w.u64(entry.offset)
        w.uvarint(len(entry.body))
        w.raw(entry.body)
    return w.getvalue()


def decode_index(payload: bytes) -> tuple[int, bytes, list[IndexEntry]]:
    r = ByteReader(payload)
    head = r.u64()
    head_hash = r.take(HASH_LEN)
    # Every entry takes at least 10 bytes.
    count = r.length(r.remaining // 10, "index entry count")
    entries = []
    for _ in range(count):
        kind = r.u8()
        offset = r.u64()
        size = r.length(r.remaining, "index entry length")
        entries.append(IndexEntry(kind, offset, r.take(size)))
    r.finish("Index frame")
    return head, head_hash, entries


def encode_footer(index_offset: int) -> bytes:
    return struct.pack("<Q", index_offset)


def decode_footer(payload: bytes) -> int:
    r = ByteReader(payload)
    offset = r.u64()
    r.finish("Footer frame")
    return offset
