"""The catalogue: the in-memory description of every log frame of a file
except the column data. It is built from an ``Index`` frame or by scanning,
and maintained incrementally by the writer."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from ._format import (
    FILE_HEADER_LEN,
    FRAME_HEADER_LEN,
    ZERO_HASH,
    AnnotationOp,
    CommitRecord,
    DataHeader,
    FrameKind,
    IndexEntry,
    decode_annotation_ops,
    decode_device,
    decode_meta,
    decode_tombstone,
    encode_annotation_ops,
    encode_device,
    encode_meta,
    encode_tombstone,
    frame_kind,
)
from .errors import CorruptedError, UnknownCommitError, UnknownDeviceError
from .types import DataType, DeviceSchema


@dataclass
class DeviceInfo:
    schema: DeviceSchema
    types: list[DataType]
    offset: int
    data: list[int] = field(default_factory=list)
    """Indices into :attr:`Catalog.data`, in file order."""
    tombstones: list[int] = field(default_factory=list)
    """Indices into :attr:`Catalog.tombstones`, in file order."""


@dataclass(frozen=True)
class DataEntry:
    offset: int
    header: DataHeader
    segment_offsets: tuple[int, ...]
    """Absolute file offsets of the segments."""


@dataclass(frozen=True)
class TombstoneEntry:
    offset: int
    device_id: int
    ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class AnnotationEntry:
    offset: int
    ops: tuple[AnnotationOp, ...]


@dataclass(frozen=True)
class CommitEntry:
    offset: int
    record: CommitRecord


@dataclass(frozen=True)
class MetaEntry:
    offset: int
    metadata: dict[str, str]


@dataclass(frozen=True)
class AnnotationState:
    device_id: int
    start: int
    end: int
    label: str
    note: str | None


# Kinds of the items of Catalog.order
_META, _DEVICE, _DATA, _TOMBSTONE, _ANNOTATION, _COMMIT, _ANCILLARY = range(7)


class Catalog:
    """The catalogue of a file."""

    def __init__(self) -> None:
        self.order: list[tuple[int, int, int, bytes]] = []
        """``(offset, item kind, frame kind code, ancillary body)`` in file order."""
        self.metas: list[MetaEntry] = []
        self.devices: list[DeviceInfo] = []
        self.by_name: dict[str, int] = {}
        self.data: list[DataEntry] = []
        self.tombstones: list[TombstoneEntry] = []
        self.annotations: list[AnnotationEntry] = []
        self.commits: list[CommitEntry] = []
        self.live_annotations: dict[int, AnnotationState] = {}
        self.max_annotation_id = 0
        self.pending_count = 0
        self.pending_first: int | None = None

    # ------------------------------------------------------------ building

    def apply(self, kind: int, offset: int, body: bytes, payload_len: int | None) -> None:
        """Adds a log frame. ``payload_len`` is known when scanning and lets
        ``Data`` frames be checked against their frame length."""
        try:
            self._apply(kind, offset, body, payload_len)
        except CorruptedError as e:
            raise e.at(offset) from None

    def _apply(self, code: int, offset: int, body: bytes, payload_len: int | None) -> None:
        if self.order and offset <= self.order[-1][0]:
            msg = "frames are not in file order"
            raise CorruptedError(msg)
        kind = frame_kind(code)
        item: int
        if kind is FrameKind.META:
            self.metas.append(MetaEntry(offset, decode_meta(body)))
            item = _META
        elif kind is FrameKind.DEVICE:
            device_id, schema = decode_device(body)
            if device_id != len(self.devices):
                msg = f"device id {device_id} out of sequence (expected {len(self.devices)})"
                raise CorruptedError(msg)
            if schema.name in self.by_name:
                msg = f"duplicate device {schema.name!r}"
                raise CorruptedError(msg)
            self.by_name[schema.name] = device_id
            self.devices.append(DeviceInfo(schema, schema.types, offset))
            item = _DEVICE
        elif kind is FrameKind.DATA:
            header = DataHeader.decode(body, [d.types for d in self.devices])
            if payload_len is not None and 4 + len(body) + header.segments_len != payload_len:
                msg = "segment lengths do not match the frame length"
                raise CorruptedError(msg)
            position = offset + FRAME_HEADER_LEN + 4 + len(body)
            offsets = []
            for segment in header.segments:
                offsets.append(position)
                position += segment.stored_len
            self.devices[header.device_id].data.append(len(self.data))
            self.data.append(DataEntry(offset, header, tuple(offsets)))
            item = _DATA
        elif kind is FrameKind.TOMBSTONE:
            device_id, ranges = decode_tombstone(body)
            if device_id >= len(self.devices):
                msg = f"tombstone for unknown device {device_id}"
                raise CorruptedError(msg)
            self.devices[device_id].tombstones.append(len(self.tombstones))
            self.tombstones.append(TombstoneEntry(offset, device_id, tuple(ranges)))
            item = _TOMBSTONE
        elif kind is FrameKind.ANNOTATION:
            ops = decode_annotation_ops(body)
            for op in ops:
                self._apply_annotation_op(op)
            self.annotations.append(AnnotationEntry(offset, tuple(ops)))
            item = _ANNOTATION
        elif kind is FrameKind.COMMIT:
            record = CommitRecord.decode(body)
            self._check_commit(offset, record)
            self.commits.append(CommitEntry(offset, record))
            self.pending_count = 0
            self.pending_first = None
            self.order.append((offset, _COMMIT, code, b""))
            return
        elif isinstance(kind, FrameKind):  # Index or Footer
            msg = "derived frame listed as a log frame"
            raise CorruptedError(msg)
        else:
            self.order.append((offset, _ANCILLARY, code, bytes(body)))
            self._pending(offset)
            return
        self.order.append((offset, item, code, b""))
        self._pending(offset)

    def _pending(self, offset: int) -> None:
        self.pending_count += 1
        if self.pending_first is None:
            self.pending_first = offset

    def _apply_annotation_op(self, op: AnnotationOp) -> None:
        if op.remove:
            if self.live_annotations.pop(op.id, None) is None:
                msg = f"removal of absent annotation {op.id}"
                raise CorruptedError(msg)
            return
        if op.device_id >= len(self.devices):
            msg = f"annotation for unknown device {op.device_id}"
            raise CorruptedError(msg)
        self.live_annotations[op.id] = AnnotationState(
            op.device_id, op.start, op.end, op.label, op.note
        )
        self.max_annotation_id = max(self.max_annotation_id, op.id)

    def _check_commit(self, offset: int, record: CommitRecord) -> None:
        expected = len(self.commits) + 1
        if record.number != expected:
            msg = f"commit number {record.number} out of sequence (expected {expected})"
            raise CorruptedError(msg)
        if self.commits and record.prev_hash != self.commits[-1].record.hash:
            msg = "commit does not link to the previous commit"
            raise CorruptedError(msg)
        first = offset if self.pending_first is None else self.pending_first
        if record.frame_count != self.pending_count or record.first_offset != first:
            msg = "commit does not describe the frames it seals"
            raise CorruptedError(msg)

    def truncate(self, offset: int) -> None:
        """Removes all frames at or after ``offset``."""
        while self.order and self.order[-1][0] >= offset:
            _, item, _, _ = self.order.pop()
            if item == _META:
                self.metas.pop()
            elif item == _DEVICE:
                device = self.devices.pop()
                del self.by_name[device.schema.name]
            elif item == _DATA:
                entry = self.data.pop()
                self.devices[entry.header.device_id].data.pop()
            elif item == _TOMBSTONE:
                tombstone = self.tombstones.pop()
                self.devices[tombstone.device_id].tombstones.pop()
            elif item == _ANNOTATION:
                self.annotations.pop()
            elif item == _COMMIT:
                self.commits.pop()
        # Rebuild the derived state; the operations were valid in this order.
        self.live_annotations = {}
        self.max_annotation_id = 0
        for annotation in self.annotations:
            for op in annotation.ops:
                self._apply_annotation_op(op)
        last_commit = self.commits[-1].offset if self.commits else 0
        after = [pos for pos, item, _, _ in self.order if pos > last_commit and item != _COMMIT]
        self.pending_count = len(after)
        self.pending_first = after[0] if after else None

    @classmethod
    def from_index(cls, entries: list[IndexEntry]) -> Catalog:
        """Builds a catalogue from the entries of an ``Index`` frame."""
        catalog = cls()
        for entry in entries:
            catalog.apply(entry.kind, entry.offset, entry.body, None)
        if catalog.pending_count:
            msg = "the index lists uncommitted frames"
            raise CorruptedError(msg)
        return catalog

    def index_entries(self) -> list[IndexEntry]:
        """Entries of an ``Index`` frame describing this catalogue."""
        counters = [0] * 7
        out = []
        for offset, item, code, body in self.order:
            i = counters[item]
            counters[item] += 1
            if item == _META:
                payload = encode_meta(self.metas[i].metadata)
            elif item == _DEVICE:
                payload = encode_device(i, self.devices[i].schema)
            elif item == _DATA:
                payload = self.data[i].header.encode()
            elif item == _TOMBSTONE:
                t = self.tombstones[i]
                payload = encode_tombstone(t.device_id, list(t.ranges))
            elif item == _ANNOTATION:
                payload = encode_annotation_ops(list(self.annotations[i].ops))
            elif item == _COMMIT:
                payload = self.commits[i].record.encode()
            else:
                payload = body
            out.append(IndexEntry(code, offset, payload))
        return out

    # ------------------------------------------------------------ queries

    def head(self) -> tuple[int, bytes]:
        """Number and hash of the last commit (``0`` and zeros if there is none)."""
        if not self.commits:
            return 0, ZERO_HASH
        record = self.commits[-1].record
        return record.number, record.hash

    def device_id(self, name: str) -> int:
        try:
            return self.by_name[name]
        except KeyError:
            raise UnknownDeviceError(name) from None

    def limit(self, as_of: int | None) -> int:
        """The exclusive offset bound of version ``as_of`` (``None``: everything)."""
        if as_of is None:
            return 1 << 64
        if as_of == 0:
            return FILE_HEADER_LEN
        if as_of < 0 or as_of > len(self.commits):
            raise UnknownCommitError(as_of)
        return self.commits[as_of - 1].offset

    def metadata_at(self, limit: int) -> dict[str, str]:
        for meta in reversed(self.metas):
            if meta.offset < limit:
                return dict(meta.metadata)
        return {}

    def annotations_at(self, limit: int) -> dict[int, AnnotationState]:
        state: dict[int, AnnotationState] = {}
        for entry in self.annotations:
            if entry.offset >= limit:
                break
            for op in entry.ops:
                if op.remove:
                    state.pop(op.id, None)
                else:
                    state[op.id] = AnnotationState(
                        op.device_id, op.start, op.end, op.label, op.note
                    )
        return dict(sorted(state.items()))

    def commit_of(self, offset: int) -> int | None:
        """Number of the commit whose frames contain ``offset``."""
        offsets = [c.offset for c in self.commits]
        i = bisect.bisect_left(offsets, offset)
        return self.commits[i].record.number if i < len(self.commits) else None
