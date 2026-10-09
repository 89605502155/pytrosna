"""History of a file: commits, annotations and differences between versions."""

from __future__ import annotations

import datetime as dt
import enum
import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, BinaryIO

from ._read import ReadSpec, execute_all, merge_intervals, plan
from .errors import InvalidArgumentError

if TYPE_CHECKING:
    from ._catalog import AnnotationState, Catalog
    from .column import Batch

__all__ = [
    "Annotation",
    "AnnotationChange",
    "ChangeKind",
    "CommitChanges",
    "CommitInfo",
    "Diff",
    "PointChange",
]

_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)


@dataclass(frozen=True)
class Annotation:
    """A labelled closed interval ``[start, end]`` on the time axis of a device.

    ``start`` and ``end`` are raw time stamps in the device's unit;
    ``start == end`` marks an instant event.
    """

    id: int
    device: str
    start: int
    end: int
    label: str
    note: str | None = None

    @classmethod
    def from_state(cls, catalog: Catalog, annotation_id: int, state: AnnotationState) -> Annotation:
        return cls(
            annotation_id,
            catalog.devices[state.device_id].schema.name,
            state.start,
            state.end,
            state.label,
            state.note,
        )


@dataclass(frozen=True)
class CommitChanges:
    """Summary of what a commit changed."""

    devices_created: tuple[str, ...] = ()
    blocks_written: int = 0
    rows_written: int = 0
    """Rows written, including rows that replace existing points."""
    ranges_deleted: int = 0
    annotation_ops: int = 0
    metadata_changed: bool = False


@dataclass(frozen=True)
class CommitInfo:
    """A commit: an atomic, hashed set of changes."""

    number: int
    """Sequence number, starting at 1."""
    time_ns: int
    """Wall-clock time of the commit, nanoseconds since the Unix epoch."""
    author: str | None
    message: str | None
    hash: str
    """SHA-256 of the commit as 64 hexadecimal digits; identifies the whole
    history up to this commit."""
    prev_hash: str
    """Hash of the previous commit (or of the origin of a compacted file)."""
    offset: int = 0
    """Offset of the ``Commit`` frame in the file."""
    changes: CommitChanges = field(default_factory=CommitChanges)

    @property
    def time(self) -> dt.datetime:
        """The commit time as an aware ``datetime`` in UTC (microsecond precision)."""
        return _EPOCH + dt.timedelta(microseconds=self.time_ns // 1000)

    @property
    def short_hash(self) -> str:
        """The first 12 hexadecimal digits of the hash."""
        return self.hash[:12]

    @classmethod
    def from_catalog(cls, catalog: Catalog, index: int) -> CommitInfo:
        entry = catalog.commits[index]
        start = 0 if index == 0 else catalog.commits[index - 1].offset
        end = entry.offset

        def within(offset: int) -> bool:
            return start < offset < end

        blocks = [d for d in catalog.data if within(d.offset)]
        changes = CommitChanges(
            devices_created=tuple(d.schema.name for d in catalog.devices if within(d.offset)),
            blocks_written=len(blocks),
            rows_written=sum(d.header.row_count for d in blocks),
            ranges_deleted=sum(len(t.ranges) for t in catalog.tombstones if within(t.offset)),
            annotation_ops=sum(len(a.ops) for a in catalog.annotations if within(a.offset)),
            metadata_changed=any(within(m.offset) for m in catalog.metas),
        )
        record = entry.record
        return cls(
            record.number,
            record.time,
            record.author,
            record.message,
            record.hash.hex(),
            record.prev_hash.hex(),
            entry.offset,
            changes,
        )


class ChangeKind(enum.Enum):
    """Kind of a change between two versions."""

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class PointChange:
    """A point that differs between two versions.

    ``before`` and ``after`` map column names to values (``None`` if the
    point does not exist in that version).
    """

    time: int
    kind: ChangeKind
    before: dict[str, Any] | None
    after: dict[str, Any] | None


@dataclass(frozen=True)
class AnnotationChange:
    """An annotation that differs between two versions."""

    id: int
    kind: ChangeKind
    before: Annotation | None
    after: Annotation | None


@dataclass(frozen=True)
class Diff:
    """Differences of one device between commits ``from_commit`` and ``to_commit``."""

    device: str
    columns: tuple[str, ...]
    from_commit: int
    to_commit: int
    points: tuple[PointChange, ...]
    """Changed points, ordered by time."""
    annotations: tuple[AnnotationChange, ...]
    """Changed annotations of the device, ordered by identifier."""

    def __bool__(self) -> bool:
        return bool(self.points or self.annotations)


def _same(a: Any, b: Any) -> bool:
    """Equality of two values; floats are compared by bit pattern (NaN equals NaN)."""
    if isinstance(a, float) and isinstance(b, float):
        return struct.pack("<d", a) == struct.pack("<d", b)
    return type(a) is type(b) and a == b


def _rows_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)


def _read_range(
    f: BinaryIO, catalog: Catalog, device_id: int, span: tuple[int, int], limit: int
) -> Batch:
    columns = tuple(range(len(catalog.devices[device_id].types)))
    spec = ReadSpec(device_id, columns, span[0], span[1], limit)
    return execute_all(f, catalog, plan(catalog, spec))


def diff(f: BinaryIO, catalog: Catalog, device_id: int, start: int, end: int) -> Diff:
    """Differences of a device between versions ``start`` and ``end``. Only the
    time ranges touched by the commits in between are read."""
    if start > end:
        msg = f"version {start} is newer than {end}"
        raise InvalidArgumentError(msg)
    old_limit = catalog.limit(start)
    new_limit = catalog.limit(end)
    device = catalog.devices[device_id]

    def touched(offset: int) -> bool:
        return old_limit <= offset < new_limit

    spans = [
        (catalog.data[i].header.t_min, catalog.data[i].header.t_max)
        for i in device.data
        if touched(catalog.data[i].offset)
    ]
    for i in device.tombstones:
        if touched(catalog.tombstones[i].offset):
            spans.extend(catalog.tombstones[i].ranges)
    points: list[PointChange] = []
    for span in merge_intervals(spans):
        old = _read_range(f, catalog, device_id, span, old_limit)
        new = _read_range(f, catalog, device_id, span, new_limit)
        old_rows = list(old.rows())
        new_rows = list(new.rows())
        i = j = 0
        while i < len(old_rows) or j < len(new_rows):
            if i < len(old_rows) and j < len(new_rows) and old_rows[i][0] == new_rows[j][0]:
                before, after = old_rows[i][1], new_rows[j][1]
                if not _rows_equal(before, after):
                    points.append(PointChange(old_rows[i][0], ChangeKind.CHANGED, before, after))
                i += 1
                j += 1
            elif i < len(old_rows) and (j >= len(new_rows) or old_rows[i][0] < new_rows[j][0]):
                points.append(PointChange(old_rows[i][0], ChangeKind.REMOVED, old_rows[i][1], None))
                i += 1
            else:
                points.append(PointChange(new_rows[j][0], ChangeKind.ADDED, None, new_rows[j][1]))
                j += 1

    def of_device(states: dict[int, AnnotationState]) -> dict[int, Annotation]:
        return {
            k: Annotation.from_state(catalog, k, s)
            for k, s in states.items()
            if s.device_id == device_id
        }

    old_annotations = of_device(catalog.annotations_at(old_limit))
    new_annotations = of_device(catalog.annotations_at(new_limit))
    changes = []
    for annotation_id in sorted(old_annotations.keys() | new_annotations.keys()):
        was = old_annotations.get(annotation_id)
        now = new_annotations.get(annotation_id)
        if now is None:
            changes.append(AnnotationChange(annotation_id, ChangeKind.REMOVED, was, None))
        elif was is None:
            changes.append(AnnotationChange(annotation_id, ChangeKind.ADDED, None, now))
        elif was != now:
            changes.append(AnnotationChange(annotation_id, ChangeKind.CHANGED, was, now))
    return Diff(
        device.schema.name,
        tuple(device.schema.column_names),
        start,
        end,
        tuple(points),
        tuple(changes),
    )
