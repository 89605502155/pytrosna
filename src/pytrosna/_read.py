"""The read engine: block selection, projection and merge-on-read (DESIGN §6).
Shared by :class:`~pytrosna.Reader` and :class:`~pytrosna.Writer`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO

import numpy as np

from ._scan import read_at
from ._segment import decode_column, decode_time
from .column import Batch, Column
from .errors import CorruptedError

if TYPE_CHECKING:
    from ._catalog import Catalog, DataEntry
    from .types import DataType, DeviceSchema

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class ReadSpec:
    device_id: int
    columns: tuple[int, ...]
    """Positions of the requested value columns in the device schema."""
    lo: int
    hi: int
    """Inclusive time bounds."""
    limit: int
    """Exclusive offset bound of the version to read."""
    verify: bool = True


@dataclass(frozen=True)
class Cluster:
    """A group of transitively time-overlapping blocks."""

    blocks: tuple[int, ...]
    """Indices into ``Catalog.data``, sorted by ``t_min``."""
    deletions: tuple[tuple[tuple[int, int], ...], ...]
    """For each block, the sorted disjoint intervals deleted by later tombstones."""


@dataclass(frozen=True)
class Plan:
    spec: ReadSpec
    schema: DeviceSchema
    names: tuple[str, ...]
    types: tuple[DataType, ...]
    clusters: tuple[Cluster, ...]


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorts intervals and merges overlapping or adjacent ones."""
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(intervals):
        if out and lo <= min(out[-1][1], I64_MAX - 1) + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def plan(catalog: Catalog, spec: ReadSpec) -> Plan:
    """Plans a query."""
    device = catalog.devices[spec.device_id]
    names = tuple(device.schema.columns[c].name for c in spec.columns)
    types = tuple(device.types[c] for c in spec.columns)
    query = (spec.lo, spec.hi)
    clusters: list[Cluster] = []
    if spec.lo <= spec.hi and device.offset < spec.limit:
        blocks = [
            i
            for i in device.data
            if catalog.data[i].offset < spec.limit
            and _overlaps((catalog.data[i].header.t_min, catalog.data[i].header.t_max), query)
        ]
        blocks.sort(key=lambda i: (catalog.data[i].header.t_min, catalog.data[i].offset))
        tombstones = [
            catalog.tombstones[i]
            for i in device.tombstones
            if catalog.tombstones[i].offset < spec.limit
        ]
        current: list[int] = []
        deletions: list[tuple[tuple[int, int], ...]] = []
        current_max = 0
        for i in blocks:
            entry = catalog.data[i]
            span = (entry.header.t_min, entry.header.t_max)
            deleted = merge_intervals(
                [
                    r
                    for t in tombstones
                    if t.offset > entry.offset
                    for r in t.ranges
                    if _overlaps(r, span) and _overlaps(r, query)
                ]
            )
            if current and span[0] <= current_max:
                current.append(i)
                deletions.append(tuple(deleted))
                current_max = max(current_max, span[1])
            else:
                if current:
                    clusters.append(Cluster(tuple(current), tuple(deletions)))
                current = [i]
                deletions = [tuple(deleted)]
                current_max = span[1]
        if current:
            clusters.append(Cluster(tuple(current), tuple(deletions)))
    return Plan(spec, device.schema, names, types, tuple(clusters))


def _read_segment(f: BinaryIO, entry: DataEntry, segment: int) -> bytes:
    size = entry.header.segments[segment].stored_len
    try:
        return read_at(f, entry.segment_offsets[segment], size)
    except CorruptedError as e:
        raise e.at(entry.offset) from None


def decode_block_time(f: BinaryIO, entry: DataEntry, *, verify: bool) -> np.ndarray:
    """Decodes and validates the time column of a block."""
    header = entry.header
    stored = _read_segment(f, entry, 0)
    try:
        time = decode_time(header.segments[0], stored, header.row_count, verify=verify)
    except CorruptedError as e:
        raise e.at(entry.offset) from None
    if (
        time.size != header.row_count
        or not bool(np.all(time[1:] > time[:-1]))
        or int(time[0]) != header.t_min
        or int(time[-1]) != header.t_max
    ):
        msg = "time stamps of a block are not strictly increasing"
        raise CorruptedError(msg, entry.offset)
    return time


def decode_block_column(
    f: BinaryIO, catalog: Catalog, entry: DataEntry, column: int, *, verify: bool
) -> Column:
    """Decodes value column ``column`` of a block."""
    data_type = catalog.devices[entry.header.device_id].types[column]
    stored = _read_segment(f, entry, column + 1)
    try:
        return decode_column(
            entry.header.segments[column + 1],
            stored,
            entry.header.row_count,
            data_type,
            verify=verify,
        )
    except CorruptedError as e:
        raise e.at(entry.offset) from None


def _deleted_mask(time: np.ndarray, intervals: tuple[tuple[int, int], ...]) -> np.ndarray:
    mask = np.zeros(time.size, dtype=np.bool_)
    for lo, hi in intervals:
        start = int(np.searchsorted(time, lo, side="left"))
        stop = int(np.searchsorted(time, hi, side="right"))
        mask[start:stop] = True
    return mask


def execute(f: BinaryIO, catalog: Catalog, query: Plan, cluster: Cluster) -> Batch | None:
    """Executes one cluster of a plan. Returns ``None`` if no rows remain."""
    spec = query.spec
    if len(cluster.blocks) == 1 and not cluster.deletions[0]:
        # Fast path: a block that nothing overlaps or deletes.
        entry = catalog.data[cluster.blocks[0]]
        time = decode_block_time(f, entry, verify=spec.verify)
        start = int(np.searchsorted(time, spec.lo, side="left"))
        stop = int(np.searchsorted(time, spec.hi, side="right"))
        if start >= stop:
            return None
        whole = start == 0 and stop == time.size
        columns = {}
        for name, c in zip(query.names, spec.columns, strict=True):
            column = decode_block_column(f, catalog, entry, c, verify=spec.verify)
            columns[name] = column if whole else column.slice(start, stop)
        return Batch(time if whole else time[start:stop], columns, query.schema)

    # Merge path: the newest version of every time stamp, minus deletions.
    times: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    for b, i in enumerate(cluster.blocks):
        entry = catalog.data[i]
        time = decode_block_time(f, entry, verify=spec.verify)
        keep = (time >= spec.lo) & (time <= spec.hi) & ~_deleted_mask(time, cluster.deletions[b])
        picked = np.flatnonzero(keep)
        times.append(time[picked])
        offsets.append(np.full(picked.size, entry.offset, dtype=np.int64))
        sources.append(np.full(picked.size, b, dtype=np.int64))
        rows.append(picked)
    t = np.concatenate(times)
    if t.size == 0:
        return None
    off = np.concatenate(offsets)
    src = np.concatenate(sources)
    row = np.concatenate(rows)
    # Sort by time, the newest (largest offset) first, and keep the first of each time.
    order = np.lexsort((-off, t))
    t, src, row = t[order], src[order], row[order]
    first = np.concatenate(([True], t[1:] != t[:-1]))
    t, src, row = t[first], src[first], row[first]
    used = np.unique(src)
    columns = {}
    for name, c in zip(query.names, spec.columns, strict=True):
        decoded = {
            int(b): decode_block_column(
                f, catalog, catalog.data[cluster.blocks[int(b)]], c, verify=spec.verify
            )
            for b in used
        }
        parts = [decoded[int(b)].take(row[src == b]) for b in used]
        positions = np.concatenate([np.flatnonzero(src == b) for b in used])
        merged = Column.concat(parts)
        inverse = np.empty_like(positions)
        inverse[positions] = np.arange(positions.size)
        columns[name] = merged.take(inverse)
    return Batch(t, columns, query.schema)


def execute_all(f: BinaryIO, catalog: Catalog, query: Plan) -> Batch:
    """Executes a whole plan and concatenates the result."""
    batches = [
        b for cluster in query.clusters if (b := execute(f, catalog, query, cluster)) is not None
    ]
    if not batches:
        return Batch.empty(dict(zip(query.names, query.types, strict=True)), query.schema)
    if len(batches) == 1:
        return batches[0]
    return Batch.concat(batches)
