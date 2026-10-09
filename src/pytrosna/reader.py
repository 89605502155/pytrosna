"""Reading Trosna files with :class:`Reader`."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO, Self

from . import _read
from ._format import FILE_HEADER_LEN, Statistics, check_file_header
from ._scan import file_size, load_index, read_at, scan
from .codecs import Codec
from .errors import (
    CorruptedError,
    NotFinalizedError,
    NotTrosnaError,
    UnknownColumnError,
    UnknownCommitError,
)
from .history import Annotation, CommitInfo, Diff, diff

if TYPE_CHECKING:
    from ._catalog import Catalog
    from .column import Batch
    from .encodings import Encoding
    from .types import DataType, DeviceSchema

__all__ = ["BlockInfo", "Query", "Reader", "SegmentInfo"]

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class SegmentInfo:
    """Description of a stored segment."""

    column: str
    data_type: DataType | None
    """Column type, ``None`` for the time column."""
    encoding: Encoding
    codec: Codec
    stored_bytes: int
    encoded_bytes: int
    """Bytes after decompression (validity bitmap plus encoded values)."""
    null_count: int
    statistics: Statistics | None


@dataclass(frozen=True)
class BlockInfo:
    """Description of a stored block (a ``Data`` frame)."""

    offset: int
    commit: int | None
    """Commit that wrote the block."""
    rows: int
    t_min: int
    t_max: int
    segments: tuple[SegmentInfo, ...]
    """The time segment followed by the value segments."""


def inclusive_bounds(start: int | None, end: int | None) -> tuple[int, int] | None:
    lo = I64_MIN if start is None else max(I64_MIN, start)
    hi = I64_MAX if end is None else min(I64_MAX, end)
    return (lo, hi) if lo <= hi else None


class Reader:
    """A read-only view of a Trosna file.

    A reader sees the file as it was when opened: later commits by a writer
    are not visible. Use it as a context manager or call :meth:`close`.

    >>> with Reader("metrics.trosna") as r:  # doctest: +SKIP
    ...     batch = r.query("vm01").columns(["cpu"]).time_range(1_000, 2_000).collect()

    ``strict=True`` refuses a file that was not closed properly instead of
    recovering its committed state by scanning; ``verify_checksums=False``
    skips the CRC check of the segments that are read.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        strict: bool = False,
        verify_checksums: bool = True,
    ) -> None:
        self.path = os.fspath(path)
        self._file: BinaryIO = open(self.path, "rb")  # noqa: SIM115 - closed in close()
        try:
            self.size = file_size(self._file)
            if self.size < FILE_HEADER_LEN:
                raise NotTrosnaError
            self.format_version = check_file_header(read_at(self._file, 0, FILE_HEADER_LEN))
            catalog: Catalog | None
            try:
                catalog = load_index(self._file, self.size)
            except CorruptedError:
                if strict:
                    raise
                catalog = None
            if catalog is None:
                if strict:
                    raise NotFinalizedError
                self.finalized = False
                catalog = scan(self._file, self.size, full=False).catalog
            else:
                self.finalized = True
        except BaseException:
            self._file.close()
            raise
        self._catalog = catalog
        self.verify = verify_checksums

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"pytrosna.Reader({self.path!r}, devices={self.device_names})"

    # ------------------------------------------------------------ catalogue

    @property
    def metadata(self) -> dict[str, str]:
        """File metadata of the latest version."""
        return self._catalog.metadata_at(1 << 64)

    def metadata_as_of(self, commit: int) -> dict[str, str]:
        """File metadata as of a commit."""
        return self._catalog.metadata_at(self._catalog.limit(commit))

    @property
    def devices(self) -> list[DeviceSchema]:
        """Schemas of all devices, in order of creation."""
        return [d.schema for d in self._catalog.devices]

    @property
    def device_names(self) -> list[str]:
        return [d.schema.name for d in self._catalog.devices]

    def device(self, name: str) -> DeviceSchema:
        """Schema of a device."""
        return self._catalog.devices[self._catalog.device_id(name)].schema

    def commits(self) -> list[CommitInfo]:
        """All commits, oldest first."""
        return [
            CommitInfo.from_catalog(self._catalog, i) for i in range(len(self._catalog.commits))
        ]

    def commit(self, number: int) -> CommitInfo:
        """A commit by number."""
        if not 1 <= number <= len(self._catalog.commits):
            raise UnknownCommitError(number)
        return CommitInfo.from_catalog(self._catalog, number - 1)

    @property
    def head(self) -> CommitInfo | None:
        """The latest commit."""
        if not self._catalog.commits:
            return None
        return CommitInfo.from_catalog(self._catalog, len(self._catalog.commits) - 1)

    def commit_at(self, time_ns: int) -> int:
        """The last commit made at or before ``time_ns`` (nanoseconds since the
        epoch), i.e. the version of the file at that moment; 0 if none."""
        numbers = [c.record.number for c in self._catalog.commits if c.record.time <= time_ns]
        return max(numbers, default=0)

    def device_exists_at(self, name: str, commit: int) -> bool:
        """True if the device existed in version ``commit`` of the file."""
        try:
            device_id = self._catalog.device_id(name)
            limit = self._catalog.limit(commit)
        except (KeyError, UnknownCommitError):
            return False
        return self._catalog.devices[device_id].offset < limit

    def annotations(
        self, device: str | None = None, *, as_of: int | None = None
    ) -> list[Annotation]:
        """Annotations of one device or of all devices, ordered by identifier."""
        device_id = None if device is None else self._catalog.device_id(device)
        states = self._catalog.annotations_at(self._catalog.limit(as_of))
        return [
            Annotation.from_state(self._catalog, k, s)
            for k, s in states.items()
            if device_id is None or s.device_id == device_id
        ]

    def diff(self, device: str, start: int, end: int | None = None) -> Diff:
        """Differences of a device between commit ``start`` and commit ``end``
        (the latest if ``None``). Commit 0 is the empty file."""
        device_id = self._catalog.device_id(device)
        if end is None:
            end = self._catalog.head()[0]
        return diff(self._file, self._catalog, device_id, start, end)

    def blocks(self, device: str) -> list[BlockInfo]:
        """The stored blocks of a device, in file order."""
        info = self._catalog.devices[self._catalog.device_id(device)]
        out = []
        for i in info.data:
            entry = self._catalog.data[i]
            segments = tuple(
                SegmentInfo(
                    info.schema.time_name if k == 0 else info.schema.columns[k - 1].name,
                    None if k == 0 else info.types[k - 1],
                    s.encoding,
                    s.codec,
                    s.stored_len,
                    s.raw_len,
                    s.null_count,
                    s.stats,
                )
                for k, s in enumerate(entry.header.segments)
            )
            out.append(
                BlockInfo(
                    entry.offset,
                    self._catalog.commit_of(entry.offset),
                    entry.header.row_count,
                    entry.header.t_min,
                    entry.header.t_max,
                    segments,
                )
            )
        return out

    # ------------------------------------------------------------ data

    def query(self, device: str) -> Query:
        """Starts a query of a device: all columns, all time stamps and the
        latest version are selected until restricted."""
        return Query(self, self._catalog.device_id(device))

    def read(
        self,
        device: str,
        *,
        columns: Iterable[str] | None = None,
        start: int | None = None,
        end: int | None = None,
        as_of: int | None = None,
    ) -> Batch:
        """Reads a device (raw time stamps ``start <= t <= end``) into one batch."""
        query = self.query(device).time_range(start, end)
        if columns is not None:
            query = query.columns(columns)
        if as_of is not None:
            query = query.as_of(as_of)
        return query.collect()


class Query:
    """A query of one device; see :meth:`Reader.query`. Methods return new queries."""

    def __init__(
        self,
        reader: Reader,
        device_id: int,
        columns: tuple[str, ...] | None = None,
        bounds: tuple[int, int] | None = (I64_MIN, I64_MAX),
        as_of: int | None = None,
    ) -> None:
        self._reader = reader
        self._device_id = device_id
        self._columns = columns
        self._bounds = bounds
        self._as_of = as_of

    def _copy(self, **changes: Any) -> Query:
        state = {
            "columns": self._columns,
            "bounds": self._bounds,
            "as_of": self._as_of,
        } | changes
        return Query(self._reader, self._device_id, **state)

    def columns(self, names: Iterable[str]) -> Query:
        """Selects value columns by name (the time column is always included)."""
        return self._copy(columns=tuple(names))

    def time_range(self, start: int | None = None, end: int | None = None) -> Query:
        """Restricts the raw time stamps to ``start <= t <= end`` (open if ``None``)."""
        return self._copy(bounds=inclusive_bounds(start, end))

    def as_of(self, commit: int) -> Query:
        """Reads the version of the file as of a commit (0 is the empty file)."""
        return self._copy(as_of=commit)

    @property
    def device(self) -> DeviceSchema:
        return self._reader._catalog.devices[self._device_id].schema

    def plan(self, *, count_only: bool = False) -> _read.Plan:
        schema = self.device
        if count_only:
            indices: tuple[int, ...] = ()
        elif self._columns is None:
            indices = tuple(range(len(schema.columns)))
        else:
            positions = []
            for name in self._columns:
                index = schema.column_index(name)
                if index is None:
                    raise UnknownColumnError(schema.name, name)
                positions.append(index)
            indices = tuple(positions)
        lo, hi = self._bounds if self._bounds is not None else (1, 0)
        catalog = self._reader._catalog
        spec = _read.ReadSpec(
            self._device_id, indices, lo, hi, catalog.limit(self._as_of), self._reader.verify
        )
        return _read.plan(catalog, spec)

    def batches(self) -> Iterator[Batch]:
        """The result as a stream of batches in time order, roughly one per stored block."""
        query = self.plan()
        reader = self._reader
        for cluster in query.clusters:
            batch = _read.execute(reader._file, reader._catalog, query, cluster)
            if batch is not None:
                yield batch

    def collect(self) -> Batch:
        """The whole result as one batch."""
        reader = self._reader
        return _read.execute_all(reader._file, reader._catalog, self.plan())

    def count(self) -> int:
        """Counts the points without reading value columns."""
        query = self.plan(count_only=True)
        reader = self._reader
        total = 0
        for cluster in query.clusters:
            batch = _read.execute(reader._file, reader._catalog, query, cluster)
            if batch is not None:
                total += len(batch)
        return total
