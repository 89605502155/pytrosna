"""The high-level API: :func:`open`, :class:`File`, :class:`Transaction` and
module-level functions that read and write whole tables.

Times can be given as ``datetime``, ``date``, ``pandas.Timestamp``,
``numpy.datetime64``, ISO 8601 text (without an offset it is read in the
device's time zone) or an integer in the device's time unit.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Self

from . import adapters
from .column import Batch
from .errors import InvalidArgumentError, UnknownAnnotationError
from .history import Annotation, CommitInfo, Diff
from .reader import BlockInfo, Reader
from .timeconv import to_datetime, to_raw
from .tools import CompactReport, RecoverReport, VerifyReport
from .tools import compact as _compact
from .tools import recover as _recover
from .tools import verify as _verify
from .types import DeviceSchema
from .writer import WriteOptions, Writer

__all__ = [
    "File",
    "Transaction",
    "create",
    "iter_batches",
    "open",
    "read",
    "read_arrow",
    "read_batches",
    "read_pandas",
    "read_polars",
    "write",
]

_Path = str | os.PathLike[str]


def _raw(device: DeviceSchema, value: Any, rounding: str = "floor") -> int:
    return to_raw(value, device.time_unit, device.timezone, rounding)


def _opt_raw(device: DeviceSchema, value: Any, rounding: str) -> int | None:
    return None if value is None else _raw(device, value, rounding)


@dataclass(frozen=True)
class _Query:
    device: str
    columns: tuple[str, ...] | None
    start: int | None
    end: int | None
    as_of: int | None


class File:
    """A Trosna file. Opening it reads only its index; data is read on demand.

    >>> f = pytrosna.open("room.trosna")  # doctest: +SKIP
    >>> f.read_pandas(start="2026-10-08 10:00", columns=["temperature"])  # doctest: +SKIP
    """

    def __init__(self, path: _Path) -> None:
        self.path = os.fspath(path)
        self.reload()

    def reload(self) -> None:
        """Re-reads the catalogue (after another program changed the file)."""
        with Reader(self.path) as reader:
            self.finalized: bool = reader.finalized
            self.format_version: tuple[int, int] = reader.format_version
            self.size: int = reader.size
            self.metadata: dict[str, str] = reader.metadata
            self.devices: dict[str, DeviceSchema] = {d.name: d for d in reader.devices}

    def __repr__(self) -> str:
        return f"pytrosna.File({self.path!r}, devices={list(self.devices)})"

    def device(self, name: str | None = None) -> DeviceSchema:
        """The named device, or the only one in the file."""
        if name is not None:
            try:
                return self.devices[name]
            except KeyError:
                from .errors import UnknownDeviceError

                raise UnknownDeviceError(name) from None
        if len(self.devices) == 1:
            return next(iter(self.devices.values()))
        if not self.devices:
            msg = "the file has no devices"
            raise KeyError(msg)
        msg = f"the file has several devices {list(self.devices)}; pass device=..."
        raise InvalidArgumentError(msg)

    def to_raw(self, value: Any, device: str | None = None, rounding: str = "floor") -> int:
        """Converts a time to a raw time stamp of a device (see :func:`pytrosna.to_raw`)."""
        return _raw(self.device(device), value, rounding)

    def to_datetime(self, raw: int, device: str | None = None) -> Any:
        """Converts a raw time stamp of a device to a ``datetime``."""
        d = self.device(device)
        return to_datetime(raw, d.time_unit, d.timezone)

    # ------------------------------------------------------------ reading

    def _query(
        self,
        device: str | None,
        columns: Iterable[str] | None,
        start: Any,
        end: Any,
        as_of: int | None,
    ) -> _Query:
        d = self.device(device)
        return _Query(
            d.name,
            None if columns is None else tuple(columns),
            _opt_raw(d, start, "ceil"),
            _opt_raw(d, end, "floor"),
            as_of,
        )

    def iter_batches(
        self,
        device: str | None = None,
        *,
        columns: Iterable[str] | None = None,
        start: Any = None,
        end: Any = None,
        as_of: int | None = None,
    ) -> Iterator[Batch]:
        """Streams the points of a device in time order as :class:`Batch`
        objects, roughly one per stored block (``start``/``end`` inclusive)."""
        q = self._query(device, columns, start, end, as_of)
        with Reader(self.path) as reader:
            query = reader.query(q.device).time_range(q.start, q.end)
            if q.columns is not None:
                query = query.columns(q.columns)
            if q.as_of is not None:
                query = query.as_of(q.as_of)
            yield from query.batches()

    def read(
        self,
        device: str | None = None,
        *,
        columns: Iterable[str] | None = None,
        start: Any = None,
        end: Any = None,
        as_of: int | None = None,
    ) -> Batch:
        """Reads a device into one :class:`Batch` (NumPy arrays, no other dependencies)."""
        q = self._query(device, columns, start, end, as_of)
        with Reader(self.path) as reader:
            return reader.read(q.device, columns=q.columns, start=q.start, end=q.end, as_of=q.as_of)

    def read_arrow(self, device: str | None = None, **query: Any) -> Any:
        """Reads a device into a ``pyarrow.Table``; options as in :meth:`read`."""
        return self.read(device, **query).to_arrow()

    def read_batches(self, device: str | None = None, **query: Any) -> Any:
        """Streams a device as a ``pyarrow.RecordBatchReader`` (one block at a time)."""
        pa = adapters._require("pyarrow", "arrow")
        d = self.device(device)
        schema = Batch.empty({c.name: c.data_type for c in d.columns}, d)
        if query.get("columns") is not None:
            schema = schema.select(query["columns"])
        arrow_schema = schema.to_arrow().schema
        batches = (
            b.to_arrow().to_batches()[0] if len(b) else None
            for b in self.iter_batches(device, **query)
        )
        return pa.RecordBatchReader.from_batches(
            arrow_schema, (b for b in batches if b is not None)
        )

    def read_pandas(
        self, device: str | None = None, *, dtype_backend: str = "numpy", **query: Any
    ) -> Any:
        """Reads a device into a ``pandas.DataFrame`` with the time as its first
        column; see :meth:`Batch.to_pandas` for ``dtype_backend``."""
        return self.read(device, **query).to_pandas(dtype_backend=dtype_backend)

    def read_polars(self, device: str | None = None, **query: Any) -> Any:
        """Reads a device into a ``polars.DataFrame``."""
        return self.read(device, **query).to_polars()

    def count(
        self,
        device: str | None = None,
        *,
        start: Any = None,
        end: Any = None,
        as_of: int | None = None,
    ) -> int:
        """Number of points (reads only the time stamps)."""
        q = self._query(device, None, start, end, as_of)
        with Reader(self.path) as reader:
            query = reader.query(q.device).time_range(q.start, q.end)
            if q.as_of is not None:
                query = query.as_of(q.as_of)
            return query.count()

    def time_range(self, device: str | None = None) -> tuple[int, int] | None:
        """The first and last raw time stamps of a device, or ``None`` if it is empty."""
        batch = self.read(device, columns=())
        if not len(batch):
            return None
        return int(batch.time[0]), int(batch.time[-1])

    # ------------------------------------------------------------ writing

    def write(self, data: Any, device: str, **options: Any) -> CommitInfo | None:
        """Appends data to a device (created if missing); see :func:`write`."""
        commit = write(self.path, data, device, mode="a", **options)
        self.reload()
        return commit

    def edit(self, message: str | None = None, author: str | None = None) -> Transaction:
        """Starts a transaction: all its changes become one commit when the
        ``with`` block ends without an exception, and none otherwise."""
        return Transaction(self, message, author)

    def writer(self, options: WriteOptions | None = None, **kwargs: Any) -> Writer:
        """Opens the low-level :class:`Writer` of the file."""
        return Writer.open(self.path, options, **kwargs)

    # ------------------------------------------------------------ history

    def commits(self) -> list[CommitInfo]:
        """All commits, oldest first."""
        with Reader(self.path) as reader:
            return reader.commits()

    @property
    def head(self) -> CommitInfo | None:
        """The latest commit."""
        with Reader(self.path) as reader:
            return reader.head

    def commit(self, number: int) -> CommitInfo:
        with Reader(self.path) as reader:
            return reader.commit(number)

    def commit_at(self, when: Any) -> int:
        """The version of the file at a moment (a ``datetime``, text or
        nanoseconds since the epoch): the last commit made at or before it."""
        nanos = to_raw(when, "ns", "UTC")
        with Reader(self.path) as reader:
            return reader.commit_at(nanos)

    def annotations(
        self, device: str | None = None, *, as_of: int | None = None
    ) -> list[Annotation]:
        """Annotations (of one device, or of all), as of a commit if given."""
        if device is not None:
            self.device(device)
        with Reader(self.path) as reader:
            return reader.annotations(device, as_of=as_of)

    def diff(
        self, from_commit: int, to_commit: int | None = None, *, device: str | None = None
    ) -> Diff:
        """What changed in a device between two commits (0 is the empty file)."""
        name = self.device(device).name
        with Reader(self.path) as reader:
            return reader.diff(name, from_commit, to_commit)

    def blocks(self, device: str | None = None) -> list[BlockInfo]:
        """The stored blocks of a device with their encodings and statistics."""
        name = self.device(device).name
        with Reader(self.path) as reader:
            return reader.blocks(name)

    # ------------------------------------------------------------ maintenance

    def verify(self) -> VerifyReport:
        return _verify(self.path)

    def compact(self, target: _Path, *, as_of: int | None = None, overwrite: bool = False) -> File:
        """Writes a version into a new file without history and opens it."""
        _compact(self.path, target, as_of=as_of, overwrite=overwrite)
        return File(target)


class Transaction:
    """Changes collected by :meth:`File.edit`; applied atomically on exit.

    >>> with f.edit(message="sensor check", author="me") as tx:  # doctest: +SKIP
    ...     tx.update("room1", "2026-10-08 10:00:20", temperature=21.65)
    ...     tx.delete("room1", "2026-10-08 10:00:10")
    """

    def __init__(self, file: File, message: str | None, author: str | None) -> None:
        self._file = file
        self._message = message
        self._author = author
        self._ops: list[tuple[Any, ...]] = []
        self.commit: CommitInfo | None = None
        """The commit created by the transaction (``None`` until it is applied
        or if it changed nothing)."""
        self.annotation_ids: list[int] = []
        """Identifiers of the annotations added by :meth:`annotate`."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None or not self._ops:
            return
        self.apply()

    def apply(self) -> CommitInfo | None:
        """Applies the collected changes now (the ``with`` block does this)."""
        ops, self._ops = self._ops, []
        writer = Writer.open(self._file.path)
        ids: list[int] = []
        try:
            for op in ops:
                result = getattr(writer, op[0])(*op[1:])
                if op[0] == "annotate":
                    ids.append(result)
            self.commit = writer.commit(message=self._message, author=self._author)
        except BaseException:
            writer.rollback()
            writer.close()
            raise
        writer.close()
        self.annotation_ids = ids
        self._file.reload()
        return self.commit

    def _point(self, device: str, time: Any) -> int:
        # A point's time must be representable exactly, or points would merge.
        return _raw(self._file.device(device), time, "exact")

    def insert(
        self, device: str, time: Any, values: Mapping[str, Any] | None = None, **kw: Any
    ) -> None:
        """Adds a point, or replaces the one at the same time; missing columns are null."""
        self._ops.append(("write_row", device, self._point(device, time), {**(values or {}), **kw}))

    def update(
        self, device: str, time: Any, values: Mapping[str, Any] | None = None, **kw: Any
    ) -> None:
        """Changes some values of an existing point."""
        self._ops.append(("update", device, self._point(device, time), {**(values or {}), **kw}))

    def delete(self, device: str, time: Any) -> None:
        """Deletes the point at ``time``, if any."""
        self._ops.append(("delete", device, self._point(device, time)))

    def delete_range(self, device: str, start: Any = None, end: Any = None) -> None:
        """Deletes all points with ``start <= time <= end`` (open ends if ``None``)."""
        d = self._file.device(device)
        self._ops.append(
            ("delete_range", device, _opt_raw(d, start, "ceil"), _opt_raw(d, end, "floor"))
        )

    def write(
        self, device: str, data: Any, *, time_column: str | None = None, unit: str | None = None
    ) -> None:
        """Writes a whole table to an existing device as part of the transaction."""
        schema = self._file.device(device)
        for source in adapters.normalize(data, time_column):
            self._ops.append(("write", device, adapters.to_batch(source, schema, unit)))

    def annotate(
        self, device: str, start: Any, end: Any, label: str, note: str | None = None
    ) -> None:
        """Adds an annotation; its identifier is in :attr:`annotation_ids` after the commit."""
        d = self._file.device(device)
        self._ops.append(("annotate", device, _raw(d, start), _raw(d, end), label, note))

    def update_annotation(
        self,
        annotation_id: int,
        *,
        start: Any = None,
        end: Any = None,
        label: str | None = None,
        note: str | None = None,
    ) -> None:
        """Changes an annotation; arguments that are not given keep their values."""
        current = next((a for a in self._file.annotations() if a.id == annotation_id), None)
        if current is None:
            raise UnknownAnnotationError(annotation_id)
        d = self._file.device(current.device)
        self._ops.append(
            (
                "update_annotation",
                annotation_id,
                _opt_raw(d, start, "floor"),
                _opt_raw(d, end, "floor"),
                label,
                note,
            )
        )

    def remove_annotation(self, annotation_id: int) -> None:
        self._ops.append(("remove_annotation", annotation_id))

    def set_metadata(self, key: str, value: str) -> None:
        self._ops.append(("set_metadata", key, value))

    def remove_metadata(self, key: str) -> None:
        self._ops.append(("remove_metadata", key))


# ---------------------------------------------------------------- module API


def open(path: _Path) -> File:  # noqa: A001 - mirrors the built-in open
    """Opens a Trosna file for reading, editing and history queries."""
    return File(path)


def read(path: _Path, device: str | None = None, **query: Any) -> Batch:
    """Reads a device into a :class:`Batch`.

    Options: ``columns``, ``start``, ``end`` (inclusive) and ``as_of`` (a commit number).
    """
    return File(path).read(device, **query)


def iter_batches(path: _Path, device: str | None = None, **query: Any) -> Iterator[Batch]:
    """Streams a device as :class:`Batch` objects; options as in :func:`read`."""
    return File(path).iter_batches(device, **query)


def read_batches(path: _Path, device: str | None = None, **query: Any) -> Any:
    """Streams a device as a ``pyarrow.RecordBatchReader``."""
    return File(path).read_batches(device, **query)


def read_arrow(path: _Path, device: str | None = None, **query: Any) -> Any:
    """Reads a device into a ``pyarrow.Table``; options as in :func:`read`."""
    return File(path).read_arrow(device, **query)


def read_pandas(path: _Path, device: str | None = None, **query: Any) -> Any:
    """Reads a device into a ``pandas.DataFrame``; options as in :func:`read`."""
    return File(path).read_pandas(device, **query)


def read_polars(path: _Path, device: str | None = None, **query: Any) -> Any:
    """Reads a device into a ``polars.DataFrame``; options as in :func:`read`."""
    return File(path).read_polars(device, **query)


def _temp_path(target: str) -> str:
    directory, name = os.path.split(target)
    return os.path.join(directory, f".{name}.{os.getpid()}.tmp")


def write(
    path: _Path,
    data: Any,
    device: str,
    *,
    time_column: str | None = None,
    unit: str | None = None,
    mode: str = "w",
    message: str | None = None,
    author: str | None = None,
    **options: Any,
) -> CommitInfo | None:
    """Writes a table to a device in one commit.

    ``data`` is a pandas or Polars DataFrame (a pandas ``DatetimeIndex`` is
    used as the time), a PyArrow Table/RecordBatch/RecordBatchReader, any
    object exporting an Arrow stream, a :class:`Batch` or a mapping
    ``{column: values}``. ``time_column`` defaults to the first timestamp
    column or a column named like "time"; integer times need ``unit``
    (``"s"``, ``"ms"``, ``"us"`` or ``"ns"``). ``mode`` is ``"w"`` (replace
    the file), ``"x"`` (create a new file) or ``"a"`` (append; a missing file
    or device is created). Rows with the time of an existing point replace
    it. If anything fails, nothing is committed and an existing file is left
    untouched. Other keyword arguments are :class:`WriteOptions`.
    """
    target = os.fspath(path)
    if mode not in ("w", "x", "a"):
        msg = f"mode must be 'w', 'x' or 'a', not {mode!r}"
        raise InvalidArgumentError(msg)
    exists = os.path.exists(target)
    if mode == "x" and exists:
        msg = f"{target!r} already exists"
        raise FileExistsError(msg)
    append = mode == "a" and exists
    temp = None if append else _temp_path(target)
    if append:
        writer = Writer.open(target, WriteOptions(**options))
    else:
        assert temp is not None  # noqa: S101
        writer = Writer.create(temp, WriteOptions(overwrite=True, **options))
    try:
        for source in adapters.normalize(data, time_column):
            if not writer.has_device(device):
                writer.create_device(adapters.infer_device(device, source, unit))
            schema = writer.device(device)
            writer.write(device, adapters.to_batch(source, schema, unit))
        commit = writer.commit(message=message, author=author)
    except BaseException:
        writer.rollback()
        writer.close()
        if temp is not None:
            _remove(temp)
        raise
    try:
        writer.close()
        if temp is not None:
            if mode == "x" and os.path.exists(target):
                msg = f"{target!r} was created by someone else meanwhile"
                raise FileExistsError(msg)
            os.replace(temp, target)
    except BaseException:
        if temp is not None:
            _remove(temp)
        raise
    return commit


def _remove(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


def create(
    path: _Path,
    *devices: DeviceSchema,
    metadata: Mapping[str, str] | None = None,
    overwrite: bool = False,
    message: str | None = None,
    author: str | None = None,
) -> File:
    """Creates a file with the given devices and metadata (one commit) and opens it.

    >>> pytrosna.create(  # doctest: +SKIP
    ...     "room.trosna",
    ...     DeviceSchema.build("room1", "s", {"temperature": "float64"}, timezone="Europe/Moscow"),
    ... )
    """
    with Writer.create(path, overwrite=overwrite) as writer:
        if metadata:
            writer.update_metadata(dict(metadata))
        for schema in devices:
            writer.create_device(schema)
        writer.commit(message=message, author=author)
    return File(path)


def verify(path: _Path) -> VerifyReport:
    """Checks all checksums, the hash chain and the decoding of all data."""
    return _verify(path)


def recover(path: _Path, *, force: bool = False) -> RecoverReport:
    """Finalizes a file whose writer was interrupted (``force`` also cuts off
    everything after a damaged frame in the middle of the file)."""
    return _recover(path, force=force)


def compact(
    source: _Path, target: _Path, *, as_of: int | None = None, overwrite: bool = False
) -> CompactReport:
    """Copies a version (the latest by default) into a new file without
    history; its first commit links to the source version's hash."""
    return _compact(source, target, as_of=as_of, overwrite=overwrite)
