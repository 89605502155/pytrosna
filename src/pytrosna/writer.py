"""Creating, appending to and editing Trosna files with :class:`Writer`."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import sys
import time as _time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, BinaryIO, Self

import numpy as np

from . import _read
from ._catalog import Catalog
from ._format import (
    FILE_HEADER_LEN,
    MAX_ROWS,
    ZERO_HASH,
    AnnotationOp,
    CommitRecord,
    DataHeader,
    FrameKind,
    check_file_header,
    encode_annotation_ops,
    encode_data_payload,
    encode_device,
    encode_footer,
    encode_frame,
    encode_index,
    encode_meta,
    encode_tombstone,
)
from ._scan import END_OF_FILE, TRUNCATED, file_size, find_valid_frame_after, load_index, read_at
from ._scan import scan as scan_file
from ._segment import EncodeOptions, encode_column, encode_time
from .codecs import Codec
from .column import Batch, Column
from .encodings import EncodingPolicy
from .errors import (
    CorruptedError,
    DeviceExistsError,
    InvalidArgumentError,
    LockedError,
    PointNotFoundError,
    SchemaError,
    TrosnaError,
    TypeMismatchError,
    UnknownAnnotationError,
    UnknownColumnError,
)
from .history import CommitInfo
from .reader import inclusive_bounds
from .types import DataType, DeviceSchema

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["RecoveryReport", "WriteOptions", "Writer"]


@dataclass(frozen=True)
class WriteOptions:
    """Options of a :class:`Writer`."""

    codec: Codec | str = Codec.ZSTD
    """General-purpose codec for segments: ``"zstd"`` (default), ``"lz4"`` or ``"none"``."""
    encoding: EncodingPolicy | str = EncodingPolicy.ADAPTIVE
    """How column encodings are chosen: ``"adaptive"`` (default), ``"classic"`` or ``"plain"``."""
    rows_per_block: int = 65_536
    """Maximum number of rows per block (at most 2^24)."""
    max_block_bytes: int = 64 << 20
    """A device's buffer is written out once it holds about this many bytes."""
    sync: bool = True
    """Call ``fsync`` when committing, so a commit survives a power failure."""
    overwrite: bool = False
    """:meth:`Writer.create` replaces an existing file."""
    repair_corruption: bool = False
    """:meth:`Writer.open` truncates a damaged file even if valid frames follow
    the damage, discarding them."""
    zstd_level: int = 3
    """Zstandard compression level."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "codec", Codec.parse(self.codec))
        object.__setattr__(self, "encoding", EncodingPolicy.parse(self.encoding))
        if not 1 <= self.rows_per_block <= MAX_ROWS:
            msg = f"rows_per_block must be between 1 and {MAX_ROWS}"
            raise InvalidArgumentError(msg)
        if self.max_block_bytes < 1:
            msg = "max_block_bytes must be positive"
            raise InvalidArgumentError(msg)


@dataclass(frozen=True)
class RecoveryReport:
    """What happened when a file that was not closed properly was opened for writing."""

    truncated_bytes: int
    """Bytes removed from the end of the file."""
    discarded_frames: int
    """Complete frames of the interrupted transaction that were discarded."""
    reason: str
    """Why the scan of the file stopped."""


@dataclass
class _DeviceBuffer:
    """Rows written to a device and not yet stored, in order of writing."""

    schema: DeviceSchema
    chunks: list[Batch] = field(default_factory=list)
    row_times: list[int] = field(default_factory=list)
    row_values: list[list[Any]] = field(default_factory=list)
    rows: int = 0
    nbytes: int = 0

    def seal_rows(self) -> None:
        """Turns the single rows written so far into a chunk."""
        if not self.row_times:
            return
        columns = {
            c.name: Column.from_values(c.data_type, [row[k] for row in self.row_values])
            for k, c in enumerate(self.schema.columns)
        }
        self.chunks.append(Batch(np.array(self.row_times, dtype=np.int64), columns))
        self.row_times = []
        self.row_values = []

    def find(self, t: int) -> dict[str, Any] | None:
        """The latest buffered row at time ``t``."""
        for i in range(len(self.row_times) - 1, -1, -1):
            if self.row_times[i] == t:
                return {c.name: self.row_values[i][k] for k, c in enumerate(self.schema.columns)}
        for chunk in reversed(self.chunks):
            hits = np.flatnonzero(chunk.time == t)
            if hits.size:
                return chunk.row(int(hits[-1]))
        return None

    def take_all(self) -> Batch | None:
        self.seal_rows()
        if not self.chunks:
            return None
        batch = self.chunks[0] if len(self.chunks) == 1 else Batch.concat(self.chunks)
        self.chunks = []
        self.rows = 0
        self.nbytes = 0
        return batch


def _batch_bytes(batch: Batch) -> int:
    total = len(batch) * 8
    for column in batch.columns.values():
        if column.data_type is DataType.STRING:
            total += sum(len(s.encode("utf-8")) for s in column.values.tolist()) + 4 * len(column)
        else:
            total += len(column) * column.data_type.plain_width
    return total


def _now_ns() -> int:
    return _time.time_ns()


_WINDOWS = sys.platform == "win32"

# Windows locks are mandatory, so the writer locks one byte far beyond the
# data, where readers never look. Some systems refuse offsets above 2 GiB;
# the second offset is the fallback for them.
_WINDOWS_LOCK_OFFSETS = (1 << 62, (1 << 31) - 2)


def _lock(f: BinaryIO) -> int | None:
    """Takes an exclusive lock on the file. Returns the locked offset on
    Windows (to be passed to :func:`_unlock`), ``None`` elsewhere, where the
    lock is released when the file is closed."""
    if not _WINDOWS:
        import fcntl

        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LockedError from None
        return None
    import errno

    msvcrt = importlib.import_module("msvcrt")
    position = f.tell()
    try:
        error: OSError | None = None
        for offset in _WINDOWS_LOCK_OFFSETS:
            f.seek(offset)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as e:
                if e.errno in (errno.EACCES, errno.EDEADLK):
                    raise LockedError from None
                error = e  # this offset is not supported; try the next one
            else:
                return offset
        assert error is not None  # noqa: S101
        raise error
    finally:
        f.seek(position)


def _unlock(f: BinaryIO, offset: int | None) -> None:
    """Releases a Windows lock taken by :func:`_lock` before the file is closed."""
    if offset is None:
        return
    msvcrt = importlib.import_module("msvcrt")
    with contextlib.suppress(OSError):
        f.seek(offset)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def _sort_rows(batch: Batch) -> Batch:
    """Sorts rows by time; for duplicate time stamps the last write wins."""
    time = batch.time
    if time.size < 2 or bool(np.all(time[1:] > time[:-1])):
        return batch
    order = np.argsort(time, kind="stable")
    sorted_time = time[order]
    last = np.append(sorted_time[1:] != sorted_time[:-1], True)
    return batch.take(order[last])


def to_column(values: Any, data_type: DataType, name: str) -> Column:
    """Converts values given for a column into a :class:`Column` of ``data_type``.

    Accepts a :class:`Column`, a NumPy array (NaN stays a value; masked
    arrays and ``None`` in object arrays are nulls) or any sequence of Python
    values (``None`` is a null)."""
    if isinstance(values, Column):
        if values.data_type is not data_type:
            msg = f"column {name!r} has type {data_type}, but {values.data_type} values were given"
            raise TypeMismatchError(msg)
        return values
    if isinstance(values, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(values)
        filled = values.filled(_fill_value(data_type))
        column = to_column(np.asarray(filled), data_type, name)
        return Column(data_type, column.values, column.valid_mask() & ~mask)
    if isinstance(values, np.ndarray) and values.dtype != object:
        return Column(data_type, _numpy_values(values, data_type, name))
    try:
        return Column.from_values(data_type, list(values))
    except TypeMismatchError as e:
        msg = f"column {name!r}: {e}"
        raise TypeMismatchError(msg) from None


def _fill_value(data_type: DataType) -> Any:
    return "" if data_type is DataType.STRING else 0


def _numpy_values(values: np.ndarray, data_type: DataType, name: str) -> np.ndarray:
    """Converts a NumPy array losslessly to the dtype of ``data_type``."""
    if values.ndim != 1:
        msg = f"column {name!r} must be one-dimensional"
        raise InvalidArgumentError(msg)
    target = data_type.numpy_dtype
    kind = values.dtype.kind
    if data_type is DataType.BOOL:
        if kind != "b":
            msg = f"column {name!r} has type bool, but {values.dtype} values were given"
            raise TypeMismatchError(msg)
        return values.astype(np.bool_)
    if data_type is DataType.STRING:
        if kind not in "US":
            msg = f"column {name!r} has type string, but {values.dtype} values were given"
            raise TypeMismatchError(msg)
        out = np.empty(values.size, dtype=object)
        out[:] = [v.decode() if isinstance(v, bytes) else str(v) for v in values.tolist()]
        return out
    if data_type in (DataType.INT32, DataType.INT64):
        if kind == "f":
            if not bool(np.all(np.isfinite(values) & (values == np.round(values)))):
                msg = f"column {name!r} has type {data_type}, but the values are not whole numbers"
                raise TypeMismatchError(msg)
        elif kind not in "iu":
            msg = f"column {name!r} has type {data_type}, but {values.dtype} values were given"
            raise TypeMismatchError(msg)
        info = np.iinfo(target)
        if values.size and (values.min() < info.min or values.max() > info.max):
            msg = f"column {name!r} has values outside the {data_type} range"
            raise TypeMismatchError(msg)
        return values.astype(target)
    if kind not in "iuf":
        msg = f"column {name!r} has type {data_type}, but {values.dtype} values were given"
        raise TypeMismatchError(msg)
    return values.astype(target)


class Writer:
    """Writes and edits a Trosna file.

    Changes are grouped into commits: rows written with :meth:`write` are
    buffered and become durable and visible to readers when :meth:`commit`
    (or :meth:`close`) returns. Use the writer as a context manager: leaving
    the ``with`` block normally commits and finalizes the file; leaving it
    with an exception discards the uncommitted changes.

    >>> with Writer.create("metrics.trosna") as w:  # doctest: +SKIP
    ...     w.create_device(DeviceSchema.build("vm01", "ms", {"cpu": "float64"}))
    ...     w.write("vm01", {"time": [1000, 2000], "cpu": [0.31, 0.35]})
    ...     w.commit(message="first measurements")
    """

    def __init__(
        self,
        path: str,
        f: BinaryIO,
        pos: int,
        catalog: Catalog,
        options: WriteOptions,
    ) -> None:
        self.path = path
        self._file = f
        self._pos = pos
        self._catalog = catalog
        self._options = options
        self._encode = EncodeOptions(options.encoding, options.codec, options.zstd_level)  # type: ignore[arg-type]
        self._buffers = [_DeviceBuffer(d.schema) for d in catalog.devices]
        self._pending_tombstones: dict[int, list[tuple[int, int]]] = {}
        self._hasher = hashlib.sha256()
        self._tx_start = pos
        self._chain_origin = ZERO_HASH
        self._dirty = False
        self._poisoned = False
        self._closed = False
        self._read_file: BinaryIO | None = None
        self._lock_offset: int | None = None
        self.recovery: RecoveryReport | None = None
        """What was repaired when the file was opened, if anything."""

    # ------------------------------------------------------------ opening

    @classmethod
    def create(
        cls, path: str | os.PathLike[str], options: WriteOptions | None = None, **kwargs: Any
    ) -> Writer:
        """Creates a new file. Fails if it exists, unless ``overwrite=True``.

        Options can be given as a :class:`WriteOptions` or as keyword arguments.
        """
        options = _options(options, kwargs)
        path = os.fspath(path)
        if options.overwrite:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o666)
        else:
            fd = os.open(
                path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666
            )
        f = os.fdopen(fd, "r+b")
        try:
            lock = _lock(f)
            f.truncate(0)
            f.seek(0)
            f.write(_header_bytes())
        except BaseException:
            f.close()
            raise
        writer = cls(path, f, FILE_HEADER_LEN, Catalog(), options)
        writer._lock_offset = lock
        writer._dirty = True
        return writer

    @classmethod
    def open(
        cls, path: str | os.PathLike[str], options: WriteOptions | None = None, **kwargs: Any
    ) -> Writer:
        """Opens an existing file for appending and editing.

        If the file was not closed properly, its uncommitted tail is removed
        (see :attr:`recovery`). If damaged frames are followed by valid ones,
        the file is not modified and :class:`~pytrosna.CorruptedError` is
        raised, unless ``repair_corruption=True``.
        """
        options = _options(options, kwargs)
        path = os.fspath(path)
        f = open(path, "r+b")  # noqa: SIM115 - owned by the writer
        lock = None
        try:
            lock = _lock(f)
            length = file_size(f)
            check_file_header(read_at(f, 0, min(FILE_HEADER_LEN, length)))
            try:
                catalog = load_index(f, length)
            except CorruptedError:
                catalog = None
            recovery = None
            end = length
            if catalog is None:
                outcome = scan_file(f, length, full=True)
                if outcome.stop != END_OF_FILE and not options.repair_corruption:
                    following = find_valid_frame_after(f, length, outcome.stopped_at)
                    if following is not None:
                        msg = (
                            f"damaged frame followed by valid data at offset {following}; "
                            "refusing to truncate (use repair_corruption=True to discard "
                            "everything after the damage)"
                        )
                        raise CorruptedError(msg, outcome.stopped_at)
                reason = {
                    END_OF_FILE: "the file was not finalized",
                    TRUNCATED: "the last frame is incomplete",
                }.get(outcome.stop, outcome.stop)
                recovery = RecoveryReport(
                    length - outcome.committed_end, outcome.discarded_frames, reason
                )
                if outcome.committed_end < length:
                    f.truncate(outcome.committed_end)
                    f.flush()
                    os.fsync(f.fileno())
                catalog = outcome.catalog
                end = outcome.committed_end
            f.seek(end)
        except BaseException:
            f.close()
            raise
        writer = cls(path, f, end, catalog, options)
        writer._lock_offset = lock
        writer.recovery = recovery
        writer._dirty = recovery is not None
        return writer

    # ------------------------------------------------------------ lifecycle

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._closed:
            return
        if exc_type is not None:
            if not self._poisoned:
                self.rollback()
            self._finish_quietly()
        else:
            self.close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            self._finish_quietly()

    def _finish_quietly(self) -> None:
        # Nothing can be reported from a finalizer or a failed block.
        with contextlib.suppress(Exception):
            if not self._poisoned and not self._file.closed:
                self._finish()
        with contextlib.suppress(Exception):
            self._release()

    def close(self) -> None:
        """Commits pending changes, writes the index and closes the file."""
        if self._closed:
            return
        try:
            self._finish()
        finally:
            self._release()

    def _finish(self) -> None:
        self._check_usable()
        self.commit()
        if self._dirty:
            self._write_index()

    def _release(self) -> None:
        self._closed = True
        if self._read_file is not None:
            self._read_file.close()
            self._read_file = None
        try:
            _unlock(self._file, self._lock_offset)
        finally:
            self._file.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"pytrosna.Writer({self.path!r}, {state})"

    # ------------------------------------------------------------ state

    @property
    def devices(self) -> list[DeviceSchema]:
        """Schemas of all devices, including those created in the current transaction."""
        return [d.schema for d in self._catalog.devices]

    def device(self, name: str) -> DeviceSchema:
        return self._catalog.devices[self._catalog.device_id(name)].schema

    def has_device(self, name: str) -> bool:
        return name in self._catalog.by_name

    @property
    def head(self) -> int:
        """Number of the last commit (0 if there is none)."""
        return self._catalog.head()[0]

    @property
    def metadata(self) -> dict[str, str]:
        """Current file metadata, including uncommitted changes."""
        return self._catalog.metadata_at(1 << 64)

    def _check_usable(self) -> None:
        if self._closed:
            msg = "the writer is closed"
            raise TrosnaError(msg)
        if self._poisoned:
            msg = "the writer is poisoned by an earlier I/O error and must be discarded"
            raise TrosnaError(msg)

    def _io(self, step: Any) -> Any:
        """Runs an I/O step, poisoning the writer if it fails."""
        try:
            return step()
        except OSError:
            self._poisoned = True
            raise

    def _write_frame(self, kind: FrameKind, payload: bytes, catalog_body: bytes) -> int:
        frame = encode_frame(kind, payload)
        offset = self._pos
        self._io(lambda: self._file.write(frame))
        self._pos += len(frame)
        if not kind.is_derived:
            if kind is not FrameKind.COMMIT:
                self._hasher.update(frame)
            self._catalog.apply(kind, offset, catalog_body, len(payload))
        return offset

    # ------------------------------------------------------------ devices

    def create_device(self, schema: DeviceSchema) -> None:
        """Creates a device."""
        self._check_usable()
        schema.validate()
        if schema.name in self._catalog.by_name:
            raise DeviceExistsError(schema.name)
        payload = encode_device(len(self._catalog.devices), schema)
        self._write_frame(FrameKind.DEVICE, payload, payload)
        self._buffers.append(_DeviceBuffer(schema))

    def ensure_device(self, schema: DeviceSchema) -> DeviceSchema:
        """Creates the device if it does not exist; otherwise checks that the
        existing device has the same time unit and columns, and returns it."""
        if schema.name not in self._catalog.by_name:
            self.create_device(schema)
            return schema
        existing = self.device(schema.name)
        if not existing.same_layout(schema):
            msg = f"device {schema.name!r} already exists with a different schema"
            raise SchemaError(msg)
        return existing

    # ------------------------------------------------------------ writing

    def write(self, device: str, data: Batch | Mapping[str, Any]) -> int:
        """Writes rows and returns their number.

        ``data`` is a :class:`Batch` or a mapping of column names to values
        whose time key is the device's time column name (or ``"time"``) and
        holds raw time stamps. Columns of the device that are not given are
        null. Rows may have any time stamps; a row with the time stamp of an
        existing point replaces it.
        """
        self._check_usable()
        device_id = self._catalog.device_id(device)
        schema = self._catalog.devices[device_id].schema
        batch = self._to_batch(schema, data)
        if len(batch) == 0:
            return 0
        buffer = self._buffers[device_id]
        incoming = _batch_bytes(batch)
        if buffer.rows and buffer.nbytes + incoming > self._options.max_block_bytes:
            self._flush_device(device_id)
        buffer.seal_rows()
        buffer.chunks.append(batch)
        buffer.rows += len(batch)
        buffer.nbytes += incoming
        self._maybe_flush(device_id)
        return len(batch)

    def _to_batch(self, schema: DeviceSchema, data: Batch | Mapping[str, Any]) -> Batch:
        if isinstance(data, Batch):
            time = data.time
            given: Mapping[str, Any] = data.columns
        else:
            given = dict(data)
            key = schema.time_name if schema.time_name in given else "time"
            if key not in given:
                msg = f"the data has no time column {schema.time_name!r}"
                raise InvalidArgumentError(msg)
            time = np.asarray(given.pop(key))
            if time.dtype.kind not in "iu":
                if time.size == 0:
                    time = time.astype(np.int64)
                else:
                    msg = "time stamps must be integers in the device's time unit"
                    raise TypeMismatchError(msg)
        time = np.asarray(time, dtype=np.int64)
        columns: dict[str, Column] = {}
        for name, values in given.items():
            index = schema.column_index(name)
            if index is None:
                raise UnknownColumnError(schema.name, name)
            column = to_column(values, schema.columns[index].data_type, name)
            if len(column) != time.size:
                msg = f"column {name!r} has {len(column)} values for {time.size} time stamps"
                raise InvalidArgumentError(msg)
            columns[name] = column
        full = {
            c.name: columns[c.name] if c.name in columns else Column.nulls(c.data_type, time.size)
            for c in schema.columns
        }
        return Batch(time, full, schema)

    def write_row(
        self, device: str, time: int, values: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Writes one row; values are given by column name, missing columns are null."""
        self._check_usable()
        device_id = self._catalog.device_id(device)
        schema = self._catalog.devices[device_id].schema
        row = self._row_from_named(
            schema, [None] * len(schema.columns), {**(values or {}), **kwargs}
        )
        self._push_row(device_id, int(time), row)

    def _row_from_named(
        self, schema: DeviceSchema, row: list[Any], values: Mapping[str, Any]
    ) -> list[Any]:
        for name, value in values.items():
            index = schema.column_index(name)
            if index is None:
                raise UnknownColumnError(schema.name, name)
            data_type = schema.columns[index].data_type
            try:
                row[index] = Column.from_values(data_type, [value]).value(0)
            except TypeMismatchError as e:
                msg = f"column {name!r}: {e}"
                raise TypeMismatchError(msg) from None
        return row

    def _push_row(self, device_id: int, time: int, row: list[Any]) -> None:
        if not -(1 << 63) <= time < (1 << 63):
            msg = f"time stamp {time} does not fit into 64 bits"
            raise InvalidArgumentError(msg)
        buffer = self._buffers[device_id]
        buffer.row_times.append(time)
        buffer.row_values.append(row)
        buffer.rows += 1
        buffer.nbytes += 8 + sum(
            len(v.encode("utf-8")) + 4 if isinstance(v, str) else c.data_type.plain_width
            for v, c in zip(row, buffer.schema.columns, strict=True)
        )
        self._maybe_flush(device_id)

    def _maybe_flush(self, device_id: int) -> None:
        buffer = self._buffers[device_id]
        if (
            buffer.rows >= self._options.rows_per_block
            or buffer.nbytes >= self._options.max_block_bytes
        ):
            self._flush_device(device_id)

    def _flush_device(self, device_id: int) -> None:
        """Writes the pending tombstones and buffered rows of a device."""
        ranges = self._pending_tombstones.pop(device_id, None)
        if ranges:
            payload = encode_tombstone(device_id, ranges)
            self._write_frame(FrameKind.TOMBSTONE, payload, payload)
        batch = self._buffers[device_id].take_all()
        if batch is None or len(batch) == 0:
            return
        batch = _sort_rows(batch)
        schema = self._catalog.devices[device_id].schema
        step = self._options.rows_per_block
        for start in range(0, len(batch), step):
            block = batch.slice(start, start + step)
            headers = []
            segments = []
            header, stored = encode_time(block.time, self._encode)
            headers.append(header)
            segments.append(stored)
            for column in schema.columns:
                header, stored = encode_column(block.columns[column.name], self._encode)
                headers.append(header)
                segments.append(stored)
            data_header = DataHeader(
                device_id,
                len(block),
                int(block.time[0]),
                int(block.time[-1]),
                tuple(headers),
            )
            payload, header_bytes = encode_data_payload(data_header, segments)
            self._write_frame(FrameKind.DATA, payload, header_bytes)

    # ------------------------------------------------------------ editing

    def get(self, device: str, time: int) -> dict[str, Any] | None:
        """The current values of the point at ``time``, including changes not
        yet committed; ``None`` if there is no such point."""
        self._check_usable()
        return self._current_row(self._catalog.device_id(device), int(time))

    def _current_row(self, device_id: int, time: int) -> dict[str, Any] | None:
        found = self._buffers[device_id].find(time)
        if found is not None:
            return found
        ranges = self._pending_tombstones.get(device_id, [])
        if any(lo <= time <= hi for lo, hi in ranges):
            return None
        self._io(self._file.flush)
        if self._read_file is None:
            self._read_file = open(self.path, "rb")  # noqa: SIM115 - closed in _release
        columns = tuple(range(len(self._catalog.devices[device_id].types)))
        spec = _read.ReadSpec(device_id, columns, time, time, 1 << 64)
        batch = _read.execute_all(self._read_file, self._catalog, _read.plan(self._catalog, spec))
        return batch.row(0) if len(batch) else None

    def update(
        self, device: str, time: int, values: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Changes values of the existing point at ``time``; raises
        :class:`~pytrosna.PointNotFoundError` if there is none."""
        self._check_usable()
        device_id = self._catalog.device_id(device)
        current = self._current_row(device_id, int(time))
        if current is None:
            raise PointNotFoundError(device, int(time))
        schema = self._catalog.devices[device_id].schema
        row = self._row_from_named(
            schema, [current[c.name] for c in schema.columns], {**(values or {}), **kwargs}
        )
        self._push_row(device_id, int(time), row)

    def delete(self, device: str, time: int) -> None:
        """Deletes the point at ``time``, if any."""
        self.delete_range(device, int(time), int(time))

    def delete_range(self, device: str, start: int | None = None, end: int | None = None) -> None:
        """Deletes all points with ``start <= time <= end`` (open ends if ``None``)."""
        self._check_usable()
        device_id = self._catalog.device_id(device)
        bounds = inclusive_bounds(start, end)
        if bounds is None:
            return
        if self._buffers[device_id].rows:
            # Rows written before the deletion must precede the tombstone in the log.
            self._flush_device(device_id)
        self._pending_tombstones.setdefault(device_id, []).append(bounds)

    # ------------------------------------------------------------ annotations

    def _write_annotation_op(self, op: AnnotationOp) -> None:
        payload = encode_annotation_ops([op])
        self._write_frame(FrameKind.ANNOTATION, payload, payload)

    @staticmethod
    def _check_interval(start: int, end: int) -> None:
        if start > end:
            msg = f"annotation start {start} is after its end {end}"
            raise InvalidArgumentError(msg)

    def annotate(
        self, device: str, start: int, end: int, label: str, note: str | None = None
    ) -> int:
        """Adds an annotation ``[start, end]`` to a device and returns its identifier."""
        self._check_usable()
        self._check_interval(start, end)
        device_id = self._catalog.device_id(device)
        annotation_id = self._catalog.max_annotation_id + 1
        self._write_annotation_op(
            AnnotationOp(annotation_id, False, device_id, int(start), int(end), label, note)
        )
        return annotation_id

    def update_annotation(
        self,
        annotation_id: int,
        start: int | None = None,
        end: int | None = None,
        label: str | None = None,
        note: str | None = None,
    ) -> None:
        """Changes an annotation; arguments that are ``None`` keep their values
        (pass ``note=""`` to clear a note)."""
        self._check_usable()
        current = self._catalog.live_annotations.get(annotation_id)
        if current is None:
            raise UnknownAnnotationError(annotation_id)
        new_start = current.start if start is None else int(start)
        new_end = current.end if end is None else int(end)
        self._check_interval(new_start, new_end)
        new_note = current.note if note is None else (note or None)
        self._write_annotation_op(
            AnnotationOp(
                annotation_id,
                False,
                current.device_id,
                new_start,
                new_end,
                current.label if label is None else label,
                new_note,
            )
        )

    def remove_annotation(self, annotation_id: int) -> None:
        """Removes an annotation."""
        self._check_usable()
        if annotation_id not in self._catalog.live_annotations:
            raise UnknownAnnotationError(annotation_id)
        self._write_annotation_op(AnnotationOp(annotation_id, remove=True))

    def restore_annotation(
        self, annotation_id: int, device: str, *, start: int, end: int, label: str, note: str | None
    ) -> None:
        """Re-creates an annotation with a given identifier (used by compaction)."""
        self._check_usable()
        device_id = self._catalog.device_id(device)
        self._write_annotation_op(
            AnnotationOp(annotation_id, False, device_id, start, end, label, note)
        )

    def annotations(self) -> dict[int, tuple[str, int, int, str, str | None]]:
        """The current annotations, including uncommitted changes, as
        ``{id: (device, start, end, label, note)}``."""
        return {
            k: (self._catalog.devices[s.device_id].schema.name, s.start, s.end, s.label, s.note)
            for k, s in sorted(self._catalog.live_annotations.items())
        }

    # ------------------------------------------------------------ metadata

    def set_metadata(self, key: str, value: str) -> None:
        """Sets a file metadata entry."""
        self._check_usable()
        metadata = self.metadata
        metadata[str(key)] = str(value)
        self._write_metadata(metadata)

    def update_metadata(self, entries: Mapping[str, str]) -> None:
        """Sets several file metadata entries with one ``Meta`` frame."""
        self._check_usable()
        metadata = self.metadata
        metadata.update({str(k): str(v) for k, v in entries.items()})
        self._write_metadata(metadata)

    def remove_metadata(self, key: str) -> None:
        """Removes a file metadata entry (nothing happens if it does not exist)."""
        self._check_usable()
        metadata = self.metadata
        if metadata.pop(key, None) is not None:
            self._write_metadata(metadata)

    def _write_metadata(self, metadata: dict[str, str]) -> None:
        payload = encode_meta(metadata)
        self._write_frame(FrameKind.META, payload, payload)

    def set_chain_origin(self, origin: bytes) -> None:
        """Sets the ``prev_hash`` of the first commit (provenance of compacted files)."""
        if self._catalog.commits:
            msg = "the file already has commits"
            raise InvalidArgumentError(msg)
        if len(origin) != len(ZERO_HASH):
            msg = "a chain origin is a 32-byte hash"
            raise InvalidArgumentError(msg)
        self._chain_origin = bytes(origin)

    # ------------------------------------------------------------ transactions

    def commit(
        self,
        message: str | None = None,
        author: str | None = None,
        *,
        time_ns: int | None = None,
    ) -> CommitInfo | None:
        """Commits pending changes; returns ``None`` if there was nothing to commit.

        ``time_ns`` overrides the commit time (nanoseconds since the epoch).
        """
        self._check_usable()
        for device_id in range(len(self._buffers)):
            self._flush_device(device_id)
        frame_count = self._catalog.pending_count
        if frame_count == 0:
            return None
        self._durable()
        head, head_hash = self._catalog.head()
        first = self._catalog.pending_first
        record = CommitRecord(
            number=head + 1,
            time=_now_ns() if time_ns is None else int(time_ns),
            first_offset=self._pos if first is None else first,
            frame_count=frame_count,
            prev_hash=self._chain_origin if head == 0 else head_hash,
            content_hash=self._hasher.digest(),
            author=author,
            message=message,
        ).seal()
        payload = record.encode()
        self._write_frame(FrameKind.COMMIT, payload, payload)
        self._durable()
        self._hasher = hashlib.sha256()
        self._tx_start = self._pos
        self._dirty = True
        return CommitInfo.from_catalog(self._catalog, len(self._catalog.commits) - 1)

    def _durable(self) -> None:
        def step() -> None:
            self._file.flush()
            if self._options.sync:
                os.fsync(self._file.fileno())

        self._io(step)

    def rollback(self) -> None:
        """Discards all changes since the last commit."""
        self._check_usable()
        self._buffers = [_DeviceBuffer(d.schema) for d in self._catalog.devices]
        self._pending_tombstones = {}
        start = self._tx_start

        def step() -> None:
            self._file.flush()
            self._file.truncate(start)
            self._file.seek(start)

        self._io(step)
        self._pos = start
        self._catalog.truncate(start)
        self._buffers = self._buffers[: len(self._catalog.devices)]
        self._hasher = hashlib.sha256()

    @property
    def has_pending_changes(self) -> bool:
        """True if there are changes that the next commit would seal."""
        return (
            self._catalog.pending_count > 0
            or any(b.rows for b in self._buffers)
            or any(self._pending_tombstones.values())
        )

    def _write_index(self) -> None:
        head, head_hash = self._catalog.head()
        payload = encode_index(head, head_hash, self._catalog.index_entries())
        index_offset = self._pos
        self._write_frame(FrameKind.INDEX, payload, b"")
        self._write_frame(FrameKind.FOOTER, encode_footer(index_offset), b"")

        def step() -> None:
            self._file.flush()
            os.fsync(self._file.fileno())

        self._io(step)
        self._dirty = False


def _header_bytes() -> bytes:
    from ._format import file_header

    return file_header()


def _options(options: WriteOptions | None, kwargs: dict[str, Any]) -> WriteOptions:
    if options is not None and kwargs:
        msg = "give either a WriteOptions or keyword options, not both"
        raise InvalidArgumentError(msg)
    return options if options is not None else WriteOptions(**kwargs)


def device_schema(
    name: str,
    time_unit: str = "ms",
    columns: Mapping[str, DataType | str] | Sequence[tuple[str, DataType | str]] = (),
    **kwargs: Any,
) -> DeviceSchema:
    """Shorthand for :meth:`DeviceSchema.build`."""
    items: Iterable[tuple[str, DataType | str]] = (
        columns.items() if isinstance(columns, Mapping) else columns
    )
    return DeviceSchema.build(name, time_unit, dict(items), **kwargs)
