# pytrosna API reference

[Русская версия](api.ru.md) · [User guide](guide.md) · [README](../README.md)

Everything listed here is importable from the top-level package
(`import pytrosna`). Times called *raw* are integers in the time unit of the
device; *any time* means a raw integer, `datetime`/`date`,
`pandas.Timestamp`, `numpy.datetime64` or ISO 8601 text (see
[Times](#times)).

* [Module functions](#module-functions)
* [File and Transaction](#file-and-transaction)
* [Writer and WriteOptions](#writer-and-writeoptions)
* [Reader and Query](#reader-and-query)
* [Batch and Column](#batch-and-column)
* [Schemas and types](#schemas-and-types)
* [History objects](#history-objects)
* [Storage information](#storage-information)
* [Maintenance reports](#maintenance-reports)
* [Times](#times)
* [Adapters](#adapters)
* [Errors](#errors)
* [Constants](#constants)

## Module functions

| Function | Description |
|---|---|
| `open(path) -> File` | Opens a file for reading, editing and history queries. |
| `read(path, device=None, *, columns=None, start=None, end=None, as_of=None) -> Batch` | Reads a device into a NumPy-backed `Batch`. `start`/`end` are inclusive (any time); `as_of` is a commit number (0 = empty file). `device` may be omitted if the file has one device. |
| `read_pandas(path, device=None, *, dtype_backend="numpy", **query)` | Reads into a `pandas.DataFrame`; the time is the first column. |
| `read_polars(path, device=None, **query)` | Reads into a `polars.DataFrame`. |
| `read_arrow(path, device=None, **query)` | Reads into a `pyarrow.Table`. |
| `read_batches(path, device=None, **query)` | Streams as a `pyarrow.RecordBatchReader`. |
| `iter_batches(path, device=None, **query) -> Iterator[Batch]` | Streams `Batch` objects, roughly one per stored block. |
| `write(path, data, device, *, time_column=None, unit=None, mode="w", message=None, author=None, **options) -> CommitInfo \| None` | Writes a table to a device in one commit. `data`: pandas/Polars DataFrame (or LazyFrame), PyArrow Table/RecordBatch/RecordBatchReader, an object with `__arrow_c_stream__`, a `Batch` or a mapping `{column: values}`. `unit` is the unit of integer time stamps. `mode`: `"w"` replace, `"x"` create new, `"a"` append (creates a missing file or device). `options` are `WriteOptions` fields. Returns `None` if nothing was written. |
| `create(path, *devices, metadata=None, overwrite=False, message=None, author=None) -> File` | Creates a file with devices (`DeviceSchema`) and metadata. |
| `verify(path) -> VerifyReport` | Full integrity check. |
| `recover(path, *, force=False) -> RecoverReport` | Finalizes a file whose writer was interrupted; `force` also discards data after damage in the middle of the file. |
| `compact(source, target, *, as_of=None, overwrite=False) -> CompactReport` | Copies a version (the latest by default) into a new file without history. |
| `to_raw(value, unit, tz=None, rounding="floor") -> int` | Converts any time to a raw time stamp. |
| `to_datetime(raw, unit, tz=None) -> datetime` | Converts a raw time stamp to a `datetime`. |
| `format_time(raw, unit, tz=None) -> str` | ISO 8601 text with the full precision of the unit. |

## File and Transaction

### `class File(path)`

A Trosna file; opening reads only its index. Attributes (refreshed by
`reload()` and after every write through the object):

| Attribute | Type | Meaning |
|---|---|---|
| `path` | `str` | the file name |
| `finalized` | `bool` | `False` if the file was not closed properly (it is still readable) |
| `format_version` | `tuple[int, int]` | from the file header |
| `size` | `int` | bytes |
| `metadata` | `dict[str, str]` | file metadata of the latest version |
| `devices` | `dict[str, DeviceSchema]` | devices by name |

| Method | Description |
|---|---|
| `device(name=None) -> DeviceSchema` | The named device or the only one. |
| `read(device=None, *, columns, start, end, as_of) -> Batch` | See `pytrosna.read`. |
| `read_pandas(device=None, *, dtype_backend="numpy", **query)` / `read_polars(...)` / `read_arrow(...)` / `read_batches(...)` / `iter_batches(...)` | As the module functions. |
| `count(device=None, *, start=None, end=None, as_of=None) -> int` | Number of points (reads only the time stamps). |
| `time_range(device=None) -> tuple[int, int] \| None` | First and last raw time stamps. |
| `to_raw(value, device=None, rounding="floor") -> int`, `to_datetime(raw, device=None)` | Time conversion in a device's unit and zone. |
| `write(data, device, **options) -> CommitInfo \| None` | Appends (`mode="a"`). |
| `edit(message=None, author=None) -> Transaction` | Starts a transaction. |
| `writer(options=None, **kwargs) -> Writer` | Opens the low-level writer. |
| `commits() -> list[CommitInfo]`, `head`, `commit(number)` | History. |
| `commit_at(when) -> int` | The version at a moment (any time, UTC for naive values). |
| `annotations(device=None, *, as_of=None) -> list[Annotation]` | Annotations, ordered by identifier. |
| `diff(from_commit, to_commit=None, *, device=None) -> Diff` | Changes between two versions. |
| `blocks(device=None) -> list[BlockInfo]` | Stored blocks with encodings and statistics. |
| `verify() -> VerifyReport`, `compact(target, *, as_of=None, overwrite=False) -> File`, `reload()` | Maintenance. |

### `class Transaction`

Returned by `File.edit()`. Changes are applied as one commit when the `with`
block ends without an exception (or by calling `apply()`); none otherwise.

| Method | Description |
|---|---|
| `insert(device, time, values=None, **kw)` | Adds a point or replaces the one at that time; missing columns are null. `time` must be exact in the device's unit. |
| `update(device, time, values=None, **kw)` | Changes some values of an existing point (`PointNotFoundError` otherwise). |
| `delete(device, time)` | Deletes the point at `time`, if any. |
| `delete_range(device, start=None, end=None)` | Deletes `start <= t <= end` (open ends if `None`). |
| `write(device, data, *, time_column=None, unit=None)` | Writes a whole table to an existing device. |
| `annotate(device, start, end, label, note=None)` | Adds an annotation. |
| `update_annotation(annotation_id, *, start=None, end=None, label=None, note=None)` | Changes an annotation; `None` keeps a value, `note=""` removes the note. |
| `remove_annotation(annotation_id)` | Removes an annotation. |
| `set_metadata(key, value)`, `remove_metadata(key)` | File metadata. |
| `apply() -> CommitInfo \| None` | Applies now. |

Attributes after the commit: `commit` (`CommitInfo` or `None`) and
`annotation_ids` (identifiers of the new annotations).

## Writer and WriteOptions

### `class WriteOptions`

A frozen dataclass; all fields are optional.

| Field | Default | Meaning |
|---|---|---|
| `codec` | `Codec.ZSTD` | `Codec` or `"zstd"`, `"lz4"`, `"none"` |
| `encoding` | `EncodingPolicy.ADAPTIVE` | `"adaptive"`, `"classic"`, `"plain"` |
| `rows_per_block` | `65536` | 1 … 2²⁴ |
| `max_block_bytes` | `64 MiB` | flush threshold of a device buffer |
| `sync` | `True` | `fsync` on commit |
| `overwrite` | `False` | `Writer.create` replaces an existing file |
| `repair_corruption` | `False` | `Writer.open` truncates after damage in the middle |
| `zstd_level` | `3` | Zstandard level |

### `class Writer`

Writes and edits a file; one writer per file at a time (exclusive lock).
Use it as a context manager: a normal exit commits and finalizes the file,
an exception discards uncommitted changes. All times are raw integers.

| Method / attribute | Description |
|---|---|
| `Writer.create(path, options=None, **kwargs)` | Creates a file (fails if it exists unless `overwrite=True`). |
| `Writer.open(path, options=None, **kwargs)` | Opens for appending; recovers an unfinished file (see `recovery`). |
| `recovery: RecoveryReport \| None` | What was removed when opening. |
| `devices`, `device(name)`, `has_device(name)`, `head`, `metadata`, `closed`, `has_pending_changes` | State, including uncommitted changes. |
| `create_device(schema)` | Creates a device (`DeviceExistsError` if it exists). |
| `ensure_device(schema) -> DeviceSchema` | Creates it, or checks the existing one has the same layout. |
| `write(device, data) -> int` | Writes a `Batch` or a mapping (time key: the device's time name or `"time"`); missing columns are null; rows may be in any order, duplicates replace. |
| `write_row(device, time, values=None, **kw)` | Writes one row. |
| `get(device, time) -> dict \| None` | The current row, including uncommitted changes. |
| `update(device, time, values=None, **kw)` | Changes values of an existing point. |
| `delete(device, time)`, `delete_range(device, start=None, end=None)` | Deletions. |
| `annotate(device, start, end, label, note=None) -> int` | Returns the new identifier. |
| `update_annotation(id, start=None, end=None, label=None, note=None)`, `remove_annotation(id)`, `annotations()` | Annotation editing. |
| `set_metadata(key, value)`, `update_metadata(mapping)`, `remove_metadata(key)` | File metadata. |
| `commit(message=None, author=None, *, time_ns=None) -> CommitInfo \| None` | Seals pending changes (`None` if there are none). |
| `rollback()` | Discards everything since the last commit. |
| `close()` | Commits, writes the index and footer, closes the file. |
| `set_chain_origin(hash32)`, `restore_annotation(...)` | Used by compaction. |

## Reader and Query

### `class Reader(path, *, strict=False, verify_checksums=True)`

A read-only snapshot of a file (later commits are not visible). `strict`
refuses a file that is not finalized; `verify_checksums=False` skips the
CRC check of the segments read. A context manager; `close()` releases it.

| Member | Description |
|---|---|
| `finalized`, `format_version`, `size`, `path`, `metadata`, `devices`, `device_names` | Catalogue. |
| `device(name)`, `metadata_as_of(commit)`, `device_exists_at(name, commit)` | Lookups. |
| `commits()`, `commit(number)`, `head`, `commit_at(time_ns)` | History. |
| `annotations(device=None, *, as_of=None)`, `diff(device, start, end=None)` | Annotations and differences. |
| `blocks(device) -> list[BlockInfo]` | Storage details. |
| `query(device) -> Query` | Starts a query. |
| `read(device, *, columns=None, start=None, end=None, as_of=None) -> Batch` | Raw-time shorthand. |

### `class Query`

Immutable; each method returns a new query.

| Method | Description |
|---|---|
| `columns(names)` | Selects value columns (the time is always included). |
| `time_range(start=None, end=None)` | Raw bounds, inclusive. |
| `as_of(commit)` | Version. |
| `collect() -> Batch` | The whole result. |
| `batches() -> Iterator[Batch]` | Streamed, in time order. |
| `count() -> int` | Number of points. |
| `device` | The schema. |

## Batch and Column

### `class Batch(time, columns=None, schema=None)`

Rows of one device: `time` (`numpy.int64` array of raw time stamps),
`columns` (`dict[str, Column]`) and `schema` (`DeviceSchema` or `None`).

| Member | Description |
|---|---|
| `len(batch)`, `batch[name]`, `names` | Size and columns. |
| `row(i) -> dict`, `rows() -> Iterator[(time, dict)]`, `to_dict()` | Python values. |
| `slice(start, stop)`, `take(indices)`, `select(names)` | Subsets. |
| `Batch.concat(batches)`, `Batch.empty(types, schema=None)` | Construction. |
| `datetimes()` | Time stamps as `numpy.datetime64` (UTC). |
| `to_pandas(*, dtype_backend="numpy")`, `to_polars()`, `to_arrow()` | Conversions. |

### `class Column(data_type, values, validity=None)`

`values` is a NumPy array of the type's dtype (`object` for strings);
`validity` a boolean mask (`True` = present) or `None` (no nulls).

| Member | Description |
|---|---|
| `Column.from_values(data_type, values, *, nan_is_null=False)` | From Python values; `None` is null; lossless conversions only. |
| `Column.nulls(data_type, length)`, `Column.concat(columns)` | Construction. |
| `data_type`, `null_count`, `len(column)` | Properties. |
| `value(i)` / `column[i]`, `to_list()`, `to_numpy()`, `dense()`, `valid_mask()`, `is_valid(i)` | Access. |
| `slice(start, stop)`, `take(indices)` | Subsets. |

## Schemas and types

* `DataType` — `BOOL`, `INT32`, `INT64`, `FLOAT32`, `FLOAT64`, `STRING`;
  `label`, `numpy_dtype`, `plain_width`, `DataType.parse("float64")`.
* `TimeUnit` — `SECOND`, `MILLISECOND`, `MICROSECOND`, `NANOSECOND`; `label`
  (`"s"` …), `per_second`, `nanos`, `from_nanos()`, `to_nanos()`,
  `TimeUnit.parse("ms")`.
* `ColumnSchema(name, data_type, metadata={})`.
* `DeviceSchema(name, time_unit="ms", columns=(), timezone=None,
  time_name="time", metadata={})` — `DeviceSchema.build(name, unit,
  {column: type}, **kwargs)`, `with_column(name, type, metadata=None)`,
  `column_names`, `types`, `column_index(name)`, `validate()`,
  `same_layout(other)`.
* `Codec` — `NONE`, `LZ4`, `ZSTD`; `Codec.parse("zstd")`.
* `Encoding` — `PLAIN`, `DELTA_BIT_PACK`, `DELTA_OF_DELTA`, `RLE`, `XOR`,
  `DICTIONARY`; `Encoding.candidates(data_type)`, `Encoding.classic(data_type)`.
* `EncodingPolicy` — `ADAPTIVE`, `CLASSIC`, `PLAIN`.

## History objects

* `CommitInfo` — `number`, `time_ns`, `time` (aware `datetime`, UTC),
  `author`, `message`, `hash` and `prev_hash` (64 hex digits), `short_hash`,
  `offset`, `changes`.
* `CommitChanges` — `devices_created`, `blocks_written`, `rows_written`,
  `ranges_deleted`, `annotation_ops`, `metadata_changed`.
* `Annotation` — `id`, `device`, `start`, `end` (raw), `label`, `note`.
* `Diff` — `device`, `columns`, `from_commit`, `to_commit`, `points`
  (`PointChange`: `time`, `kind`, `before`, `after`), `annotations`
  (`AnnotationChange`: `id`, `kind`, `before`, `after`); `bool(diff)` is true
  if anything changed.
* `ChangeKind` — `ADDED`, `REMOVED`, `CHANGED`.

## Storage information

* `BlockInfo` — `offset`, `commit`, `rows`, `t_min`, `t_max`, `segments`.
* `SegmentInfo` — `column`, `data_type` (`None` for the time), `encoding`,
  `codec`, `stored_bytes`, `encoded_bytes`, `null_count`, `statistics`.
* `Statistics` — `min`, `max`, `is_float`.

## Maintenance reports

* `VerifyReport` — `ok`, `frames`, `commits`, `blocks`, `finalized`, `head`,
  `origin`, `problems`, `warnings`; `bool(report)` is `report.ok`.
* `RecoverReport` — `was_finalized`, `recovery`, `removed_bytes`,
  `removed_frames`.
* `RecoveryReport` — `truncated_bytes`, `discarded_frames`, `reason`.
* `CompactReport` — `version`, `origin`, `rows`, `annotations`,
  `bytes_before`, `bytes_after`.

## Times

`to_raw(value, unit, tz=None, rounding="floor")` accepts:

| Value | Interpretation |
|---|---|
| `int` | already raw (returned unchanged) |
| `str` (ISO 8601) | with an offset or `Z`: an instant; without: wall-clock time in `tz` |
| `datetime` | aware: an instant; naive: wall-clock time in `tz` |
| `date` | midnight, wall-clock time in `tz` |
| `pandas.Timestamp`, `numpy.datetime64` | as `datetime`, with nanoseconds |

`rounding` handles times finer than the unit: `"floor"`, `"ceil"` or
`"exact"` (an error). Reading functions round `start` up and `end` down;
points written through a `Transaction` must be exact.

## Adapters

`pytrosna.adapters` holds the conversions used by the high-level API:
`batch_to_arrow`, `batch_to_pandas`, `batch_to_polars` (reading), and
`normalize(data, time_column=None) -> Iterator[SourceTable]`,
`infer_device(name, source, unit=None) -> DeviceSchema`,
`to_batch(source, schema, unit=None) -> Batch` (writing). They can be used to
build custom pipelines, e.g. with a low-level `Writer`:

```text
for source in pytrosna.adapters.normalize(polars_frame):
    schema = writer.ensure_device(pytrosna.adapters.infer_device("vm01", source))
    writer.write("vm01", pytrosna.adapters.to_batch(source, schema))
```

## Errors

`TrosnaError` is the base class.

| Exception | Also a | Raised when |
|---|---|---|
| `CorruptedError` | | malformed or damaged data; `reason`, `offset` |
| `NotTrosnaError` | `CorruptedError` | not a Trosna file |
| `UnsupportedVersionError` | | a different major format version |
| `UnsupportedError` | | an unknown feature, or data Trosna cannot store |
| `NotFinalizedError` | | `Reader(strict=True)` of an unfinished file |
| `LockedError` | | another writer holds the file |
| `UnknownDeviceError`, `UnknownColumnError`, `UnknownAnnotationError`, `UnknownCommitError`, `PointNotFoundError` | `KeyError` | a missing object |
| `DeviceExistsError` | | the device already exists |
| `TypeMismatchError` | `TypeError` | a value of the wrong type |
| `SchemaError`, `InvalidArgumentError`, `LimitExceededError` | `ValueError` | invalid schema, argument or size |

## Constants

* `EXTENSION = "trosna"` — the file name extension.
* `FORMAT_VERSION = (1, 0)` — the format version written.
* `__version__` — the package version.
