# pytrosna user guide

[Русская версия](guide.ru.md) · [API reference](api.md) · [README](../README.md)

This guide walks through everything pytrosna can do, from the first file to
crash recovery. Every Python example on this page is executed by the test
suite, in order, so the examples build on each other.

1. [Concepts](#1-concepts)
2. [Creating a file](#2-creating-a-file)
3. [Reading](#3-reading)
4. [Times and time zones](#4-times-and-time-zones)
5. [Appending](#5-appending)
6. [Editing with transactions](#6-editing-with-transactions)
7. [Annotations](#7-annotations)
8. [History and time travel](#8-history-and-time-travel)
9. [Several devices](#9-several-devices)
10. [The low-level Writer and Reader](#10-the-low-level-writer-and-reader)
11. [Storage, encodings and codecs](#11-storage-encodings-and-codecs)
12. [Integrity, crashes and recovery](#12-integrity-crashes-and-recovery)
13. [Compaction](#13-compaction)
14. [The command line](#14-the-command-line)
15. [Type mapping](#15-type-mapping)
16. [Errors](#16-errors)

## 1. Concepts

* A **file** (`*.trosna`) holds any number of **devices** and file-level
  **metadata** (string → string).
* A **device** is a table: a time column and named value columns of the
  types `bool`, `int32`, `int64`, `float32`, `float64` and `string`. Every
  value may be null. A device has a **time unit** (`s`, `ms`, `us`, `ns`), an
  optional **time zone** (an IANA name such as `Europe/Moscow`, `UTC` or a
  fixed offset such as `+03:00`) and its own metadata. Its schema is fixed
  when it is created.
* Time stamps are stored as integers: the number of time units since
  1970-01-01 UTC. At any version, the time stamps of a device are unique — a
  device is a function from time to a row.
* Every change is part of a **commit**, an atomic, hashed set of changes. The
  file keeps every version: you can read it "as of" any commit.
* An **annotation** is a labelled closed interval `[start, end]` of a device,
  for example an anomaly or a regime, with an optional note.

## 2. Creating a file

The simplest way is `pytrosna.write`, which takes a pandas or Polars
DataFrame, a PyArrow table, a `pytrosna.Batch` or a plain mapping:

```python
import numpy as np
import pandas as pd
import pytrosna

times = pd.date_range("2026-10-08 10:00", periods=60, freq="10s", tz="Europe/Moscow")
rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "time": times,
        "temperature": np.round(21.5 + np.cumsum(rng.normal(0, 0.02, times.size)), 2),
        "humidity": rng.integers(38, 45, times.size),
        "door_open": rng.random(times.size) < 0.1,
        "status": rng.choice(["ok", "check"], times.size, p=[0.9, 0.1]),
    }
)

commit = pytrosna.write("room.trosna", df, device="room1", message="first import", author="lab")
print(commit.number, commit.short_hash, commit.changes.rows_written)
```

What happened:

* The **time column** was found automatically: the first timestamp column,
  or a column named `time`, `timestamp`, `ts`, `datetime`, `date`, `t` or
  `время`. Pass `time_column="..."` to choose another one. A pandas
  `DatetimeIndex` is used as the time when there is no time column.
* The **time unit** and **time zone** of the device were taken from the
  column (`datetime64[us, Europe/Moscow]`).
* The **column types** were derived from the dtypes (`float64`, `int64`,
  `bool`, `string`).
* `mode="w"` (the default) replaced any existing file; `mode="x"` refuses to
  overwrite and `mode="a"` appends. A replacing write goes to a temporary file
  first, so a failure never damages an existing file.

Integer time stamps need their unit:

```python
pytrosna.write(
    "counters.trosna",
    {"time": [1_791_442_800, 1_791_442_801, 1_791_442_802], "requests": [10, 12, None]},
    device="web",
    unit="s",
)
print(pytrosna.read_pandas("counters.trosna"))
```

`None` is a null. To create an empty file with a fixed schema, use
`pytrosna.create`:

```python
from pytrosna import ColumnSchema, DeviceSchema

schema = DeviceSchema(
    "boiler",
    "ms",
    (
        ColumnSchema("pressure", "float32", {"unit": "bar"}),
        ColumnSchema("burner_on", "bool"),
    ),
    timezone="UTC",
    metadata={"site": "Bryansk"},
)
pytrosna.create("boiler.trosna", schema, metadata={"owner": "heating"}, overwrite=True)
print(pytrosna.open("boiler.trosna").devices)
```

## 3. Reading

```python
pytrosna.read_pandas("room.trosna")  # pandas.DataFrame
pytrosna.read_polars("room.trosna")  # polars.DataFrame
pytrosna.read_arrow("room.trosna")  # pyarrow.Table
batch = pytrosna.read("room.trosna")  # pytrosna.Batch (NumPy only)
print(batch, batch["temperature"].values[:3], batch.time[:3])
```

All readers take the same options:

```python
part = pytrosna.read_pandas(
    "room.trosna",
    columns=["temperature", "status"],  # the time column is always included
    start="2026-10-08 10:02:00",  # inclusive
    end="2026-10-08 10:03:00",  # inclusive
)
print(part)
```

Reading through a `File` object avoids re-opening the file for every call
and gives access to the schema and history:

```python
f = pytrosna.open("room.trosna")
print(f.devices["room1"].column_names, f.count(), f.time_range())
for chunk in f.iter_batches(columns=["temperature"]):  # streamed block by block
    print(len(chunk), chunk["temperature"].values.mean())
reader = f.read_batches()  # a pyarrow.RecordBatchReader
print(reader.schema)
```

With pandas, integer columns that contain nulls become `float64` with `NaN`
(as with PyArrow); `dtype_backend="numpy_nullable"` keeps them as `Int64`:

```python
print(pytrosna.read_pandas("counters.trosna", dtype_backend="numpy_nullable").dtypes)
```

## 4. Times and time zones

Wherever the API takes a time, it accepts:

* `datetime.datetime` / `datetime.date`, `pandas.Timestamp`, `numpy.datetime64`;
* ISO 8601 text: `"2026-10-08T10:00:00+03:00"`, `"2026-10-08 10:00:00.250"`,
  `"2026-10-08"`;
* an integer — a raw time stamp in the device's unit.

A time **without a UTC offset** is a wall-clock time of the device's time
zone. In a repeated hour (clocks moved back) the earlier moment is taken; a
time skipped when the clocks were moved forward is an error. Range bounds
that fall between two time stamps are rounded inwards; the time of a point
you insert must be exact in the device's unit.

```python
from pytrosna import to_raw, to_datetime, format_time

raw = to_raw("2026-10-08 10:00", "s", "Europe/Moscow")
print(raw, to_datetime(raw, "s", "Europe/Moscow"), format_time(raw, "s", "+03:00"))
print(f.to_raw("2026-10-08 10:00:10"), f.to_datetime(f.to_raw("2026-10-08 10:00:10")))
```

## 5. Appending

`mode="a"` appends to an existing device (or creates the device or the file
if needed). Each call is one commit. A row with the time stamp of an
existing point replaces it (the last write wins).

```python
later = df.assign(time=df["time"] + pd.Timedelta("10min"))
f.write(later, "room1", message="next ten minutes")  # same as mode="a"
print(f.count(), len(f.commits()))
```

Naive time stamps appended to a device with a time zone are read in that
zone:

```python
naive = pd.DataFrame({"time": pd.to_datetime(["2026-10-08 10:30:00"]), "temperature": [22.0]})
f.write(naive, "room1")
print(f.read_pandas(start="2026-10-08 10:30").tail(1))
```

## 6. Editing with transactions

`File.edit()` collects changes; they become **one commit** when the `with`
block ends normally and are discarded if it raises:

```python
with f.edit(message="sensor check", author="technician") as tx:
    tx.update("room1", "2026-10-08 10:00:20", temperature=21.65)  # change values
    tx.insert("room1", "2026-10-08 10:31:00", temperature=22.1)  # missing columns are null
    tx.delete("room1", "2026-10-08 10:00:10")  # one point
    tx.delete_range("room1", "2026-10-08 10:05:00", "2026-10-08 10:05:30")  # an interval
    tx.set_metadata("calibrated", "2026-10-08")
print(tx.commit.number, tx.commit.changes)
```

* `update` changes only the given columns and fails with
  `PointNotFoundError` if there is no point at that time.
* `insert` writes a full row: columns that are not given are null.
* `delete_range(device, start=None, end=None)` accepts open ends.
* `tx.write(device, data)` adds a whole table inside the transaction.

```python
try:
    with f.edit() as tx:
        tx.delete("room1", "2026-10-08 10:00:30")
        raise RuntimeError("abort")
except RuntimeError:
    pass
print(len(f.commits()))  # unchanged: the transaction left no trace
```

## 7. Annotations

Annotations label intervals of a device — anomalies, regimes, experiments —
and are versioned like the data:

```python
with f.edit(message="labels") as tx:
    tx.annotate("room1", "2026-10-08 10:02", "2026-10-08 10:04", "ventilation", "window open")
    tx.annotate("room1", "2026-10-08 10:06", "2026-10-08 10:06", "door slam")  # an instant
ids = tx.annotation_ids  # identifiers of the new annotations
print(ids)

with f.edit() as tx:
    tx.update_annotation(ids[0], label="airing")  # other fields keep their values
    tx.remove_annotation(ids[1])

for a in f.annotations():
    print(a.id, a.label, f.to_datetime(a.start), f.to_datetime(a.end), a.note)
```

Identifiers are unique within the file and never reused. `annotations(as_of=n)`
returns the annotations of an older version.

Annotations make convenient training labels:

```python
data = f.read_pandas()
raw = f.read().time
data["label"] = None
for a in f.annotations("room1"):
    data.loc[(raw >= a.start) & (raw <= a.end), "label"] = a.label
print(data["label"].value_counts())
```

## 8. History and time travel

```python
for c in f.commits():
    print(c.number, c.short_hash, c.time.isoformat(timespec="seconds"), c.author, c.message)

first = f.read_pandas(as_of=1)  # the file as imported
print(len(first), len(f.read_pandas()))

changes = f.diff(1)  # from commit 1 to the latest
for p in changes.points[:5]:
    print(p.kind.value, f.to_datetime(p.time), p.before, "->", p.after)
for a in changes.annotations:
    print(a.kind.value, a.id, a.after or a.before)

version = f.commit_at("2100-01-01T00:00:00Z")  # the version at a moment
print(version)
```

`as_of=0` is the empty file. Each commit stores the SHA-256 hash of the
previous one, so the hash of the latest commit identifies the whole history:
publish it (for example in a paper or a lab journal) and nobody can silently
change a measurement afterwards.

## 9. Several devices

```python
pytrosna.write("plant.trosna", df, device="room1")
pytrosna.write(
    "plant.trosna", df.assign(temperature=df["temperature"] + 5), device="room2", mode="a"
)
plant = pytrosna.open("plant.trosna")
print(list(plant.devices))
print(plant.read_pandas("room2").head(2))
print(plant.count(device="room1"))
```

With several devices, pass the device name to reading functions.

## 10. The low-level Writer and Reader

`Writer` and `Reader` give full control and need no data-frame library.
Times are raw integers in the device's unit.

```python
from pytrosna import Batch, Column, Reader, WriteOptions, Writer

schema = DeviceSchema.build("vm01", "ms", {"cpu": "float64", "ram_mb": "int64", "state": "string"})
options = WriteOptions(codec="zstd", rows_per_block=4096, sync=True, overwrite=True)

with Writer.create("vms.trosna", options) as w:
    w.create_device(schema)
    t0 = 1_791_442_800_000
    w.write(
        "vm01",
        {
            "time": t0 + np.arange(10_000) * 1000,
            "cpu": np.round(rng.random(10_000), 3),
            "ram_mb": 2048 + rng.integers(0, 64, 10_000),
            "state": ["running"] * 10_000,
        },
    )
    w.commit(message="bulk load", author="collector")

    w.write_row("vm01", t0 + 10_000 * 1000, cpu=0.5, ram_mb=4096)  # one row
    w.update("vm01", t0, ram_mb=1024)  # change a value
    print(w.get("vm01", t0))  # sees uncommitted changes
    w.delete_range("vm01", t0 + 1000, t0 + 5000)
    w.annotate("vm01", t0, t0 + 60_000, "warm-up")
    w.commit(message="edits")
    w.write_row("vm01", t0 - 1000, cpu=0.0)
    w.rollback()  # discard since the last commit
# leaving the block commits pending changes and writes the index;
# leaving it with an exception discards them

with Reader("vms.trosna") as r:
    q = r.query("vm01").columns(["cpu", "ram_mb"]).time_range(t0, t0 + 59_000)
    b = q.collect()
    print(len(b), b.time[:2], b["ram_mb"].values[:2])
    print(r.query("vm01").count(), r.query("vm01").as_of(1).count())
    for chunk in r.query("vm01").batches():
        pass
    print(r.metadata, r.head.message, [d.name for d in r.devices])
```

A `Batch` holds `time` (an `int64` array) and `Column` objects with
`values` (a NumPy array) and an optional `validity` mask:

```python
manual = Batch([1, 2, 3], {"cpu": Column.from_values("float64", [0.1, None, 0.3])})
print(manual.to_dict(), manual.row(1), manual["cpu"].null_count)
```

## 11. Storage, encodings and codecs

Each block (`rows_per_block` rows, 65 536 by default) stores the time column
and every value column as a separate segment. For each segment the writer
tries every applicable encoding and keeps the smallest result:

| Column type     | Encodings tried                                      |
|-----------------|------------------------------------------------------|
| time            | delta-of-delta, delta-bitpack, plain                 |
| int32, int64    | delta-bitpack, RLE, plain                            |
| float32/64      | Gorilla XOR, plain                                   |
| bool            | RLE, plain bitmap                                    |
| string          | dictionary, plain                                    |

The encoded segment is then compressed with the codec (`zstd` by default,
`lz4` or `none`) if that makes it smaller. `File.blocks()` shows the choice:

```python
for block in pytrosna.open("vms.trosna").blocks()[:1]:
    for s in block.segments:
        print(
            f"{s.column:<8} {s.encoding.label:<15} {s.codec.label:<5} {s.stored_bytes:>6} B",
            s.statistics,
        )
```

Options of `WriteOptions` (also accepted as keyword arguments by
`Writer.create`, `Writer.open` and `pytrosna.write`):

| Option              | Default        | Meaning                                                   |
|---------------------|----------------|-----------------------------------------------------------|
| `codec`             | `"zstd"`       | `"zstd"`, `"lz4"` or `"none"`                             |
| `encoding`          | `"adaptive"`   | `"adaptive"`, `"classic"` (one encoding per type), `"plain"` |
| `rows_per_block`    | 65 536         | rows per block (1 … 2²⁴)                                  |
| `max_block_bytes`   | 64 MiB         | a device's buffer is written once it holds this much      |
| `sync`              | `True`         | `fsync` at every commit                                   |
| `overwrite`         | `False`        | `Writer.create` replaces an existing file                 |
| `repair_corruption` | `False`        | `Writer.open` cuts off data after damage in the middle    |
| `zstd_level`        | 3              | Zstandard compression level                               |

```python
pytrosna.write("plain.trosna", df, device="room1", codec="none", encoding="plain")
print(
    pytrosna.open("plain.trosna").size,
    "bytes without compression vs",
    pytrosna.open("room.trosna").size,
)
```

## 12. Integrity, crashes and recovery

```python
report = pytrosna.verify("room.trosna")
print(report.ok, report.commits, report.blocks, report.finalized, report.head[:12])
print(report.problems, report.warnings)
```

`verify` checks the CRC-32C of every frame and segment, decodes all data,
recomputes the hash chain of commits and compares the index with the frames.

The file is an append-only log. If a program dies while writing, readers see
the state of the last complete commit, and the next writer (or `recover`)
removes the unfinished tail:

```python
from pathlib import Path

data = Path("room.trosna").read_bytes()
Path("crashed.trosna").write_bytes(data[:-30])  # simulate an interrupted write
print(pytrosna.open("crashed.trosna").finalized)  # False, but readable
print(pytrosna.recover("crashed.trosna"))  # removes the tail, writes a new index
print(pytrosna.verify("crashed.trosna").ok)
```

If damage is found in the middle of a file (valid frames follow it), the
writer refuses to truncate it; `recover(path, force=True)` cuts off
everything after the damage.

A file can be written by one writer at a time (an exclusive lock is taken)
and read by any number of readers.

## 13. Compaction

Edits are stored as new blocks and tombstones; reading merges them.
Compaction writes the state of one version into a new file without history
and without overlapping blocks. Its first commit links to the hash of the
source version, and its metadata records the origin:

```python
small = f.compact("room-compact.trosna", overwrite=True)
print(small.size, "<", f.size, small.metadata["trosna.compacted_from"][:12])
pytrosna.compact("room.trosna", "room-v1.trosna", as_of=1, overwrite=True)  # an older version
```

## 14. The command line

```sh
pytrosna convert data.csv data.trosna --device sensor1     # CSV → Trosna
pytrosna convert data.csv data.trosna --append --device sensor1
pytrosna convert data.trosna back.csv --as-of 2            # Trosna → CSV (a version)
pytrosna info data.trosna
pytrosna cat data.trosna --columns temperature --from "2026-10-08 10:00" --to "2026-10-08 11:00"
pytrosna cat data.trosna --format table --limit 20         # or --format json, --raw-time
pytrosna insert data.trosna --time "2026-10-08 10:01" temperature=22 -m "manual entry"
pytrosna update data.trosna --time "2026-10-08 10:00:20" temperature=21.65 --author me
pytrosna delete data.trosna --from "2026-10-08 10:30" --to "2026-10-08 10:40"
pytrosna annotate data.trosna --start "2026-10-08 10:00" --end "2026-10-08 10:05" --label warmup
pytrosna annotate data.trosna --id 1 --label "warm-up"     # change an annotation
pytrosna unannotate data.trosna 1
pytrosna annotations data.trosna
pytrosna log data.trosna
pytrosna diff data.trosna --from 1 --to 3
pytrosna verify data.trosna                                # exit status 1 if damaged
pytrosna recover data.trosna
pytrosna compact data.trosna small.trosna
```

CSV import detects the column types (integers → `int64`, decimals →
`float64`, `true`/`false` → `bool`, anything else → `string`; an empty cell is
null). The first column is the time unless `--time` says otherwise. ISO 8601
times with an offset set the device's time zone; integers are Unix epoch
values whose unit is guessed from their size or given with `--unit`.

## 15. Type mapping

| Trosna   | NumPy (`Batch`) | pandas (default / nullable)     | Polars             | PyArrow                |
|----------|-----------------|---------------------------------|--------------------|------------------------|
| time     | `int64` (raw)   | `datetime64[unit, tz]`          | `Datetime(unit, tz)` | `timestamp(unit, tz)` |
| bool     | `bool`          | `bool` / `object` / `boolean`   | `Boolean`          | `bool`                 |
| int32    | `int32`         | `int32` / `float64` / `Int32`   | `Int32`            | `int32`                |
| int64    | `int64`         | `int64` / `float64` / `Int64`   | `Int64`            | `int64`                |
| float32  | `float32`       | `float32` / `Float32`           | `Float32`          | `float32`              |
| float64  | `float64`       | `float64` / `Float64`           | `Float64`          | `float64`              |
| string   | `object` (`str`)| `object`/`str` / `string`       | `String`           | `string`               |

When writing, narrower types are widened losslessly (`int8`, `uint16` →
`int32`; `uint32`, `uint64` → `int64`; `float16` → `float32`; categorical and
dictionary strings → `string`). Writing to an existing device converts values
to its column types: integers accept whole floats, floats accept any number,
booleans and strings accept only their own type. In pandas, `NaN` means a
missing value; in NumPy arrays and Polars it is a value. Polars has no second
resolution, so second time stamps are converted to milliseconds, and fixed
offsets become `Etc/GMT±H` zones.

## 16. Errors

All errors derive from `pytrosna.TrosnaError`. The most common ones:

| Exception                | When                                                         |
|--------------------------|--------------------------------------------------------------|
| `CorruptedError`         | damaged or malformed data (`offset` tells where)             |
| `NotTrosnaError`         | the file is not a Trosna file                                |
| `UnknownDeviceError`     | no device with that name (also a `KeyError`)                 |
| `UnknownColumnError`     | no column with that name (also a `KeyError`)                 |
| `PointNotFoundError`     | `update` of a point that does not exist (also a `KeyError`)  |
| `TypeMismatchError`      | a value of the wrong type (also a `TypeError`)               |
| `InvalidArgumentError`   | an invalid argument, e.g. a time finer than the unit (`ValueError`) |
| `LockedError`            | another writer holds the file                                |
| `UnsupportedError`       | data Trosna cannot store, or a newer format feature          |

```python
try:
    with f.edit() as tx:
        tx.update("room1", "2030-01-01", temperature=0.0)
except pytrosna.PointNotFoundError as e:
    print("error:", e)
```
