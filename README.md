# pytrosna

**Read, write and edit Trosna time-series files (`.trosna`) in pure Python.**

[Русская версия](https://github.com/89605502155/pytrosna/blob/main/README.ru.md) ·
[User guide](https://github.com/89605502155/pytrosna/blob/main/docs/guide.md) ·
[API reference](https://github.com/89605502155/pytrosna/blob/main/docs/api.md) ·
[Examples](https://github.com/89605502155/pytrosna/tree/main/examples) ·
[Format specification](https://github.com/89605502155/trosna-file/blob/main/docs/SPEC.md)

Trosna is a file format for time series. A `.trosna` file is as easy to
create and read as CSV, but stores data the way time-series databases do:
column by column, with encodings designed for time stamps and measurements,
and with a verifiable history of every change.

`pytrosna` is a pure-Python implementation of format version 1.0, fully
compatible with the reference Rust implementation
([trosna-file](https://github.com/89605502155/trosna-file)): files written by
either one are read, edited and verified by the other. The writer uses the
same algorithms, down to the choice of encodings, and reproduces the golden
file of the Rust implementation byte for byte. (Where a segment is
compressed with Zstandard, the two libraries may produce different but
equivalent compressed bytes.)

* **Compact.** Every column segment is encoded with the smallest of
  delta-of-delta, TS_2DIFF-style bit packing, Gorilla XOR, run-length and
  dictionary encodings, then compressed with Zstandard or LZ4. A sensor
  series takes **1.5 bytes per point** (CSV: 16.9, Parquet + zstd: 2.6).
* **Editable, with history.** Insert, change and delete points anywhere on
  the time axis and label time intervals. Every change is an atomic commit;
  every earlier version stays readable (`as_of=`). Commits form a SHA-256 hash
  chain, so the history is tamper-evident.
* **Crash-safe.** The file is an append-only log of checksummed frames. After
  a crash it reads as of its last commit; the next writer removes the
  unfinished tail.
* **Several devices per file**, each a table with its own columns, time unit
  (s, ms, us, ns) and time zone.
* **Works with the Python data stack.** Read into and write from pandas,
  Polars and PyArrow, or use NumPy-backed batches without any of them.
* **Pure Python.** No compiler needed: the dependencies are NumPy, `lz4`,
  `crc32c` and, before Python 3.14, `zstandard`.

## Installation

```sh
pip install pytrosna                 # NumPy-backed core
pip install "pytrosna[pandas]"       # + pandas
pip install "pytrosna[polars]"       # + Polars
pip install "pytrosna[arrow]"        # + PyArrow
pip install "pytrosna[all]"          # everything

uv add "pytrosna[all]"               # with uv
```

Python 3.11 or newer is required.

## Quick start

```python
import pandas as pd
import pytrosna

df = pd.DataFrame(
    {
        "time": pd.date_range("2026-10-08 10:00", periods=5, freq="10s", tz="Europe/Moscow"),
        "temperature": [21.5, 21.6, 21.7, 21.6, 21.8],
        "humidity": [40, 41, 41, 42, 42],
    }
)

pytrosna.write("room.trosna", df, device="room1")  # create the file (one commit)

pytrosna.read_pandas("room.trosna")  # everything, as pandas
# a time range and a column, as Polars:
pytrosna.read_polars("room.trosna", start="2026-10-08 10:00:10", columns=["temperature"])
pytrosna.read_arrow("room.trosna")  # a pyarrow.Table
pytrosna.read("room.trosna")  # a NumPy-backed pytrosna.Batch
```

Times can be `datetime`, `pandas.Timestamp`, `numpy.datetime64`, ISO 8601 text
or raw integers. Text without a UTC offset is a wall-clock time of the
device's time zone.

## Editing, annotations and time travel

```python
f = pytrosna.open("room.trosna")

with f.edit(message="sensor check", author="me") as tx:  # one atomic commit
    tx.update("room1", "2026-10-08 10:00:20", temperature=21.65)
    tx.delete("room1", "2026-10-08 10:00:10")
    tx.insert("room1", "2026-10-08 10:01:00", temperature=22.0)
    tx.annotate("room1", "2026-10-08 10:00:15", "2026-10-08 10:00:25", "door open")
# if the block raises, nothing is written

f.read_pandas()  # the current data
f.read_pandas(as_of=1)  # the data as first imported
f.diff(1).points  # what changed since commit 1
f.annotations()  # labelled intervals
f.commits()  # the hash-chained history
f.verify().ok  # checksums, hash chain, decoding of all data
f.compact("small.trosna")  # a copy without history that links to it
```

Appending is just as simple:

```python
more_rows = df.assign(time=df["time"] + pd.Timedelta("1h"))
pytrosna.write("room.trosna", more_rows, device="room1", mode="a")
```

## Low-level API

`Writer` and `Reader` work with raw integer time stamps and NumPy arrays and
need no other libraries:

```python
import numpy as np
from pytrosna import DeviceSchema, Reader, Writer

with Writer.create("vms.trosna") as w:  # closing commits
    w.create_device(DeviceSchema.build("vm01", "ms", {"cpu": "float64", "ram_mb": "int64"}))
    w.write(
        "vm01",
        {
            "time": np.arange(0, 10_000, 1000),
            "cpu": np.random.rand(10),
            "ram_mb": np.full(10, 2048),
        },
    )
    w.commit(message="first measurements")
    w.update("vm01", 3000, cpu=0.99)
    w.delete_range("vm01", 7000, 9000)

with Reader("vms.trosna") as r:
    batch = r.query("vm01").columns(["cpu"]).time_range(0, 5000).collect()
    batch.time, batch["cpu"].values  # NumPy arrays
    r.query("vm01").as_of(1).count()  # 10 points in version 1
```

## Command line

The package installs a `pytrosna` command:

```sh
pytrosna convert room.csv room.trosna --device room1    # CSV → Trosna (types are detected)
pytrosna info room.trosna                               # devices, history, storage per column
pytrosna cat room.trosna --from "2026-10-08 10:00" --format table
pytrosna update room.trosna --time "2026-10-08 10:00:20" temperature=21.65 -m "sensor check"
pytrosna annotate room.trosna --start "2026-10-08 10:00:15" --end "2026-10-08 10:00:25" --label "door open"
pytrosna log room.trosna
pytrosna diff room.trosna --from 1
pytrosna verify room.trosna
pytrosna convert room.trosna back.csv                   # Trosna → CSV
```

`pytrosna --help` lists all commands: `info`, `cat`, `convert`, `insert`,
`update`, `delete`, `annotate`, `unannotate`, `annotations`, `log`, `diff`,
`verify`, `recover`, `compact`.

## Size and speed

One million readings of a slowly varying sensor (values rounded to 0.01, one
per second), with Zstandard:

| Format                      | Bytes per point |
|-----------------------------|-----------------|
| **Trosna (pytrosna)**       | **1.49**        |
| Arrow IPC (Feather), zstd   | 2.18            |
| Parquet, zstd               | 2.56            |
| CSV, gzip                   | 3.14            |
| CSV                         | 16.90           |

pytrosna writes about 0.8 and reads about 0.6 million points per second on a
laptop: its encoders and decoders are vectorized with NumPy where the format
allows it. When speed matters more than having no compiled dependencies, the
Rust implementation is faster; the files are the same.

## Compatibility and testing

* Format version 1.0, as specified in
  [SPEC.md](https://github.com/89605502155/trosna-file/blob/main/docs/SPEC.md).
* The writer reproduces the golden file of the Rust implementation **byte for
  byte**, and the test suite (about 400 tests, 98 % line coverage) includes
  interoperability tests that run the reference `trosna` tool on files written
  by pytrosna and vice versa. The code examples of this README and of the
  guides are executed by the tests as well.
* Property-based tests compare every encoder with a reference implementation
  written directly from the specification; a model-based test checks the
  editing semantics at every version; files are truncated at every byte
  offset and corrupted with random bit flips to check recovery and error
  handling.

## Development

```sh
uv sync                       # create the environment with all extras and tools
uv run pytest                 # the test suite (TROSNA_CLI=/path/to/trosna enables interop tests)
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv build                      # sdist and wheel in dist/
```

Publishing to PyPI is described in
[docs/publishing.md](https://github.com/89605502155/pytrosna/blob/main/docs/publishing.md).

## Citation

If you use Trosna in research, please cite the review that motivated it:

> Ferubko A. O., Kazakov O. D. Review of data structures and file formats for
> storage and transfer of time series with regard to their support in the
> Python and Rust ecosystems. 2026.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
