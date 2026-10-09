"""The low-level API: Writer and Reader with raw integer time stamps and NumPy.

It needs neither pandas nor Polars nor PyArrow.

Run: python examples/03_low_level.py
"""

from pathlib import Path

import numpy as np

from pytrosna import Batch, Column, DeviceSchema, Reader, WriteOptions, Writer

workdir = Path("pytrosna-examples")  # files are written here
workdir.mkdir(exist_ok=True)
path = workdir / "vms.trosna"

schema = DeviceSchema.build(
    "vm01",
    "ms",  # time stamps are integers: milliseconds since 1970-01-01 UTC
    {"cpu": "float64", "ram_mb": "int64", "state": "string"},
    metadata={"host": "node-7"},
)

options = WriteOptions(codec="zstd", rows_per_block=10_000, sync=False, overwrite=True)
with Writer.create(path, options) as w:
    w.set_metadata("project", "adaptive memory management")
    w.create_device(schema)

    n = 50_000
    t0 = 1_791_442_800_000
    rng = np.random.default_rng(1)
    w.write(
        "vm01",
        {
            "time": t0 + np.arange(n) * 1000,
            "cpu": np.round(rng.random(n), 3),
            "ram_mb": 2048 + rng.integers(0, 64, n),
            "state": rng.choice(["running", "paused"], n, p=[0.95, 0.05]),
        },
    )
    w.commit(message="bulk load")

    # Rows can also be written one by one, edited and deleted.
    w.write_row("vm01", t0 + n * 1000, cpu=0.5, ram_mb=4096, state="running")
    w.update("vm01", t0, ram_mb=1024)
    w.delete_range("vm01", t0 + 1000, t0 + 9000)
    print("current row:", w.get("vm01", t0))
    w.commit(message="edits")

with Reader(path) as r:
    query = r.query("vm01").columns(["cpu"]).time_range(t0, t0 + 59_000)
    batch = query.collect()  # a Batch of NumPy arrays
    print(len(batch), batch.time[:3], batch["cpu"].values[:3])
    print("points:", r.query("vm01").count(), "in version 1:", r.query("vm01").as_of(1).count())

    # Stream a large device block by block.
    total = sum(len(b) for b in r.query("vm01").batches())
    print("streamed", total, "rows")

    for block in r.blocks("vm01")[:2]:
        print(
            block.rows,
            "rows:",
            [(s.column, s.encoding.label, s.codec.label) for s in block.segments],
        )

# Build a Batch by hand with explicit nulls.
manual = Batch(
    [1, 2, 3],
    {"cpu": Column.from_values("float64", [0.1, None, 0.3])},
)
print(manual, manual.to_dict())
