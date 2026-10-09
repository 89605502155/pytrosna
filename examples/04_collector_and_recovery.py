"""A data collector appends measurements; a crash leaves the file readable.

Run: python examples/04_collector_and_recovery.py
"""

import time
from pathlib import Path

import pytrosna
from pytrosna import DeviceSchema, Writer

workdir = Path("pytrosna-examples")  # files are written here
workdir.mkdir(exist_ok=True)
path = workdir / "collector.trosna"

pytrosna.create(path, DeviceSchema.build("sensor", "ms", {"value": "float64"}), overwrite=True)

# The collector commits every few readings: each commit is durable (fsync).
w = Writer.open(path)
start = int(time.time() * 1000)
for i in range(35):
    w.write_row("sensor", start + i * 100, value=float(i))
    if i % 10 == 9:
        w.commit(message=f"readings up to {i}")

# Simulate a crash: the process dies before committing the last 5 readings
# and before closing the file (so it has no index).
w._release()

# Readers see every completed commit: 30 points.
f = pytrosna.open(path)
print("finalized:", f.finalized, "points:", f.count())
assert f.count() == 30

# The next writer (or pytrosna.recover) removes the unfinished tail and continues.
report = pytrosna.recover(path)
print("recovered:", not report.was_finalized, report.recovery)
print(pytrosna.verify(path))
