"""Editing with transactions, interval annotations, time travel, diff,
verification and compaction.

Run: python examples/02_editing_and_history.py
"""

from pathlib import Path

import pandas as pd

import pytrosna

workdir = Path("pytrosna-examples")  # files are written here
workdir.mkdir(exist_ok=True)
path = workdir / "metrics.trosna"

df = pd.DataFrame(
    {
        "time": pd.date_range("2026-10-08 10:00", periods=6, freq="10s", tz="UTC"),
        "cpu": [0.31, 0.35, 0.97, 0.33, 0.30, 0.32],
        "ram_mb": [2048, 2051, 2049, 2050, 2047, 2049],
    }
)
pytrosna.write(path, df, device="vm01", message="import")
f = pytrosna.open(path)

# All changes of a `with` block become ONE commit; an exception discards them.
with f.edit(message="remove the spike", author="analyst") as tx:
    tx.update("vm01", "2026-10-08 10:00:20", cpu=0.34)  # change one value
    tx.delete("vm01", "2026-10-08 10:00:40")  # delete a point
    tx.insert("vm01", "2026-10-08 10:01:00", cpu=0.29, ram_mb=2046)  # add a point
    tx.annotate(  # label an interval
        "vm01", "2026-10-08 10:00:15", "2026-10-08 10:00:25", "spike", "backup job"
    )
print("commit", tx.commit.number, "annotation ids", tx.annotation_ids)

print(f.read_pandas())  # the current version
print(f.read_pandas(as_of=1))  # the data as imported: nothing is lost

for change in f.diff(1).points:  # what commit 2 changed
    print(change.kind.value, change.time, change.before, "->", change.after)

for c in f.commits():  # the hash-chained history
    print(c.number, c.short_hash, c.time.isoformat(), c.author, c.message)

print(f.annotations())

# A failed transaction leaves no trace.
try:
    with f.edit() as tx:
        tx.delete("vm01", "2026-10-08 10:00:00")
        msg = "changed my mind"
        raise RuntimeError(msg)
except RuntimeError:
    pass
assert len(f.commits()) == 2

# Integrity: checksums of every frame and segment and the whole hash chain.
report = f.verify()
print("verify:", "OK" if report.ok else report.problems, "head", report.head[:12])

# Compaction drops the history but links the new file to it.
small = f.compact(workdir / "metrics-compact.trosna", overwrite=True)
print("compacted:", small.size, "bytes, origin", small.metadata["trosna.compacted_from"][:12])
