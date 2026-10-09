"""Interval annotations as training labels for machine learning.

Run: python examples/05_annotations_as_labels.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import pytrosna

workdir = Path("pytrosna-examples")  # files are written here
workdir.mkdir(exist_ok=True)
path = workdir / "memory.trosna"

times = pd.date_range("2026-10-08", periods=24 * 60, freq="min", tz="UTC")
rng = np.random.default_rng(3)
used = 4000 + np.cumsum(rng.normal(0, 5, times.size))
used[600:660] += 2500  # a memory spike
pytrosna.write(path, pd.DataFrame({"time": times, "used_mb": used}), device="host")

f = pytrosna.open(path)
with f.edit(message="expert labels", author="expert") as tx:
    tx.annotate("host", times[600], times[659], "overload", "memory balloon inflated")
    tx.annotate("host", times[0], times[599], "normal")
    tx.annotate("host", times[660], times[-1], "normal")

# Turn the labels into a column of the DataFrame.
df = f.read_pandas()
raw = f.read().time  # raw time stamps in the device's unit
df["label"] = None
for a in f.annotations("host"):
    df.loc[(raw >= a.start) & (raw <= a.end), "label"] = a.label
print(df["label"].value_counts())
print(df.iloc[598:603])
