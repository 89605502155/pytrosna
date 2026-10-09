"""Quick start: write a pandas DataFrame to a .trosna file and read it back.

Run: python examples/01_quickstart.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import pytrosna

workdir = Path("pytrosna-examples")  # files are written here
workdir.mkdir(exist_ok=True)
path = workdir / "room.trosna"

# Ten minutes of readings, one per second, in Moscow time.
times = pd.date_range("2026-10-08 10:00", periods=600, freq="s", tz="Europe/Moscow")
rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "time": times,
        "temperature": np.round(21.5 + np.cumsum(rng.normal(0, 0.01, times.size)), 2),
        "humidity": rng.integers(38, 45, times.size),
        "door_open": rng.random(times.size) < 0.05,
    }
)

# One call creates the file (one atomic commit). The time zone of the time
# column becomes the device's time zone.
commit = pytrosna.write(path, df, device="room1", message="first import", author="me")
print(f"wrote {path.name}: {path.stat().st_size} bytes, commit {commit.short_hash}")
csv_size = len(df.to_csv(index=False).encode())
per_row = path.stat().st_size / len(df)
print(f"{per_row:.2f} bytes per row; the same data as CSV: {csv_size / len(df):.1f}")

# Read everything back into pandas ...
back = pytrosna.read_pandas(path)
print(back.head())
assert back["temperature"].tolist() == df["temperature"].tolist()

# ... or only a time range and some columns, into Polars.
part = pytrosna.read_polars(
    path, start="2026-10-08 10:05:00", end="2026-10-08 10:05:09", columns=["temperature"]
)
print(part)

# The file object gives the schema and the stored blocks.
f = pytrosna.open(path)
device = f.device()
print(device.name, device.time_unit, device.timezone, device.column_names)
for segment in f.blocks()[0].segments:
    print(f"  {segment.column:<12} {segment.encoding.label:<15} {segment.stored_bytes} B")
