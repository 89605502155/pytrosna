"""The ``pytrosna`` command-line tool (also ``python -m pytrosna``).

Run ``pytrosna --help`` or ``pytrosna <command> --help`` for the options.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections.abc import Iterator, Sequence
from typing import Any, TextIO

from . import api
from .errors import TrosnaError
from .reader import Reader
from .timeconv import format_time, to_raw
from .tools import compact, recover, verify
from .types import DataType, DeviceSchema, TimeUnit
from .writer import Writer

__all__ = ["main"]


class CliError(Exception):
    """An error in the command line or its input, reported without a traceback."""


# ---------------------------------------------------------------- helpers


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{n} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"  # pragma: no cover


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _pick_device(reader_devices: Sequence[DeviceSchema], name: str | None) -> DeviceSchema:
    if name is not None:
        for d in reader_devices:
            if d.name == name:
                return d
        msg = f"unknown device {name!r}"
        raise CliError(msg)
    if len(reader_devices) == 1:
        return reader_devices[0]
    if not reader_devices:
        msg = "the file has no devices"
        raise CliError(msg)
    names = ", ".join(d.name for d in reader_devices)
    msg = f"the file has several devices ({names}); choose one with --device"
    raise CliError(msg)


def _time_arg(device: DeviceSchema, text: str | None, rounding: str) -> int | None:
    if text is None:
        return None
    value: Any = int(text) if text.lstrip("-").isdigit() else text
    return to_raw(value, device.time_unit, device.timezone, rounding)


def _parse_value(column: str, data_type: DataType, text: str) -> Any:
    if text == "" or text.lower() == "null":
        return None
    try:
        if data_type is DataType.BOOL:
            lowered = text.lower()
            if lowered in ("true", "1", "yes"):
                return True
            if lowered in ("false", "0", "no"):
                return False
            raise ValueError(text)
        if data_type in (DataType.INT32, DataType.INT64):
            return int(text)
        if data_type in (DataType.FLOAT32, DataType.FLOAT64):
            return float(text)
    except ValueError:
        msg = f"cannot read {text!r} as a {data_type} value for column {column!r}"
        raise CliError(msg) from None
    return text


def _assignments(device: DeviceSchema, items: Sequence[str]) -> dict[str, Any]:
    values = {}
    for item in items:
        if "=" not in item:
            msg = f"expected column=value, got {item!r}"
            raise CliError(msg)
        name, text = item.split("=", 1)
        index = device.column_index(name)
        if index is None:
            msg = f"unknown column {name!r} in device {device.name!r}"
            raise CliError(msg)
        values[name] = _parse_value(name, device.columns[index].data_type, text)
    return values


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _rows(
    reader: Reader,
    device: DeviceSchema,
    *,
    columns: list[str] | None,
    start: int | None,
    end: int | None,
    as_of: int | None,
) -> Iterator[tuple[int, list[Any]]]:
    query = reader.query(device.name).time_range(start, end)
    if columns is not None:
        query = query.columns(columns)
    if as_of is not None:
        query = query.as_of(as_of)
    for batch in query.batches():
        lists = [c.to_list() for c in batch.columns.values()]
        for i, t in enumerate(batch.time.tolist()):
            yield t, [values[i] for values in lists]


# ---------------------------------------------------------------- CSV


def _guess_unit(values: Sequence[int]) -> TimeUnit:
    """The unit of integer epoch time stamps, guessed from their size."""
    biggest = max((abs(v) for v in values), default=0)
    if biggest < 10**11:
        return TimeUnit.SECOND
    if biggest < 10**14:
        return TimeUnit.MILLISECOND
    if biggest < 10**17:
        return TimeUnit.MICROSECOND
    return TimeUnit.NANOSECOND


def _infer_csv_type(cells: Sequence[str]) -> DataType:
    kinds = set()
    for cell in cells:
        if cell == "":
            continue
        lowered = cell.lower()
        if lowered in ("true", "false"):
            kinds.add("bool")
            continue
        try:
            int(cell)
        except ValueError:
            try:
                float(cell)
            except ValueError:
                kinds.add("string")
            else:
                kinds.add("float")
        else:
            kinds.add("int")
    if not kinds or "string" in kinds or ("bool" in kinds and len(kinds) > 1):
        return DataType.STRING
    if kinds == {"bool"}:
        return DataType.BOOL
    if "float" in kinds:
        return DataType.FLOAT64
    return DataType.INT64


def import_csv(
    source: str,
    target: str,
    *,
    device: str | None,
    time_column: str | None,
    unit: str | None,
    delimiter: str,
    append: bool,
    force: bool,
    codec: str,
    message: str | None,
    author: str | None,
) -> int:
    """Converts a CSV file into a Trosna file; returns the number of rows."""
    with open(source, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            msg = f"{source} is empty"
            raise CliError(msg) from None
        rows = list(reader)
    if len(set(header)) != len(header):
        msg = "the CSV header has duplicate column names"
        raise CliError(msg)
    time_name = time_column or header[0]
    if time_name not in header:
        msg = f"the CSV has no column {time_name!r}"
        raise CliError(msg)
    for number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            msg = f"line {number} has {len(row)} fields, the header has {len(header)}"
            raise CliError(msg)
    t = header.index(time_name)
    cells = [row[t] for row in rows]
    if any(c == "" for c in cells):
        msg = "the time column has empty cells"
        raise CliError(msg)
    data: dict[str, Any] = {}
    time_unit: str | None = unit
    if cells and all(c.lstrip("-").isdigit() for c in cells):
        stamps = [int(c) for c in cells]
        time_unit = unit or _guess_unit(stamps).label
        data[time_name] = stamps
    else:
        data[time_name] = cells
    for k, name in enumerate(header):
        if k == t:
            continue
        column = [row[k] for row in rows]
        data_type = _infer_csv_type(column)
        data[name] = [_parse_value(name, data_type, cell) for cell in column]
        if data_type is DataType.STRING:
            data[name] = [None if cell == "" else cell for cell in column]
    name = device or os.path.splitext(os.path.basename(source))[0] or "device"
    exists = os.path.exists(target)
    if exists and not append and not force:
        msg = f"{target} exists; use --append to add to it or --force to replace it"
        raise CliError(msg)
    mode = "a" if append else "w"
    if not rows and not exists:
        msg = "the CSV has no rows, so the column types are unknown"
        raise CliError(msg)
    api.write(
        target,
        data,
        name,
        time_column=time_name,
        unit=time_unit if all(isinstance(v, int) for v in data[time_name]) else None,
        mode=mode,
        message=message or f"import {os.path.basename(source)}",
        author=author,
        codec=codec,
    )
    return len(rows)


def export_csv(
    source: str,
    out: TextIO,
    *,
    device: str | None,
    columns: list[str] | None,
    start: str | None,
    end: str | None,
    as_of: int | None,
    raw_time: bool,
    delimiter: str = ",",
    limit: int | None = None,
) -> int:
    with Reader(source) as reader:
        schema = _pick_device(reader.devices, device)
        names = columns if columns is not None else schema.column_names
        writer = csv.writer(out, delimiter=delimiter, lineterminator="\n")
        writer.writerow([schema.time_name, *names])
        count = 0
        for t, values in _rows(
            reader,
            schema,
            columns=columns,
            start=_time_arg(schema, start, "ceil"),
            end=_time_arg(schema, end, "floor"),
            as_of=as_of,
        ):
            if limit is not None and count >= limit:
                break
            stamp = str(t) if raw_time else format_time(t, schema.time_unit, schema.timezone)
            writer.writerow([stamp, *(_format_value(v) for v in values)])
            count += 1
        return count


# ---------------------------------------------------------------- commands


def cmd_info(args: argparse.Namespace) -> int:
    with Reader(args.file) as reader:
        state = "finalized" if reader.finalized else "NOT finalized (run: pytrosna recover)"
        major, minor = reader.format_version
        print(
            f"file:     {args.file} ({_human_bytes(reader.size)}, format {major}.{minor}, {state})"
        )
        head = reader.head
        if head is None:
            print("history:  no commits")
        else:
            print(f"history:  {_plural(head.number, 'commit')}, head {head.hash}")
        for key, value in reader.metadata.items():
            print(f"metadata: {key} = {value}")
        for schema in reader.devices:
            count = reader.query(schema.name).count()
            print()
            print(f"device {schema.name!r}: {_plural(count, 'point')}")
            if count:
                batch = reader.read(schema.name, columns=())
                first = format_time(int(batch.time[0]), schema.time_unit, schema.timezone)
                last = format_time(int(batch.time[-1]), schema.time_unit, schema.timezone)
                print(f"  time range:  {first} … {last}")
            blocks = reader.blocks(schema.name)
            stored = sum(s.stored_bytes for b in blocks for s in b.segments)
            blocks_text = _plural(len(blocks), "block")
            print(f"  storage:     {blocks_text}, {_human_bytes(stored)} of segments")
            tz = f", time zone {schema.timezone}" if schema.timezone else ""
            print(f"  {schema.time_name:<20} time, unit {schema.time_unit}{tz}")
            for column in schema.columns:
                print(f"  {column.name:<20} {column.data_type}")
            if blocks:
                print("  storage by column (all blocks, including superseded ones):")
                rows = sum(b.rows for b in blocks)
                for k in range(len(schema.columns) + 1):
                    segments = [b.segments[k] for b in blocks]
                    stored_k = sum(s.stored_bytes for s in segments)
                    data_type = None if k == 0 else schema.columns[k - 1].data_type
                    plain = rows * 8 if data_type is None else rows * data_type.plain_width
                    ratio = (
                        f", K = {plain / stored_k:.2f}"
                        if data_type is not DataType.STRING and stored_k
                        else ""
                    )
                    used: dict[str, int] = {}
                    for s in segments:
                        key = f"{s.encoding.label}+{s.codec.label}"
                        used[key] = used.get(key, 0) + 1
                    kinds = ", ".join(f"{key} ×{n}" for key, n in used.items())  # noqa: RUF001
                    name = segments[0].column
                    bits = 8 * stored_k / rows if rows else 0
                    size = _human_bytes(stored_k)
                    print(f"    {name:<18} {size}, B = {bits:.2f} bit/point{ratio}; {kinds}")
    return 0


def _render_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in header]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row, strict=True)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows]
    return "\n".join(line.rstrip() for line in lines)


def cmd_cat(args: argparse.Namespace) -> int:
    columns = args.columns.split(",") if args.columns else None
    if args.format == "csv":
        export_csv(
            args.file,
            sys.stdout,
            device=args.device,
            columns=columns,
            start=args.start,
            end=args.end,
            as_of=args.as_of,
            raw_time=args.raw_time,
            limit=args.limit,
        )
        return 0
    with Reader(args.file) as reader:
        schema = _pick_device(reader.devices, args.device)
        names = columns if columns is not None else schema.column_names
        rows: list[list[str]] = []
        records: list[dict[str, Any]] = []
        for t, values in _rows(
            reader,
            schema,
            columns=columns,
            start=_time_arg(schema, args.start, "ceil"),
            end=_time_arg(schema, args.end, "floor"),
            as_of=args.as_of,
        ):
            if args.limit is not None and len(rows) >= args.limit:
                break
            stamp = str(t) if args.raw_time else format_time(t, schema.time_unit, schema.timezone)
            rows.append([stamp, *(_format_value(v) for v in values)])
            records.append(
                {
                    schema.time_name: t if args.raw_time else stamp,
                    **dict(zip(names, values, strict=True)),
                }
            )
    if args.format == "table":
        print(_render_table([schema.time_name, *names], rows))
    else:
        for record in records:
            print(json.dumps(record, ensure_ascii=False))
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    if args.input.endswith(".trosna"):
        if os.path.exists(args.output) and not args.force:
            msg = f"{args.output} exists; use --force to replace it"
            raise CliError(msg)
        temp = f"{args.output}.{os.getpid()}.tmp"
        try:
            with open(temp, "w", newline="", encoding="utf-8") as out:
                count = export_csv(
                    args.input,
                    out,
                    device=args.device,
                    columns=None,
                    start=None,
                    end=None,
                    as_of=args.as_of,
                    raw_time=args.raw_time,
                    delimiter=args.delimiter,
                )
            os.replace(temp, args.output)
        except BaseException:
            if os.path.exists(temp):
                os.remove(temp)
            raise
        print(f"wrote {_plural(count, 'row')} to {args.output}")
        return 0
    count = import_csv(
        args.input,
        args.output,
        device=args.device,
        time_column=args.time,
        unit=args.unit,
        delimiter=args.delimiter,
        append=args.append,
        force=args.force,
        codec=args.codec,
        message=args.message,
        author=args.author,
    )
    print(f"wrote {_plural(count, 'row')} to {args.output}")
    return 0


def _finish(writer: Writer, args: argparse.Namespace) -> None:
    commit = writer.commit(message=args.message, author=args.author)
    writer.close()
    if commit is None:
        print("nothing to commit")
    else:
        print(f"commit {commit.number} {commit.short_hash}")


def cmd_point(args: argparse.Namespace) -> int:
    writer = Writer.open(args.file)
    try:
        schema = _pick_device(writer.devices, args.device)
        time = _time_arg(schema, args.time, "exact")
        assert time is not None  # noqa: S101 - required by the parser
        values = _assignments(schema, args.values)
        if args.command == "update":
            writer.update(schema.name, time, values)
        else:
            writer.write_row(schema.name, time, values)
        _finish(writer, args)
    except BaseException:
        if not writer.closed:
            writer.rollback()
            writer.close()
        raise
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    if args.time is None and args.start is None and args.end is None:
        msg = "give --time, or --from and/or --to"
        raise CliError(msg)
    writer = Writer.open(args.file)
    try:
        schema = _pick_device(writer.devices, args.device)
        if args.time is not None:
            time = _time_arg(schema, args.time, "exact")
            assert time is not None  # noqa: S101
            writer.delete(schema.name, time)
        else:
            writer.delete_range(
                schema.name,
                _time_arg(schema, args.start, "ceil"),
                _time_arg(schema, args.end, "floor"),
            )
        _finish(writer, args)
    except BaseException:
        if not writer.closed:
            writer.rollback()
            writer.close()
        raise
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    writer = Writer.open(args.file)
    try:
        if args.id is not None:
            current = writer.annotations().get(args.id)
            if current is None:
                msg = f"unknown annotation {args.id}"
                raise CliError(msg)
            schema = writer.device(current[0])
            writer.update_annotation(
                args.id,
                _time_arg(schema, args.start, "floor"),
                _time_arg(schema, args.end, "floor"),
                args.label,
                args.note,
            )
        else:
            if args.start is None or args.label is None:
                msg = "a new annotation needs --start and --label"
                raise CliError(msg)
            schema = _pick_device(writer.devices, args.device)
            start = _time_arg(schema, args.start, "floor")
            end = _time_arg(schema, args.end or args.start, "floor")
            assert start is not None and end is not None  # noqa: S101, PT018
            new_id = writer.annotate(schema.name, start, end, args.label, args.note)
            print(f"annotation {new_id}")
        _finish(writer, args)
    except BaseException:
        if not writer.closed:
            writer.rollback()
            writer.close()
        raise
    return 0


def cmd_unannotate(args: argparse.Namespace) -> int:
    writer = Writer.open(args.file)
    try:
        writer.remove_annotation(args.id)
        _finish(writer, args)
    except BaseException:
        if not writer.closed:
            writer.rollback()
            writer.close()
        raise
    return 0


def cmd_annotations(args: argparse.Namespace) -> int:
    with Reader(args.file) as reader:
        annotations = reader.annotations(args.device, as_of=args.as_of)
        if not annotations:
            print("no annotations")
        for a in annotations:
            schema = reader.device(a.device)
            start = format_time(a.start, schema.time_unit, schema.timezone)
            end = format_time(a.end, schema.time_unit, schema.timezone)
            note = f" — {a.note}" if a.note else ""
            print(f"{a.id}\t{a.device}\t{start} … {end}\t{a.label}{note}")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    with Reader(args.file) as reader:
        commits = reader.commits()
        if not commits:
            print("no commits")
        for c in reversed(commits):
            print(f"commit {c.number} {c.hash}")
            print(f"Date:    {c.time.isoformat()}")
            if c.author:
                print(f"Author:  {c.author}")
            parts = []
            ch = c.changes
            if ch.devices_created:
                parts.append("new devices: " + ", ".join(ch.devices_created))
            if ch.blocks_written:
                parts.append(
                    f"{_plural(ch.rows_written, 'row')} in {_plural(ch.blocks_written, 'block')}"
                )
            if ch.ranges_deleted:
                parts.append(_plural(ch.ranges_deleted, "deleted range"))
            if ch.annotation_ops:
                parts.append(_plural(ch.annotation_ops, "annotation change"))
            if ch.metadata_changed:
                parts.append("metadata")
            if parts:
                print("Changes: " + "; ".join(parts))
            if c.message:
                print()
                print(f"    {c.message}")
            print()
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    with Reader(args.file) as reader:
        schema = _pick_device(reader.devices, args.device)
        d = reader.diff(schema.name, args.start, args.end)
        print(f"device {d.device}: commit {d.from_commit} → {d.to_commit}")
        for p in d.points:
            stamp = format_time(p.time, schema.time_unit, schema.timezone)
            if p.before is not None and p.after is not None:
                changed = ", ".join(
                    f"{k}: {_format_value(p.before[k]) or 'null'} → "
                    f"{_format_value(p.after[k]) or 'null'}"
                    for k in d.columns
                    if p.before[k] != p.after[k]
                )
                print(f"~ {stamp}  {changed}")
            elif p.after is not None:
                print(
                    f"+ {stamp}  "
                    + ", ".join(f"{k}={_format_value(v)}" for k, v in p.after.items())
                )
            elif p.before is not None:
                print(
                    f"- {stamp}  "
                    + ", ".join(f"{k}={_format_value(v)}" for k, v in p.before.items())
                )
        for a in d.annotations:
            print(f"{a.kind.value} annotation {a.id}: {(a.after or a.before).label}")  # type: ignore[union-attr]
        if not d:
            print("no changes")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    report = verify(args.file)
    print(f"frames:    {report.frames}")
    print(f"commits:   {report.commits}")
    print(f"blocks:    {report.blocks}")
    print(f"finalized: {'yes' if report.finalized else 'no'}")
    if report.head:
        print(f"head:      {report.head}")
    if report.origin:
        print(f"origin:    {report.origin}")
    for w in report.warnings:
        print(f"warning: {w}")
    for p in report.problems:
        print(f"PROBLEM: {p}")
    print("OK" if report.ok else "DAMAGED")
    return 0 if report.ok else 1


def cmd_recover(args: argparse.Namespace) -> int:
    report = recover(args.file, force=args.force)
    if report.was_finalized:
        print("the file was finalized; nothing to do")
    else:
        assert report.recovery is not None  # noqa: S101
        print(
            f"recovered: removed {_human_bytes(report.removed_bytes)} and "
            f"{_plural(report.removed_frames, 'uncommitted frame')} ({report.recovery.reason})"
        )
    return 0


def cmd_compact(args: argparse.Namespace) -> int:
    report = compact(args.source, args.target, as_of=args.as_of, overwrite=args.force)
    print(
        f"compacted version {report.version} ({report.origin[:12]}): "
        f"{_plural(report.rows, 'row')}, {_plural(report.annotations, 'annotation')}, "
        f"{_human_bytes(report.bytes_before)} → {_human_bytes(report.bytes_after)}"
    )
    return 0


# ---------------------------------------------------------------- parser


def _commit_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--message", help="commit message")
    parser.add_argument("--author", help="author recorded in the commit")


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="pytrosna",
        description="Read, write and edit Trosna time-series files (.trosna).",
    )
    parser.add_argument("--version", action="version", version=f"pytrosna {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("info", help="devices, columns, history and storage of a file")
    p.add_argument("file")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("cat", help="print the points of a device")
    p.add_argument("file")
    p.add_argument("--device")
    p.add_argument("--columns", help="comma-separated value columns")
    p.add_argument("--from", dest="start", help="first time (inclusive)")
    p.add_argument("--to", dest="end", help="last time (inclusive)")
    p.add_argument("--as-of", type=int, help="read the data as it was after this commit")
    p.add_argument("--format", choices=("csv", "table", "json"), default="csv")
    p.add_argument("--limit", type=int, help="print at most this many rows")
    p.add_argument("--raw-time", action="store_true", help="integer time stamps")
    p.set_defaults(func=cmd_cat)

    p = sub.add_parser("convert", help="CSV → Trosna or Trosna → CSV (by the input's extension)")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--device", help="device name (CSV → Trosna: the file name by default)")
    p.add_argument("--time", help="name of the time column in the CSV (the first by default)")
    p.add_argument("--unit", choices=("s", "ms", "us", "ns"), help="unit of integer time stamps")
    p.add_argument("--delimiter", default=",")
    p.add_argument("--append", action="store_true", help="add the data to an existing file")
    p.add_argument("--force", action="store_true", help="overwrite the output if it exists")
    p.add_argument("--codec", choices=("none", "lz4", "zstd"), default="zstd")
    p.add_argument("--raw-time", action="store_true", help="Trosna → CSV: integer time stamps")
    p.add_argument("--as-of", type=int, help="Trosna → CSV: the data after this commit")
    _commit_options(p)
    p.set_defaults(func=cmd_convert)

    for name, text in (
        ("insert", "add a point (or replace the one at that time)"),
        ("update", "change values of an existing point"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("file")
        p.add_argument("--device")
        p.add_argument("--time", required=True)
        p.add_argument("values", nargs="+", metavar="COLUMN=VALUE")
        _commit_options(p)
        p.set_defaults(func=cmd_point)

    p = sub.add_parser("delete", help="delete a point or a time range")
    p.add_argument("file")
    p.add_argument("--device")
    p.add_argument("--time")
    p.add_argument("--from", dest="start")
    p.add_argument("--to", dest="end")
    _commit_options(p)
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("annotate", help="label a time interval (or change a label with --id)")
    p.add_argument("file")
    p.add_argument("--device")
    p.add_argument("--id", type=int, help="change this annotation")
    p.add_argument("--start")
    p.add_argument("--end", help="end of the interval (= start by default)")
    p.add_argument("--label")
    p.add_argument("--note")
    _commit_options(p)
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("unannotate", help="remove an annotation")
    p.add_argument("file")
    p.add_argument("id", type=int)
    _commit_options(p)
    p.set_defaults(func=cmd_unannotate)

    p = sub.add_parser("annotations", help="list annotations")
    p.add_argument("file")
    p.add_argument("--device")
    p.add_argument("--as-of", type=int)
    p.set_defaults(func=cmd_annotations)

    p = sub.add_parser("log", help="commit history")
    p.add_argument("file")
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("diff", help="changes of a device between commits")
    p.add_argument("file")
    p.add_argument("--device")
    p.add_argument("--from", dest="start", type=int, required=True, help="older commit (0 = empty)")
    p.add_argument("--to", dest="end", type=int, help="newer commit (the latest by default)")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("verify", help="check checksums, hash chain and all data")
    p.add_argument("file")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("recover", help="finalize a file whose writer was interrupted")
    p.add_argument("file")
    p.add_argument("--force", action="store_true", help="also cut off data after damage")
    p.set_defaults(func=cmd_recover)

    p = sub.add_parser("compact", help="copy a version into a new file without history")
    p.add_argument("source")
    p.add_argument("target")
    p.add_argument("--as-of", type=int)
    p.add_argument("--force", action="store_true", help="overwrite the target")
    p.set_defaults(func=cmd_compact)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``pytrosna`` command; returns the exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:  # pragma: no cover - e.g. `pytrosna cat f | head`
        sys.stderr.close()
        return 0
    except (CliError, TrosnaError, OSError, ValueError, KeyError, TypeError) as e:
        text = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"pytrosna: error: {text}", file=sys.stderr)
        return 2


def _stdout() -> TextIO:  # pragma: no cover - kept for embedding
    return io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
