# Руководство пользователя pytrosna

[English version](guide.md) · [Справочник API](api.ru.md) · [README](../README.ru.md)

Руководство шаг за шагом показывает все возможности pytrosna: от первого
файла до восстановления после сбоя. Все примеры на Python с этой страницы
выполняются тестами по порядку, поэтому каждый следующий пример опирается на
предыдущие.

1. [Основные понятия](#1-основные-понятия)
2. [Создание файла](#2-создание-файла)
3. [Чтение](#3-чтение)
4. [Время и часовые пояса](#4-время-и-часовые-пояса)
5. [Дозапись](#5-дозапись)
6. [Правки в транзакциях](#6-правки-в-транзакциях)
7. [Аннотации](#7-аннотации)
8. [История и путешествие во времени](#8-история-и-путешествие-во-времени)
9. [Несколько устройств](#9-несколько-устройств)
10. [Низкоуровневые Writer и Reader](#10-низкоуровневые-writer-и-reader)
11. [Хранение, кодирования и кодеки](#11-хранение-кодирования-и-кодеки)
12. [Целостность, сбои и восстановление](#12-целостность-сбои-и-восстановление)
13. [Уплотнение](#13-уплотнение)
14. [Командная строка](#14-командная-строка)
15. [Соответствие типов](#15-соответствие-типов)
16. [Ошибки](#16-ошибки)

## 1. Основные понятия

* **Файл** (`*.trosna`) содержит любое число **устройств** и **метаданные**
  файла (строка → строка).
* **Устройство** — это таблица: столбец времени и именованные столбцы
  значений типов `bool`, `int32`, `int64`, `float32`, `float64` и `string`.
  Любое значение может быть пустым (null). У устройства есть **единица
  времени** (`s`, `ms`, `us`, `ns`), необязательный **часовой пояс** (имя IANA,
  например `Europe/Moscow`, `UTC` или фиксированное смещение `+03:00`) и свои
  метаданные. Схема устройства задаётся при его создании и дальше не меняется.
* Метки времени хранятся целыми числами — количеством единиц времени от
  1970-01-01 UTC. В любой версии метки времени устройства уникальны:
  устройство — это функция «время → строка».
* Каждое изменение входит в **коммит** — атомарный набор изменений с хешем.
  Файл хранит все версии, и его можно прочитать «по состоянию на» любой коммит.
* **Аннотация** — помеченный замкнутый интервал `[start, end]` устройства,
  например аномалия или режим работы, с необязательным примечанием.

## 2. Создание файла

Проще всего воспользоваться `pytrosna.write`. Функция принимает DataFrame
pandas или Polars, таблицу PyArrow, `pytrosna.Batch` или обычный словарь:

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

commit = pytrosna.write(
    "room.trosna", df, device="room1", message="первый импорт", author="лаборатория"
)
print(commit.number, commit.short_hash, commit.changes.rows_written)
```

Что произошло:

* **Столбец времени** найден автоматически: это первый столбец с метками
  времени либо столбец с именем `time`, `timestamp`, `ts`, `datetime`, `date`,
  `t` или `время`. Другой столбец задаётся параметром `time_column="..."`.
  Если столбца времени нет, используется `DatetimeIndex` датафрейма pandas.
* **Единица времени** и **часовой пояс** устройства взяты из столбца
  (`datetime64[us, Europe/Moscow]`).
* **Типы столбцов** определены по dtype (`float64`, `int64`, `bool`, `string`).
* Режим `mode="w"` (по умолчанию) заменяет существующий файл; `mode="x"`
  отказывается перезаписывать, `mode="a"` дописывает. Замена идёт через
  временный файл, поэтому ошибка никогда не портит существующий файл.

Для целых меток времени нужно указать единицу:

```python
pytrosna.write(
    "counters.trosna",
    {"time": [1_791_442_800, 1_791_442_801, 1_791_442_802], "requests": [10, 12, None]},
    device="web",
    unit="s",
)
print(pytrosna.read_pandas("counters.trosna"))
```

`None` означает пустое значение. Пустой файл с заданной схемой создаёт
`pytrosna.create`:

```python
from pytrosna import ColumnSchema, DeviceSchema

schema = DeviceSchema(
    "boiler",
    "ms",
    (
        ColumnSchema("pressure", "float32", {"unit": "бар"}),
        ColumnSchema("burner_on", "bool"),
    ),
    timezone="UTC",
    metadata={"site": "Брянск"},
)
pytrosna.create("boiler.trosna", schema, metadata={"owner": "котельная"}, overwrite=True)
print(pytrosna.open("boiler.trosna").devices)
```

## 3. Чтение

```python
pytrosna.read_pandas("room.trosna")  # pandas.DataFrame
pytrosna.read_polars("room.trosna")  # polars.DataFrame
pytrosna.read_arrow("room.trosna")  # pyarrow.Table
batch = pytrosna.read("room.trosna")  # pytrosna.Batch (только NumPy)
print(batch, batch["temperature"].values[:3], batch.time[:3])
```

У всех функций чтения одинаковые параметры:

```python
part = pytrosna.read_pandas(
    "room.trosna",
    columns=["temperature", "status"],  # столбец времени включается всегда
    start="2026-10-08 10:02:00",  # включительно
    end="2026-10-08 10:03:00",  # включительно
)
print(part)
```

Объект `File` не открывает файл заново при каждом вызове и даёт доступ к
схеме и истории:

```python
f = pytrosna.open("room.trosna")
print(f.devices["room1"].column_names, f.count(), f.time_range())
for chunk in f.iter_batches(columns=["temperature"]):  # потоково, блок за блоком
    print(len(chunk), chunk["temperature"].values.mean())
reader = f.read_batches()  # pyarrow.RecordBatchReader
print(reader.schema)
```

В pandas целочисленные столбцы с пустыми значениями становятся `float64` с
`NaN` (так же, как в PyArrow); с `dtype_backend="numpy_nullable"` они остаются
`Int64`:

```python
print(pytrosna.read_pandas("counters.trosna", dtype_backend="numpy_nullable").dtypes)
```

## 4. Время и часовые пояса

Везде, где API принимает время, допустимы:

* `datetime.datetime` / `datetime.date`, `pandas.Timestamp`, `numpy.datetime64`;
* текст ISO 8601: `"2026-10-08T10:00:00+03:00"`, `"2026-10-08 10:00:00.250"`,
  `"2026-10-08"`;
* целое число — «сырая» метка времени в единице устройства.

Время **без смещения UTC** — это местное время часового пояса устройства.
Если час повторяется (часы переведены назад), берётся более ранний момент;
время, пропущенное при переводе часов вперёд, — ошибка. Границы интервала,
попавшие между двумя метками времени, округляются внутрь интервала; время
добавляемой точки должно точно выражаться в единице устройства.

```python
from pytrosna import to_raw, to_datetime, format_time

raw = to_raw("2026-10-08 10:00", "s", "Europe/Moscow")
print(raw, to_datetime(raw, "s", "Europe/Moscow"), format_time(raw, "s", "+03:00"))
print(f.to_raw("2026-10-08 10:00:10"), f.to_datetime(f.to_raw("2026-10-08 10:00:10")))
```

## 5. Дозапись

`mode="a"` дописывает данные в существующее устройство (при необходимости
создаёт устройство или файл). Каждый вызов — один коммит. Строка с меткой
времени существующей точки заменяет её (побеждает последняя запись).

```python
later = df.assign(time=df["time"] + pd.Timedelta("10min"))
f.write(later, "room1", message="следующие десять минут")  # то же, что mode="a"
print(f.count(), len(f.commits()))
```

Метки времени без часового пояса, дописываемые в устройство с часовым
поясом, читаются в этом поясе:

```python
naive = pd.DataFrame({"time": pd.to_datetime(["2026-10-08 10:30:00"]), "temperature": [22.0]})
f.write(naive, "room1")
print(f.read_pandas(start="2026-10-08 10:30").tail(1))
```

## 6. Правки в транзакциях

`File.edit()` собирает изменения. Если блок `with` завершается обычно, они
становятся **одним коммитом**; если в блоке возникает исключение, они
отбрасываются:

```python
with f.edit(message="проверка датчика", author="техник") as tx:
    tx.update("room1", "2026-10-08 10:00:20", temperature=21.65)  # изменить значения
    tx.insert("room1", "2026-10-08 10:31:00", temperature=22.1)  # остальные столбцы — null
    tx.delete("room1", "2026-10-08 10:00:10")  # одна точка
    tx.delete_range("room1", "2026-10-08 10:05:00", "2026-10-08 10:05:30")  # интервал
    tx.set_metadata("calibrated", "2026-10-08")
print(tx.commit.number, tx.commit.changes)
```

* `update` меняет только указанные столбцы; если точки с таким временем нет,
  возникает `PointNotFoundError`.
* `insert` записывает строку целиком: не указанные столбцы пусты.
* `delete_range(device, start=None, end=None)` допускает открытые концы.
* `tx.write(device, data)` добавляет в транзакцию целую таблицу.

```python
try:
    with f.edit() as tx:
        tx.delete("room1", "2026-10-08 10:00:30")
        raise RuntimeError("отмена")
except RuntimeError:
    pass
print(len(f.commits()))  # не изменилось: от транзакции не осталось следа
```

## 7. Аннотации

Аннотации размечают интервалы устройства — аномалии, режимы, эксперименты —
и версионируются так же, как данные:

```python
with f.edit(message="разметка") as tx:
    tx.annotate("room1", "2026-10-08 10:02", "2026-10-08 10:04", "проветривание", "окно открыто")
    tx.annotate(
        "room1", "2026-10-08 10:06", "2026-10-08 10:06", "хлопнула дверь"
    )  # мгновенное событие
ids = tx.annotation_ids  # идентификаторы новых аннотаций
print(ids)

with f.edit() as tx:
    tx.update_annotation(ids[0], label="вентиляция")  # остальные поля сохраняются
    tx.remove_annotation(ids[1])

for a in f.annotations():
    print(a.id, a.label, f.to_datetime(a.start), f.to_datetime(a.end), a.note)
```

Идентификаторы уникальны в пределах файла и не используются повторно.
`annotations(as_of=n)` возвращает аннотации старой версии.

Аннотации удобно использовать как метки для обучения моделей:

```python
data = f.read_pandas()
raw = f.read().time
data["label"] = None
for a in f.annotations("room1"):
    data.loc[(raw >= a.start) & (raw <= a.end), "label"] = a.label
print(data["label"].value_counts())
```

## 8. История и путешествие во времени

```python
for c in f.commits():
    print(c.number, c.short_hash, c.time.isoformat(timespec="seconds"), c.author, c.message)

first = f.read_pandas(as_of=1)  # файл сразу после импорта
print(len(first), len(f.read_pandas()))

changes = f.diff(1)  # от коммита 1 до последнего
for p in changes.points[:5]:
    print(p.kind.value, f.to_datetime(p.time), p.before, "->", p.after)
for a in changes.annotations:
    print(a.kind.value, a.id, a.after or a.before)

version = f.commit_at("2100-01-01T00:00:00Z")  # версия файла на заданный момент
print(version)
```

`as_of=0` — пустой файл. Каждый коммит хранит хеш SHA-256 предыдущего,
поэтому хеш последнего коммита идентифицирует всю историю. Если его
опубликовать (например, в статье или лабораторном журнале), никто не сможет
незаметно изменить измерение задним числом.

## 9. Несколько устройств

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

Если устройств несколько, функциям чтения нужно передать имя устройства.

## 10. Низкоуровневые Writer и Reader

`Writer` и `Reader` дают полный контроль и не требуют библиотек датафреймов.
Время — «сырые» целые числа в единице устройства.

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
    w.commit(message="массовая загрузка", author="сборщик")

    w.write_row("vm01", t0 + 10_000 * 1000, cpu=0.5, ram_mb=4096)  # одна строка
    w.update("vm01", t0, ram_mb=1024)  # изменить значение
    print(w.get("vm01", t0))  # видит незакоммиченные правки
    w.delete_range("vm01", t0 + 1000, t0 + 5000)
    w.annotate("vm01", t0, t0 + 60_000, "прогрев")
    w.commit(message="правки")
    w.write_row("vm01", t0 - 1000, cpu=0.0)
    w.rollback()  # отменить всё после коммита
# при выходе из блока ожидающие изменения коммитятся и записывается индекс;
# при выходе с исключением они отбрасываются

with Reader("vms.trosna") as r:
    q = r.query("vm01").columns(["cpu", "ram_mb"]).time_range(t0, t0 + 59_000)
    b = q.collect()
    print(len(b), b.time[:2], b["ram_mb"].values[:2])
    print(r.query("vm01").count(), r.query("vm01").as_of(1).count())
    for chunk in r.query("vm01").batches():
        pass
    print(r.metadata, r.head.message, [d.name for d in r.devices])
```

`Batch` содержит `time` (массив `int64`) и объекты `Column` со значениями
`values` (массив NumPy) и необязательной маской `validity`:

```python
manual = Batch([1, 2, 3], {"cpu": Column.from_values("float64", [0.1, None, 0.3])})
print(manual.to_dict(), manual.row(1), manual["cpu"].null_count)
```

## 11. Хранение, кодирования и кодеки

Каждый блок (`rows_per_block` строк, по умолчанию 65 536) хранит столбец
времени и каждый столбец значений отдельными сегментами. Для каждого
сегмента писатель пробует все подходящие кодирования и оставляет самый
короткий результат:

| Тип столбца     | Пробуемые кодирования                                |
|-----------------|------------------------------------------------------|
| время           | delta-of-delta, delta-bitpack, plain                 |
| int32, int64    | delta-bitpack, RLE, plain                            |
| float32/64      | Gorilla XOR, plain                                   |
| bool            | RLE, битовая маска (plain)                           |
| string          | словарное, plain                                     |

Затем закодированный сегмент сжимается кодеком (`zstd` по умолчанию, `lz4`
или `none`), если это уменьшает его размер. `File.blocks()` показывает выбор:

```python
for block in pytrosna.open("vms.trosna").blocks()[:1]:
    for s in block.segments:
        print(
            f"{s.column:<8} {s.encoding.label:<15} {s.codec.label:<5} {s.stored_bytes:>6} Б",
            s.statistics,
        )
```

Параметры `WriteOptions` (их также принимают как именованные аргументы
`Writer.create`, `Writer.open` и `pytrosna.write`):

| Параметр            | По умолчанию   | Назначение                                                    |
|---------------------|----------------|---------------------------------------------------------------|
| `codec`             | `"zstd"`       | `"zstd"`, `"lz4"` или `"none"`                                |
| `encoding`          | `"adaptive"`   | `"adaptive"`, `"classic"` (одно кодирование на тип), `"plain"` |
| `rows_per_block`    | 65 536         | строк в блоке (1 … 2²⁴)                                       |
| `max_block_bytes`   | 64 МиБ         | буфер устройства записывается, когда достигает этого объёма   |
| `sync`              | `True`         | `fsync` при каждом коммите                                    |
| `overwrite`         | `False`        | `Writer.create` заменяет существующий файл                    |
| `repair_corruption` | `False`        | `Writer.open` отрезает данные после повреждения в середине    |
| `zstd_level`        | 3              | уровень сжатия Zstandard                                      |

```python
pytrosna.write("plain.trosna", df, device="room1", codec="none", encoding="plain")
print(
    pytrosna.open("plain.trosna").size, "байт без сжатия против", pytrosna.open("room.trosna").size
)
```

## 12. Целостность, сбои и восстановление

```python
report = pytrosna.verify("room.trosna")
print(report.ok, report.commits, report.blocks, report.finalized, report.head[:12])
print(report.problems, report.warnings)
```

`verify` проверяет CRC-32C каждого фрейма и сегмента, декодирует все данные,
пересчитывает цепочку хешей коммитов и сравнивает индекс с фреймами.

Файл — журнал, в который только дописывают. Если программа завершилась во
время записи, читатели видят состояние последнего завершённого коммита, а
следующий писатель (или `recover`) удаляет незавершённый хвост:

```python
from pathlib import Path

data = Path("room.trosna").read_bytes()
Path("crashed.trosna").write_bytes(data[:-30])  # имитация прерванной записи
print(pytrosna.open("crashed.trosna").finalized)  # False, но файл читается
print(pytrosna.recover("crashed.trosna"))  # удаляет хвост, пишет новый индекс
print(pytrosna.verify("crashed.trosna").ok)
```

Если повреждение найдено в середине файла (после него идут целые фреймы),
писатель отказывается обрезать файл; `recover(path, force=True)` отрезает
всё после повреждения.

Писать в файл может один писатель одновременно (берётся исключительная
блокировка), читать — сколько угодно читателей.

## 13. Уплотнение

Правки хранятся как новые блоки и «надгробия» (tombstones); при чтении они
сливаются. Уплотнение записывает состояние одной версии в новый файл без
истории и без перекрывающихся блоков. Первый коммит нового файла ссылается
на хеш исходной версии, а метаданные запоминают происхождение:

```python
small = f.compact("room-compact.trosna", overwrite=True)
print(small.size, "<", f.size, small.metadata["trosna.compacted_from"][:12])
pytrosna.compact("room.trosna", "room-v1.trosna", as_of=1, overwrite=True)  # старая версия
```

## 14. Командная строка

```sh
pytrosna convert data.csv data.trosna --device sensor1     # CSV → Trosna
pytrosna convert data.csv data.trosna --append --device sensor1
pytrosna convert data.trosna back.csv --as-of 2            # Trosna → CSV (выбранная версия)
pytrosna info data.trosna
pytrosna cat data.trosna --columns temperature --from "2026-10-08 10:00" --to "2026-10-08 11:00"
pytrosna cat data.trosna --format table --limit 20         # или --format json, --raw-time
pytrosna insert data.trosna --time "2026-10-08 10:01" temperature=22 -m "ручной ввод"
pytrosna update data.trosna --time "2026-10-08 10:00:20" temperature=21.65 --author я
pytrosna delete data.trosna --from "2026-10-08 10:30" --to "2026-10-08 10:40"
pytrosna annotate data.trosna --start "2026-10-08 10:00" --end "2026-10-08 10:05" --label прогрев
pytrosna annotate data.trosna --id 1 --label "прогрев печи"   # изменить аннотацию
pytrosna unannotate data.trosna 1
pytrosna annotations data.trosna
pytrosna log data.trosna
pytrosna diff data.trosna --from 1 --to 3
pytrosna verify data.trosna                                # код выхода 1 при повреждении
pytrosna recover data.trosna
pytrosna compact data.trosna small.trosna
```

При импорте CSV типы столбцов определяются сами (целые → `int64`, дробные →
`float64`, `true`/`false` → `bool`, остальное → `string`; пустая ячейка —
null). Первый столбец считается временем, если `--time` не указывает
другой. Время ISO 8601 со смещением задаёт часовой пояс устройства; целые
числа считаются эпохой Unix, единица угадывается по их величине или
задаётся `--unit`.

## 15. Соответствие типов

| Trosna   | NumPy (`Batch`) | pandas (обычные / nullable)     | Polars             | PyArrow                |
|----------|-----------------|---------------------------------|--------------------|------------------------|
| время    | `int64` (сырое) | `datetime64[unit, tz]`          | `Datetime(unit, tz)` | `timestamp(unit, tz)` |
| bool     | `bool`          | `bool` / `object` / `boolean`   | `Boolean`          | `bool`                 |
| int32    | `int32`         | `int32` / `float64` / `Int32`   | `Int32`            | `int32`                |
| int64    | `int64`         | `int64` / `float64` / `Int64`   | `Int64`            | `int64`                |
| float32  | `float32`       | `float32` / `Float32`           | `Float32`          | `float32`              |
| float64  | `float64`       | `float64` / `Float64`           | `Float64`          | `float64`              |
| string   | `object` (`str`)| `object`/`str` / `string`       | `String`           | `string`               |

При записи более узкие типы расширяются без потерь (`int8`, `uint16` →
`int32`; `uint32`, `uint64` → `int64`; `float16` → `float32`; категориальные
и словарные строки → `string`). При записи в существующее устройство
значения приводятся к типам его столбцов: целые столбцы принимают дробные
числа без дробной части, дробные — любые числа, логические и строковые —
только свой тип. В pandas `NaN` означает пропуск, а в массивах NumPy и в
Polars это значение. В Polars нет секундного разрешения, поэтому секундные
метки времени преобразуются в миллисекунды, а фиксированные смещения — в
пояса `Etc/GMT±H`.

## 16. Ошибки

Все ошибки наследуются от `pytrosna.TrosnaError`. Самые частые:

| Исключение               | Когда возникает                                              |
|--------------------------|--------------------------------------------------------------|
| `CorruptedError`         | повреждённые или некорректные данные (`offset` — где)        |
| `NotTrosnaError`         | файл не является файлом Trosna                               |
| `UnknownDeviceError`     | нет устройства с таким именем (также `KeyError`)             |
| `UnknownColumnError`     | нет столбца с таким именем (также `KeyError`)                |
| `PointNotFoundError`     | `update` несуществующей точки (также `KeyError`)             |
| `TypeMismatchError`      | значение неподходящего типа (также `TypeError`)              |
| `InvalidArgumentError`   | неверный аргумент, например время точнее единицы (`ValueError`) |
| `LockedError`            | файл занят другим писателем                                  |
| `UnsupportedError`       | данные, которые Trosna не может хранить, или новая версия формата |

```python
try:
    with f.edit() as tx:
        tx.update("room1", "2030-01-01", temperature=0.0)
except pytrosna.PointNotFoundError as e:
    print("ошибка:", e)
```
