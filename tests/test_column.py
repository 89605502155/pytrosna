"""In-memory columns and batches."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pytrosna import Batch, Column, DataType, DeviceSchema
from pytrosna.errors import InvalidArgumentError, TypeMismatchError


def test_from_values_with_nulls() -> None:
    c = Column.from_values("int64", [1, None, 3])
    assert c.to_list() == [1, None, 3]
    assert c.null_count == 1
    assert len(c) == 3
    assert c[1] is None
    assert c.value(2) == 3
    assert list(c) == [1, None, 3]
    assert c.dense().tolist() == [1, 3]
    assert not c.is_valid(1)
    assert repr(c) == "Column(int64, [1, None, 3])"


def test_all_valid_mask_is_dropped() -> None:
    c = Column("float64", np.array([1.0, 2.0]), np.array([True, True]))
    assert c.validity is None
    assert c.null_count == 0
    assert c.valid_mask().all()


@pytest.mark.parametrize(
    ("data_type", "values", "expected"),
    [
        ("bool", [True, False, None], [True, False, None]),
        ("int32", [1, 2.0, -(2**31)], [1, 2, -(2**31)]),
        ("int64", [np.int64(5), 2**63 - 1], [5, 2**63 - 1]),
        ("float32", [1, 0.5], [1.0, 0.5]),
        ("float64", [1, np.float32(0.25), math.inf], [1.0, 0.25, math.inf]),
        ("string", ["a", "б", None], ["a", "б", None]),
    ],
)
def test_conversions(data_type: str, values: list[object], expected: list[object]) -> None:
    c = Column.from_values(data_type, values)
    assert c.to_list() == expected
    assert c.values.dtype == DataType.parse(data_type).numpy_dtype


@pytest.mark.parametrize(
    ("data_type", "value"),
    [
        ("bool", 1),
        ("int32", 2**31),
        ("int64", 2**63),
        ("int64", 1.5),
        ("int64", True),
        ("float64", "1.0"),
        ("float64", False),
        ("string", 5),
    ],
)
def test_rejected_conversions(data_type: str, value: object) -> None:
    with pytest.raises(TypeMismatchError):
        Column.from_values(data_type, [value])


def test_nan_is_a_value_unless_asked() -> None:
    assert Column.from_values("float64", [math.nan]).null_count == 0
    c = Column.from_values("float32", [math.nan, 1.0], nan_is_null=True)
    assert c.to_list() == [None, 1.0]


def test_nulls_and_concat() -> None:
    nulls = Column.nulls("string", 2)
    assert nulls.to_list() == [None, None]
    joined = Column.concat([nulls, Column.from_values("string", ["x"])])
    assert joined.to_list() == [None, None, "x"]
    plain = Column.concat([Column.from_values("int32", [1]), Column.from_values("int32", [2])])
    assert plain.validity is None
    with pytest.raises(InvalidArgumentError):
        Column.concat([])
    with pytest.raises(TypeMismatchError):
        Column.concat([Column.from_values("int32", [1]), Column.from_values("int64", [1])])


def test_equality_compares_bits_of_floats() -> None:
    a = Column.from_values("float64", [math.nan, 1.0])
    b = Column.from_values("float64", [math.nan, 1.0])
    assert a == b
    assert a != Column.from_values("float64", [math.nan, 2.0])
    assert Column.from_values("float64", [0.0]) != Column.from_values("float64", [-0.0])
    assert Column.from_values("int64", [1, None]) == Column(
        "int64", np.array([1, 99]), np.array([True, False])
    )
    assert Column.from_values("int64", [1]) != Column.from_values("int32", [1])
    assert Column.from_values("int64", [1, None]) != Column.from_values("int64", [1, 2])
    assert Column.from_values("int64", [1]).__eq__("x") is NotImplemented


def test_to_numpy() -> None:
    assert np.isnan(Column.from_values("float32", [None, 1.0]).to_numpy()[0])
    assert Column.from_values("int64", [None, 1]).to_numpy().tolist() == [None, 1]
    assert Column.from_values("int64", [2, 1]).to_numpy().dtype == np.int64


def test_slice_and_take() -> None:
    c = Column.from_values("int32", [1, None, 3, 4])
    assert c.slice(1, 3).to_list() == [None, 3]
    assert c.take(np.array([3, 1])).to_list() == [4, None]
    assert Column.from_values("int32", [5, 6]).take(np.array([1])).to_list() == [6]


def test_constructor_checks() -> None:
    with pytest.raises(TypeMismatchError):
        Column("int64", np.array([1.5]))
    with pytest.raises(InvalidArgumentError):
        Column("int64", np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(InvalidArgumentError):
        Column("int64", np.zeros(2, dtype=np.int64), np.array([True]))


def test_batch_basics() -> None:
    schema = DeviceSchema.build("d", "s", {"a": "int64", "b": "string"})
    b = Batch(
        [1, 2, 3],
        {
            "a": Column.from_values("int64", [1, None, 3]),
            "b": Column.from_values("string", ["x", "y", None]),
        },
        schema,
    )
    assert len(b) == 3
    assert b.names == ["a", "b"]
    assert b["a"].to_list() == [1, None, 3]
    assert b.row(1) == {"a": None, "b": "y"}
    assert list(b.rows())[2] == (3, {"a": 3, "b": None})
    assert b.to_dict() == {"time": [1, 2, 3], "a": [1, None, 3], "b": ["x", "y", None]}
    assert len(b.slice(1, 3)) == 2
    assert b.take(np.array([2, 0])).time.tolist() == [3, 1]
    assert b.select(["b"]).names == ["b"]
    assert b.datetimes().dtype == np.dtype("datetime64[s]")
    assert "device='d'" in repr(b)
    assert b == Batch.concat([b.slice(0, 1), b.slice(1, 3)])
    assert b != b.slice(0, 2)
    assert b.__eq__(5) is NotImplemented


def test_batch_checks() -> None:
    with pytest.raises(InvalidArgumentError):
        Batch([1, 2], {"a": Column.from_values("int64", [1])})
    with pytest.raises(InvalidArgumentError):
        Batch([1], {"a": [1]})  # type: ignore[dict-item]
    with pytest.raises(InvalidArgumentError):
        Batch(np.zeros((1, 1)))
    with pytest.raises(InvalidArgumentError):
        Batch.concat([])
    empty = Batch.empty({"a": DataType.FLOAT32})
    assert len(empty) == 0
    assert empty["a"].data_type is DataType.FLOAT32
    assert Batch([5]).datetimes().dtype == np.dtype("datetime64[ns]")
