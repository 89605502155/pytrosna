"""CRC-32C checksums."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from pytrosna._checksum import crc32c, crc32c_python


def test_known_value() -> None:
    # The standard check value of CRC-32C.
    assert crc32c(b"123456789") == 0xE3069283
    assert crc32c_python(b"123456789") == 0xE3069283
    assert crc32c(b"") == 0


def test_continuation() -> None:
    assert crc32c(b"6789", crc32c(b"12345")) == crc32c(b"123456789")
    assert crc32c_python(b"6789", crc32c_python(b"12345")) == 0xE3069283
    assert crc32c(memoryview(b"123456789")) == 0xE3069283
    assert crc32c(bytearray(b"123456789")) == 0xE3069283


@given(st.binary(max_size=300))
def test_pure_python_fallback_matches(data: bytes) -> None:
    assert crc32c_python(data) == crc32c(data)
