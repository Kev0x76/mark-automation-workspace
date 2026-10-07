from __future__ import annotations

import re
from numbers import Number
from typing import Any


def normalize_barcode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Number) and not isinstance(value, bool):
        if float(value).is_integer():
            value = int(value)
    return re.sub(r"\D", "", str(value))


def ean13_check_digit(first_12_digits: str) -> int:
    if not re.fullmatch(r"\d{12}", first_12_digits):
        raise ValueError("EAN-13 check digit requires exactly 12 digits")
    total = 0
    for index, char in enumerate(first_12_digits):
        digit = int(char)
        total += digit if index % 2 == 0 else digit * 3
    return (10 - (total % 10)) % 10


def is_valid_ean13(value: Any) -> bool:
    digits = normalize_barcode(value)
    if not re.fullmatch(r"\d{13}", digits):
        return False
    return ean13_check_digit(digits[:12]) == int(digits[-1])
