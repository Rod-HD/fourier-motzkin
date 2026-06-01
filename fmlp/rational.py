"""Số học hữu tỉ chính xác cho bộ giải Fourier-Motzkin (phiên bản 2).

Nhân thuật toán làm việc hoàn toàn trên trường số hữu tỉ ``Q`` bằng kiểu
:class:`fractions.Fraction`. Nhờ vậy mọi phép chia khi khử biến đều chính
xác tuyệt đối, không tích lũy sai số dấu phẩy động qua nhiều bước chiếu.
Lựa chọn này bám sát cách trình bày lý thuyết trong các tài liệu về quy
hoạch tuyến tính (xem Schrijver, Bertsimas--Tsitsiklis), nơi các phép biến
đổi đều giả định số học hữu tỉ chính xác.

Module này gom các tiện ích chuyển đổi đầu vào (số nguyên, chuỗi ``"3/4"``,
biểu thức hằng ``"sin(pi/6)"``) về ``Fraction`` và định dạng để hiển thị.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

Number = Fraction


class NotRationalError(ValueError):
    """Giá trị không quy về được một số hữu tỉ chính xác."""


def to_fraction(value: Any) -> Fraction:
    """Quy đổi *value* về :class:`Fraction` chính xác.

    Chấp nhận: ``int``, ``Fraction``, ``float`` (đọc qua chuỗi để tránh rác
    nhị phân), và ``str`` dạng ``"3"``, ``"-1/2"``, ``"0.25"``. Chuỗi chứa
    hàm/hằng ký hiệu (``"sin(pi/6)"``) được giao cho SymPy đánh giá rồi kiểm
    tra tính hữu tỉ.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):  # chặn trước int vì bool là int trong Python
        return Fraction(int(value))
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise NotRationalError("Chuỗi rỗng không phải là số.")
        try:
            return Fraction(text)
        except ValueError:
            return _fraction_from_symbolic(text)
    # Các kiểu còn lại (ví dụ số SymPy) thử ép qua chuỗi/đánh giá ký hiệu.
    return _fraction_from_symbolic(value)


def _fraction_from_symbolic(value: Any) -> Fraction:
    """Đánh giá một biểu thức hằng ký hiệu và trả về Fraction nếu hữu tỉ."""
    from sympy import Rational, nsimplify, sympify

    try:
        expr = sympify(value)
    except Exception as exc:  # pragma: no cover - phụ thuộc lỗi nội bộ SymPy
        raise NotRationalError(f"Không đọc được giá trị: {value!r}") from exc

    if not expr.free_symbols and expr.is_number:
        simplified = nsimplify(expr, rational=True)
        if simplified.is_Rational:
            r = Rational(simplified)
            return Fraction(int(r.p), int(r.q))
    raise NotRationalError(
        f"Giá trị {value!r} không phải số hữu tỉ. "
        "Phiên bản này tính trên trường Q nên hệ số phải quy về phân số."
    )


def format_fraction(value: Fraction) -> str:
    """Định dạng phân số: số nguyên in gọn, còn lại in dạng ``p/q``."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_decimal(value: Fraction, digits: int = 6) -> str:
    """Định dạng dạng thập phân (để hiển thị kèm), bỏ số 0 thừa."""
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"
