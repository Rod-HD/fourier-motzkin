"""Phân tích ràng buộc nhập dạng văn bản (phiên bản 2).

Mỗi dòng là một ràng buộc tuyến tính ``<biểu thức> <toán tử> <hằng số>`` với
toán tử thuộc {``<=``, ``>=``, ``=``}. Hệ số cho phép số nguyên, phân số, căn
và lượng giác hằng (``sqrt(2)``, ``sin(pi/6)``, ...); SymPy lo phần đọc biểu
thức, sau đó hệ số được quy về phân số chính xác qua :func:`fmlp.rational.to_fraction`.

Trả về danh sách dict ``{"coeffs", "sense", "rhs"}`` để nạp thẳng vào
:meth:`fmlp.model.LinearProgram.from_input`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from sympy import Poly, Symbol, cos, cot, pi, sin, sqrt, tan
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.polys.polyerrors import PolynomialError

from .rational import to_fraction

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_COMPARATOR = re.compile(r"(<=|>=|=)")
_VAR_TOKEN = re.compile(r"\bx(\d+)\b")


@dataclass
class ParseError(ValueError):
    line: int
    text: str
    message: str
    hint: str

    def __str__(self) -> str:
        return f"Dòng {self.line}: {self.message} | {self.text} | Gợi ý: {self.hint}"


def parse_constraints(text: str, n: int) -> List[Dict[str, object]]:
    if n < 1:
        raise ValueError("Số biến phải >= 1.")
    if not (text or "").strip():
        raise ParseError(1, "", "Chưa nhập ràng buộc nào.",
                         "Mỗi dòng một ràng buộc, ví dụ: x1 + x2 <= 4")

    variables = [Symbol(f"x{i + 1}", real=True) for i in range(n)]
    local = {"sin": sin, "cos": cos, "tan": tan, "cot": cot, "sqrt": sqrt, "pi": pi}
    for s in variables:
        local[str(s)] = s

    constraints: List[Dict[str, object]] = []
    for ln, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        constraints.append(_parse_line(line, ln, variables, local))

    if not constraints:
        raise ParseError(1, "", "Không tìm thấy ràng buộc hợp lệ.",
                         "Nhập ít nhất một dòng như: 2x1 + 3x2 <= 5")
    return constraints


def _parse_line(line: str, ln: int, variables: List[Symbol], local: Dict) -> Dict[str, object]:
    matches = list(_COMPARATOR.finditer(line))
    if len(matches) != 1:
        raise ParseError(ln, line, "Mỗi dòng phải có đúng một toán tử <=, >= hoặc =.",
                         "Ví dụ: sin(pi/6)x1 + cos(pi/3)x2 >= 1/2")
    m = matches[0]
    sense = m.group(1)
    lhs_text, rhs_text = line[: m.start()].strip(), line[m.end():].strip()
    if not lhs_text or not rhs_text:
        raise ParseError(ln, line, "Thiếu vế trái hoặc vế phải.",
                         "Mẫu: <biểu thức tuyến tính> <= <hằng số>")

    bad = [f"x{t}" for t in _VAR_TOKEN.findall(line) if not (1 <= int(t) <= len(variables))]
    if bad:
        raise ParseError(ln, line, f"Biến không hợp lệ: {', '.join(sorted(set(bad)))}.",
                         f"Chỉ dùng x1..x{len(variables)}.")

    lhs = _parse_side(lhs_text, line, ln, local)
    rhs = _parse_side(rhs_text, line, ln, local)
    expr = (lhs - rhs).expand()

    extra = sorted(str(s) for s in expr.free_symbols if s not in set(variables))
    if extra:
        raise ParseError(ln, line, f"Ký hiệu lạ: {', '.join(extra)}.",
                         f"Chỉ dùng x1..x{len(variables)} và hằng số.")

    try:
        poly = Poly(expr, *variables, domain="EX")
    except PolynomialError as exc:
        raise ParseError(ln, line, "Biểu thức không tuyến tính.",
                         "Chỉ chấp nhận a1x1 + a2x2 + ... ; không dùng x1^2, x1*x2, sin(x1).") from exc

    for mono in poly.monoms():
        if sum(mono) > 1:
            raise ParseError(ln, line, "Có hạng tử bậc cao / tích biến.",
                             "Chỉ dùng tổ hợp tuyến tính của các biến.")

    try:
        coeffs = [_coeff_to_fraction(poly.coeff_monomial(v), line, ln) for v in variables]
        rhs_val = _coeff_to_fraction(-poly.coeff_monomial(1), line, ln)
    except _IrrationalCoeff as exc:
        raise ParseError(ln, line, f"Hệ số không hữu tỉ: {exc.value}.",
                         "Phiên bản 2 tính trên trường Q; hệ số phải quy về phân số "
                         "(ví dụ sin(pi/6)=1/2 hợp lệ, sqrt(2) thì không).") from exc
    return {"coeffs": coeffs, "sense": sense, "rhs": rhs_val}


class _IrrationalCoeff(Exception):
    def __init__(self, value: str):
        self.value = value


def _coeff_to_fraction(expr, line: str, ln: int):
    from .rational import NotRationalError
    try:
        return to_fraction(str(expr))
    except NotRationalError as exc:
        raise _IrrationalCoeff(str(expr)) from exc


def _parse_side(text: str, line: str, ln: int, local: Dict):
    try:
        return parse_expr(text.replace("^", "**"), local_dict=local,
                          transformations=_TRANSFORMS, evaluate=True)
    except Exception as exc:  # pragma: no cover
        raise ParseError(ln, line, f"Không đọc được biểu thức '{text}'.",
                         "Kiểm tra ngoặc, phân số, hàm số và cú pháp biến.") from exc
