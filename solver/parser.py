"""Parser ràng buộc dạng text nhiều dòng cho Fourier-Motzkin solver."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from sympy import Poly, Symbol, cot, cos, pi, sin, sqrt, tan
from sympy.core.expr import Expr
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.polys.polyerrors import PolynomialError


TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
COMPARATOR_PATTERN = re.compile(r"(<=|>=|=)")
VARIABLE_TOKEN_PATTERN = re.compile(r"\bx(\d+)\b")


@dataclass
class ConstraintTextParseError(ValueError):
    """Lỗi parse ràng buộc text với thông tin dòng cụ thể."""

    line_number: int
    line_text: str
    message: str
    hint: str

    def __str__(self) -> str:
        return (
            f"Dòng {self.line_number}: {self.message}\n"
            f"Nội dung: {self.line_text}\n"
            f"Gợi ý: {self.hint}"
        )


def parse_constraints_text(text: str, n: int) -> List[Dict[str, object]]:
    """Parse chuỗi nhiều dòng thành danh sách constraints.

    Mỗi dòng hợp lệ có dạng:
        <biểu thức tuyến tính> <=|>=|= <hằng số>
    """
    if n < 1:
        raise ValueError("Số biến phải >= 1 trước khi parse ràng buộc text.")

    if not (text or "").strip():
        raise ConstraintTextParseError(
            line_number=1,
            line_text="",
            message="Chưa có ràng buộc nào để parse.",
            hint="Mỗi dòng hãy nhập một ràng buộc, ví dụ: x1 + x2 <= 4",
        )

    variables = [Symbol(f"x{i + 1}", real=True) for i in range(n)]
    local_dict = _build_local_dict(variables)

    constraints: List[Dict[str, object]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        constraints.append(_parse_constraint_line(line, line_number, variables, local_dict))

    if not constraints:
        raise ConstraintTextParseError(
            line_number=1,
            line_text="",
            message="Không tìm thấy ràng buộc hợp lệ nào.",
            hint="Hãy nhập ít nhất một dòng như: 2x1 + 3x2 <= 5",
        )

    return constraints


def _build_local_dict(variables: List[Symbol]) -> Dict[str, object]:
    local_dict: Dict[str, object] = {
        "sin": sin,
        "cos": cos,
        "tan": tan,
        "cot": cot,
        "sqrt": sqrt,
        "pi": pi,
    }
    for symbol in variables:
        local_dict[str(symbol)] = symbol
    return local_dict


def _parse_constraint_line(
    line: str,
    line_number: int,
    variables: List[Symbol],
    local_dict: Dict[str, object],
) -> Dict[str, object]:
    matches = list(COMPARATOR_PATTERN.finditer(line))
    if len(matches) != 1:
        raise ConstraintTextParseError(
            line_number=line_number,
            line_text=line,
            message="Mỗi dòng phải có đúng một toán tử so sánh: <=, >= hoặc =.",
            hint="Ví dụ hợp lệ: sin(pi/6)x1 + cos(pi/3)x2 >= 1/2",
        )

    match = matches[0]
    sense = match.group(1)
    lhs_text = line[: match.start()].strip()
    rhs_text = line[match.end() :].strip()

    if not lhs_text or not rhs_text:
        raise ConstraintTextParseError(
            line_number=line_number,
            line_text=line,
            message="Thiếu vế trái hoặc vế phải của ràng buộc.",
            hint="Mẫu đúng: <biểu thức tuyến tính> <= <hằng số>",
        )

    _validate_variable_tokens(line, line_number, len(variables))
    lhs_expr = _parse_side(lhs_text, line, line_number, local_dict)
    rhs_expr = _parse_side(rhs_text, line, line_number, local_dict)
    normalized_expr = (lhs_expr - rhs_expr).expand()

    unexpected_symbols = sorted(
        str(symbol) for symbol in normalized_expr.free_symbols if symbol not in set(variables)
    )
    if unexpected_symbols:
        raise ConstraintTextParseError(
            line_number=line_number,
            line_text=line,
            message=f"Biến không hợp lệ: {', '.join(unexpected_symbols)}.",
            hint=f"Chỉ dùng các biến từ x1 đến x{len(variables)}.",
        )

    try:
        poly = Poly(normalized_expr, *variables, domain="EX")
    except PolynomialError as exc:
        raise ConstraintTextParseError(
            line_number=line_number,
            line_text=line,
            message="Biểu thức không tuyến tính hoặc có hàm dùng trực tiếp trên biến.",
            hint="Chỉ cho phép dạng ax1 + bx2 + ... <= c; ví dụ: sqrt(2)x1 - 3x2 <= 7",
        ) from exc

    _ensure_linear(poly, line, line_number, len(variables))

    coeffs = [float(poly.coeff_monomial(var)) for var in variables]
    rhs = float(-poly.coeff_monomial(1))
    return {"coeffs": coeffs, "sense": sense, "rhs": rhs}


def _parse_side(
    expr_text: str,
    original_line: str,
    line_number: int,
    local_dict: Dict[str, object],
) -> Expr:
    normalized = expr_text.replace("^", "**")
    try:
        return parse_expr(
            normalized,
            local_dict=local_dict,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:  # pragma: no cover - phụ thuộc lỗi nội bộ sympy
        raise ConstraintTextParseError(
            line_number=line_number,
            line_text=original_line,
            message=f"Không parse được biểu thức '{expr_text}'.",
            hint="Kiểm tra lại ngoặc, phân số, hàm số và dùng x1, x2, ... đúng cú pháp.",
        ) from exc


def _ensure_linear(poly: Poly, line: str, line_number: int, n: int) -> None:
    for monomial in poly.monoms():
        total_degree = sum(monomial)
        if total_degree <= 1:
            continue
        raise ConstraintTextParseError(
            line_number=line_number,
            line_text=line,
            message="Phát hiện hạng tử không tuyến tính.",
            hint=(
                "Không dùng x1^2, x1*x2, sin(x1), ... "
                f"Chỉ dùng tổ hợp tuyến tính của x1..x{n} với hệ số hằng."
            ),
        )


def _validate_variable_tokens(line: str, line_number: int, n: int) -> None:
    invalid_variables = []
    for token in VARIABLE_TOKEN_PATTERN.findall(line):
        idx = int(token)
        if idx < 1 or idx > n:
            invalid_variables.append(f"x{idx}")

    if invalid_variables:
        invalid_variables = sorted(set(invalid_variables), key=lambda name: int(name[1:]))
        raise ConstraintTextParseError(
            line_number=line_number,
            line_text=line,
            message=f"Biến không hợp lệ: {', '.join(invalid_variables)}.",
            hint=f"Chỉ dùng các biến từ x1 đến x{n}.",
        )
