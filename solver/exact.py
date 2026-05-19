"""Tiện ích số học exact/numeric dùng chung cho solver."""

from __future__ import annotations

from typing import Any, Dict, List

from sympy import Float, Integer, Rational, simplify, sympify
from sympy.core.expr import Expr


def to_exact_number(value: Any) -> Expr:
    """Chuẩn hóa giá trị thành số sympy exact khi có thể."""
    if isinstance(value, Expr):
        expr = simplify(value)
    elif isinstance(value, bool):
        expr = Integer(int(value))
    elif isinstance(value, int):
        expr = Integer(value)
    elif isinstance(value, float):
        expr = Rational(str(value))
    elif isinstance(value, str):
        expr = sympify(value)
    else:
        expr = sympify(value)

    if isinstance(expr, Float):
        return Rational(str(expr))
    return simplify(expr)


def to_float_number(value: Any) -> float:
    return float(to_exact_number(value).evalf())


def is_zero(value: Any, tol: float = 1e-12) -> bool:
    return abs(to_float_number(value)) <= tol


def is_positive(value: Any, tol: float = 1e-12) -> bool:
    return to_float_number(value) > tol


def is_negative(value: Any, tol: float = 1e-12) -> bool:
    return to_float_number(value) < -tol


def is_greater(a: Any, b: Any, tol: float = 1e-12) -> bool:
    return to_float_number(to_exact_number(a) - to_exact_number(b)) > tol


def is_less(a: Any, b: Any, tol: float = 1e-12) -> bool:
    return to_float_number(to_exact_number(a) - to_exact_number(b)) < -tol


def fmt_exact(value: Any) -> str:
    expr = simplify(to_exact_number(value))
    if expr.is_Integer or expr.is_Rational:
        return str(expr)
    if isinstance(expr, Float):
        as_float = float(expr)
        if abs(as_float - round(as_float)) < 1e-9:
            return str(int(round(as_float)))
        return f"{as_float:.4g}"
    return str(expr)


def constraint_to_exact(constraint: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "coeffs": [to_exact_number(v) for v in constraint["coeffs"]],
        "rhs": to_exact_number(constraint["rhs"]),
        "sense": constraint["sense"],
    }


def constraint_to_numeric(constraint: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "coeffs": [to_float_number(v) for v in constraint["coeffs"]],
        "rhs": to_float_number(constraint["rhs"]),
        "sense": constraint["sense"],
    }


def format_linear_expression(coeffs: List[Any], var_names: List[str]) -> str:
    parts: List[str] = []
    for i, coef in enumerate(coeffs):
        if i >= len(var_names) or is_zero(coef):
            continue
        name = var_names[i]
        if is_zero(to_exact_number(coef) - 1):
            parts.append(name)
        elif is_zero(to_exact_number(coef) + 1):
            parts.append(f"-{name}")
        else:
            parts.append(f"{fmt_exact(coef)}{name}")

    if not parts:
        return "0"

    expr = parts[0]
    for term in parts[1:]:
        expr += f" - {term[1:]}" if term.startswith("-") else f" + {term}"
    return expr
