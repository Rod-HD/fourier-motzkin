"""Trình chạy kiểm thử cho bộ giải Fourier-Motzkin v2.

Chạy từ thư mục ``v2``:

    python -m tests.run_tests

Kiểm bốn nhóm:
  [1] phương pháp đại số (khử Fourier-Motzkin trên Q),
  [2] phương pháp hình học (n = 2),
  [3] parser ràng buộc dạng text,
  [4] ứng dụng kiểm tra phụ thuộc vòng lặp.
"""

import os
import sys
from fractions import Fraction

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmlp import (  # noqa: E402
    LinearProgram,
    ParseError,
    Solution,
    parse_constraints,
    solve,
    solve_geometric,
    test_loop_dependence,
)

from tests.cases import CASES, DEPENDENCE_CASES, TEXT_CASES  # noqa: E402


def _feasible_in(x, constraints, n):
    for coeffs, sense, rhs in constraints:
        lhs = sum(Fraction(coeffs[j]) * x[j] for j in range(n))
        r = Fraction(rhs)
        if sense == "<=" and lhs > r:
            return False
        if sense == ">=" and lhs < r:
            return False
        if sense == "=" and lhs != r:
            return False
    return True


def _check_algebraic(name, n, obj, sense, cons, expected):
    raw = [{"coeffs": c, "sense": s, "rhs": r} for (c, s, r) in cons]
    lp = LinearProgram.from_input(n, sense, obj, raw)
    sol = solve(lp)
    if sol.status != expected["status"]:
        return False, f"trạng thái {sol.status} ≠ {expected['status']}"
    if sol.status != Solution.FEASIBLE:
        return True, f"trạng thái đúng: {sol.status}"
    feas = _feasible_in(sol.x, cons, n)
    z_ok = sol.z == Fraction(expected["z"])
    ok = feas and z_ok
    return ok, f"z*={sol.z} (mong đợi {expected['z']}), nghiệm hợp lệ={feas}"


def _check_geometric(name, n, obj, sense, cons, expected):
    if n != 2:
        return None, "bỏ qua (hình học chỉ cho n=2)"
    if expected["status"] == "unbounded":
        return None, "bỏ qua (hình học không phát hiện không bị chặn)"
    raw = [{"coeffs": c, "sense": s, "rhs": r} for (c, s, r) in cons]
    lp = LinearProgram.from_input(n, sense, obj, raw)
    geo = solve_geometric(lp)
    if expected["status"] == "infeasible":
        ok = geo["status"] == "infeasible"
        return ok, ("vô nghiệm đúng kỳ vọng" if ok else f"trả {geo['status']}")
    if geo["status"] != "feasible":
        return False, f"đáng lẽ feasible, lại {geo['status']}"
    feas = _feasible_in(geo["x"], cons, 2)
    z_ok = geo["z"] == Fraction(expected["z"])
    return (feas and z_ok), f"z*={geo['z']} (mong đợi {expected['z']}), hợp lệ={feas}"


def _check_text(name, n, text, expected):
    if expected == "expect_error":
        try:
            parse_constraints(text, n)
            return False, "đáng lẽ báo lỗi (hệ số không hữu tỉ) nhưng lại parse được"
        except ParseError:
            return True, "báo lỗi đúng như kỳ vọng (sqrt(2) không hữu tỉ)"
    parsed = parse_constraints(text, n)
    if len(parsed) != len(expected):
        return False, f"số ràng buộc {len(parsed)} ≠ {len(expected)}"
    for got, (coeffs, sense, rhs) in zip(parsed, expected):
        if got["sense"] != sense:
            return False, f"sense {got['sense']} ≠ {sense}"
        for gc, ec in zip(got["coeffs"], coeffs):
            if Fraction(gc) != Fraction(ec):
                return False, f"hệ số {gc} ≠ {ec}"
        if Fraction(got["rhs"]) != Fraction(rhs):
            return False, f"rhs {got['rhs']} ≠ {rhs}"
    return True, "parse khớp exact"


def _check_dependence(name, a, c0, b, c1, L, U, expected):
    res = test_loop_dependence(a, c0, b, c1, L, U)
    ok = res["has_dependence"] == expected
    return ok, f"has_dependence={res['has_dependence']} (mong đợi {expected})"


def main():
    total = passed = 0

    print("=" * 72)
    print("  KIỂM THỬ FOURIER-MOTZKIN LP SOLVER — V2")
    print("=" * 72)

    print("\n[1] PHƯƠNG PHÁP ĐẠI SỐ (khử Fourier-Motzkin trên Q)")
    print("-" * 72)
    for (name, n, obj, sense, cons, exp) in CASES:
        ok, detail = _check_algebraic(name, n, obj, sense, cons, exp)
        total += 1; passed += 1 if ok else 0
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")

    print("\n[2] PHƯƠNG PHÁP HÌNH HỌC (n = 2)")
    print("-" * 72)
    for (name, n, obj, sense, cons, exp) in CASES:
        ok, detail = _check_geometric(name, n, obj, sense, cons, exp)
        if ok is None:
            print(f"  [SKIP] {name}\n        {detail}")
            continue
        total += 1; passed += 1 if ok else 0
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")

    print("\n[3] PARSER RÀNG BUỘC DẠNG TEXT")
    print("-" * 72)
    for (name, n, text, exp) in TEXT_CASES:
        ok, detail = _check_text(name, n, text, exp)
        total += 1; passed += 1 if ok else 0
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")

    print("\n[4] KIỂM TRA PHỤ THUỘC VÒNG LẶP (ứng dụng compiler)")
    print("-" * 72)
    for (name, a, c0, b, c1, L, U, exp) in DEPENDENCE_CASES:
        ok, detail = _check_dependence(name, a, c0, b, c1, L, U, exp)
        total += 1; passed += 1 if ok else 0
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")

    print("\n" + "=" * 72)
    print(f"  TỔNG KẾT: {passed}/{total} test PASS")
    print("=" * 72)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
