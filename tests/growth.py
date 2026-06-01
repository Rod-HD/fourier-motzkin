"""Đo số ràng buộc qua từng bước khử (có / không lọc dư thừa).

Dùng để minh họa định lượng đà bùng nổ tổ hợp của Fourier-Motzkin và hiệu quả
của bước lọc dư thừa. In bảng số liệu cho vài bài toán.
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmlp.eliminate import eliminate_variable, prune_redundant  # noqa: E402
from fmlp.model import LinearProgram  # noqa: E402
from fmlp.solver import _build_objective_row, _extend  # noqa: E402


def counts(n, sense, obj, cons, do_prune):
    raw = [{"coeffs": c, "sense": s, "rhs": r} for (c, s, r) in cons]
    lp = LinearProgram.from_input(n, sense, obj, raw)
    rows = [r.copy() for r in lp.rows]
    _extend(rows, n + 1)
    rows.append(_build_objective_row(lp))
    cur = prune_redundant(rows) if do_prune else rows
    seq = [len(cur)]
    for k in range(n):
        cur = eliminate_variable(cur, k)
        if do_prune:
            cur = prune_redundant(cur)
        seq.append(len(cur))
    return seq


def hypercube(n):
    """max sum x_i với sum x_i <= n và x_i >= 0 — sinh nhiều ràng buộc khi khử."""
    cons = [([1] * n, "<=", n)]
    for i in range(n):
        c = [0] * n
        c[i] = 1
        cons.append((c, ">=", 0))
    return n, "max", [1] * n, cons


def dense_mixed(n, m):
    """Hệ dày, hệ số dấu xen kẽ để mỗi biến có nhiều chặn trên lẫn chặn dưới.

    Ràng buộc i: sum_j s(i,j) * x_j <= 10 + i, với dấu s(i,j) = +1 nếu (i+j)
    chẵn, -1 nếu lẻ. Kèm hộp 0 <= x_j <= 6 để bài toán bị chặn. Khi khử một
    biến, nhóm L và U đều lớn nên số cặp ghép (|L|*|U|) tăng nhanh — minh họa
    đà bùng nổ tổ hợp.
    """
    cons = []
    for i in range(m):
        coeffs = [1 if (i + j) % 2 == 0 else -1 for j in range(n)]
        cons.append((coeffs, "<=", 10 + i))
    for j in range(n):
        c = [0] * n
        c[j] = 1
        cons.append((c, ">=", 0))
        c2 = [0] * n
        c2[j] = 1
        cons.append((c2, "<=", 6))
    return n, "max", [1] * n, cons


PROBLEMS = [
    ("Simplex 4 biến", *hypercube(4)),
    ("Simplex 6 biến", *hypercube(6)),
    ("Dày 4 biến/8 rb", *dense_mixed(4, 8)),
    ("Dày 5 biến/10 rb", *dense_mixed(5, 10)),
    ("Dày 6 biến/12 rb", *dense_mixed(6, 12)),
]

print("Số ràng buộc sau mỗi bước khử (cột 0 = hệ ban đầu kèm z)")
print("=" * 72)
print(f"{'Bài toán':<20}{'Lọc?':<8}{'Chuỗi số ràng buộc':<40}")
print("-" * 72)
for name, n, sense, obj, cons in PROBLEMS:
    no = counts(n, sense, obj, cons, do_prune=False)
    yes = counts(n, sense, obj, cons, do_prune=True)
    print(f"{name:<20}{'không':<8}{' -> '.join(map(str, no))}")
    print(f"{'':<20}{'có':<8}{' -> '.join(map(str, yes))}")
print("=" * 72)
