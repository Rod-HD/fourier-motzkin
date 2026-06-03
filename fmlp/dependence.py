"""Kiểm tra phụ thuộc dữ liệu vòng lặp bằng Fourier-Motzkin.

Cài đặt theo Allen & Kennedy, *Optimizing Compilers for Modern Architectures*,
chương Dependence Testing.

Mô hình tổng quát
-----------------
Xét ``d`` vòng lặp lồng nhau:

    for i_1 = L_1 to U_1:
      for i_2 = L_2 to U_2:
        ...
          for i_d = L_d to U_d:
              A[f_1(i), ..., f_m(i)] = ...   # GHI (write)
              ... A[g_1(i), ..., g_m(i)] ... # ĐỌC (read)

Mỗi hàm chỉ số là **affine** (tuyến tính + hằng số) theo vector lặp
``i = (i_1, ..., i_d)``:

    f_k(i) = a_{k,1}·i_1 + ... + a_{k,d}·i_d + c_k^{(w)}
    g_k(i) = b_{k,1}·i_1 + ... + b_{k,d}·i_d + c_k^{(r)}

Tồn tại phụ thuộc khi có hai vector lặp ``iw`` (ghi) và ``ir`` (đọc), cả hai
trong không gian lặp, sao cho chúng trỏ vào **cùng phần tử mảng** trên mọi
chiều:

    f_k(iw) = g_k(ir)   cho mọi k = 1..m

Tức là:

    sum_j a_{k,j}·iw_j - sum_j b_{k,j}·ir_j = c_k^{(r)} - c_k^{(w)}

Cộng với ràng buộc biên:

    L_j <= iw_j <= U_j,   L_j <= ir_j <= U_j   cho mọi j = 1..d

Hệ này có ``2d`` biến (``iw_1..iw_d, ir_1..ir_d``) và ``m + 4d`` ràng buộc.
Dùng khử Fourier-Motzkin kiểm tra tính khả thi (nới lỏng về số thực).

Trường hợp đặc biệt: 1 vòng, 1 chiều mảng → đúng bài toán cũ.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Any, Dict, List, Optional

from .model import LinearProgram
from .rational import format_fraction
from .solver import Solution, solve
from .trace import Trace


# ─────────────────────────────────────────────────────────────────────────────
# Hàm tổng quát (d vòng lặp, m chiều mảng)
# ─────────────────────────────────────────────────────────────────────────────

def test_loop_dependence_general(
    d: int,
    loops: List[Dict[str, int]],
    write_access: List[List[int]],
    read_access: List[List[int]],
    trace: Optional[Trace] = None,
) -> Dict[str, Any]:
    """Kiểm tra phụ thuộc tổng quát cho vòng lặp lồng nhau.

    Parameters
    ----------
    d : int
        Số vòng lặp lồng nhau (số chiều không gian lặp).
    loops : list of dict
        Danh sách ``d`` phần tử, mỗi phần tử ``{"L": int, "U": int}`` là cận
        của vòng lặp tương ứng.
    write_access : list of list
        Hàm chỉ số GHI. Mỗi phần tử là một chiều mảng, dạng
        ``[a_1, a_2, ..., a_d, c]`` (``d`` hệ số + 1 hằng số).
    read_access : list of list
        Hàm chỉ số ĐỌC, cùng định dạng với ``write_access``.
        Số chiều mảng (``len(write_access)``) phải bằng ``len(read_access)``.

    Returns
    -------
    dict với các trường:
        ``has_dependence`` (bool), ``status`` (str), ``witness`` (list|None),
        ``gcd_eliminated`` (bool), ``trace`` (dict).
    """
    tr = trace if trace is not None else Trace()
    m = len(write_access)  # số chiều mảng

    if len(read_access) != m:
        raise ValueError("Số chiều mảng của ghi và đọc phải bằng nhau.")
    if len(loops) != d:
        raise ValueError("Số cận vòng lặp phải bằng d.")
    for row in write_access + read_access:
        if len(row) != d + 1:
            raise ValueError(f"Mỗi hàng hệ số phải có {d+1} phần tử (d hệ số + 1 hằng số).")

    # ── Bước 0: GCD test (điều kiện cần, rất rẻ) ─────────────────────────
    # Với mỗi chiều mảng k: a_k·iw - b_k·ir = c_k^r - c_k^w
    # Phương trình này có nghiệm nguyên ⟺ gcd(a_k, b_k) | (c_k^r - c_k^w).
    # Nếu bất kỳ chiều nào không thỏa → chắc chắn độc lập.
    gcd_eliminated = False
    gcd_lines: List[str] = []
    for k in range(m):
        w_coeffs = write_access[k][:d]   # a_1..a_d
        r_coeffs = read_access[k][:d]    # b_1..b_d
        w_const  = write_access[k][d]    # c^w
        r_const  = read_access[k][d]     # c^r
        diff = r_const - w_const
        # hệ số của iw_j là a_j, của ir_j là -b_j
        all_coeffs = w_coeffs + [-b for b in r_coeffs]
        g = 0
        for v in all_coeffs:
            g = gcd(g, abs(v))
        if g == 0:
            # tất cả hệ số = 0: phương trình là 0 = diff
            if diff != 0:
                gcd_lines.append(
                    f"Chiều {k+1}: tất cả hệ số = 0 nhưng hằng số khác nhau "
                    f"({w_const} ≠ {r_const}) → vô nghiệm."
                )
                gcd_eliminated = True
                break
            continue
        if diff % g != 0:
            gcd_lines.append(
                f"Chiều {k+1}: gcd({', '.join(str(v) for v in all_coeffs)}) = {g} "
                f"không chia hết {diff} → vô nghiệm trên số nguyên."
            )
            gcd_eliminated = True
            break
        gcd_lines.append(
            f"Chiều {k+1}: gcd = {g}, diff = {diff}, {diff} % {g} = 0 → qua."
        )

    tr.add(
        "Bước 1 - GCD test (điều kiện cần, kiểm tra số nguyên)",
        "\n".join(gcd_lines) if gcd_lines else "Không có chiều nào cần kiểm tra.",
    )

    if gcd_eliminated:
        tr.result(
            "Kết luận: KHÔNG có phụ thuộc (GCD test)",
            "Phương trình phụ thuộc không có nghiệm nguyên → hai truy cập "
            "không bao giờ chạm cùng ô nhớ. Vòng lặp an toàn để song song hóa.",
        )
        return {
            "has_dependence": False,
            "status": "infeasible",
            "witness": None,
            "gcd_eliminated": True,
            "trace": tr.to_dict(),
        }

    # ── Bước 1: Xây dựng hệ ràng buộc cho FM ─────────────────────────────
    # Biến: x_1..x_d = iw_1..iw_d,  x_{d+1}..x_{2d} = ir_1..ir_d
    n_vars = 2 * d
    constraints: List[Dict[str, Any]] = []

    # Phương trình phụ thuộc: f_k(iw) = g_k(ir) cho mỗi chiều mảng k
    dep_lines: List[str] = []
    for k in range(m):
        w_coeffs = write_access[k][:d]
        r_coeffs = read_access[k][:d]
        w_const  = write_access[k][d]
        r_const  = read_access[k][d]
        # a·iw - b·ir = c^r - c^w
        coeffs = list(w_coeffs) + [-b for b in r_coeffs]
        rhs = r_const - w_const
        constraints.append({"coeffs": coeffs, "sense": "=", "rhs": rhs})
        lhs_str = _format_dep_eq(w_coeffs, r_coeffs, d)
        dep_lines.append(f"  Chiều {k+1}: {lhs_str} = {rhs}")

    # Ràng buộc biên: L_j <= iw_j <= U_j và L_j <= ir_j <= U_j
    bound_lines: List[str] = []
    for j in range(d):
        L, U = loops[j]["L"], loops[j]["U"]
        # iw_j >= L
        c = [0] * n_vars; c[j] = 1
        constraints.append({"coeffs": c, "sense": ">=", "rhs": L})
        # iw_j <= U
        c = [0] * n_vars; c[j] = 1
        constraints.append({"coeffs": c, "sense": "<=", "rhs": U})
        # ir_j >= L
        c = [0] * n_vars; c[d + j] = 1
        constraints.append({"coeffs": c, "sense": ">=", "rhs": L})
        # ir_j <= U
        c = [0] * n_vars; c[d + j] = 1
        constraints.append({"coeffs": c, "sense": "<=", "rhs": U})
        bound_lines.append(f"  Vòng {j+1}: {L} ≤ iw_{j+1} ≤ {U},  {L} ≤ ir_{j+1} ≤ {U}")

    tr.add(
        "Bước 2 - Xây dựng hệ ràng buộc cho Fourier-Motzkin",
        f"Phương trình phụ thuộc ({m} chiều mảng):\n" + "\n".join(dep_lines) +
        f"\nRàng buộc biên ({d} vòng lặp):\n" + "\n".join(bound_lines) +
        f"\nTổng cộng {n_vars} biến (iw_1..iw_{d}, ir_1..ir_{d}), "
        f"{len(constraints)} ràng buộc.",
    )

    # ── Bước 2: Chạy FM ───────────────────────────────────────────────────
    tr.add(
        "Bước 3 - Khử Fourier-Motzkin kiểm tra khả thi",
        f"Khử lần lượt {n_vars} biến. Nếu hệ vô nghiệm → không phụ thuộc; "
        "nếu khả thi → có thể phụ thuộc (nới lỏng số thực).",
    )
    obj = [0] * n_vars
    lp = LinearProgram.from_input(n_vars, "max", obj, constraints)
    sol: Solution = solve(lp, tr)

    has_dep = sol.status == Solution.FEASIBLE
    if has_dep:
        iw_vals = [format_fraction(sol.x[j]) for j in range(d)]
        ir_vals = [format_fraction(sol.x[d + j]) for j in range(d)]
        tr.result(
            "Kết luận: CÓ THỂ có phụ thuộc (nới lỏng số thực)",
            f"Hệ khả thi. Nhân chứng (số thực): "
            f"iw = ({', '.join(iw_vals)}), ir = ({', '.join(ir_vals)}).\n"
            "Trình biên dịch phải giữ thứ tự, không tự do song song hóa.",
        )
    else:
        tr.result(
            "Kết luận: KHÔNG có phụ thuộc",
            "Hệ vô nghiệm trên số thực ⇒ vô nghiệm trên số nguyên ⇒ "
            "hai truy cập không bao giờ chạm cùng ô nhớ. "
            "Vòng lặp an toàn để song song hóa.",
        )

    return {
        "has_dependence": has_dep,
        "status": sol.status,
        "witness": (
            {"iw": iw_vals, "ir": ir_vals} if has_dep else None
        ),
        "gcd_eliminated": False,
        "trace": tr.to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hàm đơn giản (1 vòng, 1 chiều) - giữ lại để tương thích ngược
# ─────────────────────────────────────────────────────────────────────────────

def test_loop_dependence(
    a: int, c0: int, b: int, c1: int, L: int, U: int,
    trace: Optional[Trace] = None,
) -> Dict[str, Any]:
    """Wrapper đơn giản: 1 vòng lặp, mảng 1 chiều, A[a*i+c0] vs A[b*i+c1]."""
    return test_loop_dependence_general(
        d=1,
        loops=[{"L": L, "U": U}],
        write_access=[[a, c0]],
        read_access=[[b, c1]],
        trace=trace,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tiện ích
# ─────────────────────────────────────────────────────────────────────────────

def _format_dep_eq(w: List[int], r: List[int], d: int) -> str:
    """Định dạng phương trình phụ thuộc dạng 'a1·iw1 + ... - b1·ir1 - ...'."""
    parts: List[str] = []
    for j in range(d):
        if w[j] != 0:
            parts.append(f"{w[j]}·iw_{j+1}" if w[j] != 1 else f"iw_{j+1}")
    for j in range(d):
        if r[j] != 0:
            parts.append(f"-{r[j]}·ir_{j+1}" if r[j] != 1 else f"-ir_{j+1}")
    return " + ".join(parts).replace("+ -", "- ") if parts else "0"
