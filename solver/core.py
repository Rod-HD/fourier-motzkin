"""Fourier-Motzkin LP Solver — thuật toán FM đúng cho bài toán tối ưu."""

from typing import Any, Dict, List, Optional, Tuple, TypedDict
from solver.reasoning import ReasoningEngine


class Constraint(TypedDict):
    coeffs: List[float]
    rhs: float
    sense: str


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------

def ten_bien(i: int) -> str:
    return f"x{i + 1}"


def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4g}"


def dinh_dang_rang_buoc(c: Constraint, var_names: List[str]) -> str:
    terms = []
    for i, coef in enumerate(c["coeffs"]):
        if i >= len(var_names) or abs(coef) < 1e-12:
            continue
        name = var_names[i]
        if abs(coef - 1) < 1e-12:
            terms.append(name)
        elif abs(coef + 1) < 1e-12:
            terms.append(f"-{name}")
        else:
            terms.append(f"{_fmt(coef)}{name}")
    if not terms:
        lhs = "0"
    else:
        lhs = terms[0]
        for t in terms[1:]:
            lhs += f" - {t[1:]}" if t.startswith("-") else f" + {t}"
    sm = {"<=": "≤", ">=": "≥", "=": "="}
    return f"{lhs} {sm.get(c['sense'], c['sense'])} {_fmt(c['rhs'])}"


# ---------------------------------------------------------------------------
# Chuẩn hoá ràng buộc (chỉ chuẩn hoá constraints, không dùng norm_obj)
# ---------------------------------------------------------------------------

def chuan_hoa_he(
    constraints: List[Constraint],
    obj_coeffs: List[float],
    obj_type: str,
    engine: ReasoningEngine
) -> List[Constraint]:
    """Chuẩn hoá về dạng ≤."""
    n = len(obj_coeffs)
    var_names = [ten_bien(i) for i in range(n)]
    result: List[Constraint] = []
    engine.ghi("Chuẩn hoá hệ ràng buộc", "Đổi tất cả ràng buộc về dạng ≤.", "INFO")
    for c in constraints:
        if c["sense"] == "<=":
            result.append({"coeffs": c["coeffs"][:], "rhs": c["rhs"], "sense": "<="})
        elif c["sense"] == ">=":
            nc: Constraint = {"coeffs": [-v for v in c["coeffs"]], "rhs": -c["rhs"], "sense": "<="}
            engine.ghi_cong_thuc("Đổi ≥ → ≤",
                dinh_dang_rang_buoc(c, var_names) + "  ×(-1)",
                dinh_dang_rang_buoc(nc, var_names))
            result.append(nc)
        elif c["sense"] == "=":
            cle: Constraint = {"coeffs": c["coeffs"][:], "rhs": c["rhs"], "sense": "<="}
            cge: Constraint = {"coeffs": [-v for v in c["coeffs"]], "rhs": -c["rhs"], "sense": "<="}
            engine.ghi_cong_thuc("Đổi = → 2 ≤",
                dinh_dang_rang_buoc(c, var_names),
                f"{dinh_dang_rang_buoc(cle, var_names)}  và  {dinh_dang_rang_buoc(cge, var_names)}")
            result.append(cle)
            result.append(cge)
    return result


# ---------------------------------------------------------------------------
# Thêm biến z
# ---------------------------------------------------------------------------

def them_bien_z(
    constraints: List[Constraint],
    obj_coeffs: List[float],
    n: int,
    obj_type: str,
    engine: ReasoningEngine
) -> List[Constraint]:
    """Thêm biến z (index n) vào hệ.

    Với max z = c^T x:  thêm  z - c^T x ≤ 0  →  z ≤ c^T x
    Với min z = c^T x:  chuyển thành max(-c^T x):
                        thêm  z + c^T x ≤ 0  →  z ≤ -c^T x
    Sau FM tìm z_max; nếu min thì z_min_gốc = -z_max.
    """
    expanded: List[Constraint] = [
        {"coeffs": c["coeffs"][:] + [0.0], "rhs": c["rhs"], "sense": "<="}
        for c in constraints
    ]
    var_names = [ten_bien(i) for i in range(n)] + ["z"]

    if obj_type == "max":
        # z ≤ c^T x  ↔  z - c^T x ≤ 0
        z_coeffs = [-coef for coef in obj_coeffs] + [1.0]
        obj_str = " + ".join(
            f"{_fmt(obj_coeffs[i])}{ten_bien(i)}" for i in range(n) if abs(obj_coeffs[i]) > 1e-12
        )
        desc = f"z ≤ {obj_str}"
    else:
        # min c^T x = -max(-c^T x); z ≤ -c^T x  ↔  z + c^T x ≤ 0
        z_coeffs = list(obj_coeffs) + [1.0]
        obj_str = " + ".join(
            f"{_fmt(obj_coeffs[i])}{ten_bien(i)}" for i in range(n) if abs(obj_coeffs[i]) > 1e-12
        )
        desc = f"z ≤ -({obj_str})"

    z_c: Constraint = {"coeffs": z_coeffs, "rhs": 0.0, "sense": "<="}
    expanded.append(z_c)
    engine.ghi_cong_thuc(
        "Thêm biến z vào hệ",
        desc,
        dinh_dang_rang_buoc(z_c, var_names)
    )
    return expanded


# ---------------------------------------------------------------------------
# Cô lập biến — trả về bound-constraint dạng f = rhs + coeffs·x
# ---------------------------------------------------------------------------

def co_lap_bien(
    constraints: List[Constraint],
    var_idx: int
) -> Tuple[List[Constraint], List[Constraint], List[Constraint]]:
    """Phân loại ràng buộc theo hệ số của biến var_idx.

    Với ràng buộc  a·var + rest ≤ rhs  (hệ số a tại var_idx):
      Biến đổi: var ≤/≥ f  với  f = rhs/a - sum(c_j/a · xj, j≠var)
                                   = (rhs/a) + sum((-c_j/a) · xj)
      Lưu dạng: {"coeffs": [-c_j/a cho mọi j, var_idx=0], "rhs": rhs/a}
      f được tính: f = stored_rhs + stored_coeffs · x

    Returns:
        nhom_U: var ≤ f  (a > 0)
        nhom_L: var ≥ f  (a < 0)
        nhom_N: không chứa var
    """
    nhom_U: List[Constraint] = []
    nhom_L: List[Constraint] = []
    nhom_N: List[Constraint] = []

    for c in constraints:
        a = c["coeffs"][var_idx] if var_idx < len(c["coeffs"]) else 0.0
        if abs(a) < 1e-12:
            nhom_N.append(c)
            continue

        n_v = len(c["coeffs"])
        # stored_coeffs[j] = -c_j / a
        sc = [-c["coeffs"][j] / a for j in range(n_v)]
        sc[var_idx] = 0.0
        sr = c["rhs"] / a
        bc: Constraint = {"coeffs": sc, "rhs": sr, "sense": "<=" if a > 0 else ">="}

        if a > 0:
            nhom_U.append(bc)
        else:
            nhom_L.append(bc)

    return nhom_U, nhom_L, nhom_N


# ---------------------------------------------------------------------------
# Ghép cặp FM
# ---------------------------------------------------------------------------

def ghep_cap_FM(
    nhom_L: List[Constraint],
    nhom_U: List[Constraint],
    nhom_N: List[Constraint],
    var_idx: int,
    var_names: List[str],
    engine: ReasoningEngine
) -> List[Constraint]:
    """Ghép mọi cặp (l ∈ nhom_L, u ∈ nhom_U) loại biến var_idx.

    f_L = l_rhs + l_coeffs · x   (lower bound)
    f_U = u_rhs + u_coeffs · x   (upper bound)
    Ràng buộc mới: f_L ≤ f_U
      → (l_coeffs - u_coeffs) · x  ≤  u_rhs - l_rhs
    """
    vname = var_names[var_idx] if var_idx < len(var_names) else f"x{var_idx+1}"
    result: List[Constraint] = list(nhom_N)
    n_v = len(var_names)

    for l in nhom_L:
        for u in nhom_U:
            lc = l["coeffs"]
            uc = u["coeffs"]
            new_coeffs = [
                (lc[j] if j < len(lc) else 0.0) - (uc[j] if j < len(uc) else 0.0)
                for j in range(n_v)
            ]
            new_rhs = u["rhs"] - l["rhs"]
            nc: Constraint = {"coeffs": new_coeffs, "rhs": new_rhs, "sense": "<="}

            # Tính biểu thức f_L, f_U để ghi reasoning
            def _expr(bc: Constraint) -> str:
                parts = [_fmt(bc["rhs"])] if abs(bc["rhs"]) > 1e-12 else []
                for j, cf in enumerate(bc["coeffs"]):
                    if j < len(var_names) and abs(cf) > 1e-12:
                        parts.append(f"{_fmt(cf)}{var_names[j]}")
                return " + ".join(parts) if parts else "0"

            fl_str = _expr(l)
            fu_str = _expr(u)
            nc_str = dinh_dang_rang_buoc(nc, var_names)
            engine.ghi_cong_thuc(
                f"Ghép cặp L×U loại {vname}",
                f"{vname} ≥ {fl_str}  và  {vname} ≤ {fu_str}\n→  {fl_str} ≤ {fu_str}",
                nc_str
            )
            result.append(nc)

    return result


# ---------------------------------------------------------------------------
# Tìm z_max
# ---------------------------------------------------------------------------

def tim_z_max(
    constraints: List[Constraint],
    z_idx: int,
    engine: ReasoningEngine
) -> Optional[float]:
    """Thu thập chặn trên của z sau khi loại hết x1..xn.

    Ràng buộc còn lại dạng  a·z ≤ b  (sau khi tất cả xj đều 0).
    a > 0 → z ≤ b/a  (chặn trên).
    a < 0 → z ≥ b/a  (chặn dưới, bỏ qua khi tìm max).
    a = 0 → 0 ≤ b, kiểm tra tính khả thi.
    """
    engine.ghi("Tìm z_max", "Thu thập chặn trên của z.", "INFO")
    uppers: List[float] = []
    bound_strs: List[str] = []

    for c in constraints:
        az = c["coeffs"][z_idx] if z_idx < len(c["coeffs"]) else 0.0
        # Kiểm tra các hệ số khác
        rest = [abs(c["coeffs"][j]) for j in range(len(c["coeffs"])) if j != z_idx]
        has_rest = any(v > 1e-12 for v in rest)
        if has_rest:
            continue

        if abs(az) < 1e-12:
            if c["rhs"] < -1e-9:
                engine.ghi("Vô nghiệm", f"0 ≤ {_fmt(c['rhs'])} vi phạm", "KET_LUAN")
                return None
            continue

        b = c["rhs"] / az
        if az > 0:
            uppers.append(b)
            bound_strs.append(f"z ≤ {_fmt(b)}")

    if not uppers:
        engine.ghi("Không bị chặn", "Không có chặn trên → bài toán không bị chặn.", "CANH_BAO")
        return None

    z_max = min(uppers)
    engine.ghi_cong_thuc(
        "Tổng hợp chặn trên của z",
        "\n".join(f"{s}  (ràng buộc {i+1})" for i, s in enumerate(bound_strs)),
        f"z_max = min({', '.join(_fmt(b) for b in uppers)}) = {_fmt(z_max)}"
    )
    return z_max


# ---------------------------------------------------------------------------
# Back substitution
# ---------------------------------------------------------------------------

def back_substitute_FM(
    steps_constraints: List[List[Constraint]],
    z_max: float,
    n: int,
    engine: ReasoningEngine
) -> List[float]:
    """Thế z=z_max và các biến đã biết, tìm từng xi từ xn về x1.

    steps_constraints[k] = hệ TRƯỚC khi loại x_k (0-indexed).
    Chứa x_k, x_{k+1},...,x_{n-1}, z — thay z và x_{k+1}..x_{n-1} để tìm x_k.
    """
    engine.ghi("Back substitution",
               f"Thế z = {_fmt(z_max)}, tìm x₁...x{n} từ xn về x1.", "INFO")
    solution = [0.0] * n
    z_idx = n

    for k in range(n - 1, -1, -1):
        vname = ten_bien(k)
        cons = steps_constraints[k] if k < len(steps_constraints) else steps_constraints[0]

        lower = -1e15
        upper = 1e15

        for c in cons:
            a = c["coeffs"][k] if k < len(c["coeffs"]) else 0.0
            if abs(a) < 1e-12:
                continue

            # Thế z và các xj đã biết (j > k)
            rhs_val = c["rhs"]
            subs: List[str] = []
            if z_idx < len(c["coeffs"]) and abs(c["coeffs"][z_idx]) > 1e-12:
                rhs_val -= c["coeffs"][z_idx] * z_max
                subs.append(f"z={_fmt(z_max)}")
            for j in range(k + 1, n):
                if j < len(c["coeffs"]) and abs(c["coeffs"][j]) > 1e-12:
                    rhs_val -= c["coeffs"][j] * solution[j]
                    subs.append(f"x{j+1}={_fmt(solution[j])}")

            bound = rhs_val / a
            sub_info = ", ".join(subs)
            suffix = f"  (thế {sub_info})" if subs else ""

            if a > 0:
                upper = min(upper, bound)
                engine.ghi_cong_thuc(
                    f"{vname} ≤ (rhs - ...)/a{suffix}",
                    f"{vname} ≤ {_fmt(rhs_val)} / {_fmt(a)}",
                    f"{vname} ≤ {_fmt(bound)}"
                )
            else:
                lower = max(lower, bound)
                engine.ghi_cong_thuc(
                    f"{vname} ≥ (rhs - ...)/a{suffix}",
                    f"{vname} ≥ {_fmt(rhs_val)} / {_fmt(a)}",
                    f"{vname} ≥ {_fmt(bound)}"
                )

        # Chọn giá trị trong [lower, upper]
        lo = lower if lower > -1e14 else 0.0
        hi = upper if upper < 1e14 else lo

        if abs(hi - lo) < 1e-9:
            val = lo
        elif lower > -1e14 and upper < 1e14:
            val = (lo + hi) / 2.0
        elif lower > -1e14:
            val = lo
        elif upper < 1e14:
            val = max(0.0, hi)
        else:
            val = 0.0

        if abs(val - round(val)) < 1e-9:
            val = float(round(val))

        solution[k] = val
        lo_str = _fmt(lower) if lower > -1e14 else "-∞"
        hi_str = _fmt(upper) if upper < 1e14 else "+∞"
        engine.ghi(f"Kết quả {vname}",
                   f"[{lo_str}, {hi_str}] → {vname} = {_fmt(val)}", "KET_LUAN")

    return solution


# ---------------------------------------------------------------------------
# Kiểm tra nghiệm
# ---------------------------------------------------------------------------

def kiem_tra_nghiem(
    solution: List[float],
    constraints: List[Constraint],
    n: int,
    engine: ReasoningEngine
) -> bool:
    engine.ghi("Kiểm tra nghiệm", "Thế vào hệ ràng buộc gốc.", "INFO")
    for c in constraints:
        lhs = sum(c["coeffs"][j] * solution[j] for j in range(min(n, len(c["coeffs"]))))
        if lhs > c["rhs"] + 1e-6:
            engine.ghi("Vi phạm", f"lhs={_fmt(lhs)} > rhs={_fmt(c['rhs'])}", "CANH_BAO")
            return False
    engine.ghi("Nghiệm hợp lệ", "Tất cả ràng buộc thoả mãn.", "KET_LUAN")
    return True


# ---------------------------------------------------------------------------
# Chart data cho bài toán 2 biến (dùng chung với phương pháp hình học)
# ---------------------------------------------------------------------------

def _tinh_chart_data_2d(
    constraints: List[Constraint],
    obj_coeffs: List[float],
    obj_type: str,
    z_final: float,
    solution: List[float],
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Tính vertices, boundary_lines, level_lines cho biểu đồ 2 biến.

    Dùng lazy import để tránh circular import với solver.geometric.
    """
    from itertools import combinations
    from math import sqrt
    from solver.geometric import (
        phan_loai_duong_bien, giai_he_phuong_trinh, _kiem_tra, tinh_huong_vec
    )

    var_names = [f"x{i+1}" for i in range(2)]

    # Boundary lines
    boundary_lines: List[Dict] = []
    for i, c in enumerate(constraints):
        info = phan_loai_duong_bien(c, var_names)
        info["idx"] = i + 1
        boundary_lines.append(info)

    # Vertices: giao điểm mọi cặp ràng buộc
    labels_abc = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    feasible_vertices: List[Dict] = []
    seen: set = set()

    for (i, j) in combinations(range(len(constraints)), 2):
        sol = giai_he_phuong_trinh(constraints[i], constraints[j])
        if sol is None:
            continue
        key = (round(sol[0], 6), round(sol[1], 6))
        if key in seen:
            continue
        seen.add(key)
        if not _kiem_tra(sol, constraints):
            continue

        lbl = (labels_abc[len(feasible_vertices)]
               if len(feasible_vertices) < len(labels_abc)
               else f"P{len(feasible_vertices)}")
        z = sum(obj_coeffs[k] * sol[k] for k in range(2))
        feasible_vertices.append({
            "point": [round(sol[0], 6), round(sol[1], 6)],
            "label": f"{lbl}({_fmt(sol[0])},{_fmt(sol[1])})",
            "feasible": True,
            "is_optimal": False,
            "from_indices": (i, j),
            "z": round(z, 6),
        })

    # Đánh dấu đỉnh tối ưu (gần nhất với solution)
    opt_x = round(solution[0], 4) if len(solution) > 0 else 0.0
    opt_y = round(solution[1], 4) if len(solution) > 1 else 0.0
    if feasible_vertices:
        best = min(feasible_vertices,
                   key=lambda v: (v["point"][0] - opt_x) ** 2 + (v["point"][1] - opt_y) ** 2)
        best["is_optimal"] = True

    # Cập nhật huong_mien theo centroid thực tế
    if feasible_vertices:
        cx_f = sum(v["point"][0] for v in feasible_vertices) / len(feasible_vertices)
        cy_f = sum(v["point"][1] for v in feasible_vertices) / len(feasible_vertices)
        for idx_c, c in enumerate(constraints):
            hv = tinh_huong_vec(c, cx_f, cy_f)
            boundary_lines[idx_c]["huong_vec"] = hv
            boundary_lines[idx_c]["huong_mien"] = hv

    # Level lines
    mag_oc = sqrt(sum(cf ** 2 for cf in obj_coeffs[:2]))
    sign = 1.0 if obj_type == "max" else -1.0
    grad = [sign * obj_coeffs[k] / mag_oc for k in range(2)] if mag_oc > 1e-12 else [0.0, 0.0]
    level_lines: Dict = {
        "z_opt":     z_final,
        "coeffs":    obj_coeffs[:2],
        "direction": grad,
        "obj_type":  obj_type,
    }

    return feasible_vertices, boundary_lines, level_lines


# ---------------------------------------------------------------------------
# Pipeline chính
# ---------------------------------------------------------------------------

def giai_bai_toan(
    n: int,
    constraints: List[Constraint],
    obj_coeffs: List[float],
    obj_type: str
) -> Dict[str, Any]:
    """Giải LP bằng Fourier-Motzkin + biến z.

    Pipeline:
    1. Chuẩn hoá ràng buộc về ≤
    2. Thêm biến z (index n) với ràng buộc z ≤ c^T x (max) hoặc z ≤ -c^T x (min)
    3. Loại x1...xn bằng FM (giữ z)
    4. Tìm z_max từ hệ còn lại
    5. Back substitute để tìm x1...xn
    6. Trả kết quả
    """
    engine = ReasoningEngine()
    var_names = [ten_bien(i) for i in range(n)]
    var_names_z = var_names + ["z"]
    z_idx = n

    engine.ghi(
        "Bắt đầu giải",
        f"n={n}, {obj_type} z = " +
        " + ".join(f"{_fmt(obj_coeffs[i])}{var_names[i]}" for i in range(n)),
        "INFO"
    )

    # Bước 1: chuẩn hoá ràng buộc
    norm_c = chuan_hoa_he(constraints, obj_coeffs, obj_type, engine)

    # Bước 2: thêm z
    system = them_bien_z(norm_c, obj_coeffs, n, obj_type, engine)

    # Lưu hệ ban đầu
    steps_constraints: List[List[Constraint]] = [system[:]]
    steps_info = [{
        "buoc": 0,
        "tieu_de": "Hệ ban đầu (sau chuẩn hoá + biến z)",
        "constraints": [dinh_dang_rang_buoc(c, var_names_z) for c in system],
        "bi_loai": [],
        "moi_tao": []
    }]

    # Bước 3: loại x1...xn
    current = system[:]
    for k in range(n):
        vname = var_names[k]
        before_strs = [dinh_dang_rang_buoc(c, var_names_z) for c in current]

        engine.ghi(f"Loại biến {vname}",
                   f"Phân loại theo hệ số {vname}.", "INFO")

        nhom_U, nhom_L, nhom_N = co_lap_bien(current, k)
        engine.ghi(
            f"Nhóm {vname}",
            f"Nhóm U (≤): {len(nhom_U)}  |  Nhóm L (≥): {len(nhom_L)}  |  Nhóm N: {len(nhom_N)}",
            "INFO"
        )

        current = ghep_cap_FM(nhom_L, nhom_U, nhom_N, k, var_names_z, engine)
        steps_constraints.append(current[:])

        after_strs = [dinh_dang_rang_buoc(c, var_names_z) for c in current]
        new_ones = [s for s in after_strs if s not in before_strs]
        removed  = [s for s in before_strs if s not in after_strs]

        steps_info.append({
            "buoc": k + 1,
            "tieu_de": f"Sau khi loại {vname}",
            "constraints": after_strs,
            "bi_loai": removed,
            "moi_tao": new_ones
        })

    # Bước 4: tìm z_max
    z_max = tim_z_max(current, z_idx, engine)
    if z_max is None:
        engine.ghi("Kết luận", "Bài toán VÔ NGHIỆM hoặc không bị chặn.", "KET_LUAN")
        return {
            "feasible": False, "solution": {}, "z": None,
            "obj_type": obj_type, "method": "algebraic",
            "steps_constraints": steps_info,
            "reasoning": engine.to_dict(), "vertices": [],
            "boundary_lines": [], "level_lines": {},
            "only_2d": (n == 2), "var_names": var_names, "n": n,
        }

    # z_max là max của (c^T x) với max, hoặc max của (-c^T x) với min
    z_final = z_max if obj_type == "max" else -z_max

    engine.ghi_cong_thuc(
        "Giá trị tối ưu",
        f"z_max từ FM = {_fmt(z_max)}" + (f"  →  z_min = -{_fmt(z_max)}" if obj_type == "min" else ""),
        f"z* = {_fmt(z_final)}"
    )

    # Bước 5: back substitute
    solution = back_substitute_FM(steps_constraints, z_max, n, engine)

    # Bước 6: verify
    feasible = kiem_tra_nghiem(solution, norm_c, n, engine)

    sol_dict = {var_names[i]: round(solution[i], 6) for i in range(n)}
    z_rounded = round(z_final, 6)

    engine.ghi(
        "Kết luận cuối",
        f"Nghiệm: {sol_dict}\nz* = {_fmt(z_rounded)}",
        "KET_LUAN"
    )

    # Chart data (chỉ tính cho n=2 khi có nghiệm)
    if n == 2 and feasible:
        verts, blines, llines = _tinh_chart_data_2d(
            constraints, obj_coeffs, obj_type, z_rounded, solution
        )
    else:
        verts, blines, llines = [], [], {}

    return {
        "feasible": feasible,
        "solution": sol_dict,
        "z": z_rounded,
        "obj_type": obj_type,
        "method": "algebraic",
        "steps_constraints": steps_info,
        "reasoning": engine.to_dict(),
        "vertices": verts,
        "boundary_lines": blines,
        "level_lines": llines,
        "only_2d": (n == 2),
        "var_names": var_names,
        "n": n,
    }
