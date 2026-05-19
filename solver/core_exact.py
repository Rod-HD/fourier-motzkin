"""Fourier-Motzkin LP Solver với ưu tiên số học exact."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict

from solver.exact import (
    constraint_to_exact,
    constraint_to_numeric,
    fmt_exact,
    format_linear_expression,
    is_greater,
    is_less,
    is_negative,
    is_positive,
    is_zero,
    to_exact_number,
    to_float_number,
)
from solver.reasoning import ReasoningEngine


class Constraint(TypedDict):
    coeffs: List[Any]
    rhs: Any
    sense: str


def ten_bien(i: int) -> str:
    return f"x{i + 1}"


def _fmt(v: Any) -> str:
    return fmt_exact(v)


def dinh_dang_rang_buoc(c: Constraint, var_names: List[str]) -> str:
    lhs = format_linear_expression(c["coeffs"], var_names)
    sm = {"<=": "≤", ">=": "≥", "=": "="}
    return f"{lhs} {sm.get(c['sense'], c['sense'])} {_fmt(c['rhs'])}"


def serialize_constraint_exact(c: Constraint) -> Dict[str, Any]:
    return {
        "coeffs": [_fmt(v) for v in c["coeffs"]],
        "rhs": _fmt(c["rhs"]),
        "sense": c["sense"],
    }


def format_objective_expression(obj_coeffs: List[Any], n: int) -> str:
    return format_linear_expression(obj_coeffs, [ten_bien(i) for i in range(n)])


def _serialize_solution(solution: List[Any], var_names: List[str]) -> Tuple[Dict[str, str], Dict[str, float]]:
    return (
        {var_names[i]: _fmt(solution[i]) for i in range(len(var_names))},
        {var_names[i]: to_float_number(solution[i]) for i in range(len(var_names))},
    )


def chuan_hoa_he(
    constraints: List[Constraint],
    obj_coeffs: List[Any],
    obj_type: str,
    engine: ReasoningEngine,
) -> List[Constraint]:
    n = len(obj_coeffs)
    var_names = [ten_bien(i) for i in range(n)]
    result: List[Constraint] = []
    engine.ghi("Chuẩn hóa hệ ràng buộc", "Đổi tất cả ràng buộc về dạng ≤.", "INFO")
    for c in constraints:
        c = constraint_to_exact(c)
        if c["sense"] == "<=":
            result.append({"coeffs": c["coeffs"][:], "rhs": c["rhs"], "sense": "<="})
        elif c["sense"] == ">=":
            nc: Constraint = {"coeffs": [-v for v in c["coeffs"]], "rhs": -c["rhs"], "sense": "<="}
            engine.ghi_cong_thuc(
                "Đổi ≥ → ≤",
                dinh_dang_rang_buoc(c, var_names) + "  ×(-1)",
                dinh_dang_rang_buoc(nc, var_names),
            )
            result.append(nc)
        elif c["sense"] == "=":
            cle: Constraint = {"coeffs": c["coeffs"][:], "rhs": c["rhs"], "sense": "<="}
            cge: Constraint = {"coeffs": [-v for v in c["coeffs"]], "rhs": -c["rhs"], "sense": "<="}
            engine.ghi_cong_thuc(
                "Đổi = → 2 ≤",
                dinh_dang_rang_buoc(c, var_names),
                f"{dinh_dang_rang_buoc(cle, var_names)}  và  {dinh_dang_rang_buoc(cge, var_names)}",
            )
            result.append(cle)
            result.append(cge)
    return result


def them_bien_z(
    constraints: List[Constraint],
    obj_coeffs: List[Any],
    n: int,
    obj_type: str,
    engine: ReasoningEngine,
) -> List[Constraint]:
    expanded: List[Constraint] = [
        {"coeffs": c["coeffs"][:] + [to_exact_number(0)], "rhs": c["rhs"], "sense": "<="}
        for c in constraints
    ]
    var_names = [ten_bien(i) for i in range(n)] + ["z"]
    obj_str = format_objective_expression(obj_coeffs, n)

    if obj_type == "max":
        z_coeffs = [-coef for coef in obj_coeffs] + [to_exact_number(1)]
        desc = f"z ≤ {obj_str}"
    else:
        z_coeffs = [to_exact_number(v) for v in obj_coeffs] + [to_exact_number(1)]
        desc = f"z ≤ -({obj_str})"

    z_c: Constraint = {"coeffs": z_coeffs, "rhs": to_exact_number(0), "sense": "<="}
    expanded.append(z_c)
    engine.ghi_cong_thuc("Thêm biến z vào hệ", desc, dinh_dang_rang_buoc(z_c, var_names))
    return expanded


def co_lap_bien(constraints: List[Constraint], var_idx: int) -> Tuple[List[Constraint], List[Constraint], List[Constraint]]:
    nhom_U: List[Constraint] = []
    nhom_L: List[Constraint] = []
    nhom_N: List[Constraint] = []

    for c in constraints:
        a = c["coeffs"][var_idx] if var_idx < len(c["coeffs"]) else to_exact_number(0)
        if is_zero(a):
            nhom_N.append(c)
            continue

        n_v = len(c["coeffs"])
        sc = [-(c["coeffs"][j]) / a for j in range(n_v)]
        sc[var_idx] = to_exact_number(0)
        sr = c["rhs"] / a
        bc: Constraint = {"coeffs": sc, "rhs": sr, "sense": "<=" if is_positive(a) else ">="}
        if is_positive(a):
            nhom_U.append(bc)
        else:
            nhom_L.append(bc)

    return nhom_U, nhom_L, nhom_N


def ghep_cap_FM(
    nhom_L: List[Constraint],
    nhom_U: List[Constraint],
    nhom_N: List[Constraint],
    var_idx: int,
    var_names: List[str],
    engine: ReasoningEngine,
) -> List[Constraint]:
    vname = var_names[var_idx] if var_idx < len(var_names) else f"x{var_idx+1}"
    result: List[Constraint] = list(nhom_N)
    n_v = len(var_names)

    for l in nhom_L:
        for u in nhom_U:
            lc = l["coeffs"]
            uc = u["coeffs"]
            new_coeffs = [
                (lc[j] if j < len(lc) else to_exact_number(0)) - (uc[j] if j < len(uc) else to_exact_number(0))
                for j in range(n_v)
            ]
            new_rhs = u["rhs"] - l["rhs"]
            nc: Constraint = {"coeffs": new_coeffs, "rhs": new_rhs, "sense": "<="}

            def _expr(bc: Constraint) -> str:
                parts = [_fmt(bc["rhs"])] if not is_zero(bc["rhs"]) else []
                for j, cf in enumerate(bc["coeffs"]):
                    if j < len(var_names) and not is_zero(cf):
                        parts.append(f"{_fmt(cf)}{var_names[j]}")
                return " + ".join(parts) if parts else "0"

            engine.ghi_cong_thuc(
                f"Ghép cặp L×U loại {vname}",
                f"{vname} ≥ {_expr(l)}  và  {vname} ≤ {_expr(u)}\n→  {_expr(l)} ≤ {_expr(u)}",
                dinh_dang_rang_buoc(nc, var_names),
            )
            result.append(nc)

    return result


def tim_z_max(constraints: List[Constraint], z_idx: int, engine: ReasoningEngine) -> Optional[Any]:
    engine.ghi("Tìm z_max", "Thu thập chặn trên của z.", "INFO")
    uppers: List[Any] = []
    bound_strs: List[str] = []

    for c in constraints:
        az = c["coeffs"][z_idx] if z_idx < len(c["coeffs"]) else to_exact_number(0)
        rest = [c["coeffs"][j] for j in range(len(c["coeffs"])) if j != z_idx]
        if any(not is_zero(v) for v in rest):
            continue

        if is_zero(az):
            if is_negative(c["rhs"]):
                engine.ghi("Vô nghiệm", f"0 ≤ {_fmt(c['rhs'])} vi phạm", "KET_LUAN")
                return None
            continue

        b = c["rhs"] / az
        if is_positive(az):
            uppers.append(b)
            bound_strs.append(f"z ≤ {_fmt(b)}")

    if not uppers:
        engine.ghi("Không bị chặn", "Không có chặn trên → bài toán không bị chặn.", "CANH_BAO")
        return None

    z_max = uppers[0]
    for upper in uppers[1:]:
        if is_less(upper, z_max):
            z_max = upper

    engine.ghi_cong_thuc(
        "Tổng hợp chặn trên của z",
        "\n".join(f"{s}  (ràng buộc {i+1})" for i, s in enumerate(bound_strs)),
        f"z_max = min({', '.join(_fmt(b) for b in uppers)}) = {_fmt(z_max)}",
    )
    return z_max


def back_substitute_FM(steps_constraints: List[List[Constraint]], z_max: Any, n: int, engine: ReasoningEngine) -> List[Any]:
    engine.ghi("Back substitution", f"Thế z = {_fmt(z_max)}, tìm x₁...x{n} từ xn về x1.", "INFO")
    solution = [to_exact_number(0) for _ in range(n)]
    z_idx = n

    for k in range(n - 1, -1, -1):
        vname = ten_bien(k)
        cons = steps_constraints[k] if k < len(steps_constraints) else steps_constraints[0]

        lower: Optional[Any] = None
        upper: Optional[Any] = None

        for c in cons:
            a = c["coeffs"][k] if k < len(c["coeffs"]) else to_exact_number(0)
            if is_zero(a):
                continue

            rhs_val = c["rhs"]
            subs: List[str] = []
            if z_idx < len(c["coeffs"]) and not is_zero(c["coeffs"][z_idx]):
                rhs_val -= c["coeffs"][z_idx] * z_max
                subs.append(f"z={_fmt(z_max)}")
            for j in range(k + 1, n):
                if j < len(c["coeffs"]) and not is_zero(c["coeffs"][j]):
                    rhs_val -= c["coeffs"][j] * solution[j]
                    subs.append(f"x{j+1}={_fmt(solution[j])}")

            bound = rhs_val / a
            suffix = f"  (thế {', '.join(subs)})" if subs else ""
            if is_positive(a):
                upper = bound if upper is None or is_less(bound, upper) else upper
                engine.ghi_cong_thuc(
                    f"{vname} ≤ (rhs - ...)/a{suffix}",
                    f"{vname} ≤ {_fmt(rhs_val)} / {_fmt(a)}",
                    f"{vname} ≤ {_fmt(bound)}",
                )
            else:
                lower = bound if lower is None or is_greater(bound, lower) else lower
                engine.ghi_cong_thuc(
                    f"{vname} ≥ (rhs - ...)/a{suffix}",
                    f"{vname} ≥ {_fmt(rhs_val)} / {_fmt(a)}",
                    f"{vname} ≥ {_fmt(bound)}",
                )

        if lower is not None and upper is not None and not is_greater(lower, upper):
            val = lower if not is_less(lower, upper) else (lower + upper) / 2
        elif lower is not None:
            val = lower
        elif upper is not None:
            val = upper if not is_less(upper, 0) else to_exact_number(0)
        else:
            val = to_exact_number(0)

        solution[k] = to_exact_number(val)
        lo_str = _fmt(lower) if lower is not None else "-∞"
        hi_str = _fmt(upper) if upper is not None else "+∞"
        engine.ghi(f"Kết quả {vname}", f"[{lo_str}, {hi_str}] → {vname} = {_fmt(solution[k])}", "KET_LUAN")

    return solution


def kiem_tra_nghiem(solution: List[Any], constraints: List[Constraint], n: int, engine: ReasoningEngine) -> bool:
    engine.ghi("Kiểm tra nghiệm", "Thế vào hệ ràng buộc gốc.", "INFO")
    for c in constraints:
        lhs = sum(c["coeffs"][j] * solution[j] for j in range(min(n, len(c["coeffs"]))))
        if is_greater(lhs, c["rhs"]):
            engine.ghi("Vi phạm", f"lhs={_fmt(lhs)} > rhs={_fmt(c['rhs'])}", "CANH_BAO")
            return False
    engine.ghi("Nghiệm hợp lệ", "Tất cả ràng buộc thỏa mãn.", "KET_LUAN")
    return True


def _tinh_chart_data_2d(
    constraints: List[Constraint],
    obj_coeffs: List[Any],
    obj_type: str,
    z_final: Any,
    solution: List[Any],
) -> Tuple[List[Dict], List[Dict], Dict]:
    from itertools import combinations
    from math import sqrt

    from solver.geometric import _kiem_tra, giai_he_phuong_trinh, phan_loai_duong_bien, tinh_huong_vec

    numeric_constraints = [constraint_to_numeric(c) for c in constraints]
    obj_coeffs_numeric = [to_float_number(c) for c in obj_coeffs]
    solution_numeric = [to_float_number(v) for v in solution]
    z_numeric = to_float_number(z_final)

    var_names = [f"x{i+1}" for i in range(2)]
    boundary_lines: List[Dict] = []
    for i, c in enumerate(numeric_constraints):
        info = phan_loai_duong_bien(c, var_names)
        info["idx"] = i + 1
        boundary_lines.append(info)

    labels_abc = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    feasible_vertices: List[Dict] = []
    seen: set = set()
    for (i, j) in combinations(range(len(numeric_constraints)), 2):
        sol = giai_he_phuong_trinh(numeric_constraints[i], numeric_constraints[j])
        if sol is None:
            continue
        key = (round(sol[0], 6), round(sol[1], 6))
        if key in seen:
            continue
        seen.add(key)
        if not _kiem_tra(sol, numeric_constraints):
            continue
        lbl = labels_abc[len(feasible_vertices)] if len(feasible_vertices) < len(labels_abc) else f"P{len(feasible_vertices)}"
        z = sum(obj_coeffs_numeric[k] * sol[k] for k in range(2))
        feasible_vertices.append({
            "point": [round(sol[0], 6), round(sol[1], 6)],
            "label": f"{lbl}({_fmt(sol[0])},{_fmt(sol[1])})",
            "feasible": True,
            "is_optimal": False,
            "from_indices": (i, j),
            "z": round(z, 6),
        })

    opt_x = round(solution_numeric[0], 4) if len(solution_numeric) > 0 else 0.0
    opt_y = round(solution_numeric[1], 4) if len(solution_numeric) > 1 else 0.0
    if feasible_vertices:
        best = min(feasible_vertices, key=lambda v: (v["point"][0] - opt_x) ** 2 + (v["point"][1] - opt_y) ** 2)
        best["is_optimal"] = True

    if feasible_vertices:
        cx_f = sum(v["point"][0] for v in feasible_vertices) / len(feasible_vertices)
        cy_f = sum(v["point"][1] for v in feasible_vertices) / len(feasible_vertices)
        for idx_c, c in enumerate(numeric_constraints):
            hv = tinh_huong_vec(c, cx_f, cy_f)
            boundary_lines[idx_c]["huong_vec"] = hv
            boundary_lines[idx_c]["huong_mien"] = hv

    mag_oc = sqrt(sum(cf ** 2 for cf in obj_coeffs_numeric[:2]))
    sign = 1.0 if obj_type == "max" else -1.0
    grad = [sign * obj_coeffs_numeric[k] / mag_oc for k in range(2)] if mag_oc > 1e-12 else [0.0, 0.0]
    level_lines: Dict[str, Any] = {
        "z_opt": z_numeric,
        "coeffs": obj_coeffs_numeric[:2],
        "direction": grad,
        "obj_type": obj_type,
    }
    return feasible_vertices, boundary_lines, level_lines


def giai_bai_toan(n: int, constraints: List[Constraint], obj_coeffs: List[Any], obj_type: str) -> Dict[str, Any]:
    engine = ReasoningEngine()
    constraints = [constraint_to_exact(c) for c in constraints]
    obj_coeffs = [to_exact_number(v) for v in obj_coeffs]

    var_names = [ten_bien(i) for i in range(n)]
    var_names_z = var_names + ["z"]
    z_idx = n

    engine.ghi("Bắt đầu giải", f"n={n}, {obj_type} z = {format_objective_expression(obj_coeffs, n)}", "INFO")

    norm_c = chuan_hoa_he(constraints, obj_coeffs, obj_type, engine)
    system = them_bien_z(norm_c, obj_coeffs, n, obj_type, engine)

    steps_constraints: List[List[Constraint]] = [system[:]]
    steps_info = [{
        "buoc": 0,
        "tieu_de": "Hệ ban đầu (sau chuẩn hoá + biến z)",
        "constraints": [dinh_dang_rang_buoc(c, var_names_z) for c in system],
        "bi_loai": [],
        "moi_tao": [],
    }]

    current = system[:]
    for k in range(n):
        vname = var_names[k]
        before_strs = [dinh_dang_rang_buoc(c, var_names_z) for c in current]
        engine.ghi(f"Loại biến {vname}", f"Phân loại theo hệ số {vname}.", "INFO")
        nhom_U, nhom_L, nhom_N = co_lap_bien(current, k)
        engine.ghi(
            f"Nhóm {vname}",
            f"Nhóm U (≤): {len(nhom_U)}  |  Nhóm L (≥): {len(nhom_L)}  |  Nhóm N: {len(nhom_N)}",
            "INFO",
        )
        current = ghep_cap_FM(nhom_L, nhom_U, nhom_N, k, var_names_z, engine)
        steps_constraints.append(current[:])
        after_strs = [dinh_dang_rang_buoc(c, var_names_z) for c in current]
        steps_info.append({
            "buoc": k + 1,
            "tieu_de": f"Sau khi loại {vname}",
            "constraints": after_strs,
            "bi_loai": [s for s in before_strs if s not in after_strs],
            "moi_tao": [s for s in after_strs if s not in before_strs],
        })

    z_max = tim_z_max(current, z_idx, engine)
    if z_max is None:
        engine.ghi("Kết luận", "Bài toán VÔ NGHIỆM hoặc không bị chặn.", "KET_LUAN")
        return {
            "feasible": False,
            "solution": {},
            "solution_numeric": {},
            "z": None,
            "z_numeric": None,
            "obj_type": obj_type,
            "method": "algebraic",
            "steps_constraints": steps_info,
            "reasoning": engine.to_dict(),
            "vertices": [],
            "boundary_lines": [],
            "level_lines": {},
            "only_2d": (n == 2),
            "var_names": var_names,
            "n": n,
        }

    z_final = z_max if obj_type == "max" else -z_max
    engine.ghi_cong_thuc(
        "Giá trị tối ưu",
        f"z_max từ FM = {_fmt(z_max)}" + (f"  →  z_min = -{_fmt(z_max)}" if obj_type == "min" else ""),
        f"z* = {_fmt(z_final)}",
    )

    solution = back_substitute_FM(steps_constraints, z_max, n, engine)
    feasible = kiem_tra_nghiem(solution, norm_c, n, engine)
    solution_exact, solution_numeric = _serialize_solution(solution, var_names)

    engine.ghi("Kết luận cuối", f"Nghiệm: {solution_exact}\nz* = {_fmt(z_final)}", "KET_LUAN")

    if n == 2 and feasible:
        verts, blines, llines = _tinh_chart_data_2d(constraints, obj_coeffs, obj_type, z_final, solution)
    else:
        verts, blines, llines = [], [], {}

    return {
        "feasible": feasible,
        "solution": solution_exact,
        "solution_numeric": solution_numeric,
        "z": _fmt(z_final),
        "z_numeric": to_float_number(z_final),
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
