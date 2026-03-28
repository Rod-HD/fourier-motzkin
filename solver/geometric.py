"""Phương pháp hình học giải bài toán LP 2 biến."""

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple
from solver.reasoning import ReasoningEngine


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def _sub(s: str) -> str:
    return s.translate(_SUB)

def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4g}"

def _fmt_coef(coef: float, name: str, first: bool = False) -> str:
    """Định dạng hạng tử cho biểu thức."""
    if abs(coef) < 1e-12:
        return ""
    if abs(coef - 1) < 1e-12:
        return name if first else f"+ {name}"
    if abs(coef + 1) < 1e-12:
        return f"-{name}" if first else f"- {name}"
    if first:
        return f"{_fmt(coef)}{name}"
    if coef > 0:
        return f"+ {_fmt(coef)}{name}"
    return f"- {_fmt(abs(coef))}{name}"

def _expr(coeffs: List[float], var_names: List[str]) -> str:
    """Tạo chuỗi biểu thức từ hệ số."""
    parts = []
    for i, c in enumerate(coeffs):
        if i >= len(var_names) or abs(c) < 1e-12:
            continue
        parts.append(_fmt_coef(c, var_names[i], first=(len(parts) == 0)))
    return " ".join(parts) if parts else "0"

def _constraint_str(c: Dict, var_names: List[str]) -> str:
    sense_map = {"<=": "≤", ">=": "≥", "=": "="}
    return f"{_expr(c['coeffs'], var_names)} {sense_map.get(c['sense'], c['sense'])} {_fmt(c['rhs'])}"

def _eq_str(c: Dict, var_names: List[str]) -> str:
    """Đường biên (dạng = )."""
    return f"{_expr(c['coeffs'], var_names)} = {_fmt(c['rhs'])}"


# ---------------------------------------------------------------------------
# Giải hệ 2 phương trình 2 ẩn
# ---------------------------------------------------------------------------

def giai_he_2x2(
    a1: float, b1: float, c1: float,
    a2: float, b2: float, c2: float
) -> Optional[Tuple[float, float]]:
    """Giải a1·x+b1·y=c1, a2·x+b2·y=c2. Trả None nếu vô nghiệm/vô số nghiệm."""
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return x, y


def giai_he_phuong_trinh(
    constraints: List[Dict],
    indices: Tuple[int, ...]
) -> Optional[List[float]]:
    """Giải hệ n=2 phương trình từ constraints[indices[0]] và constraints[indices[1]]."""
    if len(indices) != 2:
        return None
    i, j = indices
    ci, cj = constraints[i], constraints[j]
    n = max(len(ci["coeffs"]), len(cj["coeffs"]))
    if n < 2:
        return None
    a1, b1 = ci["coeffs"][0] if len(ci["coeffs"]) > 0 else 0.0, \
              ci["coeffs"][1] if len(ci["coeffs"]) > 1 else 0.0
    a2, b2 = cj["coeffs"][0] if len(cj["coeffs"]) > 0 else 0.0, \
              cj["coeffs"][1] if len(cj["coeffs"]) > 1 else 0.0
    sol = giai_he_2x2(a1, b1, ci["rhs"], a2, b2, cj["rhs"])
    if sol is None:
        return None
    return list(sol)


# ---------------------------------------------------------------------------
# Kiểm tra điểm thỏa mãn ràng buộc
# ---------------------------------------------------------------------------

def thoa_man(point: List[float], c: Dict, tol: float = 1e-6) -> bool:
    lhs = sum(c["coeffs"][j] * point[j] for j in range(min(len(c["coeffs"]), len(point))))
    if c["sense"] == "<=":
        return lhs <= c["rhs"] + tol
    if c["sense"] == ">=":
        return lhs >= c["rhs"] - tol
    return abs(lhs - c["rhs"]) <= tol


def kiem_tra_diem(point: List[float], constraints: List[Dict]) -> bool:
    return all(thoa_man(point, c) for c in constraints)


# ---------------------------------------------------------------------------
# Tính các đỉnh miền khả thi
# ---------------------------------------------------------------------------

def tim_cac_dinh(constraints: List[Dict], n: int) -> List[Dict]:
    """Duyệt tổ hợp n ràng buộc, giải hệ, lọc đỉnh khả thi.

    Returns:
        List[{"point": [x1,...], "feasible": bool, "from_indices": (i,j)}]
    """
    result = []
    seen = set()
    for idx_combo in combinations(range(len(constraints)), n):
        sol = giai_he_phuong_trinh(constraints, idx_combo)
        if sol is None:
            continue
        # Làm tròn để tránh trùng lặp
        key = tuple(round(v, 6) for v in sol)
        if key in seen:
            continue
        seen.add(key)
        feasible = kiem_tra_diem(sol, constraints)
        result.append({
            "point": sol,
            "feasible": feasible,
            "from_indices": idx_combo
        })
    return result


# ---------------------------------------------------------------------------
# Vẽ đường biên
# ---------------------------------------------------------------------------

def ve_duong_bien(constraint: Dict, var_names: List[str], idx: int) -> Dict:
    """Tính 2 điểm để vẽ đường biên (x₁=0 và x₂=0).

    Returns:
        {"p1": [0, x2], "p2": [x1, 0], "label": "...", "x2_at_0": ..., "x1_at_0": ...}
    """
    coeffs = constraint["coeffs"]
    rhs = constraint["rhs"]
    a = coeffs[0] if len(coeffs) > 0 else 0.0
    b = coeffs[1] if len(coeffs) > 1 else 0.0
    label = f"{_eq_str(constraint, var_names)}"

    # Cho x1=0 → b·x2 = rhs → x2 = rhs/b
    x2_at_0 = (rhs / b) if abs(b) > 1e-12 else None
    # Cho x2=0 → a·x1 = rhs → x1 = rhs/a
    x1_at_0 = (rhs / a) if abs(a) > 1e-12 else None

    p1 = [0.0, x2_at_0] if x2_at_0 is not None else None
    p2 = [x1_at_0, 0.0] if x1_at_0 is not None else None

    return {
        "p1": p1,
        "p2": p2,
        "label": label,
        "x2_at_x1_0": x2_at_0,
        "x1_at_x2_0": x1_at_0,
        "idx": idx
    }


# ---------------------------------------------------------------------------
# Chuẩn hoá ràng buộc (chỉ dạng ≤/≥/=, đủ n chiều)
# ---------------------------------------------------------------------------

def _ensure_n(coeffs: List[float], n: int) -> List[float]:
    """Đảm bảo coeffs có đúng n phần tử."""
    r = coeffs[:n]
    r += [0.0] * (n - len(r))
    return r


def _chuan_hoa(constraints: List[Dict], n: int) -> List[Dict]:
    """Chuẩn hoá coeffs về đúng độ dài n."""
    return [
        {"coeffs": _ensure_n(c["coeffs"], n), "rhs": c["rhs"], "sense": c["sense"]}
        for c in constraints
    ]


# ---------------------------------------------------------------------------
# Sinh bước giải thích tính đỉnh bằng Gauss (2×2)
# ---------------------------------------------------------------------------

def _giai_trinh_dinh(
    ci: Dict, cj: Dict,
    i: int, j: int,
    sol: List[float],
    var_names: List[str],
    label: str
) -> Tuple[str, str]:
    """Sinh chuỗi cong_thuc và ket_qua cho việc giải giao 2 đường."""
    a1 = ci["coeffs"][0]; b1 = ci["coeffs"][1]; c1 = ci["rhs"]
    a2 = cj["coeffs"][0]; b2 = cj["coeffs"][1]; c2 = cj["rhs"]
    v1, v2 = var_names[0], var_names[1]

    lines = [
        f"(d{i+1}): {_fmt(a1)}{v1} + {_fmt(b1)}{v2} = {_fmt(c1)}",
        f"(d{j+1}): {_fmt(a2)}{v1} + {_fmt(b2)}{v2} = {_fmt(c2)}",
    ]

    # Cách giải: nhân và trừ để loại x1
    det = a1 * b2 - a2 * b1
    if abs(a1) > 1e-12 and abs(a2) > 1e-12:
        m = a2 / a1
        lines.append(f"× ({_fmt(m)}): {_fmt(a2)}{v1} + {_fmt(b1*m)}{v2} = {_fmt(c1*m)}")
        lines.append(f"Trừ 2 phương trình: {_fmt(b2 - b1*m)}{v2} = {_fmt(c2 - c1*m)}")
        if abs(b2 - b1*m) > 1e-12:
            lines.append(f"→ {v2} = {_fmt(c2 - c1*m)}/{_fmt(b2 - b1*m)} = {_fmt(sol[1])}")
            lines.append(f"→ {v1} = ({_fmt(c1)} - {_fmt(b1)}×{_fmt(sol[1])}) / {_fmt(a1)} = {_fmt(sol[0])}")
    elif abs(b1) > 1e-12 and abs(b2) > 1e-12:
        m = b2 / b1
        lines.append(f"Loại {v2}: {_fmt(a1*m - a2)}{v1} = {_fmt(c1*m - c2)}")
        if abs(a1*m - a2) > 1e-12:
            lines.append(f"→ {v1} = {_fmt(sol[0])}")
            lines.append(f"→ {v2} = {_fmt(sol[1])}")
    else:
        lines.append(f"→ {v1}={_fmt(sol[0])}, {v2}={_fmt(sol[1])}")

    cong_thuc = "\n".join(lines)
    ket_qua = f"{label}({_fmt(sol[0])}, {_fmt(sol[1])})"
    return cong_thuc, ket_qua


# ---------------------------------------------------------------------------
# Hàm chính: giai_hinh_hoc
# ---------------------------------------------------------------------------

def giai_hinh_hoc(
    n: int,
    constraints: List[Dict],
    obj_coeffs: List[float],
    obj_type: str
) -> Dict[str, Any]:
    """Giải LP 2 biến bằng phương pháp hình học.

    Chỉ hoạt động đầy đủ với n=2 (2D). Với n≠2 trả only_2d=False.
    """
    engine = ReasoningEngine()
    var_names = [f"x{i+1}" for i in range(n)]
    only_2d = (n == 2)

    if not only_2d:
        engine.ghi("Phương pháp hình học",
                   "Chỉ áp dụng cho bài toán 2 biến.", "CANH_BAO")
        return {
            "feasible": False, "solution": {}, "z": None,
            "obj_type": obj_type, "method": "geometric",
            "steps_constraints": [], "reasoning": engine.to_dict(),
            "vertices": [], "boundary_lines": [],
            "var_names": var_names, "n": n, "only_2d": False
        }

    cs = _chuan_hoa(constraints, n)
    obj_str = _expr(obj_coeffs, var_names)

    engine.ghi("Bắt đầu giải hình học",
               f"Bài toán: {'max' if obj_type=='max' else 'min'} z = {obj_str}\n"
               f"Số ràng buộc: {len(cs)}", "INFO")

    # ── Bước 1: Vẽ đường biên ──────────────────────────────────────────────
    engine.ghi("Bước 1: Vẽ các đường biên",
               "Chuyển mỗi ràng buộc thành đường thẳng bằng cách cho từng biến = 0.",
               "INFO")

    boundary_lines = []
    for i, c in enumerate(cs):
        bd = ve_duong_bien(c, var_names, i)
        boundary_lines.append(bd)

        p1_str = f"(0, {_fmt(bd['x2_at_x1_0'])})" if bd["x2_at_x1_0"] is not None else "không xác định"
        p2_str = f"({_fmt(bd['x1_at_x2_0'])}, 0)" if bd["x1_at_x2_0"] is not None else "không xác định"

        parts = []
        if bd["x2_at_x1_0"] is not None:
            parts.append(f"Cho {var_names[0]}=0 → {var_names[1]}={_fmt(bd['x2_at_x1_0'])} → Điểm {p1_str}")
        if bd["x1_at_x2_0"] is not None:
            parts.append(f"Cho {var_names[1]}=0 → {var_names[0]}={_fmt(bd['x1_at_x2_0'])} → Điểm {p2_str}")

        engine.ghi_cong_thuc(
            f"Đường (d{i+1}): {_constraint_str(c, var_names)}",
            " | ".join(parts) if parts else "Đường song song trục",
            f"Điểm {p1_str} và {p2_str}"
        )

    # ── Bước 2: Miền khả thi & đỉnh ────────────────────────────────────────
    engine.ghi("Bước 2: Xác định miền khả thi",
               "Xác định nửa mặt phẳng thỏa mãn từng ràng buộc.", "INFO")

    origin = [0.0, 0.0]
    for i, c in enumerate(cs):
        lhs_o = sum(c["coeffs"][j] * origin[j] for j in range(n))
        ok = thoa_man(origin, c)
        sense_str = {"<=": "≤", ">=": "≥", "=": "="}.get(c["sense"], c["sense"])
        engine.ghi_cong_thuc(
            f"Ràng buộc (d{i+1}): Thay (0,0)",
            f"{_fmt(lhs_o)} {sense_str} {_fmt(c['rhs'])} → {'✓' if ok else '✗'}",
            f"Miền nghiệm {'chứa' if ok else 'không chứa'} gốc tọa độ"
        )

    # Tính đỉnh
    all_vertices_raw = tim_cac_dinh(cs, n)
    feasible_vertices = [v for v in all_vertices_raw if v["feasible"]]

    # Đặt nhãn chữ cái
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for idx_v, v in enumerate(feasible_vertices):
        v["label"] = labels[idx_v] if idx_v < len(labels) else f"P{idx_v}"

    engine.ghi("Tính các đỉnh miền khả thi",
               f"Duyệt C({len(cs)},{n}) = {len(all_vertices_raw)} giao điểm, "
               f"tìm được {len(feasible_vertices)} đỉnh khả thi.", "INFO")

    for v in all_vertices_raw:
        i_idx, j_idx = v["from_indices"]
        ci, cj = cs[i_idx], cs[j_idx]
        sol = v["point"]
        lbl = v.get("label", "?") if v["feasible"] else "✗"

        if v["feasible"] and len(sol) >= 2:
            cong_thuc, ket_qua = _giai_trinh_dinh(
                ci, cj, i_idx, j_idx, sol, var_names, lbl
            )
        else:
            cong_thuc = (
                f"(d{i_idx+1}): {_eq_str(ci, var_names)}\n"
                f"(d{j_idx+1}): {_eq_str(cj, var_names)}"
            )
            ket_qua = (
                f"({_fmt(sol[0])}, {_fmt(sol[1])}) — ngoài miền khả thi"
                if len(sol) >= 2 else "Vô nghiệm"
            )

        engine.ghi_cong_thuc(
            f"{'Đỉnh ' + lbl if v['feasible'] else 'Giao điểm ngoài miền'}: "
            f"(d{i_idx+1}) ∩ (d{j_idx+1})",
            cong_thuc,
            ket_qua
        )

    if not feasible_vertices:
        engine.ghi("Kết luận", "Miền khả thi rỗng — bài toán vô nghiệm.", "KET_LUAN")
        return {
            "feasible": False, "solution": {}, "z": None,
            "obj_type": obj_type, "method": "geometric",
            "steps_constraints": [], "reasoning": engine.to_dict(),
            "vertices": [], "boundary_lines": boundary_lines,
            "var_names": var_names, "n": n, "only_2d": True
        }

    # ── Bước 3: Tính z tại các đỉnh ────────────────────────────────────────
    engine.ghi("Bước 3: Tính z tại các đỉnh",
               "Theo Định lý điểm cực biên, z tối ưu nằm tại 1 đỉnh của miền khả thi.",
               "INFO")

    best_z = None
    best_vertex = None

    z_vals = []
    for v in feasible_vertices:
        pt = v["point"]
        z_val = sum(obj_coeffs[j] * pt[j] for j in range(min(n, len(obj_coeffs))))
        z_vals.append(z_val)

        is_best_so_far = (
            best_z is None or
            (obj_type == "max" and z_val > best_z + 1e-9) or
            (obj_type == "min" and z_val < best_z - 1e-9)
        )
        if is_best_so_far:
            best_z = z_val
            best_vertex = v

    # Sau khi duyệt xong mới biết best → ghi reasoning
    for v, z_val in zip(feasible_vertices, z_vals):
        pt = v["point"]
        lbl = v["label"]
        is_opt = (v is best_vertex)

        # Tính từng hạng: c1×x1 + c2×x2 = ...
        terms = " + ".join(
            f"{_fmt(obj_coeffs[j])}×{_fmt(pt[j])}"
            for j in range(min(n, len(obj_coeffs)))
        )
        cong_thuc = f"z = {obj_str.replace('x1', _fmt(pt[0])).replace('x2', _fmt(pt[1]))} = {terms} = {_fmt(z_val)}"

        engine.ghi_cong_thuc(
            f"Tại đỉnh {lbl}({_fmt(pt[0])}, {_fmt(pt[1])})",
            cong_thuc,
            f"z = {_fmt(z_val)}" + (" ← TỐI ƯU" if is_opt else "")
        )

    # ── Bước 4: Đường mức ───────────────────────────────────────────────────
    direction = "Tịnh tiến ra xa gốc tọa độ (tăng z)" if obj_type == "max" \
                else "Tịnh tiến về phía gốc tọa độ (giảm z)"
    touch_word = "cuối cùng" if obj_type == "max" else "đầu tiên"
    engine.ghi(
        "Bước 4: Kiểm tra bằng đường mức",
        f"Vẽ đường mức z = {obj_str}. {direction}. "
        f"Điểm {touch_word} đường mức chạm miền khả thi là điểm tối ưu.",
        "INFO"
    )

    opt_lbl = best_vertex["label"]
    opt_pt  = best_vertex["point"]
    engine.ghi(
        "Kết luận",
        f"Điểm tối ưu: {opt_lbl}({_fmt(opt_pt[0])}, {_fmt(opt_pt[1])})\n"
        f"z* = {_fmt(best_z)}",
        "KET_LUAN"
    )

    # ── Kết quả ────────────────────────────────────────────────────────────
    solution = {var_names[i]: round(opt_pt[i], 6) for i in range(n)}
    z_rounded = round(best_z, 6)

    vertices_out = []
    for v, z_val in zip(feasible_vertices, z_vals):
        pt = v["point"]
        lbl = v["label"]
        vertices_out.append({
            "point": [round(pt[0], 6), round(pt[1], 6)],
            "z": round(z_val, 6),
            "feasible": True,
            "is_optimal": (v is best_vertex),
            "label": f"{lbl}({_fmt(pt[0])},{_fmt(pt[1])})"
        })

    # steps_constraints giữ format cũ (để frontend dùng chung)
    steps_constraints = [
        {
            "buoc": 0,
            "tieu_de": "Hệ ràng buộc ban đầu",
            "constraints": [_constraint_str(c, var_names) for c in cs],
            "bi_loai": [],
            "moi_tao": []
        }
    ]

    return {
        "feasible": True,
        "solution": solution,
        "z": z_rounded,
        "obj_type": obj_type,
        "method": "geometric",
        "steps_constraints": steps_constraints,
        "reasoning": engine.to_dict(),
        "vertices": vertices_out,
        "boundary_lines": boundary_lines,
        "var_names": var_names,
        "n": n,
        "only_2d": True
    }


# ---------------------------------------------------------------------------
# Xuất file txt
# ---------------------------------------------------------------------------

def xuat_file_hinh_hoc(result: Dict[str, Any], bai_toan_info: Dict[str, Any]) -> str:
    """Tạo nội dung file txt theo format tài liệu."""
    n          = result.get("n", 2)
    var_names  = result.get("var_names", [f"x{i+1}" for i in range(n)])
    obj_type   = result.get("obj_type", "max")
    obj_coeffs = bai_toan_info.get("obj_coeffs", [])
    cs_raw     = bai_toan_info.get("constraints", [])
    cs         = _chuan_hoa(cs_raw, n) if cs_raw else []

    sense_map  = {"<=": "≤", ">=": "≥", "=": "="}

    obj_str = _expr(obj_coeffs, var_names)
    lines   = []

    # Header
    lines += [
        "═" * 51,
        "BÀI TOÁN QUY HOẠCH TUYẾN TÍNH",
        "Phương pháp: Hình học",
        "═" * 51,
        f"Mục tiêu: {'max' if obj_type=='max' else 'min'} z = {obj_str}",
        "Ràng buộc:"
    ]
    for c in cs:
        lines.append(f"  {_constraint_str(c, var_names)}")
    lines.append("")

    # Bước 1
    lines += [
        "BƯỚC 1: VẼ CÁC ĐƯỜNG BIÊN",
        "─" * 35,
    ]
    for i, c in enumerate(cs):
        bd = ve_duong_bien(c, var_names, i)
        lines.append(f"Đường (d{i+1}): {_eq_str(c, var_names)}")
        if bd["x2_at_x1_0"] is not None:
            lines.append(f"  Cho {var_names[0]}=0 → {var_names[1]}={_fmt(bd['x2_at_x1_0'])} → Điểm (0, {_fmt(bd['x2_at_x1_0'])})")
        if bd["x1_at_x2_0"] is not None:
            lines.append(f"  Cho {var_names[1]}=0 → {var_names[0]}={_fmt(bd['x1_at_x2_0'])} → Điểm ({_fmt(bd['x1_at_x2_0'])}, 0)")
    lines.append("")

    # Bước 2
    lines += [
        "BƯỚC 2: XÁC ĐỊNH MIỀN KHẢ THI",
        "─" * 35,
    ]
    origin = [0.0] * n
    for i, c in enumerate(cs):
        lhs_o = sum(c["coeffs"][j] * origin[j] for j in range(n))
        ok    = thoa_man(origin, c)
        sn    = sense_map.get(c["sense"], c["sense"])
        lines.append(f"Ràng buộc (d{i+1}): Thay (0,0): {_fmt(lhs_o)} {sn} {_fmt(c['rhs'])} {'✓' if ok else '✗'}")
        lines.append(f"  → Miền nghiệm {'chứa' if ok else 'không chứa'} gốc tọa độ")

    lines.append("")
    lines.append("Các đỉnh miền khả thi:")

    fv = [v for v in result.get("vertices", [])]
    for v in fv:
        lines.append(f"  Đỉnh {v['label']}")

    lines.append("")

    # Bước 3
    lines += [
        "BƯỚC 3: TÍNH z TẠI CÁC ĐỈNH",
        "─" * 35,
    ]
    opt_z = result.get("z")
    for v in fv:
        pt    = v["point"]
        z_val = v["z"]
        lbl   = v["label"]
        terms = " + ".join(
            f"{_fmt(obj_coeffs[j])}×{_fmt(pt[j])}"
            for j in range(min(n, len(obj_coeffs)))
        )
        suffix = "  ← TỐI ƯU" if v.get("is_optimal") else ""
        lines.append(f"  Đỉnh {lbl}: z = {terms} = {_fmt(z_val)}{suffix}")

    lines.append("")

    # Bước 4
    direction  = "Tịnh tiến ra xa gốc tọa độ (tăng z)" if obj_type == "max" \
                 else "Tịnh tiến về phía gốc tọa độ (giảm z)"
    touch_word = "cuối cùng" if obj_type == "max" else "đầu tiên"
    lines += [
        "BƯỚC 4: KIỂM TRA ĐƯỜNG MỨC",
        "─" * 35,
        f"Vẽ đường mức z = {obj_str}.",
        f"{direction}.",
        f"Điểm {touch_word} đường mức chạm miền khả thi là điểm tối ưu.",
        ""
    ]

    # Kết quả
    lines += ["KẾT QUẢ", "═" * 51]
    sol = result.get("solution", {})
    sol_str = ", ".join(f"{k} = {_fmt(v)}" for k, v in sol.items())
    lines.append(f"{sol_str}, z = {_fmt(opt_z)}")

    return "\n".join(lines)
