"""Phương pháp hình học giải bài toán LP 2 biến."""

from itertools import combinations
from math import atan2, sqrt
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from solver.core import dinh_dang_rang_buoc
from solver.reasoning import ReasoningEngine


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4g}"


def _ensure_2(coeffs: List[float]) -> Tuple[float, float]:
    a = coeffs[0] if len(coeffs) > 0 else 0.0
    b = coeffs[1] if len(coeffs) > 1 else 0.0
    return a, b


def _obj_str(obj_coeffs: List[float], var_names: List[str]) -> str:
    parts: List[str] = []
    for i, c in enumerate(obj_coeffs):
        if i >= len(var_names) or abs(c) < 1e-12:
            continue
        nm = var_names[i]
        if not parts:
            if abs(c - 1) < 1e-9:
                parts.append(nm)
            elif abs(c + 1) < 1e-9:
                parts.append(f"-{nm}")
            else:
                parts.append(f"{_fmt(c)}{nm}")
        else:
            if c < 0:
                parts.append(f"- {_fmt(abs(c))}{nm}")
            else:
                parts.append(f"+ {_fmt(c)}{nm}")
    return " ".join(parts) or "0"


# ---------------------------------------------------------------------------
# Giải hệ 2×2
# ---------------------------------------------------------------------------

def giai_he_phuong_trinh(ci: Dict, cj: Dict) -> Optional[Tuple[float, float]]:
    """Giải hệ 2 phương trình 2 ẩn từ 2 ràng buộc."""
    a1, b1 = _ensure_2(ci["coeffs"])
    a2, b2 = _ensure_2(cj["coeffs"])
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        return None
    x = (ci["rhs"] * b2 - cj["rhs"] * b1) / det
    y = (a1 * cj["rhs"] - a2 * ci["rhs"]) / det
    return x, y


# ---------------------------------------------------------------------------
# Kiểm tra điểm
# ---------------------------------------------------------------------------

def _thoa_man(point: Tuple[float, float], c: Dict, tol: float = 1e-6) -> bool:
    a, b = _ensure_2(c["coeffs"])
    lhs = a * point[0] + b * point[1]
    s = c["sense"]
    if s == "<=":
        return lhs <= c["rhs"] + tol
    if s == ">=":
        return lhs >= c["rhs"] - tol
    return abs(lhs - c["rhs"]) <= tol


def _kiem_tra(point: Tuple[float, float], constraints: List[Dict]) -> bool:
    return all(_thoa_man(point, c) for c in constraints)


# ---------------------------------------------------------------------------
# Phân loại đường biên
# ---------------------------------------------------------------------------

def phan_loai_duong_bien(c: Dict, var_names: List[str]) -> Dict[str, Any]:
    """Phân loại ràng buộc thành case đường biên và tính các trường cần thiết.

    Returns dict với:
      case        — "truc" | "doc" | "ngang" | "thuong"
      mo_ta       — mô tả tiếng Việt
      duong       — chuỗi phương trình đường
      diem_ve     — [[x1,y1],[x2,y2]] (None = extend đến giới hạn chart)
      huong       — "phải"/"trái"/"trên"/"dưới"
      huong_vec   — [dx,dy] chuẩn hoá
      huong_mien  — alias huong_vec (frontend compat)
      coeffs      — [a, b]
      rhs         — vế phải
      sense       — sense gốc
      label       — biểu thức dạng "..."
    """
    a, b = _ensure_2(c["coeffs"])
    rhs  = c["rhs"]
    sense = c["sense"]

    label_eq = dinh_dang_rang_buoc(
        {"coeffs": [a, b], "rhs": rhs, "sense": "="}, var_names
    )

    za = abs(a) < 1e-12
    zb = abs(b) < 1e-12
    zr = abs(rhs) < 1e-12

    def _norm(nx: float, ny: float) -> List[float]:
        mag = sqrt(nx * nx + ny * ny)
        return [nx / mag, ny / mag] if mag > 1e-12 else [0.0, 0.0]

    # ── CASE truc: đường đi qua gốc, song song trục ──────────────────────
    if zb and zr:
        # x1 = 0 (trục tung)
        # a*x1 = 0, ví dụ -x1 <= 0 → x1 >= 0 → phía phải
        nx_sign = -a if sense == "<=" else a
        direction = "phải" if nx_sign > 0 else "trái"
        hv = _norm(1.0 if direction == "phải" else -1.0, 0.0)
        return {
            "case": "truc",
            "mo_ta": f"Trục tung ({var_names[0]} = 0) — miền nghiệm phía {direction}",
            "duong": f"{var_names[0]} = 0",
            "diem_ve": [[0.0, 0.0], [0.0, None]],
            "huong": direction, "huong_vec": hv, "huong_mien": hv,
            "coeffs": [a, b], "rhs": rhs, "sense": sense, "label": label_eq,
        }

    if za and zr:
        # x2 = 0 (trục hoành)
        ny_sign = -b if sense == "<=" else b
        direction = "trên" if ny_sign > 0 else "dưới"
        hv = _norm(0.0, 1.0 if direction == "trên" else -1.0)
        return {
            "case": "truc",
            "mo_ta": f"Trục hoành ({var_names[1]} = 0) — miền nghiệm phía {direction}",
            "duong": f"{var_names[1]} = 0",
            "diem_ve": [[0.0, 0.0], [None, 0.0]],
            "huong": direction, "huong_vec": hv, "huong_mien": hv,
            "coeffs": [a, b], "rhs": rhs, "sense": sense, "label": label_eq,
        }

    # ── CASE doc: chỉ x1 (b≈0) ───────────────────────────────────────────
    if zb:
        x1_val = rhs / a
        nx_sign = -a if sense == "<=" else a
        direction = "phải" if nx_sign > 0 else "trái"
        hv = _norm(1.0 if direction == "phải" else -1.0, 0.0)
        return {
            "case": "doc",
            "mo_ta": f"Đường thẳng đứng {var_names[0]} = {_fmt(x1_val)}",
            "duong": f"{var_names[0]} = {_fmt(x1_val)}",
            "diem_ve": [[x1_val, 0.0], [x1_val, None]],
            "huong": direction, "huong_vec": hv, "huong_mien": hv,
            "coeffs": [a, b], "rhs": rhs, "sense": sense, "label": label_eq,
        }

    # ── CASE ngang: chỉ x2 (a≈0) ─────────────────────────────────────────
    if za:
        x2_val = rhs / b
        ny_sign = -b if sense == "<=" else b
        direction = "trên" if ny_sign > 0 else "dưới"
        hv = _norm(0.0, 1.0 if direction == "trên" else -1.0)
        return {
            "case": "ngang",
            "mo_ta": f"Đường nằm ngang {var_names[1]} = {_fmt(x2_val)}",
            "duong": f"{var_names[1]} = {_fmt(x2_val)}",
            "diem_ve": [[0.0, x2_val], [None, x2_val]],
            "huong": direction, "huong_vec": hv, "huong_mien": hv,
            "coeffs": [a, b], "rhs": rhs, "sense": sense, "label": label_eq,
        }

    # ── CASE thuong: cả 2 hệ số khác 0 ───────────────────────────────────
    x2_at_0 = rhs / b   # khi x1 = 0
    x1_at_0 = rhs / a   # khi x2 = 0
    # Hướng sơ bộ từ sense (sẽ được cập nhật lại theo centroid ở bước 2)
    nx0, ny0 = (-a, -b) if sense == "<=" else (a, b)
    hv = _norm(nx0, ny0)
    return {
        "case": "thuong",
        "mo_ta": label_eq,
        "duong": label_eq,
        "diem_ve": [[0.0, x2_at_0], [x1_at_0, 0.0]],
        "x2_at_x1_0": x2_at_0,
        "x1_at_x2_0": x1_at_0,
        "huong": "", "huong_vec": hv, "huong_mien": hv,
        "coeffs": [a, b], "rhs": rhs, "sense": sense, "label": label_eq,
    }


def tinh_huong_vec(c: Dict, cx: float, cy: float) -> List[float]:
    """Vector đơn vị chỉ vào phía miền nghiệm (dựa trên centroid)."""
    a, b = _ensure_2(c["coeffs"])
    dot = a * cx + b * cy - c["rhs"]
    nx, ny = (a, b) if dot > 0 else (-a, -b)
    mag = sqrt(nx * nx + ny * ny)
    return [nx / mag, ny / mag] if mag > 1e-12 else [0.0, 0.0]


def sap_xep_dinh_convex(vertices: List[List[float]]) -> List[List[float]]:
    """Sắp xếp đỉnh theo thứ tự convex hull (theo góc quanh centroid)."""
    if len(vertices) < 2:
        return vertices[:]
    cx = sum(v[0] for v in vertices) / len(vertices)
    cy = sum(v[1] for v in vertices) / len(vertices)
    return sorted(vertices, key=lambda v: atan2(v[1] - cy, v[0] - cx))


# ---------------------------------------------------------------------------
# Reasoning: giải thích từng bước tính giao điểm
# ---------------------------------------------------------------------------

def _giai_trinh_giao(
    ci: Dict, cj: Dict,
    i: int, j: int,
    sol: Tuple[float, float],
    var_names: List[str],
    label: str
) -> Tuple[str, str]:
    a1, b1 = _ensure_2(ci["coeffs"]); r1 = ci["rhs"]
    a2, b2 = _ensure_2(cj["coeffs"]); r2 = cj["rhs"]
    v1, v2 = var_names[0], var_names[1]
    ln = [
        f"(d{i+1}): {_fmt(a1)}{v1} + {_fmt(b1)}{v2} = {_fmt(r1)}",
        f"(d{j+1}): {_fmt(a2)}{v1} + {_fmt(b2)}{v2} = {_fmt(r2)}",
    ]
    if abs(a1) > 1e-12 and abs(a2) > 1e-12:
        m = a2 / a1
        b_diff = b2 - b1 * m
        r_diff = r2 - r1 * m
        ln.append(f"(d{i+1})×{_fmt(m)}: {_fmt(a2)}{v1} + {_fmt(b1*m)}{v2} = {_fmt(r1*m)}")
        if abs(b_diff) > 1e-12:
            ln.append(f"Trừ → {_fmt(b_diff)}{v2} = {_fmt(r_diff)}")
            ln.append(f"→ {v2} = {_fmt(sol[1])}")
            ln.append(f"→ {v1} = ({_fmt(r1)} - {_fmt(b1)}×{_fmt(sol[1])}) / {_fmt(a1)} = {_fmt(sol[0])}")
        else:
            ln.append(f"→ {v1} = {_fmt(sol[0])}, {v2} = {_fmt(sol[1])}")
    else:
        ln.append(f"→ {v1} = {_fmt(sol[0])},  {v2} = {_fmt(sol[1])}")
    return "\n".join(ln), f"{label}({_fmt(sol[0])}, {_fmt(sol[1])})"


# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------

def giai_hinh_hoc(
    n: int,
    constraints: List[Dict],
    obj_coeffs: List[float],
    obj_type: str
) -> Dict[str, Any]:
    """Giải LP 2 biến bằng phương pháp hình học."""
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
            "vertices": [], "boundary_lines": [], "level_lines": {},
            "var_names": var_names, "n": n, "only_2d": False,
        }

    obj_str = _obj_str(obj_coeffs, var_names)
    engine.ghi(
        "Bắt đầu giải hình học",
        f"{'max' if obj_type=='max' else 'min'} z = {obj_str}\n"
        f"Số ràng buộc: {len(constraints)}",
        "INFO"
    )

    # ── Bước 1: Vẽ đường biên ─────────────────────────────────────────────
    engine.ghi(
        "Bước 1: Vẽ các đường biên",
        "Chuyển mỗi ràng buộc thành đường thẳng (cho dấu =)", "INFO"
    )

    boundary_lines: List[Dict] = []
    for i, c in enumerate(constraints):
        info = phan_loai_duong_bien(c, var_names)
        info["idx"] = i + 1
        case = info["case"]

        if case in ("truc", "doc", "ngang"):
            engine.ghi_cong_thuc(
                f"Đường (d{i+1}): {info['mo_ta']}",
                info["duong"],
                f"Miền nghiệm phía {info['huong']}" if info["huong"] else "Xem mô tả"
            )
        else:
            x2_0 = info.get("x2_at_x1_0")
            x1_0 = info.get("x1_at_x2_0")
            parts = []
            if x2_0 is not None:
                parts.append(f"Cho {var_names[0]}=0 → {var_names[1]}={_fmt(x2_0)}")
            if x1_0 is not None:
                parts.append(f"Cho {var_names[1]}=0 → {var_names[0]}={_fmt(x1_0)}")
            ket = (f"Điểm (0, {_fmt(x2_0)}) và ({_fmt(x1_0)}, 0)"
                   if x2_0 is not None and x1_0 is not None else "Xem tọa độ")
            engine.ghi_cong_thuc(
                f"Đường (d{i+1}): {dinh_dang_rang_buoc(c, var_names)}",
                "\n".join(parts),
                ket
            )

        boundary_lines.append(info)

    # ── Bước 2: Miền khả thi & đỉnh ──────────────────────────────────────
    engine.ghi(
        "Bước 2: Xác định miền khả thi",
        "Xác định nửa mặt phẳng thỏa mãn từng ràng buộc.", "INFO"
    )

    smap = {"<=": "≤", ">=": "≥", "=": "="}
    for i, c in enumerate(constraints):
        if boundary_lines[i]["case"] != "thuong":
            continue
        ok = _thoa_man((0.0, 0.0), c)
        sn = smap.get(c["sense"], c["sense"])
        engine.ghi_cong_thuc(
            f"Ràng buộc (d{i+1}): thay (0,0)",
            f"0 {sn} {_fmt(c['rhs'])} → {'✓' if ok else '✗'}",
            f"Miền nghiệm {'chứa' if ok else 'không chứa'} gốc tọa độ"
        )

    # Tính các đỉnh khả thi
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
        cf_str, kr_str = _giai_trinh_giao(
            constraints[i], constraints[j], i, j, sol, var_names, lbl
        )
        engine.ghi_cong_thuc(
            f"Đỉnh {lbl}: Giao của (d{i+1}) và (d{j+1})",
            cf_str, kr_str
        )
        feasible_vertices.append({
            "point": [round(sol[0], 6), round(sol[1], 6)],
            "label": lbl,
            "feasible": True,
            "is_optimal": False,
            "from_indices": (i, j),
        })

    if not feasible_vertices:
        engine.ghi("Kết luận", "Miền khả thi rỗng — bài toán vô nghiệm.", "KET_LUAN")
        return {
            "feasible": False, "solution": {}, "z": None,
            "obj_type": obj_type, "method": "geometric",
            "steps_constraints": [], "reasoning": engine.to_dict(),
            "vertices": [], "boundary_lines": boundary_lines, "level_lines": {},
            "var_names": var_names, "n": n, "only_2d": True,
        }

    # Cập nhật huong_mien theo centroid thực tế
    cx_f = sum(v["point"][0] for v in feasible_vertices) / len(feasible_vertices)
    cy_f = sum(v["point"][1] for v in feasible_vertices) / len(feasible_vertices)
    for i, c in enumerate(constraints):
        hv = tinh_huong_vec(c, cx_f, cy_f)
        boundary_lines[i]["huong_vec"]  = hv
        boundary_lines[i]["huong_mien"] = hv

    engine.ghi(
        "Miền khả thi",
        f"Tìm được {len(feasible_vertices)} đỉnh: " +
        ", ".join(
            f"{v['label']}({_fmt(v['point'][0])},{_fmt(v['point'][1])})"
            for v in feasible_vertices
        ),
        "KET_LUAN"
    )

    # ── Bước 3: Tính z tại các đỉnh ──────────────────────────────────────
    engine.ghi(
        "Bước 3: Tính z tại các đỉnh",
        "Theo Định lý điểm cực biên, z tối ưu nằm tại 1 đỉnh của miền khả thi.",
        "INFO"
    )

    best_z: Optional[float] = None
    best_idx = 0
    z_vals: List[float] = []

    for v in feasible_vertices:
        pt = v["point"]
        z = sum(obj_coeffs[k] * pt[k] for k in range(min(n, len(obj_coeffs))))
        z_vals.append(z)
        is_best = (best_z is None
                   or (obj_type == "max" and z > best_z + 1e-9)
                   or (obj_type == "min" and z < best_z - 1e-9))
        if is_best:
            best_z = z
            best_idx = len(z_vals) - 1

    # Ghi reasoning và đánh dấu optimal
    for idx_v, (v, z) in enumerate(zip(feasible_vertices, z_vals)):
        pt  = v["point"]
        lbl = v["label"]
        is_opt = (idx_v == best_idx)
        terms = " + ".join(
            f"{_fmt(obj_coeffs[k])}×{_fmt(pt[k])}"
            for k in range(min(n, len(obj_coeffs)))
        )
        engine.ghi_cong_thuc(
            f"Tại đỉnh {lbl}({_fmt(pt[0])}, {_fmt(pt[1])})",
            f"z = {terms} = {_fmt(z)}",
            f"z = {_fmt(z)}" + (" ← TỐI ƯU" if is_opt else "")
        )
        v["z"] = round(z, 6)
        v["is_optimal"] = is_opt
        v["label"] = f"{lbl}({_fmt(pt[0])},{_fmt(pt[1])})"

    # ── Bước 4: Đường mức ─────────────────────────────────────────────────
    opt_v = feasible_vertices[best_idx]
    huong_str   = "ra xa gốc tọa độ" if obj_type == "max" else "về gốc tọa độ"
    tinh_str    = "cuối cùng" if obj_type == "max" else "đầu tiên"
    engine.ghi(
        "Bước 4: Kiểm tra bằng đường mức",
        f"Vẽ đường mức z = {obj_str} = 0.\n"
        f"Tịnh tiến đường mức {huong_str} "
        f"(hướng {'tăng' if obj_type=='max' else 'giảm'} z).\n"
        f"Điểm {tinh_str} đường mức chạm miền khả thi là điểm tối ưu.\n"
        f"→ Điểm tối ưu: {opt_v['label']} với z = {_fmt(best_z)}",
        "KET_LUAN"
    )

    # ── Level lines ───────────────────────────────────────────────────────
    mag_oc = sqrt(sum(c ** 2 for c in obj_coeffs[:2]))
    grad = [obj_coeffs[k] / mag_oc for k in range(2)] if mag_oc > 1e-12 else [0.0, 0.0]
    level_lines = {
        "z_opt":     best_z,
        "coeffs":    obj_coeffs[:2],
        "direction": grad,
        "obj_type":  obj_type,
    }

    solution = {var_names[i]: opt_v["point"][i] for i in range(n)}

    return {
        "feasible": True,
        "solution": solution,
        "z": round(best_z, 6),
        "obj_type": obj_type,
        "method": "geometric",
        "steps_constraints": [],
        "reasoning": engine.to_dict(),
        "vertices": feasible_vertices,
        "boundary_lines": boundary_lines,
        "level_lines": level_lines,
        "var_names": var_names,
        "n": n,
        "only_2d": True,
    }


# ---------------------------------------------------------------------------
# Xuất file txt
# ---------------------------------------------------------------------------

def xuat_file_hinh_hoc(result: Dict[str, Any], bai_toan_info: Dict[str, Any]) -> str:
    """Xuất kết quả bài toán hình học ra chuỗi txt theo format tài liệu."""
    n          = result.get("n", 2)
    var_names  = result.get("var_names", [f"x{i+1}" for i in range(n)])
    obj_type   = result.get("obj_type", "max")
    obj_coeffs = bai_toan_info.get("obj_coeffs", [])
    cs_raw     = bai_toan_info.get("constraints", [])
    obj_str    = _obj_str(obj_coeffs, var_names) if obj_coeffs else "?"
    smap       = {"<=": "≤", ">=": "≥", "=": "="}
    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Header ────────────────────────────────────────────────────────────
    W = 51
    lines += [
        "═" * W,
        "BÀI TOÁN QUY HOẠCH TUYẾN TÍNH",
        "Phương pháp: Hình học",
        f"Ngày: {now}",
        "═" * W,
        f"Hàm mục tiêu: {'max' if obj_type=='max' else 'min'} z = {obj_str}",
        "Ràng buộc:",
    ]
    for i, c in enumerate(cs_raw):
        fmt_c = dinh_dang_rang_buoc(c, var_names) if obj_coeffs else str(c)
        lines.append(f"  ({i+1}) {fmt_c}")
    lines.append("")

    # ── Bước 1 ────────────────────────────────────────────────────────────
    bls = result.get("boundary_lines", [])
    lines += ["BƯỚC 1: VẼ CÁC ĐƯỜNG BIÊN", "─" * 35]
    for bl in bls:
        idx  = bl.get("idx", "?")
        case = bl.get("case", "thuong")
        lbl  = bl.get("label", "")
        lines.append(f"Đường (d{idx}): {lbl}")
        if case in ("truc", "doc", "ngang"):
            lines.append(f"  {bl.get('mo_ta', '')}")
            if bl.get("huong"):
                lines.append(f"  Miền nghiệm phía {bl['huong']}")
        else:
            dv = bl.get("diem_ve") or []
            if len(dv) == 2 and dv[0] is not None and dv[1] is not None:
                x2_0, x1_0 = dv[0][1], dv[1][0]
                if x2_0 is not None:
                    lines.append(f"  Cho {var_names[0]}=0 → {var_names[1]}={_fmt(x2_0)} → Điểm (0, {_fmt(x2_0)})")
                if x1_0 is not None:
                    lines.append(f"  Cho {var_names[1]}=0 → {var_names[0]}={_fmt(x1_0)} → Điểm ({_fmt(x1_0)}, 0)")
    lines.append("")

    # ── Bước 2 ────────────────────────────────────────────────────────────
    lines += ["BƯỚC 2: XÁC ĐỊNH MIỀN KHẢ THI", "─" * 35]
    for bl in bls:
        if bl.get("case") != "thuong":
            continue
        idx = bl.get("idx", "?")
        a_b = bl.get("coeffs", [0, 0])
        rhs = bl.get("rhs", 0)
        sn  = smap.get(bl.get("sense", "<="), "≤")
        ok  = (0.0 <= rhs) if bl.get("sense") == "<=" else (0.0 >= rhs)
        lines.append(f"Ràng buộc (d{idx}): Thay (0,0): 0 {sn} {_fmt(rhs)} {'✓' if ok else '✗'}")
        lines.append(f"  → Miền nghiệm {'chứa' if ok else 'không chứa'} gốc tọa độ")

    verts = [v for v in result.get("vertices", []) if v.get("feasible")]
    lines.append("")
    lines.append("Các đỉnh miền khả thi:")
    for v in verts:
        lines.append(f"  Đỉnh {v.get('label', '?')}")
    lines.append(
        f"Miền khả thi: đa giác {{{', '.join(v.get('label','?') for v in verts)}}}"
    )
    lines.append("")

    # ── Bước 3: Bảng ─────────────────────────────────────────────────────
    lines += ["BƯỚC 3: TÍNH z TẠI CÁC ĐỈNH", "─" * 35]
    cw = [10, 12, 28, 12]
    sep  = "┌" + "┬".join("─" * w for w in cw) + "┐"
    hdr  = "│" + "│".join(h.center(w) for h, w in zip(
        ["Đỉnh", "Tọa độ", "Tính toán", "z"], cw)) + "│"
    mid  = "├" + "┼".join("─" * w for w in cw) + "┤"
    bot  = "└" + "┴".join("─" * w for w in cw) + "┘"
    lines += [sep, hdr, mid]

    for v in verts:
        pt    = v.get("point", [0, 0])
        lbl   = v.get("label", "?")
        z     = v.get("z", 0.0)
        is_op = v.get("is_optimal", False)
        star  = "★ " if is_op else "  "
        coord = f"({_fmt(pt[0])}, {_fmt(pt[1])})"
        if obj_coeffs and len(obj_coeffs) >= 2:
            calc = " + ".join(
                f"{_fmt(obj_coeffs[k])}({_fmt(pt[k])})"
                for k in range(min(n, len(obj_coeffs)))
            ) + f" = {_fmt(z)}"
        else:
            calc = _fmt(z)
        z_tag = _fmt(z) + (" MIN" if is_op and obj_type=="min"
                           else " MAX" if is_op else "")
        row = "│" + "│".join(
            s[:w].center(w) for s, w in zip(
                [star + lbl, coord, calc, z_tag], cw)
        ) + "│"
        lines.append(row)
    lines.append(bot)
    lines.append("")

    # ── Bước 4 ────────────────────────────────────────────────────────────
    huong_str = "ra xa gốc tọa độ" if obj_type == "max" else "về gốc tọa độ"
    tinh_str  = "cuối cùng" if obj_type == "max" else "đầu tiên"
    lines += [
        "BƯỚC 4: KIỂM TRA ĐƯỜNG MỨC", "─" * 35,
        f"Vẽ đường mức z = {obj_str}.",
        f"Tịnh tiến đường mức {huong_str} "
        f"(hướng {'tăng' if obj_type=='max' else 'giảm'} z).",
        f"Điểm {tinh_str} đường mức chạm miền khả thi là điểm tối ưu.",
        "",
    ]

    # ── Kết quả ───────────────────────────────────────────────────────────
    sol      = result.get("solution", {})
    best_z   = result.get("z")
    sol_line = ", ".join(f"{k} = {_fmt(v)}" for k, v in sol.items())
    lines += [
        "═" * W,
        "KẾT QUẢ TỐI ƯU",
        "═" * W,
        sol_line,
        f"z  = {_fmt(best_z)}  ({'MIN' if obj_type=='min' else 'MAX'})",
    ]

    return "\n".join(lines)
