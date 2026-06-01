"""Phương pháp hình học cho bài toán LP hai biến (phiên bản 2).

Cơ sở lý thuyết (Bertsimas--Tsitsiklis): nếu một bài toán LP có nghiệm tối ưu
hữu hạn thì tối ưu đạt tại ít nhất một *đỉnh* (điểm cực biên) của đa diện khả
thi. Với n = 2, đa diện là một đa giác lồi trong mặt phẳng nên ta có thể:

  1. coi mỗi ràng buộc là một đường thẳng (lấy dấu =),
  2. giao mọi cặp đường để lấy các đỉnh ứng viên,
  3. giữ lại đỉnh nằm trong miền khả thi (thỏa mọi ràng buộc),
  4. tính z tại từng đỉnh và chọn đỉnh tối ưu.

Mọi tọa độ tính trên ``Fraction`` để chính xác; dữ liệu vẽ thì xuất ra float
cho frontend (Chart.js).
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from .model import LinearProgram, Row
from .rational import Number, format_fraction
from .trace import Trace


def _intersect(r1: Row, r2: Row) -> Optional[Tuple[Number, Number]]:
    """Giao điểm hai đường ``a·x = b`` (lấy dấu = của ràng buộc 2 biến)."""
    a1, b1, c1 = r1.coeff(0), r1.coeff(1), r1.rhs
    a2, b2, c2 = r2.coeff(0), r2.coeff(1), r2.rhs
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return x, y


def _satisfies(point: Tuple[Number, Number], rows: List[Row]) -> bool:
    x, y = point
    for r in rows:
        if r.coeff(0) * x + r.coeff(1) * y > r.rhs:
            return False
    return True


def solve_geometric(lp: LinearProgram, trace: Optional[Trace] = None) -> Dict[str, Any]:
    """Giải LP 2 biến bằng liệt kê đỉnh; trả về dict kết quả + dữ liệu vẽ."""
    if lp.n != 2:
        raise ValueError("Phương pháp hình học chỉ áp dụng cho bài toán 2 biến.")
    tr = trace if trace is not None else Trace()
    rows = lp.rows

    tr.add(
        "Phương pháp hình học (n = 2)",
        f"Cần {lp.sense} z = {_obj_str(lp)} với {len(rows)} ràng buộc. "
        "Mỗi ràng buộc là một nửa mặt phẳng; giao của chúng là đa giác khả thi. "
        "Theo định lý điểm cực biên, nghiệm tối ưu nằm ở một ĐỈNH của đa giác, "
        "nên ta tìm mọi đỉnh rồi so giá trị z.",
    )

    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    vertices: List[Dict[str, Any]] = []
    seen: set = set()
    for i, j in combinations(range(len(rows)), 2):
        pt = _intersect(rows[i], rows[j])
        if pt is None:
            continue
        key = (pt[0], pt[1])
        if key in seen:
            continue
        seen.add(key)
        if not _satisfies(pt, rows):
            continue
        z = lp.obj[0] * pt[0] + lp.obj[1] * pt[1]
        label = labels[len(vertices)] if len(vertices) < len(labels) else f"P{len(vertices)}"
        vertices.append({
            "label": label,
            "x": pt[0], "y": pt[1], "z": z,
            "from": (i + 1, j + 1),
        })
        tr.add(
            f"Đỉnh {label}: giao của ràng buộc #{i + 1} và #{j + 1}",
            f"Giải hệ 2 phương trình hai đường biên, được "
            f"{label} = ({format_fraction(pt[0])}, {format_fraction(pt[1])}); điểm này "
            f"thỏa mọi ràng buộc nên là một đỉnh khả thi. z tại đây = {format_fraction(z)}.",
        )

    if not vertices:
        tr.result("Kết luận: miền khả thi rỗng → VÔ NGHIỆM.")
        return {
            "status": "infeasible",
            "vertices": [], "boundary_lines": _boundary_lines(lp, []),
            "level_lines": {}, "trace": tr.to_dict(),
        }

    best = vertices[0]
    for v in vertices[1:]:
        if (lp.sense == "max" and v["z"] > best["z"]) or (
            lp.sense == "min" and v["z"] < best["z"]
        ):
            best = v
    for v in vertices:
        v["is_optimal"] = v is best

    table = "; ".join(f"{v['label']}: z = {format_fraction(v['z'])}" for v in vertices)
    tr.add(
        "So sánh z tại các đỉnh",
        f"Giá trị z tại các đỉnh — {table}. "
        f"Chọn giá trị {'lớn nhất' if lp.sense == 'max' else 'nhỏ nhất'}.",
    )
    tr.result(
        "Kết luận: đỉnh tối ưu",
        f"{best['label']} = ({format_fraction(best['x'])}, {format_fraction(best['y'])}), "
        f"z* = {format_fraction(best['z'])}. Trên biểu đồ, khi trượt đường mức theo hướng tối ưu, "
        f"vị trí cuối cùng nó còn chạm miền khả thi chính là đỉnh này.",
    )

    return {
        "status": "feasible",
        "x": [best["x"], best["y"]],
        "z": best["z"],
        "vertices": vertices,
        "boundary_lines": _boundary_lines(lp, vertices),
        "level_lines": _level_lines(lp, best["z"]),
        "trace": tr.to_dict(),
    }


def _obj_str(lp: LinearProgram) -> str:
    a, b = lp.obj[0], lp.obj[1]
    return f"{format_fraction(a)}x1 + {format_fraction(b)}x2"


def _line_label(r: Row) -> str:
    """Nhãn dạng phương trình đường biên 'a x1 + b x2 = rhs'."""
    a, b, c = r.coeff(0), r.coeff(1), r.rhs
    parts: List[str] = []
    for coef, name in ((a, "x1"), (b, "x2")):
        if coef == 0:
            continue
        if not parts:
            parts.append(name if coef == 1 else f"-{name}" if coef == -1 else f"{format_fraction(coef)}{name}")
        else:
            if coef == 1:
                parts.append(f"+ {name}")
            elif coef == -1:
                parts.append(f"- {name}")
            elif coef > 0:
                parts.append(f"+ {format_fraction(coef)}{name}")
            else:
                parts.append(f"- {format_fraction(-coef)}{name}")
    lhs = " ".join(parts) if parts else "0"
    return f"{lhs} = {format_fraction(c)}"


def _boundary_lines(lp: LinearProgram, vertices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dữ liệu vẽ từng đường biên: hệ số, nhãn, và HƯỚNG vào miền khả thi.

    Hướng vào miền (``feasible_dir``) là vector đơn vị pháp tuyến trỏ về phía
    miền nghiệm, dùng để vẽ mũi tên cho biết nửa mặt phẳng nào được chọn. Xác
    định bằng trọng tâm các đỉnh khả thi: nếu trọng tâm nằm phía ``a·x < rhs``
    thì hướng vào là ``-(a, b)`` (chuẩn hóa), ngược lại là ``(a, b)``.
    """
    from math import sqrt

    if vertices:
        cx = sum(float(v["x"]) for v in vertices) / len(vertices)
        cy = sum(float(v["y"]) for v in vertices) / len(vertices)
    else:
        cx = cy = 0.0

    out: List[Dict[str, Any]] = []
    for i, r in enumerate(lp.rows):
        a, b = float(r.coeff(0)), float(r.coeff(1))
        rhs = float(r.rhs)
        # Pháp tuyến trỏ về phía CHỨA trọng tâm (phía khả thi). Trọng tâm luôn
        # nằm trong miền khả thi, nên hướng vào miền là hướng làm a·x + b·y tiến
        # về phía giá trị của trọng tâm: dot>0 ⇒ (a,b); dot<0 ⇒ (-a,-b).
        dot = a * cx + b * cy - rhs
        nx, ny = (a, b) if dot > 0 else (-a, -b)
        mag = sqrt(nx * nx + ny * ny)
        fdir = [nx / mag, ny / mag] if mag > 1e-12 else [0.0, 0.0]
        out.append({
            "idx": i + 1,
            "a": a,
            "b": b,
            "rhs": rhs,
            "label": _line_label(r),
            "feasible_dir": fdir,
        })
    return out


def _level_lines(lp: LinearProgram, z_opt: Number) -> Dict[str, Any]:
    from math import sqrt
    c1, c2 = float(lp.obj[0]), float(lp.obj[1])
    mag = sqrt(c1 * c1 + c2 * c2)
    sign = 1.0 if lp.sense == "max" else -1.0
    direction = [sign * c1 / mag, sign * c2 / mag] if mag > 1e-12 else [0.0, 0.0]
    return {
        "z_opt": float(z_opt),
        "coeffs": [c1, c2],
        "direction": direction,
        "sense": lp.sense,
    }
