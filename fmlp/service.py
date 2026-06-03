"""Lớp dịch vụ: nối bộ giải lõi với tầng web (JSON) và xuất file văn bản.

Mọi giá trị ``Fraction`` được serialize thành cả dạng phân số (chuỗi) lẫn dạng
thập phân (float) để frontend hiển thị linh hoạt.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from .geometry import solve_geometric
from .model import LinearProgram, format_row, var_name
from .parse import parse_constraints
from .rational import format_decimal, format_fraction
from .solver import Solution, solve
from .trace import Trace


def solve_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Giải một yêu cầu từ frontend, trả về dict JSON-ready."""
    n = int(payload["n"])
    sense = payload.get("sense", "max").lower()
    method = payload.get("method", "algebraic").lower()
    obj = payload["obj"]

    if payload.get("input_mode", "form").lower() == "text":
        raw = parse_constraints(payload.get("constraints_text", ""), n)
    else:
        raw = payload["constraints"]

    lp = LinearProgram.from_input(n, sense, obj, raw)

    if method == "geometric":
        if n != 2:
            return {"error": "Phương pháp hình học chỉ dùng cho bài toán 2 biến."}
        return _serialize_geometric(lp, raw)
    return _serialize_algebraic(lp, raw)


def _input_block(lp: LinearProgram) -> Dict[str, Any]:
    return {
        "n": lp.n,
        "sense": lp.sense,
        "objective": _linear_str(lp.obj, lp.n),
        "constraints": [format_row(r, lp.n) for r in lp.rows],
    }


def _serialize_algebraic(lp: LinearProgram, raw: List[Dict]) -> Dict[str, Any]:
    trace = Trace()
    sol: Solution = solve(lp, trace)
    steps = getattr(sol, "steps", [])

    result: Dict[str, Any] = {
        "method": "algebraic",
        "input": _input_block(lp),
        "status": sol.status,
        "steps": steps,
        "trace": trace.to_dict(),
    }
    if sol.status == Solution.FEASIBLE:
        result["solution"] = {
            var_name(i, lp.n): {
                "fraction": format_fraction(sol.x[i]),
                "decimal": format_decimal(sol.x[i]),
            }
            for i in range(lp.n)
        }
        result["z"] = {"fraction": format_fraction(sol.z), "decimal": format_decimal(sol.z)}
        if lp.n == 2:
            result["chart"] = _chart_from_geometry(lp)
    return result


def _serialize_geometric(lp: LinearProgram, raw: List[Dict]) -> Dict[str, Any]:
    trace = Trace()
    geo = solve_geometric(lp, trace)
    result: Dict[str, Any] = {
        "method": "geometric",
        "input": _input_block(lp),
        "status": geo["status"],
        "trace": geo["trace"],
        "chart": {
            "vertices": [_vertex_json(v) for v in geo.get("vertices", [])],
            "boundary_lines": geo.get("boundary_lines", []),
            "level_lines": geo.get("level_lines", {}),
        },
    }
    if geo["status"] == "feasible":
        result["solution"] = {
            var_name(i, lp.n): {
                "fraction": format_fraction(geo["x"][i]),
                "decimal": format_decimal(geo["x"][i]),
            }
            for i in range(lp.n)
        }
        result["z"] = {
            "fraction": format_fraction(geo["z"]),
            "decimal": format_decimal(geo["z"]),
        }
    return result


def _chart_from_geometry(lp: LinearProgram) -> Dict[str, Any]:
    """Tính dữ liệu vẽ (đỉnh, đường biên, đường mức) cho lời giải đại số 2 biến."""
    geo = solve_geometric(lp, Trace())
    return {
        "vertices": [_vertex_json(v) for v in geo.get("vertices", [])],
        "boundary_lines": geo.get("boundary_lines", []),
        "level_lines": geo.get("level_lines", {}),
    }


def _vertex_json(v: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": v["label"],
        "point": [float(v["x"]), float(v["y"])],
        "x_frac": format_fraction(v["x"]),
        "y_frac": format_fraction(v["y"]),
        "z": float(v["z"]),
        "z_frac": format_fraction(v["z"]),
        "is_optimal": v.get("is_optimal", False),
    }


def _linear_str(coeffs, n: int) -> str:
    parts: List[str] = []
    for i in range(n):
        a = coeffs[i]
        if a == 0:
            continue
        name = var_name(i, n)
        if not parts:
            parts.append(name if a == 1 else f"-{name}" if a == -1 else f"{format_fraction(a)}{name}")
        else:
            if a == 1:
                parts.append(f"+ {name}")
            elif a == -1:
                parts.append(f"- {name}")
            elif a > 0:
                parts.append(f"+ {format_fraction(a)}{name}")
            else:
                parts.append(f"- {format_fraction(-a)}{name}")
    return " ".join(parts) if parts else "0"


def export_txt(result: Dict[str, Any]) -> str:
    """Dựng nội dung file .txt giải trình từ kết quả đã serialize."""
    lines: List[str] = []
    bar = "=" * 64
    thin = "-" * 64
    inp = result.get("input", {})

    lines += [
        bar,
        "  FOURIER-MOTZKIN LP SOLVER (v2) - GIẢI TRÌNH",
        f"  Thời điểm: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Phương pháp: {result.get('method', '?')}",
        bar,
        "",
        "BÀI TOÁN",
        thin,
        f"  {inp.get('sense', '').upper()} z = {inp.get('objective', '')}",
        "  Ràng buộc (đã chuẩn hóa về ≤):",
    ]
    for c in inp.get("constraints", []):
        lines.append(f"    {c}")
    lines.append("")

    steps = result.get("steps")
    if steps:
        lines += ["CÁC BƯỚC KHỬ FOURIER-MOTZKIN", thin]
        for st in steps:
            lines.append(f"\n  [{st['step']}] {st['title']}")
            for r in st["rows"]:
                lines.append(f"      {r}")
        lines.append("")

    trace = result.get("trace", {})
    if trace.get("items"):
        lines += ["DIỄN GIẢI CHI TIẾT", thin]
        for it in trace["items"]:
            lines.append(f"\n  ({it['index']}) [{it['kind']}] {it['title']}")
            if it.get("body"):
                for sub in it["body"].split("\n"):
                    lines.append(f"      {sub}")
            if it.get("formula"):
                lines.append(f"      công thức: {it['formula']}")
            if it.get("outcome"):
                lines.append(f"      => {it['outcome']}")
        lines.append("")

    lines += ["KẾT QUẢ", thin]
    if result.get("status") == "feasible":
        for name, val in result.get("solution", {}).items():
            lines.append(f"  {name} = {val['fraction']}  (≈ {val['decimal']})")
        z = result.get("z", {})
        lines.append(f"  z* = {z.get('fraction')}  (≈ {z.get('decimal')})")
    elif result.get("status") == "unbounded":
        lines.append("  Bài toán KHÔNG BỊ CHẶN.")
    else:
        lines.append("  Bài toán VÔ NGHIỆM.")
    lines.append(bar)
    return "\n".join(lines)
