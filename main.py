"""Flask web app cho Fourier-Motzkin LP Solver."""

import io
from datetime import datetime
from typing import Any, Dict, List

from flask import Flask, Response, jsonify, render_template, request

from solver.core import dinh_dang_rang_buoc, format_objective_expression, giai_bai_toan
from solver.exact import constraint_to_exact, constraint_to_numeric, to_exact_number, to_float_number
from solver.geometric import giai_hinh_hoc, xuat_file_hinh_hoc
from solver.parser import ConstraintTextParseError, parse_constraints_text

app = Flask(__name__)


def _var_names(n: int) -> List[str]:
    return [f"x{i + 1}" for i in range(n)]


def _build_input_display(constraints_exact: List[Dict[str, Any]], n: int) -> List[str]:
    var_names = _var_names(n)
    return [dinh_dang_rang_buoc(c, var_names) for c in constraints_exact]


@app.route("/")
def index() -> str:
    """Trả về trang chủ."""
    return render_template("index.html")


@app.route("/api/solve", methods=["POST"])
def solve() -> Response:
    """Nhận bài toán LP và trả về kết quả giải."""
    try:
        data: Dict[str, Any] = request.get_json(force=True) or {}
        n = int(data["n"])
        if n < 1:
            return jsonify({"error": "Số biến phải ≥ 1"}), 400

        input_mode = data.get("input_mode", "form").lower()
        raw_constraints = (
            parse_constraints_text(data.get("constraints_text", ""), n)
            if input_mode == "text"
            else data["constraints"]
        )
        obj_type = data.get("obj_type", "max").lower()
        method = data.get("method", "algebraic").lower()

        constraints_exact = [constraint_to_exact(c) for c in raw_constraints]
        constraints_numeric = [constraint_to_numeric(c) for c in raw_constraints]
        obj_coeffs_exact = [to_exact_number(v) for v in data["obj_coeffs"]]
        obj_coeffs_numeric = [to_float_number(v) for v in data["obj_coeffs"]]

        if len(obj_coeffs_exact) != n:
            return jsonify({"error": "Số hệ số hàm mục tiêu phải bằng số biến"}), 400
        for c in constraints_exact:
            if len(c["coeffs"]) != n:
                return jsonify({"error": "Số hệ số ràng buộc phải bằng số biến"}), 400

        if method == "geometric":
            result = giai_hinh_hoc(n, constraints_numeric, obj_coeffs_numeric, obj_type)
        else:
            method = "algebraic"
            result = giai_bai_toan(n, constraints_exact, obj_coeffs_exact, obj_type)

        if "z_numeric" not in result and result.get("z") is not None and isinstance(result.get("z"), (int, float)):
            result["z_numeric"] = float(result["z"])

        result["parsed_constraints"] = constraints_numeric
        result["parsed_constraints_exact"] = [
            {
                "coeffs": [str(v) for v in c["coeffs"]],
                "rhs": str(c["rhs"]),
                "sense": c["sense"],
            }
            for c in constraints_exact
        ]
        result["input_constraints_display"] = _build_input_display(constraints_exact, n)
        result["obj_expr_display"] = format_objective_expression(obj_coeffs_exact, n)
        result["obj_coeffs_numeric"] = obj_coeffs_numeric
        result["input_mode"] = input_mode
        result["method"] = method
        return jsonify(result)

    except KeyError as e:
        return jsonify({"error": f"Thiếu trường: {e}"}), 400
    except ConstraintTextParseError as e:
        return jsonify(
            {
                "error": e.message,
                "details": [
                    f"Dòng {e.line_number}",
                    f"Nội dung: {e.line_text or '(trống)'}",
                    f"Gợi ý: {e.hint}",
                ],
            }
        ), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export", methods=["POST"])
def export() -> Response:
    """Xuất giải trình ra file txt để tải về."""
    body: Dict[str, Any] = request.get_json(force=True) or {}
    result = body.get("result")
    inp = body.get("inp")
    method = body.get("method", "algebraic")

    if not result or not inp:
        return jsonify({"error": "Thiếu result hoặc inp trong body."}), 400

    if method == "geometric":
        content = xuat_file_hinh_hoc(result, inp)
        buf = io.BytesIO(content.encode("utf-8"))
        return Response(
            buf.getvalue(),
            mimetype="text/plain; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=giai_trinh_hinh_hoc.txt"},
        )

    lines: List[str] = []
    sep = "=" * 60
    thin = "-" * 60

    lines.append(sep)
    lines.append("  FOURIER-MOTZKIN LP SOLVER - GIẢI TRÌNH CHI TIẾT")
    lines.append(f"  Ngày giờ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    n = int(inp["n"])
    obj_type = inp["obj_type"]
    var_names = result.get("var_names", _var_names(n))
    obj_str = result.get("obj_expr_display") or format_objective_expression(
        [to_exact_number(v) for v in inp.get("obj_coeffs", [])], n
    )
    constraint_lines = result.get("input_constraints_display")
    if not constraint_lines:
        export_constraints = inp.get("constraints") or result.get("parsed_constraints") or []
        constraint_lines = [
            dinh_dang_rang_buoc(constraint_to_exact(c), var_names) for c in export_constraints
        ]

    lines.append("BÀI TOÁN ĐẦU VÀO")
    lines.append(thin)
    lines.append(f"  {obj_type.upper()} z = {obj_str}")
    lines.append("  Ràng buộc:")
    for line in constraint_lines:
        lines.append(f"    {line}")
    lines.append("")

    lines.append("CÁC BƯỚC FOURIER-MOTZKIN")
    lines.append(thin)
    for step in result.get("steps_constraints", []):
        lines.append(f"\n  Bước {step['buoc']}: {step['tieu_de']}")
        for cs in step["constraints"]:
            lines.append(f"    • {cs}")
        if step.get("bi_loai"):
            lines.append(f"  -> Đã loại: {', '.join(step['bi_loai'])}")
        if step.get("moi_tao"):
            lines.append(f"  -> Mới tạo: {', '.join(step['moi_tao'])}")
    lines.append("")

    lines.append("REASONING CHI TIẾT")
    lines.append(thin)
    reasoning = result.get("reasoning", {})
    for s in reasoning.get("steps", []):
        lines.append(f"\n  [{s['so_buoc']}] [{s['loai']}] {s['tieu_de']}")
        if s.get("noi_dung") and s["noi_dung"] != s["tieu_de"]:
            lines.append(f"      Nội dung: {s['noi_dung']}")
        if s.get("cong_thuc"):
            lines.append(f"      Công thức: {s['cong_thuc']}")
        if s.get("ket_qua"):
            lines.append(f"      Kết quả: {s['ket_qua']}")
    lines.append("")

    lines.append("KẾT QUẢ")
    lines.append(thin)
    if result.get("feasible"):
        lines.append("  Khả thi: CÓ")
        for var, val in result.get("solution", {}).items():
            lines.append(f"  {var} = {val}")
        lines.append(f"  z* = {result.get('z')}")
    else:
        lines.append("  Bài toán VÔ NGHIỆM")
    lines.append("")
    lines.append(sep)

    content = "\n".join(lines)
    buf = io.BytesIO(content.encode("utf-8"))
    return Response(
        buf.getvalue(),
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=giai_trinh.txt"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
