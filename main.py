"""Flask web app cho Fourier-Motzkin LP Solver."""

import io
from datetime import datetime
from typing import Any, Dict

from flask import Flask, Response, jsonify, render_template, request

from solver.core import giai_bai_toan
from solver.geometric import giai_hinh_hoc, xuat_file_hinh_hoc
from solver.parser import ConstraintTextParseError, parse_constraints_text

app = Flask(__name__)


@app.route("/")
def index() -> str:
    """Trả về trang chủ."""
    return render_template("index.html")


@app.route("/api/solve", methods=["POST"])
def solve() -> Response:
    """Nhận bài toán LP và trả về kết quả giải.

    Body JSON:
        n (int): Số biến.
        constraints (list): Danh sách ràng buộc.
        obj_coeffs (list): Hệ số hàm mục tiêu.
        obj_type (str): "max" hoặc "min".

    Returns:
        JSON kết quả hoặc {"error": "..."}.
    """
    try:
        data: Dict[str, Any] = request.get_json(force=True)
        n: int = int(data["n"])
        if n < 1:
            return jsonify({"error": "Sá»‘ biáº¿n pháº£i â‰¥ 1"}), 400
        input_mode: str = data.get("input_mode", "form").lower()
        if input_mode == "text":
            constraints = parse_constraints_text(data.get("constraints_text", ""), n)
        else:
            constraints = data["constraints"]
        obj_coeffs = [float(v) for v in data["obj_coeffs"]]
        obj_type: str = data.get("obj_type", "max").lower()

        # Validate
        if n < 1:
            return jsonify({"error": "Số biến phải ≥ 1"}), 400
        if len(obj_coeffs) != n:
            return jsonify({"error": "Số hệ số hàm mục tiêu phải bằng số biến"}), 400
        for c in constraints:
            if len(c["coeffs"]) != n:
                return jsonify({"error": "Số hệ số ràng buộc phải bằng số biến"}), 400

        method: str = data.get("method", "algebraic").lower()

        if method == "geometric":
            result = giai_hinh_hoc(n, constraints, obj_coeffs, obj_type)
        else:
            method = "algebraic"
            result = giai_bai_toan(n, constraints, obj_coeffs, obj_type)

        result["parsed_constraints"] = constraints
        result["input_mode"] = input_mode
        result["method"] = method
        return jsonify(result)

    except KeyError as e:
        return jsonify({"error": f"Thiếu trường: {e}"}), 400
    except ConstraintTextParseError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export", methods=["POST"])
def export() -> Response:
    """Xuất giải trình ra file txt để tải về.

    Body JSON:
        result (dict): Kết quả từ /api/solve.
        inp (dict): Input gốc (n, constraints, obj_coeffs, obj_type).

    Returns:
        File text "giai_trinh.txt".
    """
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
            headers={"Content-Disposition": "attachment; filename=giai_trinh_hinh_hoc.txt"}
        )

    lines = []
    sep = "=" * 60
    thin = "-" * 60

    # Header
    lines.append(sep)
    lines.append("  FOURIER-MOTZKIN LP SOLVER — GIẢI TRÌNH CHI TIẾT")
    lines.append(f"  Ngày giờ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    # Thông tin bài toán
    n = inp["n"]
    obj_type = inp["obj_type"]
    obj_coeffs = inp["obj_coeffs"]
    export_constraints = inp.get("constraints") or result.get("parsed_constraints") or []
    var_names = result.get("var_names", [f"x{i+1}" for i in range(n)])

    obj_str = " + ".join(f"{obj_coeffs[i]}{var_names[i]}" for i in range(n))
    lines.append("BÀI TOÁN ĐẦU VÀO")
    lines.append(thin)
    lines.append(f"  {obj_type.upper()} z = {obj_str}")
    lines.append("  Ràng buộc:")
    for c in export_constraints:
        sense_map = {"<=": "≤", ">=": "≥", "=": "="}
        sign = sense_map.get(c["sense"], c["sense"])
        terms = []
        for i, coef in enumerate(c["coeffs"]):
            if coef == 0:
                continue
            coef_str = int(coef) if coef == int(coef) else coef
            if coef == 1:
                terms.append(var_names[i])
            elif coef == -1:
                terms.append(f"-{var_names[i]}")
            else:
                terms.append(f"{coef_str}{var_names[i]}")
        if not terms:
            lhs = "0"
        else:
            lhs = terms[0]
            for t in terms[1:]:
                lhs += f" - {t[1:]}" if t.startswith("-") else f" + {t}"
        rhs = c["rhs"]
        rhs_str = int(rhs) if rhs == int(rhs) else rhs
        lines.append(f"    {lhs} {sign} {rhs_str}")
    lines.append("")

    # Các bước Fourier-Motzkin
    lines.append("CÁC BƯỚC FOURIER-MOTZKIN")
    lines.append(thin)
    for step in result.get("steps_constraints", []):
        lines.append(f"\n  Bước {step['buoc']}: {step['tieu_de']}")
        for cs in step["constraints"]:
            lines.append(f"    • {cs}")
        if step.get("bi_loai"):
            lines.append(f"  → Đã loại: {', '.join(step['bi_loai'])}")
        if step.get("moi_tao"):
            lines.append(f"  → Mới tạo: {', '.join(step['moi_tao'])}")
    lines.append("")

    # Reasoning chi tiết
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

    # Kết quả cuối
    lines.append("KẾT QUẢ")
    lines.append(thin)
    if result.get("feasible"):
        lines.append(f"  Khả thi: CÓ")
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
        headers={"Content-Disposition": "attachment; filename=giai_trinh.txt"}
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
