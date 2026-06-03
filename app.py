"""Ứng dụng web Flask cho bộ giải Fourier-Motzkin v2.

Ba endpoint:
  * ``GET  /``            - trang giao diện chính.
  * ``POST /api/solve``   - giải LP (đại số hoặc hình học), trả JSON.
  * ``POST /api/depend``  - kiểm tra phụ thuộc vòng lặp (ứng dụng compiler).
  * ``POST /api/export``  - xuất file .txt giải trình.
"""

from __future__ import annotations

import io
from typing import Any, Dict

from flask import Flask, Response, jsonify, render_template, request

from fmlp.dependence import test_loop_dependence, test_loop_dependence_general
from fmlp.parse import ParseError
from fmlp.service import export_txt, solve_request
from fmlp.trace import Trace

app = Flask(__name__)


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/api/solve")
def api_solve() -> Response:
    try:
        payload: Dict[str, Any] = request.get_json(force=True) or {}
        result = solve_request(payload)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except ParseError as e:
        return jsonify({"error": e.message, "details": [f"Dòng {e.line}", e.text, e.hint]}), 400
    except KeyError as e:
        return jsonify({"error": f"Thiếu trường: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@app.post("/api/depend")
def api_depend() -> Response:
    """Kiểm tra phụ thuộc vòng lặp đơn (1 vòng, 1 chiều mảng)."""
    try:
        d = request.get_json(force=True) or {}
        res = test_loop_dependence(
            int(d["a"]), int(d["c0"]), int(d["b"]), int(d["c1"]),
            int(d["L"]), int(d["U"]),
        )
        return jsonify(res)
    except KeyError as e:
        return jsonify({"error": f"Thiếu trường: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/depend_general")
def api_depend_general() -> Response:
    """Kiểm tra phụ thuộc tổng quát: d vòng lặp lồng nhau, mảng m chiều."""
    try:
        body = request.get_json(force=True) or {}
        d = int(body["d"])
        loops = body["loops"]           # [{"L":..,"U":..}, ...]
        write_access = body["write"]    # [[a1,a2,...,ad, c], ...]  mỗi chiều mảng
        read_access  = body["read"]     # [[b1,b2,...,bd, c], ...]
        res = test_loop_dependence_general(d, loops, write_access, read_access)
        return jsonify(res)
    except KeyError as e:
        return jsonify({"error": f"Thiếu trường: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/export")
def api_export() -> Response:
    body = request.get_json(force=True) or {}
    result = body.get("result")
    if not result:
        return jsonify({"error": "Thiếu 'result' trong body."}), 400
    content = export_txt(result)
    buf = io.BytesIO(content.encode("utf-8"))
    return Response(
        buf.getvalue(),
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=giai_trinh_v2.txt"},
    )


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", "5001"))
    app.run(debug=True, port=port)
