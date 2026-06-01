"""Smoke test nhanh cho tầng dịch vụ + xuất file (không khởi động web)."""

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmlp.service import export_txt, solve_request  # noqa: E402


def show(title, payload):
    print("=" * 60)
    print(title)
    print("-" * 60)
    res = solve_request(payload)
    print("status:", res.get("status"))
    if res.get("status") == "feasible":
        print("solution:", {k: v["fraction"] for k, v in res["solution"].items()})
        print("z*:", res["z"]["fraction"])
        if "chart" in res:
            print("vertices:", [(v["label"], v["x_frac"], v["y_frac"], v["z_frac"]) for v in res["chart"]["vertices"]])
    print("trace items:", res.get("trace", {}).get("count"))
    return res


r1 = show("Đại số 2 biến (max 3x1+2x2)", {
    "n": 2, "sense": "max", "obj": [3, 2], "method": "algebraic", "input_mode": "form",
    "constraints": [
        {"coeffs": [1, 1], "sense": "<=", "rhs": 4},
        {"coeffs": [2, 1], "sense": "<=", "rhs": 6},
        {"coeffs": [1, 0], "sense": ">=", "rhs": 0},
        {"coeffs": [0, 1], "sense": ">=", "rhs": 0},
    ],
})

show("Hình học 2 biến", {
    "n": 2, "sense": "max", "obj": [3, 2], "method": "geometric", "input_mode": "form",
    "constraints": [
        {"coeffs": [1, 1], "sense": "<=", "rhs": 4},
        {"coeffs": [2, 1], "sense": "<=", "rhs": 6},
        {"coeffs": [1, 0], "sense": ">=", "rhs": 0},
        {"coeffs": [0, 1], "sense": ">=", "rhs": 0},
    ],
})

show("Text mode (phân số)", {
    "n": 2, "sense": "max", "obj": [1, 1], "method": "algebraic", "input_mode": "text",
    "constraints_text": "1/3 x1 + 2/3 x2 <= 1\n-x1 <= 0\n-x2 <= 0",
})

print("\n" + "=" * 60)
print("Kiểm tra JSON serializable + xuất file txt")
print("-" * 60)
json.dumps(r1)  # ném lỗi nếu có giá trị không serialize được
print("JSON OK")
txt = export_txt(r1)
print("export_txt dài", len(txt), "ký tự — 3 dòng đầu:")
print("\n".join(txt.splitlines()[:3]))
print("\nTẤT CẢ SMOKE TEST CHẠY XONG.")
