"""Demo trực tiếp: gọi server v2 đang chạy và in kết quả các tính năng.

Dùng:
    python -m tests.demo_live            # mặc định cổng 5055
    python -m tests.demo_live 5001       # chỉ định cổng
"""

import json
import sys
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

PORT = sys.argv[1] if len(sys.argv) > 1 else "5055"
BASE = f"http://127.0.0.1:{PORT}"


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def line(c="-", n=68):
    print(c * n)


def show_solve(title, payload):
    line("=")
    print(" " + title)
    line("=")
    r = post("/api/solve", payload)
    inp = r.get("input", {})
    print(f"Bài toán: {inp.get('sense','').upper()} z = {inp.get('objective','')}")
    print("Ràng buộc:")
    for c in inp.get("constraints", []):
        print("   ", c)
    print()
    if r["status"] == "feasible":
        sol = "  ".join(f"{k} = {v['fraction']}" for k, v in r["solution"].items())
        print(f"KẾT QUẢ: {sol}   |   z* = {r['z']['fraction']} (≈ {r['z']['decimal']})")
    elif r["status"] == "unbounded":
        print("KẾT QUẢ: KHÔNG BỊ CHẶN")
    else:
        print("KẾT QUẢ: VÔ NGHIỆM")
    if r.get("steps"):
        print("\nCác bước khử Fourier-Motzkin:")
        for s in r["steps"]:
            print(f"  [{s['step']}] {s['title']}")
            for row in s["rows"]:
                print(f"       {row}")
    if r.get("chart") and r["chart"].get("vertices"):
        print("\nĐỉnh miền khả thi (hình học):")
        for v in r["chart"]["vertices"]:
            star = "  <= TỐI ƯU" if v["is_optimal"] else ""
            print(f"   {v['label']}({v['x_frac']}, {v['y_frac']})  z = {v['z_frac']}{star}")
    # In chứng chỉ Farkas nếu có
    for it in r.get("trace", {}).get("items", []):
        if it["kind"] == "warn" and "Farkas" in (it.get("body") or ""):
            print("\n" + it["title"])
            print("   " + it["body"].replace("\n", "\n   "))
    print()


def show_depend(title, payload):
    line("=")
    print(" " + title)
    line("=")
    r = post("/api/depend", payload)
    verdict = "CÓ THỂ CÓ phụ thuộc" if r["has_dependence"] else "KHÔNG có phụ thuộc"
    print(f"a={payload['a']}, c0={payload['c0']}, b={payload['b']}, c1={payload['c1']}, "
          f"L={payload['L']}, U={payload['U']}")
    print(f"=> {verdict}  (status={r['status']})")
    if r.get("witness"):
        print(f"   nhân chứng: iw={r['witness'][0]}, ir={r['witness'][1]}")
    print()


print(f"\n# DEMO FOURIER-MOTZKIN LP SOLVER v2  @ {BASE}\n")

show_solve("TC1 — Đại số: max 3x1 + 2x2", {
    "n": 2, "sense": "max", "obj": [3, 2], "method": "algebraic", "input_mode": "form",
    "constraints": [
        {"coeffs": [1, 1], "sense": "<=", "rhs": 4},
        {"coeffs": [2, 1], "sense": "<=", "rhs": 6},
        {"coeffs": [1, 0], "sense": ">=", "rhs": 0},
        {"coeffs": [0, 1], "sense": ">=", "rhs": 0},
    ],
})

show_solve("TC1 — Hình học: cùng bài toán", {
    "n": 2, "sense": "max", "obj": [3, 2], "method": "geometric", "input_mode": "form",
    "constraints": [
        {"coeffs": [1, 1], "sense": "<=", "rhs": 4},
        {"coeffs": [2, 1], "sense": "<=", "rhs": 6},
        {"coeffs": [1, 0], "sense": ">=", "rhs": 0},
        {"coeffs": [0, 1], "sense": ">=", "rhs": 0},
    ],
})

show_solve("TC5 — Số học hữu tỉ (TEXT): max x1 + x2", {
    "n": 2, "sense": "max", "obj": [1, 1], "method": "algebraic", "input_mode": "text",
    "constraints_text": "1/3 x1 + 2/3 x2 <= 1\n-x1 <= 0\n-x2 <= 0",
})

show_solve("TC4 — Vô nghiệm + chứng chỉ Farkas: max x1", {
    "n": 1, "sense": "max", "obj": [1], "method": "algebraic", "input_mode": "form",
    "constraints": [
        {"coeffs": [1], "sense": "<=", "rhs": -1},
        {"coeffs": [1], "sense": ">=", "rhs": 0},
    ],
})

show_solve("TC9 — Không bị chặn: max x1 + x2", {
    "n": 2, "sense": "max", "obj": [1, 1], "method": "algebraic", "input_mode": "form",
    "constraints": [
        {"coeffs": [1, 0], "sense": ">=", "rhs": 0},
        {"coeffs": [0, 1], "sense": ">=", "rhs": 0},
    ],
})

show_depend("DEP1 — Ứng dụng compiler: ghi A[i], đọc A[i-1]",
            {"a": 1, "c0": 0, "b": 1, "c1": -1, "L": 0, "U": 10})
show_depend("DEP2 — ghi A[i], đọc A[i+100]",
            {"a": 1, "c0": 0, "b": 1, "c1": 100, "L": 0, "U": 10})

print("DEMO HOÀN TẤT.")
