"""Bộ dữ liệu kiểm thử cho bộ giải Fourier-Motzkin v2.

Mỗi bài toán LP được mô tả bằng (tên, n, hệ số mục tiêu, kiểu, ràng buộc, kỳ
vọng). Vì LP có thể có nhiều nghiệm tối ưu, ta kiểm hai tiêu chí độc lập thay
vì so trùng từng x_i:

  1. trạng thái (feasible / infeasible / unbounded) đúng kỳ vọng;
  2. với bài khả thi: nghiệm thỏa MỌI ràng buộc gốc và z* khớp đáp án giải tay.
"""

from fractions import Fraction

# (tên, n, obj, sense, [(coeffs, sense, rhs), ...], expected)
CASES = [
    (
        "TC1 — 2 biến, max cơ bản",
        2, [3, 2], "max",
        [([1, 1], "<=", 4), ([2, 1], "<=", 6), ([1, 0], ">=", 0), ([0, 1], ">=", 0)],
        {"status": "feasible", "z": Fraction(10)},
    ),
    (
        "TC2 — 3 biến, nhiều nghiệm tối ưu",
        3, [2, 3, 1], "max",
        [([1, 1, 1], "<=", 6), ([0, 1, 1], "<=", 4), ([1, 1, 0], "<=", 4),
         ([1, 0, 0], ">=", 0), ([0, 1, 0], ">=", 0), ([0, 0, 1], ">=", 0)],
        {"status": "feasible", "z": Fraction(12)},
    ),
    (
        "TC3 — min, ràng buộc hỗn hợp ≤/≥",
        2, [-1, -2], "min",
        [([2, 1], "<=", 6), ([1, 1], ">=", 2), ([-1, 1], ">=", 3),
         ([1, 0], ">=", 0), ([0, 1], ">=", 0)],
        {"status": "feasible", "z": Fraction(-12)},
    ),
    (
        "TC4 — vô nghiệm (miền rỗng)",
        1, [1], "max",
        [([1], "<=", -1), ([1], ">=", 0)],
        {"status": "infeasible"},
    ),
    (
        "TC5 — hệ số phân số, exact",
        2, [1, 1], "max",
        [([Fraction(1, 3), Fraction(2, 3)], "<=", 1), ([1, 0], ">=", 0), ([0, 1], ">=", 0)],
        {"status": "feasible", "z": Fraction(3)},
    ),
    (
        "TC6 — ràng buộc đẳng thức (=)",
        2, [2, 1], "max",
        [([1, 1], "=", 4), ([1, 0], ">=", 0), ([0, 1], ">=", 0)],
        {"status": "feasible", "z": Fraction(8)},
    ),
    (
        "TC7 — một biến, min có chặn dưới",
        1, [1], "min",
        [([1], "<=", 5), ([1], ">=", 2)],
        {"status": "feasible", "z": Fraction(2)},
    ),
    (
        "TC8 — 4 biến, max trên simplex",
        4, [1, 2, 3, 4], "max",
        [([1, 1, 1, 1], "<=", 10),
         ([1, 0, 0, 0], ">=", 0), ([0, 1, 0, 0], ">=", 0),
         ([0, 0, 1, 0], ">=", 0), ([0, 0, 0, 1], ">=", 0)],
        {"status": "feasible", "z": Fraction(40)},
    ),
    (
        "TC9 — không bị chặn (unbounded)",
        2, [1, 1], "max",
        [([1, 0], ">=", 0), ([0, 1], ">=", 0)],
        {"status": "unbounded"},
    ),
    (
        "TC10 — min với hệ số phân số",
        2, [Fraction(1, 2), 1], "min",
        [([1, 1], ">=", 4), ([1, 0], "<=", 6), ([1, 0], ">=", 0), ([0, 1], ">=", 0)],
        {"status": "feasible", "z": Fraction(2)},
    ),
]

# Parser: (tên, n, text, [hệ số kỳ vọng dạng (coeffs, sense, rhs)])
TEXT_CASES = [
    (
        "TXT1 — lượng giác và căn",
        2,
        "sin(pi/6)x1 + cos(pi/3)x2 <= 1\nsqrt(2)x1 + x2 >= 1/2",
        # sqrt(2) KHÔNG hữu tỉ → dòng 2 sẽ gây lỗi parse có kiểm soát.
        "expect_error",
    ),
    (
        "TXT2 — phân số, hệ số liền biến",
        2,
        "2x1 + 1/3 x2 <= 5\n-x1 <= 0",
        [([Fraction(2), Fraction(1, 3)], "<=", Fraction(5)),
         ([Fraction(-1), Fraction(0)], "<=", Fraction(0))],
    ),
    (
        "TXT3 — chỉ hằng hữu tỉ",
        2,
        "sin(pi/6)x1 + cos(pi/3)x2 <= 1\nx1 + x2 = 3",
        [([Fraction(1, 2), Fraction(1, 2)], "<=", Fraction(1)),
         ([Fraction(1), Fraction(1)], "=", Fraction(3))],
    ),
]

# Phụ thuộc vòng lặp: (tên, a, c0, b, c1, L, U, kỳ vọng has_dependence)
# Kiểm thử dựa trên *nới lỏng thực*: hệ thực vô nghiệm ⇒ chắc chắn không phụ
# thuộc; hệ thực khả thi ⇒ kết luận "có thể có phụ thuộc" (an toàn, bảo thủ).
# Phủ đủ các trường hợp: tự phụ thuộc, phụ thuộc mang qua vòng (loop-carried),
# vượt biên, chạm đúng biên, hệ số 0 (chỉ số hằng), hệ số âm, và ca biên 1 lần
# lặp. Riêng DEP7 minh họa GIỚI HẠN của nới lỏng thực (nghiệm phân số).
DEPENDENCE_CASES = [
    # --- CÓ phụ thuộc ---
    ("DEP1 — tự phụ thuộc: ghi A[i], đọc A[i]", 1, 0, 1, 0, 0, 10, True),
    ("DEP2 — mang qua vòng: ghi A[i], đọc A[i-1]", 1, 0, 1, -1, 0, 10, True),
    ("DEP3 — ghi A[i], đọc A[i+1]", 1, 0, 1, 1, 0, 10, True),
    ("DEP4 — chạm đúng biên: ghi A[i], đọc A[i+10], [0,10]", 1, 0, 1, 10, 0, 10, True),
    ("DEP5 — hệ số khác nhau: ghi A[2i], đọc A[i]", 2, 0, 1, 0, 0, 10, True),
    ("DEP6 — hệ số âm: ghi A[i], đọc A[-i]", 1, 0, -1, 0, 0, 10, True),
    # --- KHÔNG phụ thuộc ---
    ("DEP7 — vượt biên: ghi A[i], đọc A[i+11], [0,10]", 1, 0, 1, 11, 0, 10, False),
    ("DEP8 — vượt xa biên: ghi A[i], đọc A[i+100]", 1, 0, 1, 100, 0, 10, False),
    ("DEP9 — chỉ số hằng khác nhau: A[5] vs A[7]", 0, 5, 0, 7, 0, 10, False),
    ("DEP10 — biên 1 lần lặp [0,0]: ghi A[i], đọc A[i-1]", 1, 0, 1, -1, 0, 0, False),
    # --- Ca minh họa GCD test bắt được ca chẵn/lẻ ---
    # A[2i] vs A[2i+1]: gcd(2,2)=2 không chia hết 1 → vô nghiệm số nguyên.
    # GCD test loại ngay, không cần FM. Đây là điểm MẠNH hơn nới lỏng thực.
    ("DEP11 — GCD test: A[2i] vs A[2i+1] (chẵn≠lẻ, gcd=2∤1)", 2, 0, 2, 1, 0, 10, False),
]
