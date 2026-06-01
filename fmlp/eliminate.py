"""Các phép khử Fourier-Motzkin trên hệ bất đẳng thức (phiên bản 2).

Module này chứa *hạt nhân* của thuật toán: tách ràng buộc theo dấu hệ số của
biến cần khử và ghép cặp để chiếu hệ xuống không gian ít hơn một chiều.

Trực giác hình học (Bertsimas--Tsitsiklis): khử một biến tương đương với phép
*chiếu* đa diện ``P = {x : Ax <= b}`` xuống siêu phẳng còn lại n-1 chiều. Một
điểm thuộc hình chiếu khi và chỉ khi tồn tại giá trị của biến bị khử nằm trong
mọi cận dưới và cận trên do các ràng buộc áp đặt — chính là điều kiện
"mọi cận dưới <= mọi cận trên".

Vì số ràng buộc có thể bùng nổ sau mỗi bước (điểm yếu cố hữu nêu trong
Vanderbei, Schrijver), ta thêm một bước *lọc dư thừa cú pháp* rẻ tiền để giữ
hệ gọn mà vẫn tương đương về tập nghiệm.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import List, Tuple

from .model import Row, combine


def split_by_sign(rows: List[Row], k: int) -> Tuple[List[Row], List[Row], List[Row]]:
    """Chia các ràng buộc thành ba nhóm theo dấu hệ số của biến *k*.

    Returns:
        (lower, upper, none) với:
          * ``upper`` — hệ số ``a_k > 0`` (ràng buộc cho *chặn trên* của x_k),
          * ``lower`` — hệ số ``a_k < 0`` (ràng buộc cho *chặn dưới* của x_k),
          * ``none``  — hệ số ``a_k = 0`` (không liên quan tới x_k).
    """
    lower: List[Row] = []
    upper: List[Row] = []
    none: List[Row] = []
    for r in rows:
        a = r.coeff(k)
        if a > 0:
            upper.append(r)
        elif a < 0:
            lower.append(r)
        else:
            none.append(r)
    return lower, upper, none


def eliminate_variable(rows: List[Row], k: int) -> List[Row]:
    """Khử biến thứ *k* khỏi hệ ``rows`` bằng phép chiếu Fourier-Motzkin.

    Hệ kết quả gồm: các ràng buộc không chứa x_k (nhóm none) cộng với một ràng
    buộc mới cho *mỗi cặp* (chặn dưới, chặn trên). Tập nghiệm chiếu xuống các
    biến còn lại được bảo toàn chính xác.
    """
    lower, upper, none = split_by_sign(rows, k)
    out: List[Row] = [r.copy() for r in none]
    for lo in lower:
        for up in upper:
            out.append(combine(lo, up, k))
    return out


def _lhs_key_and_scale(row: Row) -> Tuple[Tuple, Fraction]:
    """Khóa hướng của vế trái + hệ số tỉ lệ dương để quy chuẩn vế phải.

    Quy vector hệ số ``a`` về số nguyên tối giản *giữ nguyên dấu* (chia cho ước
    chung lớn nhất). Hai ràng buộc có cùng khóa khi và chỉ khi vế trái của
    chúng là *bội dương* của nhau — lúc đó chúng cùng một "hướng" nửa không
    gian và có thể so độ chặt qua vế phải đã quy chuẩn.

    Trả về (khóa_hướng, scale) với ``scale`` là số dương sao cho
    ``a / scale`` chính là vector nguyên tối giản; nhờ đó ``rhs / scale`` là vế
    phải đã quy về cùng tỉ lệ để so sánh.
    """
    coeffs = list(row.coeffs)
    dens = [c.denominator for c in coeffs]
    lcm = 1
    for d in dens:
        lcm = lcm * d // gcd(lcm, d)
    ints = [int(c * lcm) for c in coeffs]
    g = 0
    for v in ints:
        g = gcd(g, abs(v))
    if g == 0:
        return tuple(ints), Fraction(1)
    norm = tuple(v // g for v in ints)
    # a / scale = norm  =>  scale = a / norm = lcm-bù... cụ thể scale = g / lcm.
    scale = Fraction(g, lcm)
    return norm, scale


def prune_redundant(rows: List[Row]) -> List[Row]:
    """Lọc dư thừa cú pháp: bỏ ràng buộc luôn đúng và ràng buộc bị trội.

    Đây là phép lọc *an toàn* (không đổi tập nghiệm) và *rẻ* (gần tuyến tính
    theo số ràng buộc), nhằm hãm bớt đà bùng nổ tổ hợp đặc trưng của
    Fourier-Motzkin (Vanderbei, Schrijver):

      * Ràng buộc tầm thường ``0 <= b`` với ``b >= 0`` được bỏ (luôn đúng).
      * Với nhóm ràng buộc có vế trái là *bội dương* của nhau (cùng một hướng
        nửa không gian), chỉ giữ ràng buộc *chặt nhất* — vế phải nhỏ nhất sau
        khi quy về cùng tỉ lệ; các ràng buộc còn lại bị nó trội và là dư thừa.

    Phép lọc này không loại được dư thừa "ẩn" (chỉ phát hiện được bằng cách
    giải một LP phụ), nhưng xử lý gọn các dư thừa hiển nhiên về cú pháp. Ràng
    buộc mâu thuẫn ``0 <= b`` với ``b < 0`` luôn được giữ làm bằng chứng vô
    nghiệm.
    """
    best: dict = {}
    order: List[Tuple] = []
    passthrough: List[Row] = []

    for r in rows:
        if all(a == 0 for a in r.coeffs):
            if r.rhs >= 0:
                continue  # 0 <= b>=0: luôn đúng, bỏ
            passthrough.append(r)  # 0 <= b<0: bằng chứng mâu thuẫn, giữ
            continue

        key, scale = _lhs_key_and_scale(r)
        norm_rhs = r.rhs / scale
        if key not in best:
            best[key] = (norm_rhs, r)
            order.append(key)
        elif norm_rhs < best[key][0]:
            best[key] = (norm_rhs, r)

    return [best[k][1] for k in order] + passthrough
