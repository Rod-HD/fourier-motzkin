"""Mô hình dữ liệu quy hoạch tuyến tính ở dạng ma trận (phiên bản 2).

Cách biểu diễn ở đây theo đúng *dạng ma trận chuẩn tắc* dùng trong các giáo
trình quy hoạch tuyến tính: một hệ ràng buộc luôn được đưa về

    A x <= b,   A in Q^{m x n},  b in Q^m,

và hàm mục tiêu tuyến tính ``z = c^T x``. Mọi ràng buộc ``>=`` được nhân -1,
ràng buộc ``=`` tách thành cặp ``<=`` và ``>=`` (Bertsimas--Tsitsiklis,
Vanderbei).

Mỗi bất đẳng thức được lưu trong lớp :class:`Row` cùng một *chứng chỉ* - tổ
hợp tuyến tính không âm của các bất đẳng thức gốc đã sinh ra nó. Chứng chỉ này
chính là bội số Farkas, cho phép truy vết nguồn gốc một mâu thuẫn ``0 <= -k``
(Schrijver, Bổ đề Farkas).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

from .rational import Number, format_fraction, to_fraction


@dataclass
class Row:
    """Một bất đẳng thức ``coeffs · x <= rhs`` trên trường Q.

    Attributes:
        coeffs: vector hệ số ``[a_1, ..., a_d]`` (có thể gồm cả biến phụ z).
        rhs: vế phải ``b``.
        cert: chứng chỉ Farkas - ánh xạ {chỉ số ràng buộc gốc -> bội số >= 0}.
              Tổng có trọng số của các ràng buộc gốc theo ``cert`` tái tạo lại
              chính ``Row`` này, dùng để giải thích vì sao hệ vô nghiệm.
    """

    coeffs: List[Number]
    rhs: Number
    cert: Dict[int, Number] = field(default_factory=dict)

    def coeff(self, idx: int) -> Number:
        """Hệ số tại vị trí *idx* (0 nếu vượt quá chiều dài)."""
        return self.coeffs[idx] if 0 <= idx < len(self.coeffs) else Fraction(0)

    def width(self) -> int:
        return len(self.coeffs)

    def copy(self) -> "Row":
        return Row(list(self.coeffs), self.rhs, dict(self.cert))


def _normalize_cert(cert: Dict[int, Number]) -> Dict[int, Number]:
    """Bỏ các bội số bằng 0 để chứng chỉ gọn gàng."""
    return {k: v for k, v in cert.items() if v != 0}


def combine(low: Row, up: Row, k: int) -> Row:
    """Khử biến thứ *k* từ một chặn dưới và một chặn trên (bước Fourier-Motzkin).

    Cho:
        low:  a_low · x <= b_low  với hệ số ``a_low[k] < 0`` (suy ra chặn dưới),
        up:   a_up  · x <= b_up   với hệ số ``a_up[k]  > 0`` (suy ra chặn trên).

    Ta nhân *low* với ``alpha = a_up[k]`` và *up* với ``beta = -a_low[k]`` (cả
    hai đều dương) rồi cộng lại; hệ số của ``x_k`` triệt tiêu. Kết quả là một
    bất đẳng thức mới không còn ``x_k``. Chứng chỉ Farkas được cộng tương ứng
    theo cùng trọng số ``alpha``, ``beta`` (Schrijver, mục về Bổ đề Farkas).
    """
    alpha = up.coeff(k)        # > 0
    beta = -low.coeff(k)       # > 0
    width = max(low.width(), up.width())
    coeffs = [alpha * low.coeff(j) + beta * up.coeff(j) for j in range(width)]
    coeffs[k] = Fraction(0)    # ép đúng 0, tránh -0/sai số bề mặt
    rhs = alpha * low.rhs + beta * up.rhs

    cert: Dict[int, Number] = {}
    for idx, mult in low.cert.items():
        cert[idx] = cert.get(idx, Fraction(0)) + alpha * mult
    for idx, mult in up.cert.items():
        cert[idx] = cert.get(idx, Fraction(0)) + beta * mult
    return Row(coeffs, rhs, _normalize_cert(cert))


@dataclass
class LinearProgram:
    """Bài toán LP ở dạng ma trận, sẵn sàng cho phép khử.

    Attributes:
        n: số biến quyết định ``x_1..x_n`` (chưa kể biến phụ z).
        sense: ``"max"`` hoặc ``"min"``.
        obj: hệ số hàm mục tiêu ``c`` (độ dài n).
        rows: danh sách ràng buộc đã chuẩn hóa về ``<=`` (ma trận A | b).
    """

    n: int
    sense: str
    obj: List[Number]
    rows: List[Row]
    names: Optional[List[str]] = None

    @staticmethod
    def from_input(
        n: int,
        sense: str,
        obj: Sequence,
        constraints: Sequence[Dict],
        names: Optional[Sequence[str]] = None,
    ) -> "LinearProgram":
        """Dựng LP từ dữ liệu thô, chuẩn hóa mọi ràng buộc về dạng ``<=``.

        ``constraints`` là danh sách dict ``{"coeffs": [...], "sense": ..., "rhs": ...}``.
        Ràng buộc ``=`` được tách đôi; ``>=`` nhân -1. Mỗi ràng buộc gốc giữ
        nguyên một chỉ số chứng chỉ để truy vết Farkas.
        """
        sense = sense.lower().strip()
        if sense not in ("max", "min"):
            raise ValueError("sense phải là 'max' hoặc 'min'.")
        if n < 1:
            raise ValueError("Số biến phải >= 1.")

        obj_q = [to_fraction(v) for v in obj]
        if len(obj_q) != n:
            raise ValueError("Số hệ số hàm mục tiêu phải bằng số biến.")

        rows: List[Row] = []
        cert_id = 0
        for c in constraints:
            coeffs = [to_fraction(v) for v in c["coeffs"]]
            if len(coeffs) != n:
                raise ValueError("Số hệ số ràng buộc phải bằng số biến.")
            rhs = to_fraction(c["rhs"])
            s = c["sense"].strip()
            if s == "<=":
                rows.append(Row(coeffs, rhs, {cert_id: Fraction(1)}))
                cert_id += 1
            elif s == ">=":
                rows.append(Row([-a for a in coeffs], -rhs, {cert_id: Fraction(1)}))
                cert_id += 1
            elif s == "=":
                rows.append(Row(list(coeffs), rhs, {cert_id: Fraction(1)}))
                cert_id += 1
                rows.append(Row([-a for a in coeffs], -rhs, {cert_id: Fraction(1)}))
                cert_id += 1
            else:
                raise ValueError(f"Toán tử không hợp lệ: {s!r}")
        return LinearProgram(
            n=n, sense=sense, obj=obj_q, rows=rows,
            names=list(names) if names is not None else None,
        )


def var_name(i: int, n: int, names: Optional[Sequence[str]] = None) -> str:
    """Tên biến: ``x1..xn`` cho biến quyết định, ``z`` cho biến mục tiêu (idx n).

    Nếu *names* được cung cấp, dùng tên tùy biến cho các biến quyết định
    (vd ``iw₁, ir₁`` trong bài toán phụ thuộc vòng lặp); biến mục tiêu vẫn là ``z``.
    """
    if i == n:
        return "z"
    if names is not None and 0 <= i < len(names):
        return names[i]
    return f"x{i + 1}"


def format_row(row: Row, n: int, names: Optional[Sequence[str]] = None) -> str:
    """In một ràng buộc thành chuỗi ``a1x1 + a2x2 + ... ≤ b`` dễ đọc."""
    terms: List[str] = []
    for i, a in enumerate(row.coeffs):
        if a == 0:
            continue
        name = var_name(i, n, names)
        if a == 1:
            terms.append(f"+ {name}")
        elif a == -1:
            terms.append(f"- {name}")
        elif a > 0:
            terms.append(f"+ {format_fraction(a)}{name}")
        else:
            terms.append(f"- {format_fraction(-a)}{name}")
    if not terms:
        lhs = "0"
    else:
        lhs = " ".join(terms)
        if lhs.startswith("+ "):
            lhs = lhs[2:]
        elif lhs.startswith("- "):
            lhs = "-" + lhs[2:]
    return f"{lhs} ≤ {format_fraction(row.rhs)}"
