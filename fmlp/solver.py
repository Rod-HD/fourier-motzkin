"""Bộ giải LP bằng khử Fourier-Motzkin (phiên bản 2).

Chiến lược (Bertsimas--Tsitsiklis, Vanderbei):

1.  Đưa hệ về dạng ma trận ``Ax <= b`` (đã làm trong :mod:`fmlp.model`).
2.  Thêm biến mục tiêu ``z``:
        * ``max c^T x``  ->  thêm  ``z <= c^T x``   (tức ``z - c^T x <= 0``);
        * ``min c^T x``  ->  thêm  ``z <= -c^T x``  rồi lấy ``z* = -z_max``
          (dùng đẳng thức ``min f = -max(-f)``).
3.  Khử lần lượt ``x_1, ..., x_n`` (mỗi lần là một phép chiếu đa diện), giữ
    lại biến ``z``. Sau mỗi bước lọc bớt ràng buộc dư thừa hiển nhiên.
4.  Hệ còn lại chỉ chứa ``z``. Cận trên nhỏ nhất cho ``z_max``; nếu có dòng
    mâu thuẫn ``0 <= b<0`` thì vô nghiệm (kèm chứng chỉ Farkas), nếu không có
    cận trên thì không bị chặn.
5.  Thế ngược (back-substitution) để khôi phục một nghiệm tối ưu ``x``.

Toàn bộ phép tính chạy trên ``Fraction`` nên chính xác tuyệt đối.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List, Optional

from .eliminate import eliminate_variable, prune_redundant, split_by_sign
from .model import LinearProgram, Row, format_row, var_name
from .rational import Number, format_decimal, format_fraction
from .trace import Trace

POS_INF = None  # quy ước: None nghĩa là không có cận


class Solution:
    """Kết quả giải LP."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"

    def __init__(self, status: str, x: Optional[List[Number]] = None,
                 z: Optional[Number] = None):
        self.status = status
        self.x = x or []
        self.z = z


def _build_objective_row(lp: LinearProgram) -> Row:
    """Tạo ràng buộc gắn biến z vào hệ, theo kiểu max/min."""
    n = lp.n
    if lp.sense == "max":
        # z <= c^T x  <=>  z - c^T x <= 0
        coeffs = [-c for c in lp.obj] + [Fraction(1)]
    else:
        # z <= -c^T x <=>  z + c^T x <= 0
        coeffs = [c for c in lp.obj] + [Fraction(1)]
    return Row(coeffs, Fraction(0), cert={})


def _extend(rows: List[Row], width: int) -> None:
    """Bổ sung cột 0 để mọi Row có cùng số chiều (kể cả cột z)."""
    for r in rows:
        while len(r.coeffs) < width:
            r.coeffs.append(Fraction(0))


def _pick_value(lo: Optional[Number], hi: Optional[Number]) -> Number:
    """Chọn một giá trị hợp lệ trong khoảng [lo, hi].

    Ưu tiên cận dưới hữu hạn (cho nghiệm tại đỉnh), rồi đến cận trên, cuối cùng
    là 0 khi cả hai vô cùng. Tại nghiệm tối ưu mọi giá trị trong khoảng đều cho
    cùng ``z*`` nên lựa chọn này luôn hợp lệ.
    """
    if lo is not None:
        return lo
    if hi is not None:
        return hi
    return Fraction(0)


def _expr_without(row: Row, k: int, n: int, denom: Number,
                  names: Optional[List[str]] = None) -> str:
    """Diễn đạt vế phải khi cô lập ``x_k`` từ một ràng buộc.

    Dùng để in dạng ``x_k <= <biểu thức>`` (khi ``a_k > 0``) hoặc
    ``x_k >= <biểu thức>`` (khi ``a_k < 0``). ``denom`` là hệ số ``a_k`` CÓ DẤU:
    cô lập ``x_k`` nghĩa là chia hai vế cho ``a_k``, nên phải dùng dấu thật của
    ``a_k`` thì biểu thức mới đúng (chia cho ``|a_k|`` sẽ sai dấu khi ``a_k < 0``,
    ví dụ ``z - x1 <= 0`` cho ``x1 >= z`` chứ không phải ``x1 >= -z``).
    """
    parts: List[str] = []
    const = row.rhs / denom
    if const != 0 or all(row.coeff(j) == 0 for j in range(row.width()) if j != k):
        parts.append(format_fraction(const))
    for j in range(row.width()):
        if j == k or row.coeff(j) == 0:
            continue
        coef = -row.coeff(j) / denom  # chuyển vế (đổi dấu) rồi chia cho a_k
        name = var_name(j, n, names)
        if coef == 1:
            term = f"+ {name}"
        elif coef == -1:
            term = f"- {name}"
        elif coef > 0:
            term = f"+ {format_fraction(coef)}{name}"
        else:
            term = f"- {format_fraction(-coef)}{name}"
        parts.append(term)
    if not parts:
        return "0"
    text = " ".join(parts)
    # gọn dấu đầu
    if text.startswith("+ "):
        text = text[2:]
    elif text.startswith("- "):
        text = "-" + text[2:]
    return text


def _as_bound(row: Row, k: int, n: int, names: Optional[List[str]] = None):
    """Trả về (kiểu, chuỗi) mô tả ràng buộc khi cô lập ``x_k``.

    kiểu ∈ {"upper", "lower", "none"}:
      * "upper": ``x_k <= <expr>``  (hệ số a_k > 0),
      * "lower": ``x_k >= <expr>``  (hệ số a_k < 0),
      * "none":  ràng buộc không chứa x_k.
    Chuỗi trả về là vế phải ``<expr>`` đã chia cho |a_k|.
    """
    a = row.coeff(k)
    if a == 0:
        return "none", format_row(row, n, names)
    expr = _expr_without(row, k, n, a, names)
    return ("upper" if a > 0 else "lower"), expr


def solve(lp: LinearProgram, trace: Optional[Trace] = None) -> Solution:
    """Giải bài toán LP, ghi vết các bước nếu *trace* được cung cấp."""
    tr = trace if trace is not None else Trace()
    n = lp.n
    z_idx = n
    width = n + 1
    names = lp.names  # tên biến tùy biến (vd iw₁, ir₁); None ⟹ mặc định x1..xn

    obj_str = _format_linear(lp.obj, n, names)
    tr.add(
        "Khởi tạo bài toán",
        f"Cần {lp.sense} z = {obj_str}, với {len(lp.rows)} ràng buộc đã đưa về dạng ≤.\n"
        "Ý tưởng: thêm biến z gắn với hàm mục tiêu, rồi khử lần lượt "
        f"{', '.join(var_name(i, n, names) for i in range(n))} để hệ chỉ còn z, từ đó đọc ra z tối ưu.",
    )

    # Bước thêm biến z.
    rows: List[Row] = [r.copy() for r in lp.rows]
    _extend(rows, width)
    obj_row = _build_objective_row(lp)
    rows.append(obj_row)
    if lp.sense == "max":
        why_z = f"Với bài toán max, ta muốn z ≤ {obj_str}, viết lại thành {format_row(obj_row, n, names)}."
    else:
        why_z = (f"Với bài toán min, dùng đẳng thức min = -max(-mục tiêu): đặt z ≤ -({obj_str}), "
                 f"viết lại thành {format_row(obj_row, n, names)}. Tìm z_max rồi lấy z* = -z_max.")
    tr.add("Gắn biến mục tiêu z vào hệ", why_z)

    snapshots: List[List[Row]] = []  # hệ ngay trước mỗi lần khử biến x_k
    steps: List[Dict[str, Any]] = [{
        "step": 0,
        "title": "Hệ ban đầu (sau khi gắn biến z)",
        "rows": [format_row(r, n, names) for r in rows],
    }]

    current = prune_redundant(rows)

    # Khử x_1 .. x_n.
    for k in range(n):
        snapshots.append([r.copy() for r in current])
        vk = var_name(k, n, names)
        lower, upper, none = split_by_sign(current, k)

        # (1) Cô lập x_k trong từng ràng buộc có chứa nó.
        upper_exprs = [_expr_without(r, k, n, r.coeff(k), names) for r in upper]
        lower_exprs = [_expr_without(r, k, n, r.coeff(k), names) for r in lower]
        lines = [f"Cô lập {vk} trong mỗi ràng buộc có chứa nó:"]
        for r, e in zip(upper, upper_exprs):
            lines.append(f"   từ  {format_row(r, n, names)}   ⟹   {vk} ≤ {e}")
        for r, e in zip(lower, lower_exprs):
            lines.append(f"   từ  {format_row(r, n, names)}   ⟹   {vk} ≥ {e}")
        if none:
            lines.append(f"   {len(none)} ràng buộc KHÔNG chứa {vk} (sẽ giữ nguyên): "
                         + "; ".join(format_row(r, n, names) for r in none))
        tr.add(f"Khử {vk} - bước 1: tách thành chặn trên / chặn dưới", "\n".join(lines))

        # (2) Nguyên lý ghép cặp.
        tr.add(
            f"Khử {vk} - bước 2: điều kiện tồn tại {vk}",
            f"{vk} tồn tại khi và chỉ khi MỌI chặn dưới ≤ MỌI chặn trên. "
            f"Vậy ghép từng cặp (chặn dưới, chặn trên): {len(lower)} × {len(upper)} "
            f"= {len(lower) * len(upper)} bất phương trình mới (đã loại {vk}).",
        )

        # (3) Liệt kê từng cặp ghép cụ thể.
        eliminated = eliminate_variable(current, k)
        pair_lines: List[str] = []
        for le in lower_exprs:
            for ue in upper_exprs:
                pair_lines.append(f"   {le} ≤ {vk} ≤ {ue}   ⟹   {le} ≤ {ue}")
        if pair_lines:
            tr.add(f"Khử {vk} - bước 3: ghép cặp và rút gọn", "\n".join(pair_lines))

        # (4) Lọc dư thừa, nêu rõ giữ gì / bỏ gì.
        before = len(eliminated)
        kept = prune_redundant(eliminated)
        removed = before - len(kept)
        keep_lines = [f"Tổng cộng {before} ràng buộc sau khi ghép "
                      f"({len(lower) * len(upper)} mới + {len(none)} không chứa {vk})."]
        if removed > 0:
            keep_lines.append(
                f"Lọc bỏ {removed} ràng buộc dư thừa (luôn đúng dạng 0 ≤ số dương, hoặc bị một "
                f"ràng buộc chặt hơn cùng hướng làm trội). Giữ lại {len(kept)} ràng buộc:")
        else:
            keep_lines.append(f"Không có ràng buộc dư thừa. Giữ lại {len(kept)} ràng buộc:")
        for r in kept:
            keep_lines.append(f"   {format_row(r, n, names)}")
        tr.add(f"Khử {vk} - bước 4: giữ lại hệ rút gọn", "\n".join(keep_lines))

        # (5) Chú thích sư phạm: nếu MỌI dòng còn chứa z đều bị loại ở bước này
        # thì z mất liên kết với hệ - đó chính là dấu hiệu bài toán không bị chặn.
        had_z = any(r.coeff(z_idx) != 0 for r in current)
        still_z = any(r.coeff(z_idx) != 0 for r in kept)
        if had_z and not still_z:
            z_row = next(r for r in current if r.coeff(z_idx) != 0)
            bound_lines: List[str] = []
            if z_row.coeff(k) < 0:
                # x_k chỉ có chặn DƯỚI (kể cả từ dòng z); thiếu chặn TRÊN để ghép cặp.
                kind, missing, grow = "chặn dưới", "chặn trên", "+∞"
                for r, e in zip(lower, lower_exprs):
                    tag = " ← dòng mục tiêu (chứa z)" if r.coeff(z_idx) != 0 else ""
                    bound_lines.append(f"   {vk} ≥ {e}   (từ {format_row(r, n, names)}){tag}")
            else:
                # x_k chỉ có chặn TRÊN; thiếu chặn DƯỚI để ghép cặp.
                kind, missing, grow = "chặn trên", "chặn dưới", "-∞"
                for r, e in zip(upper, upper_exprs):
                    tag = " ← dòng mục tiêu (chứa z)" if r.coeff(z_idx) != 0 else ""
                    bound_lines.append(f"   {vk} ≤ {e}   (từ {format_row(r, n, names)}){tag}")
            body = [f"Xét {vk} trong hệ hiện tại: {vk} CHỈ có {kind}, KHÔNG có {missing}:"]
            body.extend(bound_lines)
            body.append(
                f"Phép khử Fourier-Motzkin chỉ sinh ràng buộc mới bằng cách ghép MỖI chặn dưới "
                f"với MỖI chặn trên. Vì {vk} thiếu hẳn {missing} nên không ghép được cặp nào → "
                f"toàn bộ dòng chứa {vk} bị loại, trong đó có dòng mục tiêu gắn z.")
            body.append(
                f"Hệ quả: {vk} có thể tiến tới {grow} mà không vi phạm ràng buộc nào. Dòng mục "
                f"tiêu ràng buộc z theo {vk} (z ≤ hàm mục tiêu); khi {vk} không bị chặn thì z "
                f"cũng không còn bị chặn. Đây chính là dấu hiệu bài toán KHÔNG BỊ CHẶN "
                f"(sẽ được xác nhận ở bước phân tích hệ chỉ còn z).")
            tr.warn(f"Vì sao khử {vk} làm mất luôn z?", "\n".join(body))

        current = kept
        steps.append({
            "step": k + 1,
            "title": f"Sau khi khử {vk}",
            "rows": [format_row(r, n, names) for r in current],
        })

    # Phân tích hệ chỉ còn biến z.
    result = _analyze_z_system(current, z_idx, lp, tr)
    if result.status != Solution.FEASIBLE:
        return _finish(result, lp, steps, tr, snapshots, None)

    z_max = result.z
    z_star = z_max if lp.sense == "max" else -z_max
    if lp.sense == "max":
        tr.add("Giá trị tối ưu",
               f"z_max = {format_fraction(z_max)} (≈ {format_decimal(z_star)}). "
               f"Vì bài toán max nên z* = z_max = {format_fraction(z_star)}.")
    else:
        tr.add("Giá trị tối ưu",
               f"z_max = {format_fraction(z_max)}. Bài toán min nên z* = -z_max = "
               f"{format_fraction(z_star)} (≈ {format_decimal(z_star)}).")

    # Back-substitution: khôi phục x từ x_n về x_1.
    x = _back_substitute(snapshots, z_max, n, tr, names)
    sol = Solution(Solution.FEASIBLE, x=x, z=z_star)
    return _finish(sol, lp, steps, tr, snapshots, x)


def _analyze_z_system(rows: List[Row], z_idx: int, lp: LinearProgram,
                      trace: Trace) -> Solution:
    """Đọc nghiệm từ hệ chỉ còn biến z: vô nghiệm / không bị chặn / z_max."""
    uppers: List[Number] = []
    upper_strs: List[str] = []
    for r in rows:
        # Bỏ qua ràng buộc còn sót hệ số x_j (không phải cận thực của z).
        if any(r.coeff(j) != 0 for j in range(z_idx)):
            continue
        a = r.coeff(z_idx)
        if a == 0:
            if r.rhs < 0:
                cert_str = _format_cert(r.cert)
                trace.warn(
                    "Phát hiện mâu thuẫn → bài toán VÔ NGHIỆM",
                    f"Sau khi khử hết các biến, còn lại ràng buộc 0 ≤ {format_fraction(r.rhs)} "
                    "- điều không thể xảy ra. Vậy miền khả thi rỗng.\n"
                    f"Chứng chỉ Farkas (cộng các ràng buộc gốc với hệ số không âm sẽ ra mâu "
                    f"thuẫn này): {cert_str}.",
                )
                return Solution(Solution.INFEASIBLE)
            continue
        if a > 0:
            uppers.append(r.rhs / a)
            upper_strs.append(f"z ≤ {format_fraction(r.rhs / a)}")

    if not uppers:
        trace.warn(
            "Không có cận trên cho z → bài toán KHÔNG BỊ CHẶN",
            "Sau khi khử hết các biến, không còn ràng buộc nào chặn trên z. "
            "Hàm mục tiêu tăng vô hạn theo hướng tối ưu.",
        )
        return Solution(Solution.UNBOUNDED)

    z_max = min(uppers)
    trace.add(
        "Đọc z tối ưu từ hệ chỉ còn biến z",
        f"Các ràng buộc chỉ còn z cho: {'; '.join(upper_strs)}.\n"
        f"z lớn nhất thỏa tất cả là z_max = min({', '.join(format_fraction(u) for u in uppers)}) "
        f"= {format_fraction(z_max)}.",
    )
    return Solution(Solution.FEASIBLE, z=z_max)


_SUB_DIGITS = "₀₁₂₃₄₅₆₇₈₉"


def _name_prefix(label: str) -> str:
    """Phần chữ của tên biến, bỏ chỉ số đuôi (cả số thường lẫn chỉ số dưới chân).

    Ví dụ: ``x12`` → ``x``; ``iw₃`` → ``iw``. Dùng để gom biến cùng nhóm.
    """
    i = len(label)
    while i > 0 and (label[i - 1].isdigit() or label[i - 1] in _SUB_DIGITS):
        i -= 1
    return label[:i]


def _var_list_label(n: int, names: Optional[List[str]] = None) -> str:
    """Liệt kê n biến gọn gàng, gom theo nhóm tiền tố, không dùng '…' khi nhóm ≤ 2.

    LP thường: ``x1`` → ``x1, x2`` → ``x1, …, x4``.
    Phụ thuộc:  ``iw₁, ir₁`` (d=1); ``iw₁, iw₂, ir₁, ir₂`` (d=2);
                ``iw₁, …, iw₃, ir₁, …, ir₃`` (d=3).
    """
    labels = [var_name(i, n, names) for i in range(n)]
    if not labels:
        return ""
    groups: List[List[str]] = [[labels[0]]]
    for lab in labels[1:]:
        if _name_prefix(lab) == _name_prefix(groups[-1][0]):
            groups[-1].append(lab)
        else:
            groups.append([lab])
    parts: List[str] = []
    for g in groups:
        if len(g) <= 2:
            parts.extend(g)               # nhóm 1–2 biến: liệt kê hết
        else:
            parts.append(f"{g[0]}, …, {g[-1]}")
    return ", ".join(parts)


def _back_substitute(snapshots: List[List[Row]], z_max: Number, n: int,
                     trace: Trace, names: Optional[List[str]] = None) -> List[Number]:
    """Thế ngược tìm x_n..x_1 từ các ảnh chụp hệ trước mỗi bước khử."""
    trace.add(
        f"Thế ngược để tìm nghiệm ({_var_list_label(n, names)})",
        f"Đã biết z = {format_fraction(z_max)}. Đi ngược từ "
        f"{var_name(n - 1, n, names)} về {var_name(0, n, names)}: tại mỗi biến, thế các giá trị đã biết vào "
        "hệ ngay trước khi khử biến đó để được khoảng giá trị, rồi chọn một giá trị trong khoảng.",
    )
    known: Dict[int, Number] = {n: z_max}
    x: List[Number] = [Fraction(0)] * n
    for k in range(n - 1, -1, -1):
        vk = var_name(k, n, names)
        lo, hi, detail = _isolate_with_steps(snapshots[k], k, known, n, names)
        val = _pick_value(lo, hi)
        known[k] = val
        x[k] = val
        lo_s = format_fraction(lo) if lo is not None else "-∞"
        hi_s = format_fraction(hi) if hi is not None else "+∞"

        if lo is not None and hi is not None and lo == hi:
            why = f"cận dưới và cận trên trùng nhau nên {vk} bị ép đúng bằng {lo_s}"
        elif lo is not None:
            why = f"lấy cận dưới {lo_s} (chọn một nghiệm nằm tại đỉnh của miền)"
        elif hi is not None:
            why = f"lấy cận trên {hi_s}"
        else:
            why = "cả hai cận đều vô hạn nên chọn 0"

        body = [f"Đã biết: {_format_known(known, k, n, names)}.",
                f"Lấy hệ ngay trước khi khử {vk}, xét từng ràng buộc có chứa {vk}:"]
        if detail:
            body.extend(detail)
        else:
            body.append(f"   (không ràng buộc nào chặn {vk})")
        body.append(f"Gộp các cận: {lo_s} ≤ {vk} ≤ {hi_s}.")
        body.append(f"⟹ Chọn {vk} = {format_fraction(val)} ({why}).")
        trace.add(f"Thế ngược tìm {vk}", "\n".join(body))
    return x


def _format_known(known: Dict[int, Number], k: int, n: int,
                  names: Optional[List[str]] = None) -> str:
    """Liệt kê các giá trị đã biết (z và các biến tìm sau k)."""
    parts: List[str] = []
    for j in sorted(known):
        if j == k:
            continue
        name = "z" if j == n else var_name(j, n, names)
        parts.append(f"{name} = {format_fraction(known[j])}")
    return ", ".join(parts) if parts else "(chưa có biến nào)"


def _isolate_with_steps(rows: List[Row], k: int, known: Dict[int, Number], n: int,
                        names: Optional[List[str]] = None):
    """Cô lập x_k trong từng ràng buộc, ghi lại phép thế và cận thu được.

    Trả về (lo, hi, lines): mỗi ràng buộc có chứa x_k cho một cận, lấy cận trên
    nhỏ nhất và cận dưới lớn nhất. Đồng thời ghi lại diễn giải từng bước thế.
    """
    vk = var_name(k, n, names)
    lo: Optional[Number] = None
    hi: Optional[Number] = None
    lines: List[str] = []
    for r in rows:
        a = r.coeff(k)
        if a == 0:
            continue
        # Thế các biến đã biết, chuyển sang vế phải.
        residual = r.rhs
        subs: List[str] = []
        for j, aj in enumerate(r.coeffs):
            if j == k or aj == 0:
                continue
            if j in known:
                residual -= aj * known[j]
                name = "z" if j == n else var_name(j, n, names)
                subs.append(f"{name} = {format_fraction(known[j])}")
        bound = residual / a
        # Vế trái: a·x_k (rút gọn khi |a| = 1).
        if a == 1:
            lhs = vk
        elif a == -1:
            lhs = f"-{vk}"
        else:
            lhs = f"{format_fraction(a)}·{vk}"
        sub_txt = ("thế " + ", ".join(subs)) if subs else "không còn biến nào khác"
        rel, sym = ("≤", "≤") if a > 0 else ("≥", "≤")
        # a > 0: a·x_k ≤ residual ⟹ x_k ≤ bound; a < 0: ⟹ x_k ≥ bound (đổi chiều khi chia số âm).
        lines.append(
            f"   • {format_row(r, n, names)}\n"
            f"       {sub_txt} ⟹ {lhs} {sym} {format_fraction(residual)} "
            f"⟹ {vk} {rel} {format_fraction(bound)}"
        )
        if a > 0:
            hi = bound if hi is None or bound < hi else hi
        else:
            lo = bound if lo is None or bound > lo else lo
    return lo, hi, lines


def _finish(sol: Solution, lp: LinearProgram, steps: List[Dict[str, Any]],
            trace: Trace, snapshots: List[List[Row]],
            x: Optional[List[Number]]) -> Solution:
    """Ghi kết luận cuối vào trace và kiểm tra nghiệm (nếu khả thi)."""
    n = lp.n
    if sol.status == Solution.FEASIBLE and x is not None:
        ok = _verify(lp, x)
        sol_str = ", ".join(f"{var_name(i, n, lp.names)} = {format_fraction(x[i])}" for i in range(n))
        trace.result(
            "Kết luận: bài toán có nghiệm tối ưu",
            f"{sol_str}\nz* = {format_fraction(sol.z)}\n"
            f"Kiểm tra lại toàn bộ ràng buộc gốc: {'đạt' if ok else 'KHÔNG đạt'}.",
        )
    elif sol.status == Solution.INFEASIBLE:
        trace.result("Kết luận: bài toán VÔ NGHIỆM (miền khả thi rỗng).")
    elif sol.status == Solution.UNBOUNDED:
        trace.result("Kết luận: bài toán KHÔNG BỊ CHẶN.")
    sol.steps = steps  # type: ignore[attr-defined]
    return sol


def _verify(lp: LinearProgram, x: List[Number]) -> bool:
    """Thế nghiệm vào mọi ràng buộc đã chuẩn hóa để xác nhận khả thi."""
    for r in lp.rows:
        lhs = sum((r.coeff(j) * x[j] for j in range(lp.n)), Fraction(0))
        if lhs > r.rhs:
            return False
    return True


def _format_linear(coeffs: List[Number], n: int,
                   names: Optional[List[str]] = None) -> str:
    parts: List[str] = []
    for i in range(n):
        a = coeffs[i]
        if a == 0:
            continue
        name = var_name(i, n, names)
        if not parts:
            if a == 1:
                parts.append(name)
            elif a == -1:
                parts.append(f"-{name}")
            else:
                parts.append(f"{format_fraction(a)}{name}")
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


def _format_cert(cert: Dict[int, Number]) -> str:
    if not cert:
        return "(rỗng)"
    return " + ".join(
        f"{format_fraction(v)}·(ràng buộc #{idx + 1})"
        for idx, v in sorted(cert.items())
    )
