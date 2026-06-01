"""Gói giải LP bằng khử Fourier-Motzkin — phiên bản 2.

Hạt nhân làm việc trên trường số hữu tỉ Q (Fraction) để bảo đảm chính xác
tuyệt đối, biểu diễn bài toán ở dạng ma trận ``Ax <= b``, và truy vết mọi bước
khử kèm chứng chỉ Farkas cho trường hợp vô nghiệm.
"""

from .model import LinearProgram, Row, format_row, var_name
from .solver import Solution, solve
from .geometry import solve_geometric
from .parse import ParseError, parse_constraints
from .dependence import test_loop_dependence, test_loop_dependence_general
from .trace import Trace

__all__ = [
    "LinearProgram",
    "Row",
    "Solution",
    "Trace",
    "solve",
    "solve_geometric",
    "parse_constraints",
    "ParseError",
    "test_loop_dependence",
    "test_loop_dependence_general",
    "format_row",
    "var_name",
]
