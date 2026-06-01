"""Bộ ghi vết (trace) các bước giải để hiển thị và xuất file.

Mỗi mục vết có một *loại* (``info``, ``calc``, ``result``, ``warn``) giúp giao
diện tô màu và phân nhóm. Bộ giải chỉ việc gọi :meth:`Trace.add` /
:meth:`Trace.calc`; phần trình bày tách hẳn khỏi phần thuật toán.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceItem:
    index: int
    kind: str           # info | calc | result | warn
    title: str
    body: str = ""
    formula: str = ""
    outcome: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trace:
    items: List[TraceItem] = field(default_factory=list)

    def _next(self) -> int:
        return len(self.items) + 1

    def add(self, title: str, body: str = "", kind: str = "info") -> None:
        self.items.append(TraceItem(self._next(), kind, title, body=body))

    def calc(self, title: str, formula: str, outcome: str) -> None:
        self.items.append(
            TraceItem(self._next(), "calc", title, formula=formula, outcome=outcome)
        )

    def result(self, title: str, body: str = "") -> None:
        self.items.append(TraceItem(self._next(), "result", title, body=body))

    def warn(self, title: str, body: str = "") -> None:
        self.items.append(TraceItem(self._next(), "warn", title, body=body))

    def to_dict(self) -> Dict[str, Any]:
        return {"items": [it.to_dict() for it in self.items], "count": len(self.items)}
