"""Module quản lý ghi chép các bước lập luận trong quá trình giải bài toán."""

from typing import List, Dict, Any


class ReasoningEngine:
    """Engine ghi lại từng bước lập luận trong quá trình giải bài toán LP."""

    def __init__(self) -> None:
        """Khởi tạo engine với danh sách bước rỗng."""
        self.steps: List[Dict[str, Any]] = []
        self._counter: int = 0

    def ghi(self, tieu_de: str, noi_dung: str, loai: str = "INFO") -> None:
        """Ghi một bước lập luận thông thường.

        Args:
            tieu_de: Tiêu đề ngắn gọn của bước.
            noi_dung: Nội dung chi tiết.
            loai: Loại bước — "INFO", "TINH_TOAN", "KET_LUAN", "CANH_BAO".
        """
        self._counter += 1
        self.steps.append({
            "so_buoc": self._counter,
            "loai": loai,
            "tieu_de": tieu_de,
            "noi_dung": noi_dung,
            "cong_thuc": "",
            "ket_qua": ""
        })

    def ghi_cong_thuc(self, mo_ta: str, cong_thuc: str, ket_qua: str) -> None:
        """Ghi một bước có kèm công thức tính toán.

        Args:
            mo_ta: Mô tả ngắn về phép tính.
            cong_thuc: Công thức toán học dạng chuỗi.
            ket_qua: Kết quả sau khi tính.
        """
        self._counter += 1
        self.steps.append({
            "so_buoc": self._counter,
            "loai": "TINH_TOAN",
            "tieu_de": mo_ta,
            "noi_dung": mo_ta,
            "cong_thuc": cong_thuc,
            "ket_qua": ket_qua
        })

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển toàn bộ các bước thành dict để trả về JSON.

        Returns:
            Dict chứa danh sách steps và tổng số bước.
        """
        return {
            "steps": self.steps,
            "tong_buoc": len(self.steps)
        }
