# Fourier–Motzkin LP Solver (phiên bản 2)

Ứng dụng web giải bài toán **quy hoạch tuyến tính** bằng phép **khử Fourier–Motzkin**,
cài đặt hoàn toàn trên **trường số hữu tỉ** (`fractions.Fraction`) để chính xác tuyệt đối.

Bài tập lớn môn **CS112 — Phân tích và Thiết kế Thuật toán**.

Phiên bản này được xây dựng độc lập, bám sát bốn tài liệu tham khảo:

- **Bertsimas & Tsitsiklis**, *Introduction to Linear Optimization* — trực giác hình học,
  phép chiếu đa diện, định lý điểm cực biên.
- **Vanderbei**, *Linear Programming: Foundations and Extensions* — dạng ma trận,
  góc nhìn thuật toán, đà bùng nổ số ràng buộc.
- **Allen & Kennedy**, *Optimizing Compilers for Modern Architectures* — dùng
  Fourier–Motzkin làm kiểm thử khả thi cho phân tích phụ thuộc vòng lặp.
- **Schrijver**, *Theory of Linear and Integer Programming* — Bổ đề Farkas,
  độ phức tạp, cấu trúc nón đa diện.

---

## Điểm khác biệt của phiên bản 2

| Khía cạnh | Cài đặt |
| --- | --- |
| Số học | Trường hữu tỉ `Q` qua `Fraction` (chính xác tuyệt đối, không float) |
| Biểu diễn | Dạng ma trận `Ax ≤ b`, mỗi hàng là một `Row` |
| Vô nghiệm | Xuất **chứng chỉ Farkas** (tổ hợp không âm các ràng buộc gốc) |
| Bùng nổ tổ hợp | **Lọc dư thừa** (bỏ ràng buộc bị trội) sau mỗi bước khử |
| Ứng dụng CS | **Kiểm tra phụ thuộc vòng lặp** (compiler) bằng kiểm thử khả thi |
| Phương pháp | Đại số (mọi `n`) + Hình học (`n = 2`) |

---

## Cấu trúc thư mục

```
v2/
├── app.py                 # Flask: /, /api/solve, /api/depend, /api/export
├── requirements.txt
├── Procfile               # gunicorn app:app
├── render.yaml
├── report.tex / report.pdf
├── UIT.png
├── fmlp/                  # gói lõi
│   ├── rational.py        # số học hữu tỉ (Fraction)
│   ├── model.py           # Row, LinearProgram, combine (ghép cặp + Farkas)
│   ├── eliminate.py       # phân nhóm theo dấu, khử biến, lọc dư thừa
│   ├── solver.py          # pipeline: gắn z, khử, đọc z_max, thế ngược
│   ├── geometry.py        # phương pháp hình học cho n = 2
│   ├── parse.py           # parser ràng buộc dạng text
│   ├── dependence.py      # kiểm tra phụ thuộc vòng lặp
│   ├── trace.py           # ghi vết các bước
│   └── service.py         # serialize JSON + xuất .txt
├── templates/index.html
├── static/{style.css, app.js}
└── tests/
    ├── cases.py           # dữ liệu test
    ├── run_tests.py       # trình chạy test (21 phép kiểm)
    ├── smoke.py           # smoke test tầng dịch vụ
    └── growth.py          # đo đà bùng nổ số ràng buộc
```

---

## Cài đặt và chạy ở local

```powershell
# từ thư mục v2/
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Mở trình duyệt tại <http://localhost:5001>.

> Dự án dùng chung `.venv` ở thư mục gốc cũng được; chỉ cần cài `requirements.txt`.

---

## Chạy kiểm thử

```powershell
# từ thư mục v2/
python -m tests.run_tests      # 21/21 PASS
python -m tests.smoke          # smoke test tầng dịch vụ
python -m tests.growth         # bảng đà bùng nổ số ràng buộc
```

---

## Cú pháp nhập ràng buộc dạng text

Mỗi dòng một ràng buộc: `<biểu thức tuyến tính> <toán tử> <hằng số>`.

- Toán tử: `<=`, `>=`, `=`.
- Hệ số: số nguyên, phân số (`1/3`), hằng lượng giác **hữu tỉ** (`sin(pi/6)` = `1/2`).
- **Không** chấp nhận hệ số vô tỉ (`sqrt(2)`) vì solver làm việc trên trường `Q`.
- Không chấp nhận `x1^2`, `x1*x2`, `sin(x1)` (phi tuyến).

Ví dụ:

```
x1 + x2 <= 4
2x1 + x2 <= 6
-x1 <= 0
-x2 <= 0
```

---

## API

### `POST /api/solve`

```json
{
  "n": 2, "sense": "max", "obj": [3, 2],
  "method": "algebraic", "input_mode": "form",
  "constraints": [
    {"coeffs": [1, 1], "sense": "<=", "rhs": 4},
    {"coeffs": [2, 1], "sense": "<=", "rhs": 6},
    {"coeffs": [1, 0], "sense": ">=", "rhs": 0},
    {"coeffs": [0, 1], "sense": ">=", "rhs": 0}
  ]
}
```

Phản hồi gồm `status`, `solution` (mỗi biến có `fraction` + `decimal`), `z`,
`steps`, `trace`, và (khi `n = 2`) `chart`.

### `POST /api/depend`

Kiểm tra phụ thuộc vòng lặp cho `A[a*iw + c0]` và `A[b*ir + c1]` trên `[L, U]`:

```json
{ "a": 1, "c0": 0, "b": 1, "c1": -1, "L": 0, "U": 10 }
```

### `POST /api/export`

Xuất file `.txt` giải trình. Body: `{ "result": <phản hồi của /api/solve> }`.

---

## Triển khai trên Render

Repo có sẵn `render.yaml`. Build: `pip install -r requirements.txt`,
Start: `gunicorn app:app`.

---

## Tác giả

**Hoàng Quốc Duy** — Đồ án môn CS112, Phân tích và Thiết kế Thuật toán.
