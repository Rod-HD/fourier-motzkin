# Fourier-Motzkin LP Solver

Ứng dụng web giải bài toán **Quy hoạch tuyến tính (Linear Programming)** bằng hai phương pháp:
**Khử Fourier-Motzkin** (đại số, hoạt động cho mọi số biến) và **Hình học** (cho bài toán 2 biến,
trực quan bằng đồ thị).

Đồ án môn **CS112 — Phân tích và Thiết kế Thuật toán**.

---

## Mục lục

- [Tính năng](#tính-năng)
- [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Cài đặt và chạy ở local](#cài-đặt-và-chạy-ở-local)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Cú pháp nhập ràng buộc dạng text](#cú-pháp-nhập-ràng-buộc-dạng-text)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [API endpoints](#api-endpoints)
- [Thuật toán Fourier-Motzkin](#thuật-toán-fourier-motzkin)
- [Triển khai trên Render](#triển-khai-trên-render)
- [Test case mẫu](#test-case-mẫu)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)

---

## Tính năng

- 🎯 **Hai phương pháp giải**
  - **Đại số (Fourier-Motzkin)**: khử lần lượt từng biến `x₁, x₂, …, xₙ`, tìm `z*` rồi back-substitute để khôi phục nghiệm. Hoạt động với mọi số biến.
  - **Hình học**: liệt kê các đỉnh khả thi của đa giác miền nghiệm trong mặt phẳng `Ox₁x₂`, tính `z` tại từng đỉnh để xác định điểm tối ưu (chỉ cho `n = 2`).

- 🔢 **Tính toán chính xác (exact arithmetic)**
  - Sử dụng `sympy` để giữ hệ số ở dạng phân số / căn / hằng số ký hiệu (`Rational`, `sqrt`, `pi`, …).
  - Tránh sai số dấu phẩy động khi khử biến nhiều bước.
  - Kết quả hiển thị ở cả dạng phân số và thập phân.

- ✍️ **Hai chế độ nhập liệu**
  - **Form**: nhập từng hệ số ràng buộc qua input riêng biệt.
  - **Text**: nhập nhiều ràng buộc cùng lúc dạng tự nhiên, ví dụ `2x1 + 3x2 <= 12`. Trình parser hỗ trợ phân số, căn, lượng giác (`sin`, `cos`, `pi`, `sqrt`, …).

- 📋 **Giải trình từng bước**
  - Module `reasoning` ghi lại từng phép biến đổi (chuẩn hóa, ghép cặp ≤/≥, khử biến, back-substitute, kiểm tra nghiệm).
  - Hiển thị công thức và kết quả sau mỗi bước, kèm phân loại (INFO / KẾT_LUẬN / CẢNH_BÁO).

- 📊 **Trực quan hóa** (cho bài toán 2 biến)
  - Vẽ các đường biên ràng buộc, miền khả thi (convex polygon), các đỉnh, đường mức `z = const`.
  - Đánh dấu hướng tịnh tiến đường mức (về phía tăng/giảm `z`).

- 📥 **Xuất giải trình**
  - Tải file `.txt` chứa toàn bộ bài toán đầu vào, các bước giải, reasoning chi tiết và kết quả.
  - Có hai mẫu xuất: dành cho phương pháp đại số và phương pháp hình học.

- 🌐 **Web app nhẹ**, không database, không session — toàn bộ kết quả tính trên server và trả về JSON cho frontend.

---

## Kiến trúc tổng quan

```
┌──────────────────────┐      JSON      ┌─────────────────────────┐
│  Frontend (HTML/JS)  │ ──────────────▶│  Flask app (main.py)    │
│  - Form / text input │ ◀──────────────│  /api/solve  /api/export│
│  - Chart.js render   │                └────────────┬────────────┘
└──────────────────────┘                             │
                                                     ▼
                                       ┌────────────────────────────┐
                                       │  solver/                   │
                                       │   ├── parser.py  (text→LP)│
                                       │   ├── exact.py   (sympy)  │
                                       │   ├── core.py    (FM+z)   │
                                       │   ├── core_exact.py       │
                                       │   ├── geometric.py        │
                                       │   └── reasoning.py        │
                                       └────────────────────────────┘
```

---

## Yêu cầu môi trường

- **Python** 3.10 trở lên (test trên 3.12)
- **pip** + môi trường ảo (`venv`)
- Hệ điều hành: Windows / macOS / Linux

---

## Cài đặt và chạy ở local

### Windows (PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/Rod-HD/fourier-motzkin.git
cd fourier-motzkin

# 2. Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Chạy app
python main.py
```

### macOS / Linux

```bash
git clone https://github.com/Rod-HD/fourier-motzkin.git
cd fourier-motzkin

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python main.py
```

Mở trình duyệt tại: <http://localhost:5000>

---

## Hướng dẫn sử dụng

1. Chọn **số biến** `n` (≥ 1).
2. Chọn **kiểu hàm mục tiêu**: `max` hoặc `min`.
3. Nhập **hệ số hàm mục tiêu** `c₁, c₂, …, cₙ` cho `z = c₁x₁ + c₂x₂ + … + cₙxₙ`.
4. Nhập **các ràng buộc** theo một trong hai cách:
   - **Form**: thêm từng dòng, mỗi dòng gồm các hệ số, dấu so sánh (`<=`, `>=`, `=`) và vế phải.
   - **Text**: gõ trực tiếp tất cả ràng buộc, mỗi dòng một ràng buộc.
5. Chọn **phương pháp giải**:
   - `Đại số (Fourier-Motzkin)` — luôn dùng được.
   - `Hình học` — chỉ khả dụng khi `n = 2`.
6. Nhấn **Giải** → app hiển thị nghiệm `(x₁*, x₂*, …, xₙ*)`, giá trị `z*`, các bước giải và biểu đồ (nếu `n = 2`).
7. (Tuỳ chọn) Nhấn **Tải giải trình** để lưu file `.txt`.

---

## Cú pháp nhập ràng buộc dạng text

Mỗi dòng là **một ràng buộc** dạng:

```
<biểu thức tuyến tính theo x1..xn> <toán tử> <hằng số>
```

- Toán tử: `<=`, `>=`, `=`
- Hệ số có thể là: số nguyên, phân số (`1/2`), căn (`sqrt(2)`), `pi`, lượng giác (`sin(pi/6)`, `cos(pi/3)`, …).
- Cho phép viết hệ số liền biến: `2x1`, `-3x2`. Dấu nhân ngầm hoạt động (`sqrt(2)x1` ≡ `sqrt(2)*x1`).
- KHÔNG cho phép: `x1^2`, `x1*x2`, `sin(x1)` — chỉ chấp nhận tổ hợp **tuyến tính**.

### Ví dụ

```
x1 + x2 <= 4
2x1 + x2 <= 6
x1 >= 0
x2 >= 0
```

```
sin(pi/6)x1 + cos(pi/3)x2 >= 1/2
sqrt(2)x1 - 3x2 = 7
```

Lỗi parser (sai cú pháp, biến lạ, không tuyến tính) sẽ trả về số dòng và gợi ý sửa cụ thể.

---

## Cấu trúc thư mục

```
fourier-motzkin/
├── main.py                  # Flask app: routes /, /api/solve, /api/export
├── Procfile                 # Lệnh start cho Render (gunicorn main:app)
├── render.yaml              # Cấu hình Render Blueprint
├── requirements.txt         # Dependencies Python
├── solver/
│   ├── __init__.py
│   ├── core.py              # Pipeline FM (numeric) + chart data 2D
│   ├── core_exact.py        # Pipeline FM với sympy exact
│   ├── exact.py             # Tiện ích số học chính xác
│   ├── geometric.py         # Phương pháp hình học cho n = 2
│   ├── parser.py            # Parse ràng buộc dạng text
│   └── reasoning.py         # Engine ghi lại các bước giải
├── static/
│   └── style.css
├── templates/
│   └── index.html           # Frontend SPA (HTML + JS thuần + Chart.js)
└── README.md
```

---

## API endpoints

### `POST /api/solve`

Giải bài toán LP.

**Body** (JSON):

```json
{
  "n": 2,
  "obj_type": "max",
  "obj_coeffs": [3, 2],
  "input_mode": "form",
  "constraints": [
    { "coeffs": [1, 1], "sense": "<=", "rhs": 4 },
    { "coeffs": [2, 1], "sense": "<=", "rhs": 6 },
    { "coeffs": [1, 0], "sense": ">=", "rhs": 0 },
    { "coeffs": [0, 1], "sense": ">=", "rhs": 0 }
  ],
  "method": "algebraic"
}
```

Hoặc với `input_mode = "text"`:

```json
{
  "n": 2,
  "obj_type": "max",
  "obj_coeffs": [3, 2],
  "input_mode": "text",
  "constraints_text": "x1 + x2 <= 4\n2x1 + x2 <= 6\nx1 >= 0\nx2 >= 0",
  "method": "algebraic"
}
```

**Response** (rút gọn):

```json
{
  "feasible": true,
  "solution": { "x1": 2, "x2": 2 },
  "z": 10,
  "method": "algebraic",
  "steps_constraints": [...],
  "reasoning": { "steps": [...] },
  "vertices": [...],
  "boundary_lines": [...],
  "level_lines": {...}
}
```

### `POST /api/export`

Xuất giải trình ra file `.txt`. Body chứa `result` (response của `/api/solve`), `inp` (input gốc) và `method`.

---

## Thuật toán Fourier-Motzkin

Pipeline trong `solver/core_exact.py`:

1. **Chuẩn hoá ràng buộc**: đổi tất cả về dạng `≤`. Ràng buộc `=` được tách thành hai `≤` và `≥`, ràng buộc `≥` nhân `−1`.
2. **Thêm biến `z`**:
   - Với `max z = cᵀx`: thêm `z − cᵀx ≤ 0`.
   - Với `min z = cᵀx`: chuyển thành `max(−cᵀx)`, thêm `z + cᵀx ≤ 0`. Sau khi tìm `z_max`, suy ra `z_min = −z_max`.
3. **Khử lần lượt** `x₁, x₂, …, xₙ`:
   - Phân loại ràng buộc theo dấu hệ số biến đang khử: nhóm chặn trên (`U`), chặn dưới (`L`), không chứa biến (`N`).
   - Ghép mọi cặp `(L, U)` để sinh ra ràng buộc mới `f_L ≤ f_U`.
4. **Tìm `z_max`** từ hệ chỉ còn biến `z` (kiểm tra cả vô nghiệm `0 ≤ −k` và không bị chặn).
5. **Back-substitute**: thay `z = z_max` và các biến đã biết, tìm khoảng `[lo, hi]` cho từng biến rồi chọn giá trị (ưu tiên đầu mút).
6. **Kiểm tra nghiệm**: thay vào hệ ràng buộc gốc.

Mọi phép tính ở các bước trên đều dùng `sympy.Rational` / `sympy.Expr` để giữ giá trị chính xác.

---

## Triển khai trên Render

Repo đã có sẵn **`render.yaml`** (Render Blueprint), chỉ cần:

1. Đăng nhập <https://render.com> bằng GitHub.
2. **New +** → **Blueprint** → chọn repo `fourier-motzkin`.
3. Render tự đọc `render.yaml`, tạo **Web Service** với:
   - **Runtime**: Python
   - **Build**: `pip install -r requirements.txt`
   - **Start**: `gunicorn main:app`
   - **Plan**: Free
4. Bấm **Apply** → đợi build xong → app chạy ở `https://<service-name>.onrender.com`.

Nếu muốn deploy thủ công không qua Blueprint:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn main:app`

> ℹ️ Plan Free của Render sẽ "ngủ" sau 15 phút không có request. Lần truy cập đầu sau khi ngủ sẽ chậm khoảng 30–60 giây để khởi động lại.

---

## Test case mẫu

### 2 biến

```
max z = 3x1 + 2x2
s.t.   x1 + x2  <= 4
       2x1 + x2 <= 6
       x1, x2   >= 0
```

→ Nghiệm tối ưu: `x1 = 2`, `x2 = 2`, `z* = 10`.

### 3 biến

```
max z = 2x1 + 3x2 + x3
s.t.   x1 + x2 + x3 <= 6
       x2 + x3      <= 4
       x1 + x2      <= 4
       x1, x2, x3   >= 0
```

→ Nghiệm tối ưu: `x1 = 2`, `x2 = 2`, `x3 = 2`, `z* = 12`.

### Hệ số ký hiệu (text mode)

```
sin(pi/6)x1 + cos(pi/3)x2 <= 1
sqrt(2)x1 + x2 >= 1/2
x1 >= 0
x2 >= 0
```

App tính chính xác ở dạng phân số / căn, không bị tròn số.

---

## Công nghệ sử dụng

| Thành phần       | Thư viện                       |
| ---------------- | ------------------------------ |
| Web framework    | Flask 3                        |
| WSGI server      | Gunicorn                       |
| Tính toán exact  | SymPy 1.14                     |
| Frontend         | HTML5 + CSS3 + JavaScript thuần |
| Biểu đồ          | Chart.js (CDN)                 |
| Deploy           | Render.com                     |

Toàn bộ dependencies xem trong [`requirements.txt`](./requirements.txt).

---

## Tác giả

**Hoàng Quốc Duy** — `Rod-HD`
Đồ án môn CS112 — Phân tích và Thiết kế Thuật toán.
