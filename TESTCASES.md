# Bộ testcase — Fourier–Motzkin LP Solver

File này liệt kê toàn bộ testcase trong `tests/cases.py`.

Cách kiểm thủ công nhanh nhất:

1. Chạy app: `python app.py` rồi mở <http://localhost:5001>.
2. Đặt **số biến**, **kiểu mục tiêu (MAX/MIN)** và **hệ số hàm mục tiêu** theo từng case.
3. Chuyển ô ràng buộc sang chế độ **TEXT**, dán khối ràng buộc tương ứng.
4. Bấm **Giải** và đối chiếu với cột **Kỳ vọng**.

Hoặc chạy tự động toàn bộ:

```powershell
python -m tests.run_tests
```

---

## 1. Bài toán LP (phương pháp đại số + hình học)

### TC1 — 2 biến, max cơ bản
- **n = 2**, mục tiêu **MAX**, hệ số `z = 3, 2` → `z = 3x1 + 2x2`
- Ràng buộc (TEXT):
  ```
  x1 + x2 <= 4
  2x1 + x2 <= 6
  -x1 <= 0
  -x2 <= 0
  ```
- **Kỳ vọng:** khả thi, `z* = 10` (ví dụ `x1 = 2, x2 = 2`).

### TC2 — 3 biến, nhiều nghiệm tối ưu
- **n = 3**, mục tiêu **MAX**, hệ số `z = 2, 3, 1` → `z = 2x1 + 3x2 + x3`
- Ràng buộc (TEXT):
  ```
  x1 + x2 + x3 <= 6
  x2 + x3 <= 4
  x1 + x2 <= 4
  -x1 <= 0
  -x2 <= 0
  -x3 <= 0
  ```
- **Kỳ vọng:** khả thi, `z* = 12` (có thể trả về đỉnh tối ưu khác nhưng cùng z*).

### TC3 — min, ràng buộc hỗn hợp ≤/≥
- **n = 2**, mục tiêu **MIN**, hệ số `z = -1, -2` → `z = -x1 - 2x2`
- Ràng buộc (TEXT):
  ```
  2x1 + x2 <= 6
  x1 + x2 >= 2
  -x1 + x2 >= 3
  -x1 <= 0
  -x2 <= 0
  ```
- **Kỳ vọng:** khả thi, `z* = -12`.

### TC4 — vô nghiệm (miền rỗng)
- **n = 1**, mục tiêu **MAX**, hệ số `z = 1` → `z = x1`
- Ràng buộc (TEXT):
  ```
  x1 <= -1
  x1 >= 0
  ```
- **Kỳ vọng:** **VÔ NGHIỆM** (kèm chứng chỉ Farkas trong tab Diễn giải).

### TC5 — hệ số phân số, exact
- **n = 2**, mục tiêu **MAX**, hệ số `z = 1, 1` → `z = x1 + x2`
- Ràng buộc (TEXT):
  ```
  1/3 x1 + 2/3 x2 <= 1
  -x1 <= 0
  -x2 <= 0
  ```
- **Kỳ vọng:** khả thi, `z* = 3` (giữ chính xác phân số, không làm tròn).

### TC6 — ràng buộc đẳng thức (=)
- **n = 2**, mục tiêu **MAX**, hệ số `z = 2, 1` → `z = 2x1 + x2`
- Ràng buộc (TEXT):
  ```
  x1 + x2 = 4
  -x1 <= 0
  -x2 <= 0
  ```
- **Kỳ vọng:** khả thi, `z* = 8`.

### TC7 — một biến, min có chặn dưới
- **n = 1**, mục tiêu **MIN**, hệ số `z = 1` → `z = x1`
- Ràng buộc (TEXT):
  ```
  x1 <= 5
  x1 >= 2
  ```
- **Kỳ vọng:** khả thi, `z* = 2`.

### TC8 — 4 biến, max trên simplex
- **n = 4**, mục tiêu **MAX**, hệ số `z = 1, 2, 3, 4` → `z = x1 + 2x2 + 3x3 + 4x4`
- Ràng buộc (TEXT):
  ```
  x1 + x2 + x3 + x4 <= 10
  -x1 <= 0
  -x2 <= 0
  -x3 <= 0
  -x4 <= 0
  ```
- **Kỳ vọng:** khả thi, `z* = 40` (dồn toàn bộ vào `x4 = 10`).

### TC9 — không bị chặn (unbounded)
- **n = 2**, mục tiêu **MAX**, hệ số `z = 1, 1` → `z = x1 + x2`
- Ràng buộc (TEXT):
  ```
  -x1 <= 0
  -x2 <= 0
  ```
- **Kỳ vọng:** **KHÔNG BỊ CHẶN** (chỉ phương pháp đại số phát hiện đúng; hình học bỏ qua).

### TC10 — min với hệ số phân số
- **n = 2**, mục tiêu **MIN**, hệ số `z = 1/2, 1` → `z = 1/2 x1 + x2`
- Ràng buộc (TEXT):
  ```
  x1 + x2 >= 4
  x1 <= 6
  -x1 <= 0
  -x2 <= 0
  ```
- **Kỳ vọng:** khả thi, `z* = 2`.

#### Bảng tóm tắt LP

| TC | n | Mục tiêu | Kỳ vọng |
| --- | --- | --- | --- |
| TC1 | 2 | max 3x1+2x2 | z* = 10 |
| TC2 | 3 | max 2x1+3x2+x3 | z* = 12 |
| TC3 | 2 | min -x1-2x2 | z* = -12 |
| TC4 | 1 | max x1 | vô nghiệm |
| TC5 | 2 | max x1+x2 | z* = 3 |
| TC6 | 2 | max 2x1+x2 | z* = 8 |
| TC7 | 1 | min x1 | z* = 2 |
| TC8 | 4 | max x1+2x2+3x3+4x4 | z* = 40 |
| TC9 | 2 | max x1+x2 | không bị chặn |
| TC10 | 2 | min 1/2 x1+x2 | z* = 2 |

> Phương pháp **hình học** chỉ dùng cho `n = 2` và miền giới nội: áp dụng được
> cho TC1, TC3, TC5, TC6, TC10; bỏ qua TC2/TC8 (n≠2), TC4 (n≠2), TC9 (không bị chặn).

---

## 2. Parser ràng buộc dạng TEXT

Các case này kiểm khả năng đọc hệ số của trình parser (chạy tự động trong
`run_tests`). Khi thử thủ công, dán vào ô TEXT rồi bấm Giải; quan trọng là
parser đọc đúng hệ số (hoặc báo lỗi đúng chỗ).

### TXT1 — hệ số vô tỉ phải bị từ chối
- **n = 2**
- Nhập (TEXT):
  ```
  sin(pi/6)x1 + cos(pi/3)x2 <= 1
  sqrt(2)x1 + x2 >= 1/2
  ```
- **Kỳ vọng:** **BÁO LỖI có kiểm soát** ở dòng 2 — vì `sqrt(2)` không phải số
  hữu tỉ (v2 làm việc trên trường Q). `sin(pi/6)` = `cos(pi/3)` = `1/2` thì hợp lệ.

### TXT2 — phân số và hệ số liền biến
- **n = 2**
- Nhập (TEXT):
  ```
  2x1 + 1/3 x2 <= 5
  -x1 <= 0
  ```
- **Kỳ vọng:** parse đúng thành
  `[2, 1/3] <= 5` và `[-1, 0] <= 0`.

### TXT3 — chỉ dùng hằng lượng giác hữu tỉ
- **n = 2**
- Nhập (TEXT):
  ```
  sin(pi/6)x1 + cos(pi/3)x2 <= 1
  x1 + x2 = 3
  ```
- **Kỳ vọng:** parse đúng thành
  `[1/2, 1/2] <= 1` và `[1, 1] = 3`.

---

## 3. Kiểm tra phụ thuộc vòng lặp (ứng dụng compiler)

### 3.1. Mô hình và cách nhập liệu

Tab **"Ứng dụng"** trên giao diện dùng mô hình tổng quát: **d vòng lặp lồng nhau**, mảng **m chiều**.

```python
for i1 in range(L1, U1+1):
  for i2 in range(L2, U2+1):
    ...
      A[ f1(i), …, fm(i) ]  =  …         # GHI (write)
      …  A[ g1(i), …, gm(i) ]  …         # ĐỌC (read)

# fk(i) = ak1·i1 + … + akd·id + ck_w    (hàm chỉ số GHI chiều k)
# gk(i) = bk1·i1 + … + bkd·id + ck_r    (hàm chỉ số ĐỌC chiều k)
```

**Cách nhập trên UI:**

| Trường | Ý nghĩa |
| --- | --- |
| **d** | Số vòng lặp lồng nhau |
| **m** | Số chiều của mảng |
| **Cận vòng lặp** | Mỗi vòng j nhập L và U (biến đếm `ij ∈ [L, U]`) |
| **Hàm chỉ số GHI** | Bảng m hàng × (d cột hệ số + 1 cột hằng số). Hàng k = chiều k của mảng. |
| **Hàm chỉ số ĐỌC** | Cùng định dạng với hàm GHI. |

> **Quy trình 2 tầng:**
> ① **GCD test** — điều kiện chia hết, chính xác trên số nguyên, rất rẻ.
> Nếu không thỏa → chắc chắn không phụ thuộc, dừng ngay.
> Nếu thỏa → ② **FM trên nới lỏng số thực** (bảo thủ: có thể báo thừa).

---

### 3.2. Bộ test case — 1 vòng, mảng 1 chiều (`d=1, m=1`)

Nhập: **d = 1, m = 1**, Cận vòng 1: `L, U`, Hàm GHI hàng k=1: `[a, c0]`, Hàm ĐỌC hàng k=1: `[b, c1]`.

| Case | Vòng lặp mô phỏng | write `[×i1, +c]` | read `[×i1, +c]` | L | U | Kỳ vọng |
| --- | --- | --- | --- | --- | --- | --- |
| DEP1 | `A[i] = A[i]` (tự phụ thuộc) | `[1, 0]` | `[1, 0]` | 0 | 10 | **Có** |
| DEP2 | `A[i] = A[i-1]` (loop-carried) | `[1, 0]` | `[1, -1]` | 0 | 10 | **Có** |
| DEP3 | `A[i] = A[i+1]` | `[1, 0]` | `[1, 1]` | 0 | 10 | **Có** |
| DEP4 | `A[i]` vs `A[i+10]`, chạm đúng biên | `[1, 0]` | `[1, 10]` | 0 | 10 | **Có** |
| DEP5 | `A[2i]` vs `A[i]` (hệ số khác nhau) | `[2, 0]` | `[1, 0]` | 0 | 10 | **Có** |
| DEP6 | `A[i]` vs `A[-i]` (hệ số âm) | `[1, 0]` | `[-1, 0]` | 0 | 10 | **Có** |
| DEP7 | `A[i]` vs `A[i+11]`, vượt biên | `[1, 0]` | `[1, 11]` | 0 | 10 | **Không** |
| DEP8 | `A[i]` vs `A[i+100]`, vượt xa biên | `[1, 0]` | `[1, 100]` | 0 | 10 | **Không** |
| DEP9 | `A[5]` vs `A[7]`, chỉ số hằng khác nhau | `[0, 5]` | `[0, 7]` | 0 | 10 | **Không** |
| DEP10 | `A[i] = A[i-1]`, chỉ 1 lần lặp `[0,0]` | `[1, 0]` | `[1, -1]` | 0 | 0 | **Không** |
| DEP11 | `A[2i]` vs `A[2i+1]`, GCD test loại ngay | `[2, 0]` | `[2, 1]` | 0 | 10 | **Không** |

**Cột write/read** ghi theo định dạng bảng UI: `[hệ số × i1, + hằng số]`.

---

### 3.3. Bộ test case — 2 vòng lồng, mảng 2 chiều

Nhấn nút **"2 vòng lồng"** hoặc **"2 chiều mảng"** trên UI để nạp sẵn.

#### DEP-2D1 — 2 vòng lồng, `A[i1][i2] = A[i1-1][i2+1]`

- **d = 2, m = 2**, Cận: vòng 1 `[0,5]`, vòng 2 `[0,5]`
- Hàm GHI:

  | Chiều k | × i1 | × i2 | + c |
  | --- | --- | --- | --- |
  | k=1 | 1 | 0 | 0 |
  | k=2 | 0 | 1 | 0 |

- Hàm ĐỌC:

  | Chiều k | × i1 | × i2 | + c |
  | --- | --- | --- | --- |
  | k=1 | 1 | 0 | -1 |
  | k=2 | 0 | 1 | 1 |

- Vòng lặp mô phỏng: `A[i1][i2] = ... A[i1-1][i2+1] ...`
- **Kỳ vọng: Có phụ thuộc** — nhân chứng `iw=(0,1), ir=(1,0)`: lần ghi `(i1=0,i2=1)` ghi vào `A[0][1]`, lần đọc `(i1=1,i2=0)` đọc `A[1-1][0+1] = A[0][1]` — cùng ô.

#### DEP-2D2 — 1 vòng, mảng 2 chiều, `A[i][2i] = A[i+1][2i-1]`

- **d = 1, m = 2**, Cận: vòng 1 `[0,8]`
- Hàm GHI:

  | Chiều k | × i1 | + c |
  | --- | --- | --- |
  | k=1 | 1 | 0 |
  | k=2 | 2 | 0 |

- Hàm ĐỌC:

  | Chiều k | × i1 | + c |
  | --- | --- | --- |
  | k=1 | 1 | 1 |
  | k=2 | 2 | -1 |

- Vòng lặp mô phỏng: `A[i][2i] = ... A[i+1][2i-1] ...`
- **Kỳ vọng: Không có phụ thuộc** — GCD test loại ngay: chiều 2 yêu cầu
  `2·iw - 2·ir = -1`, nhưng `gcd(2, 2) = 2` không chia hết `1` → vô nghiệm trên
  số nguyên, `gcd_eliminated = True`.

---

## 4. Tiêu chí đánh giá đúng/sai

Vì một bài LP có thể có **nhiều nghiệm tối ưu**, không so trùng từng `xᵢ`. Bộ
test dùng hai tiêu chí độc lập:

1. Nghiệm trả về thỏa **mọi** ràng buộc gốc (kiểm tra khả thi).
2. Giá trị `z*` khớp đáp án giải tay (so sánh **chính xác** trên phân số).

Với bài đặc biệt: kiểm **trạng thái** (`feasible` / `infeasible` / `unbounded`)
đúng kỳ vọng.
