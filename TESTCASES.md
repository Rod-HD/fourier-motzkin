# Bộ testcase — Fourier–Motzkin LP Solver (v2)

File này liệt kê toàn bộ testcase trong `tests/cases.py` ở dạng dễ đọc, kèm
**bản nhập TEXT** để dán thẳng vào ô nhập của giao diện (chế độ TEXT).

Cách kiểm thủ công nhanh nhất:

1. Chạy app: `python app.py` rồi mở <http://localhost:5001>.
2. Đặt **số biến**, **kiểu mục tiêu (MAX/MIN)** và **hệ số hàm mục tiêu** theo từng case.
3. Chuyển ô ràng buộc sang chế độ **TEXT**, dán khối ràng buộc tương ứng.
4. Bấm **Giải** và đối chiếu với cột **Kỳ vọng**.

> Lưu ý: ở chế độ TEXT, **bỏ tick** "tự thêm xᵢ ≥ 0" vì các điều kiện không âm
> đã được viết sẵn trong khối ràng buộc của từng case.

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

### 3.1. Vòng lặp này là gì? `A`, `iw`, `ir`, `a, c0, b, c1` là gì?

Xét một vòng lặp duyệt mảng `A` (một dãy ô nhớ `A[0], A[1], A[2], ...`):

```python
for i in range(L, U+1):
    A[a*i + c0] = ... A[b*i + c1] ...   # mỗi lần lặp: GHI vào A[a*i+c0], ĐỌC từ A[b*i+c1]
```

Giải nghĩa từng ký hiệu:

| Ký hiệu | Ý nghĩa |
| --- | --- |
| `A` | một **mảng** (dãy ô nhớ liên tiếp). `A[k]` là ô nhớ thứ `k`. |
| `i` | **biến đếm** của vòng lặp, chạy từ `L` đến `U`. |
| `a*i + c0` | **chỉ số ô được GHI** ở lần lặp `i` (biểu thức tuyến tính theo `i`). |
| `b*i + c1` | **chỉ số ô được ĐỌC** ở lần lặp `i`. |
| `a, b` | hệ số nhân với `i` (bước nhảy chỉ số). |
| `c0, c1`| hằng số cộng thêm (độ dịch). |
| `iw` | giá trị `i` ở **lần lặp ghi** (write). |
| `ir` | giá trị `i` ở **lần lặp đọc** (read). |

Ví dụ cụ thể: với `a=1, c0=0, b=1, c1=-1` thì vòng lặp là

```python
for i in range(0, 11):
    A[i] = A[i-1] + 1     # GHI A[i], ĐỌC A[i-1]
```

### 3.2. "Phụ thuộc" nghĩa là gì? Vì sao cần kiểm tra?

Trình biên dịch muốn **chạy song song** hoặc **đảo thứ tự** các lần lặp để tăng tốc.
Việc đó chỉ **an toàn** nếu các lần lặp **không giẫm chân nhau** — tức không có
lần lặp nào ĐỌC một ô mà lần lặp khác đã/đang GHI vào đúng ô đó.

Hai lần lặp `iw` (ghi) và `ir` (đọc) chạm **cùng một ô nhớ** khi chỉ số ghi bằng
chỉ số đọc:

```
a*iw + c0 = b*ir + c1,    với L ≤ iw ≤ U  và  L ≤ ir ≤ U
```

Đây là một **hệ ràng buộc tuyến tính** theo `iw, ir`. Câu hỏi "có phụ thuộc không?"
chính là "**hệ này có nghiệm (khả thi) không?**" — và đó là việc Fourier–Motzkin
làm: khử biến để kiểm tra khả thi.

- Hệ **vô nghiệm** ⟹ không cặp `(iw, ir)` nào trùng ô ⟹ **KHÔNG phụ thuộc** ⟹ song song hóa an toàn.
- Hệ **khả thi** ⟹ có thể trùng ô ⟹ **CÓ THỂ phụ thuộc** ⟹ phải giữ nguyên thứ tự.

> Kiểm thử chạy trên **số thực** (nới lỏng): nếu vô nghiệm trên số thực thì chắc
> chắn vô nghiệm trên số nguyên (kết luận an toàn). Nếu khả thi trên số thực thì
> kết luận "có thể phụ thuộc" theo hướng **bảo thủ** — không bao giờ bỏ sót phụ
> thuộc thật (xem DEP11).

### 3.3. Cách thử trên giao diện

Cuộn xuống khối **"Kiểm tra phụ thuộc vòng lặp"**, nhập `a, c0, b, c1, L, U` rồi
bấm **Kiểm tra**.

### 3.4. Bộ test case

| Case | Vòng lặp (ý nghĩa) | a | c0 | b | c1 | L | U | Kỳ vọng |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEP1 | `A[i] = A[i]` (tự phụ thuộc) | 1 | 0 | 1 | 0 | 0 | 10 | **Có** |
| DEP2 | `A[i] = A[i-1]` (mang qua vòng) | 1 | 0 | 1 | -1 | 0 | 10 | **Có** |
| DEP3 | `A[i] = A[i+1]` | 1 | 0 | 1 | 1 | 0 | 10 | **Có** |
| DEP4 | `A[i]` vs `A[i+10]`, biên `[0,10]` (chạm đúng biên) | 1 | 0 | 1 | 10 | 0 | 10 | **Có** |
| DEP5 | `A[2i]` vs `A[i]` (hệ số khác nhau) | 2 | 0 | 1 | 0 | 0 | 10 | **Có** |
| DEP6 | `A[i]` vs `A[-i]` (hệ số âm) | 1 | 0 | -1 | 0 | 0 | 10 | **Có** |
| DEP7 | `A[i]` vs `A[i+11]`, biên `[0,10]` (vượt biên) | 1 | 0 | 1 | 11 | 0 | 10 | **Không** |
| DEP8 | `A[i]` vs `A[i+100]` (vượt xa biên) | 1 | 0 | 1 | 100 | 0 | 10 | **Không** |
| DEP9 | `A[5]` vs `A[7]` (chỉ số hằng khác nhau) | 0 | 5 | 0 | 7 | 0 | 10 | **Không** |
| DEP10 | `A[i] = A[i-1]` nhưng chỉ 1 lần lặp `[0,0]` | 1 | 0 | 1 | -1 | 0 | 0 | **Không** |
| DEP11 | `A[2i]` vs `A[2i+1]` (chẵn≠lẻ) — *giới hạn nới lỏng thực* | 2 | 0 | 2 | 1 | 0 | 10 | **Có** |

**Giải thích vài ca tiêu biểu:**

- **DEP2** `A[i] = A[i-1]`: lần lặp `i` đọc kết quả lần lặp `i-1` vừa ghi → phụ
  thuộc "mang qua vòng" (loop-carried), KHÔNG được song song hóa. Nghiệm minh
  chứng: `iw=0, ir=1` (lần ghi i=0 tạo ra A[0], lần đọc i=1 đọc A[0]).
- **DEP7/DEP8** `A[i+11]`, `A[i+100]`: chỉ số đọc luôn lớn hơn mọi chỉ số ghi
  trong `[0,10]` → không bao giờ trùng → vô nghiệm → an toàn song song hóa.
- **DEP9** `A[5]` vs `A[7]`: hai chỉ số hằng số (a=b=0), `5 ≠ 7` luôn đúng → hệ
  cho `0 = 2`, vô nghiệm → không phụ thuộc.
- **DEP10**: cùng công thức DEP2 nhưng vòng lặp chỉ chạy `i=0` (một lần) → không
  có cặp `(iw, ir)` hợp lệ → không phụ thuộc.
- **DEP11** `A[2i]` vs `A[2i+1]`: chỉ số chẵn không bao giờ bằng chỉ số lẻ trên
  **số nguyên**, nên thực tế KHÔNG phụ thuộc. Nhưng nới lỏng **số thực** cho
  nghiệm `iw=1/2` (khả thi) nên hệ thống kết luận "**có thể** có phụ thuộc". Đây
  là điểm **bảo thủ** đã biết của kiểm thử bằng Fourier–Motzkin: thà báo thừa
  (an toàn) còn hơn bỏ sót. Muốn loại ca này cần thêm kiểm tra GCD trên số nguyên.

---

## 4. Tiêu chí đánh giá đúng/sai

Vì một bài LP có thể có **nhiều nghiệm tối ưu**, không so trùng từng `xᵢ`. Bộ
test dùng hai tiêu chí độc lập:

1. Nghiệm trả về thỏa **mọi** ràng buộc gốc (kiểm tra khả thi).
2. Giá trị `z*` khớp đáp án giải tay (so sánh **chính xác** trên phân số).

Với bài đặc biệt: kiểm **trạng thái** (`feasible` / `infeasible` / `unbounded`)
đúng kỳ vọng.
