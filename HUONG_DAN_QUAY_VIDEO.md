# Hướng dẫn quay video demo (phiên bản 2)

Mục tiêu: video **5–7 phút** thể hiện đầy đủ các tính năng của ứng dụng.

---

## Chuẩn bị

```powershell
# Từ thư mục v2/
..\.venv\Scripts\Activate.ps1
$env:PORT = '5055'
python app.py
```

Mở trình duyệt tại **http://localhost:5055**. Bật phần mềm quay màn hình (OBS hoặc `Win + G`).

---

## Kịch bản quay

### Phần 1 — Giới thiệu (~20 giây)

Nói: *"Đây là ứng dụng giải quy hoạch tuyến tính bằng thuật toán khử Fourier–Motzkin, phiên bản 2. Điểm khác biệt: tính toán trên trường số hữu tỉ chính xác tuyệt đối, kèm chứng chỉ Farkas khi vô nghiệm, và ứng dụng kiểm tra phụ thuộc vòng lặp theo sách Allen & Kennedy."*

---

### Phần 2 — Giải bài toán 2 biến, phương pháp đại số (~80 giây)

Bấm **"Ví dụ mẫu"** (tự điền sẵn `max z = 3x1 + 2x2`). Chọn **Đại số**, bấm **Giải**.

**Tab "Các bước khử"** — chỉ vào và nói:
> *"Hệ ban đầu sau khi gắn biến z. Bước 1 khử x1: cô lập x1 trong từng ràng buộc, tách thành chặn trên và chặn dưới. Bước 2: điều kiện tồn tại x1 là mọi chặn dưới ≤ mọi chặn trên, nên ghép 2×2 = 4 cặp. Bước 3: liệt kê từng cặp ghép cụ thể. Bước 4: lọc bỏ ràng buộc dư thừa, giữ lại hệ rút gọn. Tương tự cho x2. Cuối cùng chỉ còn 3z ≤ 30, suy ra z* = 10."*

**Tab "Diễn giải"** — nói:
> *"Mỗi bước được giải thích chi tiết bằng ngôn ngữ tự nhiên, không chỉ in ra con số."*

---

### Phần 3 — Phương pháp hình học + biểu đồ (~70 giây)

Cùng bài toán, chọn **Hình học**, bấm **Giải**. Mở tab **Biểu đồ**.

Chỉ vào từng thành phần và nói:
> *"Miền khả thi được tô màu xanh — đây là tập hợp tất cả điểm thỏa mọi ràng buộc. Mỗi đường ràng buộc một màu riêng, mũi tên chỉ về phía nửa mặt phẳng được chọn. Đường cam nét liền là đường mức z* = 10 đi qua nghiệm tối ưu. Đường cam nét đứt đang trượt — đây là đường mức z = const đang tịnh tiến theo hướng tăng z; điểm chạm cuối cùng với miền khả thi chính là nghiệm tối ưu, được đánh dấu nhấp nháy."*

Hover vào đỉnh A(2,2) để thấy tooltip.

---

### Phần 4 — Số học hữu tỉ chính xác (~30 giây)

Chuyển sang **TEXT**, nhập:
```
1/3 x1 + 2/3 x2 <= 1
-x1 <= 0
-x2 <= 0
```
Bấm **Giải**. Nói:
> *"Hệ số phân số 1/3 và 2/3 được giữ chính xác tuyệt đối, kết quả z* = 3 không hề làm tròn. Đây là điểm khác biệt so với các solver dùng số thực dấu phẩy động."*

---

### Phần 5 — Vô nghiệm + chứng chỉ Farkas (~35 giây)

Số biến **1**, MAX, hệ số **1**. Bỏ tick "không âm". Nhập TEXT:
```
x1 <= -1
x1 >= 0
```
Bấm **Giải**. Mở tab **Diễn giải**, chỉ vào dòng cảnh báo:
> *"Bài toán vô nghiệm. Chương trình xuất chứng chỉ Farkas: cộng ràng buộc #1 và #2 với hệ số 1 và 1 cho ra 0 ≤ −1 — một mâu thuẫn. Đây là bằng chứng toán học chứng minh miền khả thi rỗng."*

---

### Phần 6 — Ứng dụng kiểm tra phụ thuộc vòng lặp (~90 giây)

Cuộn xuống khối **"Ứng dụng: kiểm tra phụ thuộc dữ liệu vòng lặp"**.

**Giải thích ngắn trước khi demo:**
> *"Trình biên dịch muốn song song hóa vòng lặp để tăng tốc. Nhưng nếu lần lặp này đọc ô mà lần lặp khác đang ghi thì không được. Bài toán 'có phụ thuộc không?' quy về kiểm tra một hệ ràng buộc tuyến tính có nghiệm không — đúng việc Fourier–Motzkin làm."*

**Demo 1 — 1 vòng có phụ thuộc:**

Bấm **"1 vòng A[i]=A[i-1]"**. Nói:
> *"Vòng lặp `for i: A[i] = A[i-1]`. Ghi A[i] — hệ số i là 1, hằng số 0. Đọc A[i-1] — hệ số i là 1, hằng số -1. Biên [0, 10]."*

Bấm **Kiểm tra**. Chỉ vào kết quả:
> *"Có phụ thuộc. Nhân chứng: lần ghi i=0 tạo ra A[0], lần đọc i=1 đọc A[0] — trùng ô nhớ. Không được song song hóa."*

**Demo 2 — 2 vòng lồng nhau:**

Bấm **"2 vòng lồng"**. Nói:
> *"Bây giờ 2 vòng lặp lồng nhau, mảng 2 chiều. Ghi A[i₁][i₂], đọc A[i₁-1][i₂+1]. Hệ có 4 biến: iw₁, iw₂, ir₁, ir₂."*

Bấm **Kiểm tra**. Nói:
> *"Có phụ thuộc. Nhân chứng: iw=(0,1), ir=(1,0) — lần ghi tại (0,1) và lần đọc tại (1,0) trỏ vào cùng ô A[0][1]."*

**Demo 3 — GCD test loại ngay:**

Bấm **"GCD loại (chẵn/lẻ)"**. Nói:
> *"Ghi A[2i], đọc A[2i+1]. Chỉ số chẵn không bao giờ bằng chỉ số lẻ."*

Bấm **Kiểm tra**. Chỉ vào kết quả màu xanh teal:
> *"Không phụ thuộc — loại bởi GCD test. gcd(2,2)=2 không chia hết 1, nên phương trình 2·iw = 2·ir+1 không có nghiệm nguyên. Kết luận chính xác trên số nguyên, không cần chạy FM."*

---

### Phần 7 — Chạy bộ test tự động (~25 giây)

Mở terminal thứ hai, chạy:
```powershell
python -m tests.run_tests
```
Cho thấy dòng tổng kết: *"29/29 test PASS."*

Nói:
> *"Bộ test phủ 10 bài LP, 3 case parser, 11 case phụ thuộc vòng lặp — tất cả đều PASS."*

---

### Phần 8 — Kết (~15 giây)

> *"Phiên bản 2 giải LP bằng Fourier–Motzkin trên trường số hữu tỉ, kèm chứng chỉ Farkas, biểu đồ hình học đầy đủ, và ứng dụng kiểm tra phụ thuộc vòng lặp tổng quát theo sách Allen & Kennedy. Cảm ơn đã xem."*

---

## Mẹo

- Nói chậm, chờ kết quả hiện ra rồi mới thuyết minh.
- Bấm **Đặt lại** nếu lỡ thao tác sai.
- Nên zoom trình duyệt lên 110–125% để chữ dễ đọc hơn khi quay.
- Tổng thời gian mục tiêu: **5–7 phút**.
