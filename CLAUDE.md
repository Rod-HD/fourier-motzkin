# Project: Fourier-Motzkin LP Solver — Web App

## Môi trường
- Windows, PowerShell, Python 3.x
- Thư mục code: D:\Duy\Docs\School\CS112. Phân tích và thiết kế thuật toán\Big_Assignments\fourier-motzkin
- .venv nằm trong thư mục code, Flask đã cài sẵn
- Frontend: HTML + CSS + JS thuần, Chart.js từ CDN cdnjs.cloudflare.com
- Không cần cài thêm thư viện nào

## Chạy lệnh — PowerShell (KHÔNG dùng ! prefix)
- Test:     python -c "..."
- Chạy app: python main.py
- Xem file: Get-Content giai_trinh.txt
- Freeze:   pip freeze > requirements.txt

## Cấu trúc file
fourier-motzkin\
├── .venv\
├── .vscode\settings.json
├── main.py
├── solver\
│   ├── __init__.py
│   ├── core.py
│   └── reasoning.py
├── templates\
│   └── index.html
├── static\
│   └── style.css
├── requirements.txt
└── CLAUDE.md

## Test case chuẩn
2 biến:
  x1 + x2  <= 4
  2x1 + x2 <= 6
  x1 >= 0, x2 >= 0
  max z = 3x1 + 2x2
  Kết quả đúng: x1=2.0, x2=2.0, z=10.0

3 biến:
  x1 + x2 + x3 <= 6
  x2 + x3       <= 4
  x1 + x2       <= 4
  x1,x2,x3 >= 0
  max z = 2x1 + 3x2 + x3
  Kết quả đúng: x1=2.0, x2=2.0, x3=2.0, z=12.0

## Quy tắc code
- Type hints đầy đủ
- Docstring tiếng Việt
- Flask trả JSON cho mọi API call
- Frontend dùng fetch() không reload trang

Báo cáo kết quả từng bước ✅/❌
