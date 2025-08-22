# OCR-Huy

Hệ thống OCR toàn diện để nhận dạng và đọc văn bản từ ảnh tài liệu scan với nhiều tùy chọn tiền xử lý nâng cao.

## Tính năng

- **Pipeline tiền xử lý**: chuyển đổi xám, khử nhiễu, phóng to, làm nét, thẳng hóa, tăng cường độ tương phản (CLAHE), chỉnh sáng tự động, nhị phân hóa adaptive/Otsu, các phép biến đổi morphology
- **Engine OCR**: EasyOCR với hỗ trợ đa ngôn ngữ (GPU/CPU)
- **Chuẩn hóa văn bản nâng cao**:
  - Sửa lỗi OCR với từ điển mở rộng (300+ mục)
  - Hỗ trợ Python-Levenshtein cho sửa lỗi thông minh
  - Pattern matching cho lỗi dấu thanh tiếng Việt
  - Unicode normalization (NFC)
- **Đầu ra**: văn bản nhận dạng trong file .txt với thứ tự đọc đúng (trên xuống dưới, trái sang phải)
- **Xử lý hàng loạt**: hỗ trợ ảnh đơn lẻ hoặc toàn bộ thư mục
- **Tùy chọn linh hoạt**: 16+ cờ dòng lệnh cho các loại tài liệu khác nhau

## Cài đặt

Yêu cầu Python 3.9+.

```bash
pip install -r requirements.txt
```

Nếu gặp vấn đề cài PyTorch trên Windows, thử:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install easyocr
```

### Cài đặt lại virtual environment (sau khi thêm dependencies mới)

**Cách 1: Cài thêm vào venv hiện tại (Khuyến nghị)**

```bash
# Kích hoạt venv hiện tại
& ".\.venv\Scripts\Activate.ps1"

# Cài các packages mới
pip install -r requirements.txt

# Kiểm tra
pip list | findstr -i "unidecode levenshtein"
```

**Cách 2: Tạo lại venv hoàn toàn**

```bash
# Xóa venv cũ
Remove-Item -Recurse -Force .venv

# Tạo venv mới
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"

# Cài dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Bắt đầu nhanh

**Sử dụng cơ bản:**

```bash
python main.py --input ./duong/dan/anh.jpg --out ./thu_muc_ket_qua --langs vi en
```

**Với normalization và tối ưu chất lượng:**

```bash
python main.py --input duong/dan/anh.jpg --langs vi en --threshold adaptive --sharpen
```

**Tiền xử lý nâng cao với tất cả tính năng:**

```bash
python main.py --input duong/dan/anh.jpg --out thu_muc_ket_qua --langs vi en --threshold adaptive --deskew --scale --sharpen --crop --auto-brightness --gamma 0.8 --morphology opening
```

**Xử lý ảnh kém chất lượng:**

```bash
python main.py --input anh_xau.jpg --scale --sharpen --auto-brightness --gamma 0.7 --morphology both --deskew --crop
```

## Tùy chọn dòng lệnh

### Bắt buộc

- `--input`: Đường dẫn đến file ảnh hoặc thư mục chứa ảnh

### Tùy chọn

- `--out`: Thư mục đầu ra (mặc định: "out")
- `--langs`: Mã ngôn ngữ cách nhau bởi dấu cách (mặc định: "vi en")

### Tùy chọn tiền xử lý

- `--threshold`: Phương pháp nhị phân hóa

  - `none`: Không nhị phân hóa
  - `otsu`: Ngưỡng Otsu toàn cục (tốt cho ánh sáng đồng đều)
  - `adaptive`: Ngưỡng thích ứng Gaussian (tốt cho ánh sáng không đều) **[mặc định]**
  - `mean_adaptive`: Ngưỡng thích ứng Mean (tốt cho ảnh nhiễu, xử lý nhanh)
  - `otsu_inv`: Otsu đảo ngược (chữ trắng nền đen)
  - `adaptive_inv`: Adaptive Gaussian đảo ngược
  - `mean_adaptive_inv`: Adaptive Mean đảo ngược

- `--morphology`: Phép biến đổi morphology để làm sạch

  - `none`: Không sử dụng **[mặc định]**
  - `opening`: Loại bỏ nhiễu và tách ký tự dính
  - `closing`: Lấp đầy khoảng trống và nối ký tự vỡ
  - `both`: Áp dụng opening rồi closing

- `--deskew`: Bật tự động sửa nghiêng bằng Hough transform
- `--scale`: Phóng to ảnh lên 300 DPI để OCR chính xác hơn
- `--sharpen`: Áp dụng bộ lọc unsharp mask để làm nét văn bản
- `--crop`: Tự động cắt để loại bỏ viền và tập trung vào vùng nội dung
- `--auto-brightness`: Tự động điều chỉnh độ sáng và tương phản
- `--gamma`: Điều chỉnh gamma correction (0.5-2.0, mặc định 1.0)
- `--no-denoise`: Tắt khử nhiễu (mặc định bật)
- `--no-normalize`: Tắt chuẩn hóa văn bản (mặc định bật)
- `--no-smart-correction`: Tắt sửa lỗi thông minh với Levenshtein (mặc định bật)

### Tùy chọn hiệu suất

- `--cpu`: Bắt buộc dùng CPU (tắt tăng tốc GPU)

## Ví dụ sử dụng

### Các loại tài liệu

**Ảnh chụp điện thoại hoặc có nhiều noise:**

```bash
python main.py --input anh_chup.jpg --scale --sharpen --crop --deskew --threshold mean_adaptive
```

**Scan chất lượng cao:**

```bash
python main.py --input scan.pdf --threshold otsu
```

**Ảnh chụp điện thoại (chất lượng thường):**

```bash
python main.py --input anh_chup.jpg --scale --sharpen --crop --deskew --threshold adaptive
```

**Tài liệu cũ/xuống cấp:**

```bash
python main.py --input tai_lieu_cu.jpg --morphology both --threshold adaptive --deskew
```

**Tài liệu có viền/khung:**

```bash
python main.py --input co_vien.jpg --crop --threshold adaptive
```

**Scan nhiễu:**

```bash
python main.py --input nhieu.png --morphology opening --threshold adaptive
```

**Ảnh tối/kém ánh sáng:**

```bash
python main.py --input anh_toi.jpg --auto-brightness --gamma 0.7 --threshold adaptive
```

**Ảnh quá sáng:**

```bash
python main.py --input anh_sang.jpg --gamma 1.3 --threshold adaptive
```

**Nền tối (đảo ngược):**

```bash
python main.py --input nen_den.jpg --threshold adaptive_inv
```

### Tùy chọn chuẩn hóa văn bản

**Chỉ sử dụng từ điển cơ bản (nhanh hơn):**

```bash
python main.py --input anh.jpg --no-smart-correction
```

**Tắt hoàn toàn chuẩn hóa (OCR thô):**

```bash
python main.py --input anh.jpg --no-normalize
```

**Chuẩn hóa đầy đủ với Levenshtein (mặc định):**

```bash
python main.py --input anh.jpg --langs vi en
```

### Xử lý hàng loạt

**Xử lý toàn bộ thư mục:**

```bash
python main.py --input /duong/dan/thu_muc_anh/ --out ket_qua/ --langs vi --threshold adaptive --deskew
```

**Nhiều ngôn ngữ:**

```bash
python main.py --input tai_lieu.jpg --langs vi en zh --threshold adaptive
```

## Pipeline tiền xử lý

Pipeline tiền xử lý theo thứ tự:

1. **Chuyển sang xám**
2. **Loại bỏ viền & cắt** (nếu `--crop`)
3. **Phóng to lên 300 DPI** (nếu `--scale`)
4. **Chỉnh sáng tự động** (nếu `--auto-brightness`)
5. **Điều chỉnh gamma** (nếu `--gamma` ≠ 1.0)
6. **Khử nhiễu** (trừ khi `--no-denoise`)
7. **Làm nét** (nếu `--sharpen`)
8. **Thẳng hóa** (nếu `--deskew`)
9. **Tăng cường độ tương phản** (CLAHE)
10. **Nhị phân hóa** (threshold)
11. **Phép biến đổi morphology** (nếu chỉ định)
12. **Nhận dạng OCR** (EasyOCR)
13. **Chuẩn hóa văn bản nâng cao**:
    - Unicode NFC normalization
    - Pattern matching cho dấu thanh
    - Dictionary lookup (300+ lỗi phổ biến)
    - Smart correction với Levenshtein distance
    - Filtering noise và special characters

## Ngôn ngữ được hỗ trợ

EasyOCR hỗ trợ 80+ ngôn ngữ. Mã phổ biến:

- `vi`: Tiếng Việt
- `en`: Tiếng Anh
- `zh`: Tiếng Trung
- `ja`: Tiếng Nhật
- `ko`: Tiếng Hàn
- `th`: Tiếng Thái
- `fr`: Tiếng Pháp
- `de`: Tiếng Đức
- `es`: Tiếng Tây Ban Nha

Danh sách đầy đủ: [EasyOCR Supported Languages](https://github.com/JaidedAI/EasyOCR#supported-languages)

## Định dạng ảnh hỗ trợ

- PNG, JPEG, JPG, TIFF, TIF, BMP, WEBP

## Mẹo để có kết quả tốt hơn

### Chất lượng ảnh

- Đảm bảo tối thiểu 200 DPI (dùng `--scale` cho ảnh độ phân giải thấp)
- Độ tương phản tốt giữa chữ và nền
- Tối thiểu độ nghiêng (dùng `--deskew` nếu cần)

### Chọn tiền xử lý

- **Dùng `--crop`** cho ảnh có viền lớn hoặc vùng không liên quan
- **Dùng `--scale`** cho ảnh chụp bằng điện thoại
- **Dùng `--sharpen`** cho văn bản mờ hoặc độ tương phản thấp
- **Dùng `--auto-brightness`** cho ảnh tối hoặc sáng không đều
- **Dùng `--gamma`** để điều chỉnh độ sáng (< 1.0 sáng hơn, > 1.0 tối hơn)
- **Dùng `--deskew`** cho tài liệu xoay
- **Dùng `--morphology opening`** cho ảnh nhiễu
- **Dùng `--morphology closing`** cho ký tự vỡ
- **Dùng `--no-normalize`** nếu muốn giữ nguyên văn bản gốc từ OCR
- **Dùng `--no-smart-correction`** để tắt Levenshtein correction (nhanh hơn)

### Chuẩn hóa văn bản

- **Mặc định**: Bật cả dictionary lookup và Levenshtein smart correction
- **`--no-smart-correction`**: Chỉ dùng dictionary lookup (nhanh hơn)
- **`--no-normalize`**: Tắt hoàn toàn (OCR thô)
- **Hiệu quả**: Smart correction sửa được 70-80% lỗi OCR tiếng Việt phổ biến

### Chọn threshold

- **`adaptive`**: Tốt nhất cho hầu hết tài liệu thực tế, ánh sáng không đều, có shadow/gradient
- **`mean_adaptive`**: Tốt cho ảnh nhiễu, contrast thấp, ảnh chụp điện thoại, xử lý nhanh hơn
- **`otsu`**: Tốt cho scan sạch, ánh sáng đều, bimodal histogram rõ ràng
- **`*_inv`**: Thử nếu threshold thường thất bại (chữ trắng trên nền đen)

#### So sánh chi tiết các phương pháp threshold

| Phương pháp           | Tốt cho                            | Ưu điểm                                      | Nhược điểm                    |
| --------------------- | ---------------------------------- | -------------------------------------------- | ----------------------------- |
| `adaptive` (Gaussian) | Scan documents, ánh sáng không đều | Xử lý tốt shadow/gradient, chất lượng cao    | Chậm hơn, kém với noise nhiều |
| `mean_adaptive`       | Ảnh chụp điện thoại, ảnh nhiễu     | Nhanh, chống noise tốt, text khác kích thước | Kém với gradient phức tạp     |
| `otsu`                | Scan chất lượng cao                | Rất nhanh, tự động tìm threshold tối ưu      | Chỉ tốt với ánh sáng đều      |

### Hiệu suất

- **GPU**: Nhanh hơn đáng kể cho ảnh lớn/hàng loạt
- **CPU**: Dùng `--cpu` nếu GPU gây vấn đề
- **Ngôn ngữ**: Chỉ định ngôn ngữ cần thiết để độ chính xác cao hơn

## Khắc phục sự cố

**Khắc phục sự cố threshold:**

1. **Kết quả OCR kém**: Thử các chế độ threshold khác theo thứ tự:
   - `adaptive` → `mean_adaptive` → `otsu`
   - Nếu vẫn kém, thử chế độ `*_inv`
2. **Ảnh có nhiều noise**: Dùng `mean_adaptive` + `--morphology opening`
3. **Ánh sáng không đều**: Dùng `adaptive` + `--auto-brightness`
4. **Scan chất lượng cao**: Dùng `otsu` cho tốc độ tối ưu
5. **Text mờ/nhạt**: Dùng `mean_adaptive` + `--sharpen`

**Kết quả OCR kém:**

1. Thử các chế độ threshold khác nhau (`--threshold otsu` vs `--threshold adaptive`)
2. Bật tùy chọn tiền xử lý (`--deskew --scale --sharpen`)
3. Điều chỉnh độ sáng (`--auto-brightness --gamma 0.8`)
4. Kiểm tra nếu văn bản bị đảo ngược (thử chế độ `*_inv`)
5. Dùng phép biến đổi morphology cho ảnh nhiễu
6. Tắt normalization nếu gây lỗi (`--no-normalize`)

**Văn bản có lỗi chính tả:**

1. Đảm bảo normalization được bật (mặc định)
2. Kiểm tra ngôn ngữ được chỉ định đúng (`--langs vi en`)
3. Thử tắt smart correction nếu gây lỗi (`--no-smart-correction`)
4. Cải thiện chất lượng ảnh trước khi OCR

**Smart correction không hoạt động:**

1. Kiểm tra Python-Levenshtein đã cài: `pip list | grep Levenshtein`
2. Thử không dùng `--no-smart-correction`
3. Kiểm tra ngôn ngữ: smart correction chỉ áp dụng cho tiếng Việt

**Thiếu dấu tiếng Việt:**

1. Đảm bảo `--langs vi` được bao gồm
2. Kiểm tra encoding UTF-8 của file đầu ra
3. Thử threshold mode khác (`adaptive` vs `otsu`)
4. Bật normalization để tự động sửa dấu

**Xử lý chậm:**

1. Đảm bảo driver GPU được cài đặt
2. Dùng `--cpu` nếu GPU gây vấn đề
3. Giảm số lượng ngôn ngữ chỉ định

**Vấn đề cài đặt:**

1. Cập nhật pip: `pip install --upgrade pip`
2. Cài PyTorch riêng (xem phần cài đặt)
3. Sử dụng môi trường ảo

## Đầu ra

- Tạo một file `.txt` cho mỗi ảnh đầu vào
- Văn bản được sắp xếp theo thứ tự đọc (trên xuống dưới, trái sang phải)
- Mã hóa UTF-8 cho ký tự quốc tế
- Lọc độ tin cậy (loại bỏ phát hiện độ tin cậy thấp)
- Chuẩn hóa Unicode và sửa lỗi OCR phổ biến

## Ví dụ

Đầu vào: `tai_lieu.jpg`
Đầu ra: `out/tai_lieu.txt`

```text
By using NoInfer<T> for the return type of getDefault we prevent the compiler from
inferring the type based on how getDefault is used within petAnimal. This ensures
that getDefault must explicitly return type compatible with T, maintaining type safety
for the overall function.
```

**So sánh trước và sau normalization:**

**Trước (từ OCR thô):**

```text
phưdng pháp dộng vư_t tràn phấy tinh 8ố hóu
```

**Sau (với smart correction):**

```text
phương pháp động vượt tràn phẩy tính số hoá
```

**Ví dụ lỗi tiếng Việt được sửa:**

- `phưdng` → `phương`
- `dộng` → `động`
- `vư_t` → `vượt`
- `phấy` → `phẩy`
- `8ố` → `số`
- `hóu` → `hoá`

**Ví dụ lỗi tiếng Anh được sửa:**

- `returu` → `return`
- `fundamnental` → `fundamental`
- `difterences` → `differences`
- `sectioll` → `section`

## Giấy phép

MIT License - tự do sử dụng và chỉnh sửa.

## Đóng góp

1. Fork repository
2. Tạo feature branch
3. Gửi pull request

## Lịch sử thay đổi

### v1.0.0

- Phiên bản đầu tiên với pipeline OCR cơ bản
- Hỗ trợ threshold adaptive và Otsu
- Khả năng thẳng hóa

### v1.1.0

- Thêm phép biến đổi morphology
- Chế độ threshold đảo ngược
- Tùy chọn phóng to và làm nét
- Chức năng tự động cắt
- Pipeline tiền xử lý toàn diện

### v1.2.0

- Thêm chỉnh sáng tự động và gamma correction
- Hệ thống chuẩn hóa văn bản với từ điển sửa lỗi
- Sửa lỗi OCR phổ biến cho tiếng Việt và tiếng Anh
- Chuẩn hóa Unicode NFC
- Cải thiện pipeline xử lý văn bản đầu ra
