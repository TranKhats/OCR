import cv2, json
import easyocr
import numpy as np
from pipeline.preprocess import ocr_with_preprocessing
from pipeline.detect import detect

    # Load OCR model với tham số tốt hơn
reader = easyocr.Reader(['vi', 'en'], gpu=False, 
                        model_storage_directory='./models',
                        download_enabled=True)

# Sắp xếp box thành dòng (giản lược)
def group_boxes_into_lines(boxes, y_threshold=20):
    boxes = sorted(boxes, key=lambda b: b[1])
    lines, current_line = [], [boxes[0]]
    for box in boxes[1:]:
        if abs(box[1] - current_line[-1][1]) < y_threshold:
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]
    lines.append(sorted(current_line, key=lambda b: b[0]))
    return lines



# Lưu kết quả vào file
def save_result(all_text):
    """Lưu kết quả OCR vào file."""
    with open("ocr_results.txt", "w", encoding="utf-8") as f:
        f.write("=== KẾT QUẢ OCR ===\n")
        for i, text in enumerate(all_text):
            f.write(f"Line {i+1}: {text}\n")

def recognize_text(lines,img_pp):
    all_text = []
    for i, line in enumerate(lines):
        words = []
        print(f"\n--- Line {i+1} ---")
        
        for j, (x1, y1, x2, y2) in enumerate(line):
            # Cắt vùng text
            crop = img_pp[y1:y2, x1:x2]
            
            # OCR với tham số tốt hơn (tăng mag_ratio, canvas_size, giảm ngưỡng)
            result = reader.readtext(
                crop, 
                detail=0,  # chỉ lấy text
                paragraph=False,  # không gộp thành paragraph
                height_ths=0.5,
                width_ths=0.5,
                text_threshold=0.5,
                link_threshold=0.3,
                low_text=0.2,
                canvas_size=4096,
                mag_ratio=2.0
            )
            
            if result:
                recognized_text = result[0]
                words.append(recognized_text)
                print(f"  Word {j+1}: '{recognized_text}' (bbox: {x1},{y1},{x2},{y2})")
            else:
                print(f"  Word {j+1}: Không nhận dạng được (bbox: {x1},{y1},{x2},{y2})")
        
        line_text = " ".join(words)
        all_text.append(line_text)
        print(f"  Line text: '{line_text}'")

    return all_text

def main(image_path=None):


    # Load ảnh gốc
    imgPath = "data/02_detect/image.png"

    # img_pp = ocr_with_preprocessing(imgPath, do_binarize=False)  # Sử dụng adaptive binarization
    data = detect(imgPath)  # Sử dụng hàm detect để lấy ảnh đã preprocess
    image_preprocessed_path = data['pp_path']  # Lấy ảnh đã preprocess từ kết quả detect
    print(f"Đã lưu ảnh đã preprocess vào: {image_preprocessed_path}")
    img_pp = cv2.imread(image_preprocessed_path)
    # Load detect result
    with open(data['json_path'], "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = data["boxes"]
    lines = group_boxes_into_lines(boxes)
    all_text = recognize_text(lines, img_pp)
    save_result(all_text)

if __name__ == "__main__":
    main()