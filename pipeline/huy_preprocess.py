import cv2
from image_preprocess.image_preprocessing import (ImagePreprocessor,
                                                  preprocess_image_quick)


def ocr_with_preprocessing(image_path):
    """OCR với tiền xử lý tối ưu"""
    
    # Bước 1: Tiền xử lý tạo binary
    binary_image = preprocess_image_quick(
        input_path=image_path,
        output_path=None,  # Không lưu file
        method='text_enhanced'
    )
    
    # Bước 2: OCR trực tiếp từ binary array
    # text = pytesseract.image_to_string(
    #     binary_image, 
    #     lang='vie+eng',
    #     config='--ps
