import cv2
from pipeline.image_preprocess.image_preprocessing import (preprocess)


def ocr_with_preprocessing(image_path):
    """OCR với tiền xử lý tối ưu"""
    
    # Bước 1: Tiền xử lý tạo binary
    binary_image = preprocess(
        image_path
    )
    return binary_image
