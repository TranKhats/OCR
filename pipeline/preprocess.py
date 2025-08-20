import cv2
from pipeline.image_preprocess.image_preprocessing import (preprocess)


def ocr_with_preprocessing(image_path, do_binarize=True):
    """OCR với tiền xử lý tối ưu"""
    
    # Bước 1: Tiền xử lý tạo binary
    binary_image = preprocess(
        image_path,
        do_binarize=False,  # Sử dụng adaptive binarization
    )
    return binary_image
