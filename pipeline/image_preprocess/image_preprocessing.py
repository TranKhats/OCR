import os
from typing import Optional, Union

import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    """
    Đọc ảnh BGR từ đường dẫn.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Chuyển ảnh sang grayscale (nếu chưa).
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_to_max_side(image: np.ndarray, max_side: Optional[int] = 1600) -> np.ndarray:
    """
    Resize sao cho cạnh dài nhất = max_side (giữ tỉ lệ). Bỏ qua nếu max_side=None.
    """
    if not max_side:
        return image
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return image
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def deskew(image: np.ndarray, max_angle: float = 20.0) -> np.ndarray:
    """
    Hiệu chỉnh nghiêng bằng cách ước lượng góc xoay từ minAreaRect trên mask nhị phân.
    Nếu góc vượt quá max_angle (độ), bỏ qua.
    """
    gray = to_grayscale(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Đảo mask nếu nền sáng chữ tối
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 10:
        return image

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # OpenCV trả angle trong [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > max_angle:
        return image

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def binarize_otsu(image: np.ndarray) -> np.ndarray:
    """
    Nhị phân hóa bằng Otsu. Trả về ảnh 1 kênh (uint8, {0,255}).
    """
    gray = to_grayscale(image)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bin_img


def binarize_adaptive(image: np.ndarray) -> np.ndarray:
    """
    Nhị phân hóa bằng Adaptive Threshold (Gaussian). Trả về ảnh 1 kênh.
    Hữu ích khi nền không đồng đều sáng tối.
    """
    gray = to_grayscale(image)
    bin_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,  # blockSize (nên là số lẻ, ví dụ 35, 51,...)
        11   # C (hệ số trừ đi từ trung bình)
    )
    return bin_img


def preprocess(
    image_or_path: Union[str, np.ndarray],
    output_path: Optional[str] = None,
    max_side: Optional[int] = 1600,
    do_deskew: bool = True,
    do_binarize: bool = True,
    use_adaptive: bool = False,
) -> np.ndarray:
    """
    Pipeline preprocess:
    1) (Optional) Resize cạnh dài nhất về max_side
    2) (Optional) Deskew
    3) (Optional) Binarize (Otsu hoặc Adaptive)

    Trả về ảnh đã chuẩn hóa (grayscale/binary) sẵn sàng cho bước Detect.
    """
    img = read_image(image_or_path) if isinstance(image_or_path, str) else image_or_path
    img = resize_to_max_side(img, max_side=max_side)
    if do_deskew:
        img = deskew(img)
    if do_binarize:
        if use_adaptive:
            img = binarize_adaptive(img)
        else:
            img = binarize_otsu(img)
    else:
        img = to_grayscale(img)
    if output_path:
        _imwrite_safe(output_path, img)
    return img


def call_preprocess(
    image_or_path: Union[str, np.ndarray],
    max_side: Optional[int] = 1600,
    do_deskew: bool = True,
    do_binarize: bool = True,
    use_adaptive: bool = False,
) -> np.ndarray:
    """
    Hàm wrapper để gọi preprocess từ module khác (ví dụ detect.py).
    """
    return preprocess(
        image_or_path,
        max_side=max_side,
        do_deskew=do_deskew,
        do_binarize=do_binarize,
        use_adaptive=use_adaptive,
    )


def _imwrite_safe(path: str, image: np.ndarray) -> None:
    """
    Ghi ảnh, tự tạo thư mục nếu chưa tồn tại.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = cv2.imwrite(path, image)
    if not ok:
        raise IOError(f"Không thể ghi ảnh ra: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bước 1: Preprocess ảnh cho OCR.")
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh đầu vào.")
    parser.add_argument(
        "--out",
        required=False,
        default="outputs/preprocessed.png",
        help="Đường dẫn lưu ảnh sau preprocess.",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=1600,
        help="Resize cạnh dài nhất về giá trị này (<=0 để bỏ qua).",
    )
    parser.add_argument("--no_deskew", action="store_true", help="Tắt bước deskew.")
    parser.add_argument(
        "--no_binarize", action="store_true", help="Tắt bước binarize (chỉ grayscale)."
    )
    parser.add_argument(
        "--adaptive", action="store_true", help="Dùng adaptive threshold thay vì Otsu."
    )
    args = parser.parse_args()

    max_side = args.max_side if args.max_side and args.max_side > 0 else None
    out = preprocess(
        args.image,
        max_side=max_side,
        do_deskew=not args.no_deskew,
        do_binarize=not args.no_binarize,
        use_adaptive=args.adaptive,
    )
    _imwrite_safe(args.out, out)
    print(f"Đã lưu ảnh preprocess: {args.out}")
