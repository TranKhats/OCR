from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(img: np.ndarray, strength: int = 5) -> np.ndarray:
    """Fast denoise for grayscale. Adjust strength for noise level."""
    if img.size == 0:
        return img
    return cv2.fastNlMeansDenoising(img, h=strength)


def scale_image(img: np.ndarray, target_dpi: int = 300, current_dpi: int = 72) -> np.ndarray:
    """
    Scale image to achieve target DPI for better OCR accuracy.
    
    Args:
        img: Input image
        target_dpi: Target DPI (default 300)
        current_dpi: Estimated current DPI (default 72 for screen)
    """
    if img.size == 0:
        return img
    
    scale_factor = target_dpi / current_dpi
    
    # Only upscale if needed and reasonable
    if scale_factor > 1.0 and scale_factor <= 4.0:
        h, w = img.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Use INTER_CUBIC for upscaling
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return scaled
    
    return img


def sharpen_image(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply unsharp mask to sharpen text.
    
    Args:
        img: Input grayscale image
        strength: Sharpening strength (0.5-2.0, default 1.0)
    """
    if img.size == 0:
        return img
    
    # Create unsharp mask
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    mask = cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)
    
    return np.clip(mask, 0, 255).astype(np.uint8)


def remove_borders(img: np.ndarray, border_threshold: int = 10) -> np.ndarray:
    """
    Remove black/white borders around the document.
    
    Args:
        img: Input grayscale image
        border_threshold: Minimum border width to remove
    """
    if img.size == 0:
        return img
    
    h, w = img.shape[:2]
    
    # Create binary image to find content area
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find content by looking for non-white pixels
    # Invert if background is black
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    
    # Find contours to get main content area
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    # Get bounding box of largest contour (main content)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w_content, h_content = cv2.boundingRect(largest_contour)
    
    # Add small padding
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    x2 = min(w, x + w_content + 2 * padding)
    y2 = min(h, y + h_content + 2 * padding)
    
    # Only crop if we're removing significant borders
    if (x > border_threshold or y > border_threshold or 
        (w - x2) > border_threshold or (h - y2) > border_threshold):
        return img[y:y2, x:x2]
    
    return img


def auto_crop_content(img: np.ndarray) -> np.ndarray:
    """
    Auto-crop to text content area using projection profiles.
    """
    if img.size == 0:
        return img
    
    # Get binary image
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if background is black
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    
    # Get projection profiles
    h_profile = np.sum(binary == 0, axis=1)  # Black pixels per row
    v_profile = np.sum(binary == 0, axis=0)  # Black pixels per column
    
    # Find content boundaries
    h_nonzero = np.where(h_profile > 0)[0]
    v_nonzero = np.where(v_profile > 0)[0]
    
    if len(h_nonzero) == 0 or len(v_nonzero) == 0:
        return img
    
    # Get boundaries with small padding
    y1 = max(0, h_nonzero[0] - 5)
    y2 = min(img.shape[0], h_nonzero[-1] + 5)
    x1 = max(0, v_nonzero[0] - 5)
    x2 = min(img.shape[1], v_nonzero[-1] + 5)
    
    return img[y1:y2, x1:x2]


def enhance_contrast(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE for better contrast."""
    if img.size == 0:
        return img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def threshold(img: np.ndarray, mode: str = "adaptive") -> np.ndarray:
    """Apply thresholding with support for inverse modes."""
    if img.size == 0:
        return img
    
    mode = (mode or "").lower()
    if mode == "none":
        return img
    
    # Check for inverse modes
    invert = mode.endswith("_inv")
    base_mode = mode.replace("_inv", "")
    
    if base_mode == "otsu":
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, th = cv2.threshold(img, 0, 255, thresh_type + cv2.THRESH_OTSU)
        return th
    
    elif base_mode == "mean_adaptive":
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresh_type, 35, 15)
        return th
    
    # adaptive/gaussian_adaptive (default)
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               thresh_type, 35, 15)
    return th


def _skew_angle_hough(binary: np.ndarray) -> float:
    """Estimate skew angle using Hough transform."""
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return 0.0
    
    angles = []
    for rho_theta in lines[:100]:  # Reduce from 200 to 100 for speed
        rho, theta = rho_theta[0]
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)
    
    if not angles:
        return 0.0
    return float(np.median(angles))


def _skew_angle_minrect(binary: np.ndarray) -> float:
    """Backup method: estimate skew using minimum area rectangle."""
    coords = np.column_stack(np.where(binary > 0))
    if coords.size < 10:  # Need minimum points
        return 0.0
    
    # Convert (y,x) to (x,y) for minAreaRect
    coords_xy = coords[:, ::-1].astype(np.float32)
    rect = cv2.minAreaRect(coords_xy)
    angle = rect[-1]
    
    # Normalize angle to [-45, 45]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    return -angle  # Negative for correction


def deskew(img_gray: np.ndarray) -> np.ndarray:
    """Deskew image using Hough lines with minAreaRect fallback."""
    if img_gray.size == 0:
        return img_gray
    
    # Get binary for angle detection
    _, bin_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Try Hough method first
    angle = _skew_angle_hough(bin_img)
    
    # Fallback to minAreaRect if Hough fails
    if abs(angle) < 0.1:
        angle = _skew_angle_minrect(bin_img)
    
    # Skip rotation if angle is too small
    if abs(angle) < 0.5:
        return img_gray
    
    # Rotate image
    h, w = img_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), 
                            flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_REPLICATE)
    return rotated


def morphology_clean(img_binary: np.ndarray, operation: str = "opening") -> np.ndarray:
    """Apply morphological operations to clean binary image."""
    if img_binary.size == 0 or operation == "none":
        return img_binary
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    if operation == "opening":
        return cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    elif operation == "closing":
        return cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    elif operation == "both":
        opened = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return img_binary


def auto_brightness_contrast(img: np.ndarray, clip_hist_percent: float = 1.0) -> np.ndarray:
    """
    Automatically adjust brightness and contrast.
    
    Args:
        img: Input grayscale image
        clip_hist_percent: Percentage of histogram to clip
    """
    if img.size == 0:
        return img
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from histogram
    accumulator = [hist[0]]
    for i in range(1, hist_size):
        accumulator.append(accumulator[i - 1] + hist[i])
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)  # make percent as threshold
    clip_hist_percent /= 2.0  # divide by 2 to apply to both sides
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    # Apply brightness and contrast adjustment
    auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return auto_result

def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to adjust brightness.
    
    Args:
        img: Input grayscale image  
        gamma: Gamma value (< 1 = brighter, > 1 = darker)
    """
    if img.size == 0:
        return img
    
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction
    return cv2.LUT(img, table)

def preprocess_image(img_bgr: np.ndarray,
                     do_denoise: bool = True,
                     do_deskew: bool = False,
                     do_scale: bool = False,
                     do_sharpen: bool = False,
                     do_crop: bool = False,
                     do_auto_brightness: bool = False,
                     gamma: float = 1.0,
                     thr_mode: str = "adaptive",
                     morphology: str = "none") -> np.ndarray:
    """
    Preprocess image for OCR with comprehensive pipeline.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image is empty or None")
    
    # Convert to grayscale
    gray = to_gray(img_bgr)
    
    # Step 1: Remove borders and crop (early to reduce processing time)
    if do_crop:
        gray = remove_borders(gray)
        gray = auto_crop_content(gray)
    
    # Step 2: Scale image for better DPI
    if do_scale:
        gray = scale_image(gray, target_dpi=300, current_dpi=72)
    
    # Step 3: Auto brightness/contrast (before denoise)
    if do_auto_brightness:
        gray = auto_brightness_contrast(gray)
    
    # Step 4: Gamma correction
    if gamma != 1.0:
        gray = gamma_correction(gray, gamma)
    
    # Step 5: Denoise (if enabled)
    if do_denoise:
        gray = denoise(gray)
    
    # Step 6: Sharpen (if enabled, after denoise to avoid amplifying noise)
    if do_sharpen:
        gray = sharpen_image(gray, strength=1.0)
    
    # Step 7: Deskew (before contrast enhancement to avoid artifacts)
    if do_deskew:
        gray = deskew(gray)
    
    # Step 8: Enhance contrast
    gray = enhance_contrast(gray)
    
    # Step 9: Threshold
    binary = threshold(gray, thr_mode)
    
    # Step 10: Morphological operations (optional)
    if morphology != "none":
        binary = morphology_clean(binary, morphology)
    
    return binary