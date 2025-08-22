from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import cv2
import easyocr
import numpy as np

from .preprocess import preprocess_image

try:
    from .text_utils import post_process_lines
    HAS_TEXT_UTILS = True
except ImportError:
    HAS_TEXT_UTILS = False


def _reading_order_key(box: np.ndarray | list) -> Tuple[int, int]:
    # box: 4x2 points; compute top-left
    box = np.asarray(box, dtype=np.float32)
    if box.ndim != 2 or box.shape[1] != 2:
        box = box.reshape(-1, 2)
    ys = box[:, 1]
    xs = box[:, 0]
    y_top = int(np.min(ys))
    x_left = int(np.min(xs))
    # Cluster lines by y with tolerance
    return (y_top // 10, x_left)


def _group_lines(results: List[Tuple[np.ndarray, str, float]]) -> List[List[Tuple[np.ndarray, str, float]]]:
    # Sort by reading order, then group by line bucket (y // 10)
    results_sorted = sorted(results, key=lambda r: _reading_order_key(r[0]))
    lines: List[List[Tuple[np.ndarray, str, float]]] = []
    current_bucket = None
    for item in results_sorted:
        bucket = _reading_order_key(item[0])[0]
        if current_bucket is None or bucket != current_bucket:
            lines.append([item])
            current_bucket = bucket
        else:
            lines[-1].append(item)
    # Within each line, sort left-to-right
    for i in range(len(lines)):
        lines[i] = sorted(lines[i], key=lambda r: int(np.min(r[0][:, 0])))
    return lines


def recognize_image(img_path: str,
                    langs: List[str],
                    *,
                    reader: easyocr.Reader | None = None,
                    gpu: bool = True,
                    **preprocess_kwargs) -> Tuple[np.ndarray, List[Tuple[np.ndarray, str, float]]]:
    reader = reader or easyocr.Reader(langs, gpu=gpu)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    pre = preprocess_image(img, **preprocess_kwargs)
    raw_results = reader.readtext(pre)
    # Normalize boxes to ndarray
    norm_results: List[Tuple[np.ndarray, str, float]] = []
    for box, text, conf in raw_results:
        b = np.asarray(box, dtype=np.float32)
        if b.ndim != 2 or b.shape[-1] != 2:
            b = b.reshape(-1, 2)
        norm_results.append((b, text, float(conf)))
    return pre, norm_results


def results_to_lines(results: List[Tuple[np.ndarray, str, float]], 
                     do_normalize: bool = True, 
                     language: str = 'mixed',
                     use_smart_correction: bool = True) -> List[str]:
    lines = _group_lines(results)
    out_lines: List[str] = []
    for group in lines:
        texts = [t for _, t, conf in group if t and conf > 0.2]
        line = " ".join(texts).strip()
        if line:
            out_lines.append(line)
    
    # Apply post-processing and normalization
    if do_normalize and HAS_TEXT_UTILS:
        return post_process_lines(out_lines, language, use_smart_correction=use_smart_correction)
    else:
        return out_lines


def process_and_write(img_path: str, out_dir: str, langs: List[str], *, 
                      reader: easyocr.Reader | None = None,
                      gpu: bool = True, 
                      thr_mode: str = "adaptive", 
                      morphology: str = "none",
                      do_denoise: bool = True, 
                      do_deskew: bool = False,
                      do_scale: bool = False,
                      do_sharpen: bool = False,
                      do_crop: bool = False,
                      do_auto_brightness: bool = False,
                      gamma: float = 1.0,
                      do_normalize: bool = True,
                      use_smart_correction: bool = True) -> str:
    
    # Determine language for normalization
    language = 'mixed'
    if len(langs) == 1:
        if 'vi' in langs:
            language = 'vi'
        elif 'en' in langs:
            language = 'en'
    
    # Prepare preprocessing parameters
    preprocess_kwargs = {
        'do_denoise': do_denoise,
        'do_deskew': do_deskew,
        'do_scale': do_scale,
        'do_sharpen': do_sharpen,
        'do_crop': do_crop,
        'do_auto_brightness': do_auto_brightness,
        'gamma': gamma,
        'thr_mode': thr_mode,
        'morphology': morphology
    }
    
    _, results = recognize_image(img_path, langs, reader=reader, gpu=gpu, **preprocess_kwargs)
    lines = results_to_lines(results, do_normalize, language, use_smart_correction)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_txt = os.path.join(out_dir, f"{base}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_txt
