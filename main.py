from __future__ import annotations

import argparse
import os
from typing import List

import easyocr

from src.ocr_pipeline import process_and_write


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OCR scanned documents to TXT (EasyOCR)")
    p.add_argument("--input", required=True, default="img2.png", help="Image file or folder")
    p.add_argument("--out", default="out", help="Output folder")
    p.add_argument("--langs", nargs="+", default=["vi", "en"], help="Languages, e.g., vi en")
    p.add_argument("--threshold", default="adaptive", 
                   choices=["none", "otsu", "adaptive", "mean_adaptive", "otsu_inv", "adaptive_inv", "mean_adaptive_inv"],
                   help="Threshold mode")
    p.add_argument("--morphology", default="none",
                   choices=["none", "opening", "closing", "both"],
                   help="Morphological operations for cleaning")
    p.add_argument("--deskew", action="store_true", help="Enable deskewing")
    p.add_argument("--scale", action="store_true", help="Scale image to 300 DPI")
    p.add_argument("--sharpen", action="store_true", help="Apply sharpening filter")
    p.add_argument("--crop", action="store_true", help="Auto-crop to content area")
    p.add_argument("--auto-brightness", action="store_true", help="Auto adjust brightness/contrast")
    p.add_argument("--gamma", type=float, default=1.0, help="Gamma correction (0.5-2.0)")
    p.add_argument("--no-denoise", action="store_true", help="Disable denoising")
    p.add_argument("--no-normalize", action="store_true", help="Disable text normalization")
    p.add_argument("--no-smart-correction", action="store_true", help="Disable Levenshtein-based spell checking")
    p.add_argument("--cpu", action="store_true", help="Force CPU (disable GPU)")
    return p.parse_args()


def iter_images(path: str):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    yield os.path.join(root, f)
    else:
        yield path


def main():
    args = parse_args()
    langs: List[str] = args.langs
    gpu = not args.cpu
    reader = easyocr.Reader(langs, gpu=gpu)
    os.makedirs(args.out, exist_ok=True)
    
    for img_path in iter_images(args.input):
        out_txt = process_and_write(
            img_path, args.out, langs,
            reader=reader, 
            gpu=gpu,
            thr_mode=args.threshold,
            morphology=args.morphology,
            do_denoise=not args.no_denoise,
            do_deskew=args.deskew,
            do_scale=args.scale,
            do_sharpen=args.sharpen,
            do_crop=args.crop,
            do_auto_brightness=args.auto_brightness,
            gamma=args.gamma,
            do_normalize=not args.no_normalize,
            use_smart_correction=not args.no_smart_correction
        )
        print(f"Wrote: {out_txt}")


if __name__ == "__main__":
    main()
