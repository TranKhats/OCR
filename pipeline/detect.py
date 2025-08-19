# pipeline/02_detect_easyocr.py
import os, json, argparse
from typing import Optional, Dict, Any
import fnmatch
import cv2, easyocr, numpy as np
from typing import List, Tuple
# from image_preprocess.image_preprocessing import (ImagePreprocessor,
#                                                   ocr_with_preprocessing)
#—— GỌI STEP 1 (PREPROCESS) bằng ocr_with_preprocessing ——
try:
    from pipeline.preprocess import ocr_with_preprocessing
    print("Sử dụng ocr_with_preprocessing từ pipeline.image_preprocess.image_preprocessing")
except Exception as e:
    raise e
    print("Warning: Không import được ocr_with_preprocessing, sẽ dùng ảnh gốc.")
    ocr_with_preprocessing = None

# -------------- utils --------------
def poly_to_xyxy(poly) -> Tuple[int,int,int,int]:
    p = np.asarray(poly, dtype=np.float32)
    xs, ys = p[:,0], p[:,1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def nms_xyxy(boxes: List[List[int]], scores: List[float], iou_thr=0.3):
    if not boxes: return [], []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    idxs = s.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        xx1 = np.maximum(b[i,0], b[idxs,0])
        yy1 = np.maximum(b[i,1], b[idxs,1])
        xx2 = np.minimum(b[i,2], b[idxs,2])
        yy2 = np.minimum(b[i,3], b[idxs,3])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        area_i = (b[i,2]-b[i,0])*(b[i,3]-b[i,1])
        area_j = (b[idxs,2]-b[idxs,0])*(b[idxs,3]-b[idxs,1])
        iou = inter / (area_i + area_j - inter + 1e-9)
        idxs = idxs[1:][iou[1:] <= iou_thr]
    return [boxes[i] for i in keep], [scores[i] for i in keep]

def draw_boxes(img, boxes, color=(0,255,0)):
    vis = img.copy()
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
    return vis

# —— group theo dòng (tuỳ chọn) —
def group_to_lines(boxes: List[List[int]], max_dy_ratio=0.6, max_gap=30):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    lines = []
    for x1,y1,x2,y2 in boxes:
        cy = (y1+y2)//2
        placed = False
        for L in lines:
            lx1,ly1,lx2,ly2 = L
            lcy = (ly1+ly2)//2
            lh  = max(1, ly2-ly1)
            if abs(cy - lcy) <= max_dy_ratio*lh and x1 - lx2 <= max_gap:
                L[0] = min(lx1, x1); L[1] = min(ly1, y1)
                L[2] = max(lx2, x2); L[3] = max(ly2, y2)
                placed = True; break
        if not placed:
            lines.append([x1,y1,x2,y2])
    lines.sort(key=lambda b: (b[1], b[0]))
    return lines

def _load_yaml_config(default_path: str = "configs/detect.yaml") -> Dict[str, Any]:
    cfg_path = os.environ.get("DETECT_CONFIG", default_path)
    if not os.path.isfile(cfg_path):
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        print("Warning: PyYAML chưa cài đặt, bỏ qua file cấu hình.")
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as e:
        print(f"Warning: Không đọc được config {cfg_path}: {e}")
        return {}


def _apply_config_defaults(ap: argparse.ArgumentParser, cfg: Dict[str, Any]) -> None:
    """Thiết lập giá trị mặc định từ YAML cho các tham số CLI nếu có."""
    if not cfg:
        return
    # Map khóa YAML -> tên đối số
    defaults = {}
    # Detect params
    for k in [
        "langs","gpu","mode","text_thresh","low_text","link_thresh","mag_ratio",
        "min_area","min_h","max_h","min_ar","max_ar","nms","iou_nms","outdir"
    ]:
        if k in cfg:
            defaults[k] = cfg[k]
    # Preprocess params
    for k in ["pp_max_side","pp_no_deskew","pp_binarize","pp_adaptive","pp_auto"]:
        if k in cfg:
            defaults[k] = cfg[k]
    if defaults:
        ap.set_defaults(**defaults)


# Áp dụng override theo tên file từ YAML
def _apply_image_overrides(cfg: Dict[str, Any], image_path: str, args: argparse.Namespace) -> argparse.Namespace:
    overrides = cfg.get("overrides") if isinstance(cfg, dict) else None
    if not overrides:
        return args
    base = os.path.basename(image_path)
    for rule in overrides:
        if not isinstance(rule, dict):
            continue
        pattern = rule.get("pattern")
        if not pattern:
            continue
        if fnmatch.fnmatch(base, str(pattern)):
            for k, v in rule.items():
                if k == "pattern":
                    continue
                if hasattr(args, k):
                    setattr(args, k, v)
    return args


# -------------- detect --------------
def detect_easyocr(
    img_bgr,
    langs=('vi','en'),
    gpu=False,
    text_threshold=0.7, low_text=0.4, link_threshold=0.4, mag_ratio=1.5,
    min_area=80, min_h=10, max_h=1_000, min_ar=0.1, max_ar=20.0,
    do_nms=True, iou_nms=0.25,
    mode="word"   # "word" | "line"
):
    reader = easyocr.Reader(list(langs), gpu=gpu)

    results = reader.readtext(
        img_bgr,
        detail=1,                # [(poly, text, conf), ...]
        paragraph=False,
        text_threshold=text_threshold,
        low_text=low_text,
        link_threshold=link_threshold,
        mag_ratio=mag_ratio
    )

    boxes, scores = [], []
    for poly, _txt, sc in results:
        x1,y1,x2,y2 = poly_to_xyxy(poly)
        w,h = x2-x1, y2-y1
        if w*h < min_area: continue
        if not (min_h <= h <= max_h): continue
        ar = w / max(1,h)
        if not (min_ar <= ar <= max_ar): continue
        boxes.append([x1,y1,x2,y2]); scores.append(float(sc))

    if do_nms:
        boxes, _ = nms_xyxy(boxes, scores, iou_thr=iou_nms)

    boxes.sort(key=lambda b: (b[1], b[0]))

    if mode == "line":
        boxes = group_to_lines(boxes)

    return boxes

# -------------- CLI --------------
def main():
    ap = argparse.ArgumentParser("Step 2: EasyOCR Detect (auto call Step 1 preprocess)")
    # Đọc cấu hình YAML trước để áp vào mặc định
    cfg = _load_yaml_config()
    _apply_config_defaults(ap, cfg)
    ap.add_argument("image", help="Ảnh đầu vào (ảnh gốc)")
    ap.add_argument("--langs", default="vi,en", help="Ngôn ngữ, ví dụ: vi,en")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--mode", choices=["word","line"], default="word",
                    help="word: box theo từ; line: gộp theo dòng")
    ap.add_argument("--outdir", default="runs/detect_easyocr")


    # tham số tinh chỉnh EasyOCR detect
    ap.add_argument("--text_thresh", type=float, default=0.7)
    ap.add_argument("--low_text", type=float, default=0.4)
    ap.add_argument("--link_thresh", type=float, default=0.4)
    ap.add_argument("--mag_ratio", type=float, default=1.5)
    ap.add_argument("--min_area", type=int, default=80)
    ap.add_argument("--min_h", type=int, default=10)
    ap.add_argument("--max_h", type=int, default=1000)
    ap.add_argument("--min_ar", type=float, default=0.1)
    ap.add_argument("--max_ar", type=float, default=20.0)
    ap.add_argument("--nms", action="store_true")
    ap.add_argument("--iou_nms", type=float, default=0.25)

    # Không cần tham số preprocess — dùng ocr_with_preprocessing cố định

    args = ap.parse_args()
    # Áp dụng override theo tên ảnh (nếu có trong YAML)
    args = _apply_image_overrides(cfg, args.image, args)

    img0 = cv2.imread(args.image)
    if img0 is None:
        raise FileNotFoundError(args.image)

    os.makedirs(args.outdir, exist_ok=True)

    # —— gọi Step 1 nếu dùng được ocr_with_preprocessing, ngược lại dùng ảnh gốc
    if ocr_with_preprocessing is not None:
        img_pp = ocr_with_preprocessing(
            args.image
        )

        img_infer = img_pp
        # Lưu ảnh sau preprocess
        pp_path = os.path.join(args.outdir, f"pp_{os.path.basename(args.image)}")
        cv2.imwrite(pp_path, img_infer)
        print(f"Kết thúc bước 1: Ảnh sau khi preprocessing được lưu ở {pp_path}")
        preprocess_used = True
    else:
        img_infer = img0
        preprocess_used = False
    

    langs = tuple(s.strip() for s in args.langs.split(",") if s.strip())

    boxes = detect_easyocr(
        img_infer, langs=langs, gpu=args.gpu,
        text_threshold=args.text_thresh, low_text=args.low_text,
        link_threshold=args.link_thresh, mag_ratio=args.mag_ratio,
        min_area=args.min_area, min_h=args.min_h, max_h=args.max_h,
        min_ar=args.min_ar, max_ar=args.max_ar,
        do_nms=args.nms, iou_nms=args.iou_nms,
        mode=args.mode
    )

    vis = draw_boxes(img0, boxes)
    vis_path = os.path.join(args.outdir, os.path.basename(args.image))
    cv2.imwrite(vis_path, vis)

    js_path = vis_path.rsplit(".",1)[0] + ".json"
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump({
            "file": args.image,
            "mode": args.mode,
            "preprocess_used": preprocess_used,
            "boxes": boxes
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] {len(boxes)} boxes")
    print(f"  • vis : {vis_path}")
    print(f"  • json: {js_path}")

if __name__ == "__main__":
    main()
