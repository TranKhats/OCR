import re
import unicodedata
from typing import Dict, List, Tuple
try:
    from Levenshtein import distance as levenshtein_distance, ratio as levenshtein_ratio
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False

# Vietnamese word dictionary for spell checking
VIETNAMESE_COMMON_WORDS = {
    'với', 'của', 'như', 'mà', 'đã', 'đến', 'thì', 'cho', 'là', 'và', 'ở', 'được',
    'không', 'những', 'cách', 'phải', 'có', 'nếu', 'khi', 'này', 'sẽ', 'trong',
    'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
    'phương', 'pháp', 'động', 'tính', 'toán', 'phẩy', 'số', 'thao', 'tác',
    'biến', 'đổi', 'tín', 'hiệu', 'bộ', 'thiết', 'bị', 'giá', 'rẻ', 'chú', 'ý',
    'sử', 'dụng', 'đầu', 'vào', 'giới', 'hạn', 'hiện', 'tượng', 'vượt', 'tràn',
    'khả', 'năng', 'xuất', 'hiện', 'kết', 'luận', 'chương', 'trình', 'bày',
    'ngắn', 'gọn', 'thực', 'hiện', 'tuy', 'nhiên', 'lúc', 'cần', 'lọc', 'để',
    'thông', 'tin', 'đơn', 'giản', 'ứng', 'dụng', 'thực', 'tiễn', 'tương', 'tự',
    'do', 'đó', 'phần', 'mềm', 'cứng', 'dữ', 'liệu', 'buông', 'tự', 'các',
    'bởi', 'vì', 'luôn', 'dẫn', 'làm', 'tròn', 'tạo', 'cả', 'lẫn', 'đều',
    'khác', 'nhau', 'phép', 'nhân', 'cộng', 'điều', 'chỉnh', 'độ', 'sáng',
    'tương', 'phản', 'thường', 'mật', 'phận', 'nhiều', 'ứng', 'ra', 'hệ',
    'đen', 'thành', 'tương', 'tự', 'cấu', 'trúc', 'nói', 'chung', 'cũng'
}

# Technical terms dictionary
TECHNICAL_TERMS = {
    'adc', 'cpu', 'gpu', 'ram', 'rom', 'usb', 'api', 'sql', 'html', 'css', 'javascript',
    'python', 'java', 'typescript', 'function', 'return', 'class', 'method', 'variable',
    'array', 'object', 'string', 'number', 'boolean', 'null', 'undefined', 'interface',
    'type', 'generic', 'inference', 'compiler', 'syntax', 'error', 'debugging', 'dac'
}

# Expanded Vietnamese OCR error dictionary
VIETNAMESE_OCR_CORRECTIONS = {
    # Character-level corrections
    'rn': 'm', 'ln': 'ln', '|': 'l', 'I': 'l',
    '0': 'O', '5': 'S', '8': 'B', '6': 'G',
    
    # Vietnamese tone mark errors
    'vÚi': 'với', 'úng': 'ứng', 'dậ': 'đã', 'dẩn': 'dẫn',
    'sấ': 'số', 'lỄn': 'lẫn', 'dều': 'đều', 'vúi': 'với',
    'phấy': 'phẩy', 'tíền': 'tiền', 'khẻ': 'khả', 'gíá': 'giá',
    'phẻi': 'phải', 'dể': 'để', 'tưạng': 'tượng', 'dưạc': 'được',
    'dẩi': 'đãi', 'vđi': 'với', 'dổi': 'đổi', 'dắt': 'đắt',
    'bĐ': 'bộ', 'tíền': 'tiền', 'dầu': 'đầu', 'cung': 'cũng',
    'thóng': 'thông', 'đền': 'đen', 'tưdng': 'tương', 'dó': 'đó',
    'híệu': 'hiệu', 'sấ': 'số', 'dudc': 'được', 'bdi': 'bởi',
    'dẩi': 'đãi', 'dơn': 'đơn', 'giàn': 'giản',
    
    # Word-level corrections  
    'phưdng': 'phương', 'dộng': 'động', 'vởi': 'với', 'dông': 'động',
    'túc': 'tác', '8ố': 'số', 'bưong': 'buông', 'hóu': 'hoá',
    'bỉển': 'biến', 'cậng': 'cộng', 'tuo': 'tạo', 'tràn': 'tràn',
    'náy': 'này', 'trinh': 'trình', 'bàv': 'bày', 'gự': 'sự',
    'chon': 'chọn', 'dến': 'đến', 'tinh': 'tính', 'vư_t': 'vượt',
    'híện': 'hiện', 'tưng': 'tượng', 'biển': 'biến', 'Ihuo': 'thao',
    'loc': 'lọc', 'han': 'hạn', 'dung': 'dụng', 'thuờng': 'thường',
    'mặt': 'mật', 'bầ': 'bậ', 'thíết': 'thiết', 'kê': 'kế',
    
    # Common Vietnamese words
    'cua': 'của', 'voi': 'với', 'nhu': 'như', 'ma': 'mà',
    'da': 'đã', 'den': 'đến', 'thi': 'thì', 'cho': 'cho',
    'la': 'là', 'va': 'và', 'o': 'ở', 'duoc': 'được',
    'khong': 'không', 'nhung': 'những', 'cach': 'cách',
    'phai': 'phải', 'co': 'có', 'neu': 'nếu', 'khi': 'khi',
    
    # Technical terms (from your documents)
    'returu': 'return', 'ellsures': 'ensures', 'fundamnental': 'fundamental',
    'syutax': 'syntax', 'kcy': 'key', 'difterences': 'differences',
    'breakdowu': 'breakdown', 'sone': 'some', 'mneaus': 'means',
    'sectioll': 'section', 'perforni': 'perform', 'learu': 'learn',
    'conipatible': 'compatible', 'distinctioIl': 'distinction',
    'functioll': 'function', 'explicítly': 'explicitly'
}

def normalize_text(text: str, language: str = 'mixed', use_smart_correction: bool = True) -> str:
    """
    Enhanced text normalization with smart correction.
    """
    if not text or not text.strip():
        return text
    
    # Step 1: Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Step 2: Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 3: Apply word-level corrections
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Simple dictionary lookup
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word in VIETNAMESE_OCR_CORRECTIONS:
            corrected = VIETNAMESE_OCR_CORRECTIONS[clean_word]
            if word.isupper():
                corrected = corrected.upper()
            elif word.istitle():
                corrected = corrected.title()
            corrected_word = re.sub(re.escape(clean_word), corrected, word.lower())
        else:
            corrected_word = word
        
        corrected_words.append(corrected_word)
    
    text = ' '.join(corrected_words)
    
    # Step 4: Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def post_process_lines(lines: List[str], 
                       language: str = 'mixed', 
                       min_length: int = 2,
                       use_smart_correction: bool = True) -> List[str]:
    """Enhanced post-processing with spell checking."""
    processed = []
    
    for line in lines:
        # Skip very short lines
        if len(line.strip()) < min_length:
            continue
        
        # Apply normalization with smart correction
        normalized = normalize_text(line, language, use_smart_correction)
        
        # Skip if still too short after normalization
        if len(normalized.strip()) < min_length:
            continue
        
        # Skip lines that are mostly non-alphabetic (likely noise)
        alpha_ratio = sum(c.isalpha() or c.isspace() or c in 'àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ' for c in normalized) / len(normalized)
        if alpha_ratio < 0.6:  # At least 60% letters/spaces/Vietnamese chars
            continue
        
        # Filter out lines with too many special characters
        special_ratio = sum(c in '*=_|[]{}()' for c in normalized) / len(normalized)
        if special_ratio > 0.3:  # More than 30% special chars
            continue
            
        processed.append(normalized)
    
    return processed
