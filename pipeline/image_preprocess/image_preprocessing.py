import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import filters, morphology
from skimage import transform as transforms
from skimage.feature import canny
from skimage.transform import rotate


class ImagePreprocessor:
    """
    Lớp xử lý tiền xử lý ảnh cho OCR
    Mục tiêu: Làm sạch và chuẩn bị ảnh để tối ưu hóa cho việc nhận dạng văn bản
    """
    
    # Hằng số thông báo lỗi
    ERROR_NO_IMAGE = "Chưa có ảnh để xử lý"
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.processing_steps = []
    
    def load_image(self, image_path):
        """Tải ảnh từ đường dẫn"""
        try:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError(f"Không thể tải ảnh từ {image_path}")
            self.processed_image = self.original_image.copy()
            print(f"✓ Đã tải ảnh: {image_path}")
            print(f"  Kích thước: {self.original_image.shape}")
            return True
        except Exception as e:
            print(f"✗ Lỗi tải ảnh: {e}")
            return False
    
    def step1_convert_to_grayscale(self, method='cv2'):
        """
        Bước 1: Chuyển đổi sang ảnh xám
        Args:
            method: 'cv2', 'weighted', 'luminance'
        """
        if self.processed_image is None:
            raise ValueError("Chưa tải ảnh")
        
        if len(self.processed_image.shape) == 3:
            if method == 'cv2':
                # Phương pháp tiêu chuẩn CV2
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            elif method == 'weighted':
                # Weighted average: R*0.299 + G*0.587 + B*0.114
                gray = np.dot(self.processed_image[...,:3], [0.114, 0.587, 0.299])
                gray = gray.astype(np.uint8)
            elif method == 'luminance':
                # Chỉ lấy channel sáng nhất
                gray = np.max(self.processed_image, axis=2)
            else:
                raise ValueError("Method phải là 'cv2', 'weighted', hoặc 'luminance'")
        else:
            gray = self.processed_image.copy()
        
        self.processed_image = gray
        self.processing_steps.append(f"Chuyển sang ảnh xám ({method})")
        print(f"✓ Bước 1: Chuyển đổi sang ảnh xám - {method}")
        return gray
    
    def step2_deskew_image(self, method='hough', max_angle=45):
        """
        Bước 2: Làm thẳng ảnh (Deskew)
        Args:
            method: 'hough', 'projection', 'contour'
            max_angle: góc tối đa để kiểm tra (độ)
        """
        if self.processed_image is None:
            raise ValueError(self.ERROR_NO_IMAGE)
        
        if method == 'hough':
            angle = self._deskew_hough_lines(max_angle)
        elif method == 'projection':
            angle = self._deskew_projection_profile(max_angle)
        elif method == 'contour':
            angle = self._deskew_contour_analysis(max_angle)
        else:
            raise ValueError("Method phải là 'hough', 'projection', hoặc 'contour'")
        
        if abs(angle) > 0.1:  # Chỉ xoay nếu góc nghiêng đáng kể
            self.processed_image = self._rotate_image(self.processed_image, angle)
            self.processing_steps.append(f"Deskew {method}: {angle:.2f}°")
            print(f"✓ Bước 2: Deskew - {method}, góc: {angle:.2f}°")
        else:
            print(f"✓ Bước 2: Deskew - không cần thiết (góc: {angle:.2f}°)")
        
        return angle
    
    def _deskew_hough_lines(self, max_angle):
        """Deskew bằng Hough Lines"""
        # Tạo edges
        edges = canny(self.processed_image, sigma=1, low_threshold=50, high_threshold=150)
        
        # Hough transform
        angles = np.deg2rad(np.arange(-max_angle, max_angle, 0.5))
        h, theta, d = transforms.hough_line(edges, theta=angles)
        
        # Tìm peaks
        peaks = transforms.hough_line_peaks(h, theta, d, num_peaks=20)
        
        if len(peaks[1]) > 0:
            # Tính góc trung vị
            line_angles = [np.rad2deg(angle) for angle in peaks[1]]
            # Lọc các góc gần ngang
            horizontal_angles = [a for a in line_angles if abs(a) < max_angle]
            if horizontal_angles:
                return np.median(horizontal_angles)
        
        return 0.0
    
    def _deskew_projection_profile(self, max_angle):
        """Deskew bằng Projection Profile"""
        best_angle = 0
        max_variance = 0
        
        for angle in np.arange(-max_angle, max_angle, 0.5):
            # Xoay ảnh thử
            rotated = self._rotate_image(self.processed_image, angle)
            
            # Tính projection profile ngang
            projection = np.sum(rotated, axis=1)
            
            # Variance cao = các dòng text rõ ràng hơn
            variance = np.var(projection)
            
            if variance > max_variance:
                max_variance = variance
                best_angle = angle
        
        return best_angle
    
    def _deskew_contour_analysis(self, max_angle):
        """Deskew bằng phân tích contours"""
        # Tìm contours
        _, binary = cv2.threshold(self.processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        angles = []
        for contour in contours:
            # Chỉ xét contours đủ lớn (có thể là text)
            if cv2.contourArea(contour) > 100:
                # Fit ellipse để tìm góc
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    angle = ellipse[2]
                    # Chuẩn hóa góc về [-45, 45]
                    if angle > 45:
                        angle = angle - 90
                    if abs(angle) < max_angle:
                        angles.append(angle)
        
        return np.median(angles) if angles else 0.0
    
    def _rotate_image(self, image, angle):
        """Xoay ảnh theo góc cho trước"""
        # Sử dụng skimage rotate để tránh mất thông tin
        rotated = rotate(image, angle, resize=True, preserve_range=True)
        return rotated.astype(np.uint8)
    
    def step3_denoise_deblur(self, method='gaussian', **kwargs):
        """
        Bước 3: Khử nhiễu và làm sắc nét
        Args:
            method: 'gaussian', 'median', 'bilateral', 'non_local_means'
        """
        if self.processed_image is None:
            raise ValueError(self.ERROR_NO_IMAGE)
        
        if method == 'gaussian':
            # Gaussian blur để khử nhiễu nhẹ
            sigma = kwargs.get('sigma', 0.8)
            denoised = gaussian_filter(self.processed_image, sigma=sigma)
            
        elif method == 'median':
            # Median filter để khử salt-pepper noise
            kernel_size = kwargs.get('kernel_size', 3)
            denoised = cv2.medianBlur(self.processed_image, kernel_size)
            
        elif method == 'bilateral':
            # Bilateral filter - giữ edges, khử nhiễu
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            denoised = cv2.bilateralFilter(self.processed_image, d, sigma_color, sigma_space)
            
        elif method == 'non_local_means':
            # Non-local means - khử nhiễu mạnh
            h = kwargs.get('h', 10)
            template_window_size = kwargs.get('template_window_size', 7)
            search_window_size = kwargs.get('search_window_size', 21)
            denoised = cv2.fastNlMeansDenoising(self.processed_image, None, h, 
                                              template_window_size, search_window_size)
        else:
            raise ValueError("Method phải là 'gaussian', 'median', 'bilateral', hoặc 'non_local_means'")
        
        self.processed_image = denoised.astype(np.uint8)
        self.processing_steps.append(f"Khử nhiễu: {method}")
        print(f"✓ Bước 3: Khử nhiễu/làm sắc nét - {method}")
        return denoised
    
    def step4_binarize(self, method='adaptive', **kwargs):
        """
        Bước 4: Nhị phân hóa ảnh
        Args:
            method: 'otsu', 'adaptive', 'sauvola', 'niblack'
        """
        if self.processed_image is None:
            raise ValueError(self.ERROR_NO_IMAGE)
        
        if method == 'otsu':
            # Otsu thresholding - tự động tìm threshold
            _, binary = cv2.threshold(self.processed_image, 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == 'adaptive':
            # Adaptive thresholding - phù hợp với lighting không đều
            max_value = kwargs.get('max_value', 255)
            adaptive_method = kwargs.get('adaptive_method', cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            thresh_type = kwargs.get('thresh_type', cv2.THRESH_BINARY)
            block_size = kwargs.get('block_size', 11)
            C = kwargs.get('C', 2)
            
            binary = cv2.adaptiveThreshold(self.processed_image, max_value, 
                                         adaptive_method, thresh_type, block_size, C)
            
        elif method == 'sauvola':
            # Sauvola thresholding - tốt cho văn bản
            window_size = kwargs.get('window_size', 15)
            k = kwargs.get('k', 0.2)
            binary = self._sauvola_threshold(self.processed_image, window_size, k)
            
        elif method == 'niblack':
            # Niblack thresholding
            window_size = kwargs.get('window_size', 15)
            k = kwargs.get('k', -0.2)
            binary = self._niblack_threshold(self.processed_image, window_size, k)
            
        else:
            raise ValueError("Method phải là 'otsu', 'adaptive', 'sauvola', hoặc 'niblack'")
        
        self.processed_image = binary
        self.processing_steps.append(f"Nhị phân hóa: {method}")
        print(f"✓ Bước 4: Nhị phân hóa - {method}")
        return binary
    
    def _sauvola_threshold(self, image, window_size, k):
        """Sauvola local thresholding"""
        from skimage.filters import threshold_sauvola
        thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=k)
        binary = image > thresh_sauvola
        return (binary * 255).astype(np.uint8)
    
    def _niblack_threshold(self, image, window_size, k):
        """Niblack local thresholding"""
        from skimage.filters import threshold_niblack
        thresh_niblack = threshold_niblack(image, window_size=window_size, k=k)
        binary = image > thresh_niblack
        return (binary * 255).astype(np.uint8)
    
    def post_process_morphology(self, operation='closing', kernel_size=3):
        """
        Hậu xử lý morphological
        Args:
            operation: 'erosion', 'dilation', 'opening', 'closing'
            kernel_size: kích thước kernel
        """
        if self.processed_image is None:
            raise ValueError(self.ERROR_NO_IMAGE)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        if operation == 'erosion':
            result = cv2.erode(self.processed_image, kernel, iterations=1)
        elif operation == 'dilation':
            result = cv2.dilate(self.processed_image, kernel, iterations=1)
        elif operation == 'opening':
            result = cv2.morphologyEx(self.processed_image, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            result = cv2.morphologyEx(self.processed_image, cv2.MORPH_CLOSE, kernel)
        else:
            raise ValueError("Operation phải là 'erosion', 'dilation', 'opening', hoặc 'closing'")
        
        self.processed_image = result
        self.processing_steps.append(f"Morphology: {operation}")
        print(f"✓ Hậu xử lý: {operation}")
        return result
    
    def remove_small_noise(self, min_area=50):
        """Loại bỏ các connected components nhỏ (nhiễu)"""
        if self.processed_image is None:
            raise ValueError(self.ERROR_NO_IMAGE)
        
        # Tìm connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            self.processed_image, connectivity=8)
        
        # Tạo ảnh output
        output = np.zeros_like(self.processed_image)
        
        # Giữ lại components đủ lớn
        for i in range(1, num_labels):  # Bỏ qua background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                output[labels == i] = 255
        
        self.processed_image = output
        self.processing_steps.append(f"Loại bỏ nhiễu nhỏ < {min_area}")
        print(f"✓ Loại bỏ nhiễu nhỏ (< {min_area} pixels)")
        return output
    
    def process_full_pipeline(self, image_path, save_path=None):
        """
        Chạy toàn bộ pipeline tiền xử lý
        """
        print("=" * 50)
        print("BẮT ĐẦU PIPELINE TIỀN XỬ LÝ ẢNH")
        print("=" * 50)
        
        # Tải ảnh
        if not self.load_image(image_path):
            return None
        
        # Bước 1: Chuyển sang ảnh xám
        self.step1_convert_to_grayscale(method='cv2')
        
        # Bước 2: Deskew
        self.step2_deskew_image(method='hough', max_angle=15)
        
        # Bước 3: Khử nhiễu
        self.step3_denoise_deblur(method='bilateral', d=9, sigma_color=75, sigma_space=75)
        
        # Bước 4: Nhị phân hóa
        self.step4_binarize(method='adaptive', block_size=11, C=2)
        
        # Hậu xử lý
        self.post_process_morphology(operation='closing', kernel_size=2)
        self.remove_small_noise(min_area=30)
        
        # Lưu kết quả
        if save_path:
            cv2.imwrite(save_path, self.processed_image)
            print(f"✓ Đã lưu ảnh xử lý: {save_path}")
        
        print("=" * 50)
        print("HOÀN THÀNH PIPELINE")
        print("Các bước đã thực hiện:")
        for i, step in enumerate(self.processing_steps, 1):
            print(f"  {i}. {step}")
        print("=" * 50)
        
        return self.processed_image
    
    def compare_results(self, show_plot=True):
        """So sánh ảnh gốc và ảnh đã xử lý"""
        if self.original_image is None or self.processed_image is None:
            print("Không có ảnh để so sánh")
            return
        
        if show_plot:
            _, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Ảnh gốc
            if len(self.original_image.shape) == 3:
                axes[0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
            else:
                axes[0].imshow(self.original_image, cmap='gray')
            axes[0].set_title('Ảnh gốc')
            axes[0].axis('off')
            
            # Ảnh đã xử lý
            axes[1].imshow(self.processed_image, cmap='gray')
            axes[1].set_title('Ảnh đã tiền xử lý')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
    
    def save_result(self, output_path):
        """Lưu ảnh đã xử lý"""
        if self.processed_image is None:
            print("Không có ảnh để lưu")
            return False
        
        cv2.imwrite(output_path, self.processed_image)
        print(f"✓ Đã lưu ảnh: {output_path}")
        return True


# Hàm sử dụng nhanh
def preprocess_image_quick(input_path, output_path=None, method='full'):
    """
    Hàm tiện ích để xử lý nhanh một ảnh
    Args:
        input_path: đường dẫn ảnh input
        output_path: đường dẫn lưu ảnh output (optional)
        method: 'full', 'simple', 'text_enhanced'
    """
    processor = ImagePreprocessor()
    
    if method == 'full':
        result = processor.process_full_pipeline(input_path, output_path)
    elif method == 'simple':
        processor.load_image(input_path)
        processor.step1_convert_to_grayscale()
        processor.step4_binarize(method='otsu')
        result = processor.processed_image
        if output_path:
            processor.save_result(output_path)
    elif method == 'text_enhanced':
        processor.load_image(input_path)
        processor.step1_convert_to_grayscale()
        processor.step2_deskew_image(method='hough')
        processor.step3_denoise_deblur(method='bilateral')
        processor.step4_binarize(method='sauvola', window_size=15, k=0.2)
        processor.post_process_morphology('closing', 2)
        result = processor.processed_image
        if output_path:
            processor.save_result(output_path)
    else:
        raise ValueError("Method phải là 'full', 'simple', hoặc 'text_enhanced'")
    
    return result


# Demo sử dụng
if __name__ == "__main__":
    # Cách 1: Sử dụng class chi tiết
    processor = ImagePreprocessor()
    result = processor.process_full_pipeline('image.png', 'preprocessed_image.png')
    
    # Cách 2: Sử dụng hàm nhanh
    # result = preprocess_image_quick('image.png', 'output.png', method='text_enhanced')
    
    # So sánh kết quả
    # processor.compare_results(show_plot=True)
