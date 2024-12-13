import os
import io
import base64
import numpy as np
import pandas as pd
import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Advanced OCR Libraries
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

class AdvancedDocumentExtractor:
    def __init__(self, 
                 tesseract_path: str = None, 
                 use_gpu: bool = False, 
                 languages: list = ['en']):
        """
        Advanced document extraction with multiple OCR engines
        
        :param tesseract_path: Custom Tesseract executable path
        :param use_gpu: Enable GPU acceleration for deep learning OCR
        :param languages: List of languages to support
        """
        # Tesseract configuration
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # EasyOCR Initialization
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(
                    languages, 
                    gpu=use_gpu
                )
            except Exception as e:
                print(f"EasyOCR initialization error: {e}")
        
        # PaddleOCR Initialization
        self.paddleocr_reader = None
        if PADDLEOCR_AVAILABLE:
            try:
                self.paddleocr_reader = paddleocr.PaddleOCR(
                    use_angle_cls=True, 
                    lang='en', 
                    use_gpu=use_gpu
                )
            except Exception as e:
                print(f"PaddleOCR initialization error: {e}")
        
        # Advanced preprocessing parameters
        self.preprocessing_config = {
            'resize_factor': 2,  # Upscale image for better OCR
            'contrast_enhancement': True,
            'sharpen': True
        }

    def _advanced_preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced image preprocessing for optimal OCR
        
        :param image: Input image as numpy array
        :return: Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize
        if self.preprocessing_config['resize_factor'] > 1:
            gray = cv2.resize(
                gray, 
                None, 
                fx=self.preprocessing_config['resize_factor'], 
                fy=self.preprocessing_config['resize_factor'], 
                interpolation=cv2.INTER_CUBIC
            )
        
        # Contrast enhancement
        if self.preprocessing_config['contrast_enhancement']:
            gray = cv2.equalizeHist(gray)
        
        # Sharpening
        if self.preprocessing_config['sharpen']:
            kernel = np.array([
                [-1,-1,-1],
                [-1, 9,-1],
                [-1,-1,-1]
            ])
            gray = cv2.filter2D(gray, -1, kernel)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        return binary

    def extract_text_multi_engine(self, image_path: str) -> dict:
        """
        Extract text using multiple OCR engines
        
        :param image_path: Path to image or PDF
        :return: Dictionary of extraction results
        """
        # Convert PDF to images if needed
        if image_path.lower().endswith('.pdf'):
            images = convert_from_path(image_path)
            image = np.array(images[0].convert('RGB'))
        else:
            image = cv2.imread(image_path)
        
        # Preprocessing
        preprocessed_image = self._advanced_preprocess(image)
        
        # Results dictionary
        extraction_results = {
            'tesseract_extraction': self._tesseract_extract(preprocessed_image),
            'easyocr_extraction': self._easyocr_extract(preprocessed_image),
            'paddleocr_extraction': self._paddleocr_extract(preprocessed_image)
        }
        
        return extraction_results

    def _tesseract_extract(self, image: np.ndarray) -> str:
        """
        Tesseract OCR extraction
        
        :param image: Preprocessed image
        :return: Extracted text
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Advanced Tesseract configuration
            text = pytesseract.image_to_string(
                pil_image, 
                config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?():;\'"-'
            )
            return text.strip()
        except Exception as e:
            print(f"Tesseract extraction error: {e}")
            return ""

    def _easyocr_extract(self, image: np.ndarray) -> str:
        """
        EasyOCR extraction
        
        :param image: Preprocessed image
        :return: Extracted text
        """
        if not self.easyocr_reader:
            return ""
        
        try:
            results = self.easyocr_reader.readtext(image)
            return " ".join([result[1] for result in results])
        except Exception as e:
            print(f"EasyOCR extraction error: {e}")
            return ""

    def _paddleocr_extract(self, image: np.ndarray) -> str:
        """
        PaddleOCR extraction
        
        :param image: Preprocessed image
        :return: Extracted text
        """
        if not self.paddleocr_reader:
            return ""
        
        try:
            results = self.paddleocr_reader.ocr(image, cls=True)
            return " ".join([line[1][0] for result in results for line in result])
        except Exception as e:
            print(f"PaddleOCR extraction error: {e}")
            return ""

    def consensus_extraction(self, extraction_results: dict) -> str:
        """
        Create consensus from multiple OCR extractions
        
        :param extraction_results: Results from multiple OCR engines
        :return: Consensus text extraction
        """
        # Remove empty extractions
        valid_extractions = [
            text for text in extraction_results.values() 
            if text and len(text) > 10
        ]
        
        # If no valid extractions, return empty string
        if not valid_extractions:
            return ""
        
        # If only one extraction, return it
        if len(valid_extractions) == 1:
            return valid_extractions[0]
        
        # Basic consensus - most common extraction
        from collections import Counter
        
        # Split extractions into words
        word_lists = [set(text.split()) for text in valid_extractions]
        
        # Find words that appear in most extractions
        consensus_words = []
        for word in set.union(*word_lists):
            word_count = sum(1 for word_set in word_lists if word in word_set)
            if word_count > len(valid_extractions) // 2:
                consensus_words.append(word)
        
        return " ".join(consensus_words)

    def analyze_extraction_quality(self, extraction_results: dict) -> dict:
        """
        Analyze quality of text extraction
        
        :param extraction_results: Results from multiple OCR engines
        :return: Extraction quality metrics
        """
        quality_metrics = {
            'total_extractions': len(extraction_results),
            'valid_extractions': sum(1 for text in extraction_results.values() if text),
            'average_length': np.mean([len(text) for text in extraction_results.values() if text]),
            'extraction_variance': np.std([len(text) for text in extraction_results.values() if text])
        }
        
        # Confidence calculation
        quality_metrics['confidence_score'] = (
            quality_metrics['valid_extractions'] / len(extraction_results)
        ) * (quality_metrics['average_length'] / 100)
        
        return quality_metrics

# Example Usage
def main():
    # Initialize extractor
    extractor = AdvancedDocumentExtractor(
        use_gpu=True,  # Enable GPU if available
        languages=['en', 'fr']  # Support multiple languages hopefully , please?
    )
    
    # Process document
    try:
        # Extract text from PDF or image
        extraction_results = extractor.extract_text_multi_engine('document.pdf')
        
        # Get consensus extraction
        consensus_text = extractor.consensus_extraction(extraction_results)
        
        # Analyze extraction quality
        quality_metrics = extractor.analyze_extraction_quality(extraction_results)
        
        print("Consensus Text:", consensus_text)
        print("Extraction Quality Metrics:", quality_metrics)
        
    except Exception as e:
        print(f"Document processing error: {e}")

if __name__ == "__main__":
    main()