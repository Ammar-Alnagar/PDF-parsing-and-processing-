import os
import io
import base64
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from pdf2image import convert_from_path
import docx
import csv
import openai
import anthropic
from typing import List, Dict, Union

class AdvancedDocumentProcessor:
    def __init__(self, 
                 openai_api_key: str = None, 
                 anthropic_api_key: str = None,
                 tesseract_path: str = None):
        """
        Initialize advanced document processor with preprocessing capabilities
        
        :param openai_api_key: OpenAI API key for GPT vision
        :param anthropic_api_key: Anthropic API key for Claude vision
        :param tesseract_path: Custom path to Tesseract executable
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.openai_client = None
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        self.anthropic_client = None
        if anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Advanced image preprocessing for improved OCR
        
        :param image: PIL Image to preprocess
        :return: Preprocessed image as numpy array
        """
        # Convert PIL Image to OpenCV format
        open_cv_image = np.array(image.convert('RGB'))
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        # Preprocessing steps
        # 1. Convert to grayscale
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Noise Reduction
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 3. Binarization (Adaptive Thresholding)
        binary = cv2.adaptiveThreshold(
            denoised, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # 4. Deskew (Correct document rotation)
        coords = np.column_stack(np.where(binary > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Correct the angle of rotation
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotation matrix
        (h, w) = binary.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary, 
            M, 
            (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated

    def process_pdf(self, pdf_path: str, use_vision_model: str = None) -> str:
        """
        Process PDF with advanced preprocessing and OCR
        
        :param pdf_path: Path to PDF file
        :param use_vision_model: 'openai', 'claude', or None
        :return: Extracted text from PDF
        """
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        # Enhanced text extraction
        full_text = ""
        for image in images:
            # Preprocess image
            preprocessed_image = self._preprocess_image(image)
            
            # Convert preprocessed numpy array back to PIL Image
            pil_image = Image.fromarray(preprocessed_image)
            
            # Extract text using Tesseract with enhanced configuration
            text = pytesseract.image_to_string(
                pil_image, 
                config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?():;\'"-'
            )
            
            full_text += text + "\n"
        
        # Optional AI Vision Enhancement
        if use_vision_model == 'openai' and self.openai_client:
            full_text = self._enhance_with_openai_vision(images)
        elif use_vision_model == 'claude' and self.anthropic_client:
            full_text = self._enhance_with_claude_vision(images)
        
        return full_text

    def _enhance_with_openai_vision(self, images: List[Image.Image]) -> str:
        """
        Enhance OCR results using OpenAI GPT Vision
        Enhanced to handle multiple images
        
        :param images: List of PIL Images
        :return: Enhanced text extraction
        """
        if not self.openai_client:
            return ""
        
        enhanced_text = ""
        for image in images:
            # Preprocess image
            preprocessed_image = self._preprocess_image(image)
            
            # Convert to base64
            buffered = io.BytesIO()
            Image.fromarray(preprocessed_image).save(buffered, format="PNG")
            image_bytes = base64.b64encode(buffered.getvalue()).decode()
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text and information from this document image with maximum accuracy."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_bytes}"}}
                        ]
                    }]
                )
                enhanced_text += response.choices[0].message.content + "\n"
            except Exception as e:
                print(f"OpenAI Vision API Error: {e}")
        
        return enhanced_text

    # ... [rest of the previous implementation remains the same]

    def validate_extraction(self, extracted_text: str, confidence_threshold: float = 0.7) -> Dict:
        """
        Validate text extraction quality
        
        :param extracted_text: Text extracted from document
        :param confidence_threshold: Minimum confidence level for acceptance
        :return: Validation results
        """
        # Basic validation metrics
        validation_result = {
            'total_characters': len(extracted_text),
            'word_count': len(extracted_text.split()),
            'avg_word_length': np.mean([len(word) for word in extracted_text.split()]),
            'is_valid': False
        }
        
        # Confidence calculation
        # More sophisticated validation could involve language models or specific document structure checks
        if validation_result['total_characters'] > 100 and validation_result['word_count'] > 20:
            validation_result['is_valid'] = True
        
        return validation_result

# Example Usage with Enhanced Error Handling
def main():
    try:
        processor = AdvancedDocumentProcessor(
            openai_api_key='your_openai_api_key', 
            anthropic_api_key='your_anthropic_api_key'
        )
        
        # Process scanned PDF
        pdf_text = processor.process_document('scanned_document.pdf', use_vision_model='claude')
        
        # Validate extraction
        validation = processor.validate_extraction(pdf_text)
        
        if not validation['is_valid']:
            print("Warning: Document extraction might be incomplete or low quality.")
        
    except Exception as e:
        print(f"Document processing error: {e}")

if __name__ == "__main__":
    main()