import os
import io
import torch
from PIL import Image
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class DocumentAnalyzer:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct"):
        """
        Initialize the document analyzer with Qwen2-VL model
        
        :param model_id: Hugging Face model identifier
        """
        # Model and Processor Loading
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to("cuda").eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )

    def extract_text_from_pdf(self, pdf_path, max_pages=10):
        """
        Extract text from PDF using multiple methods:
        1. Text extraction
        2. OCR for scanned or image-based PDFs
        
        :param pdf_path: Path to the PDF file
        :param max_pages: Maximum number of pages to process
        :return: Extracted text as a string
        """
        extracted_texts = []
        
        # First, try direct text extraction
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Limit to max_pages to prevent excessive processing
                for page_num in range(min(len(pdf_reader.pages), max_pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text().strip()
                    if page_text:
                        extracted_texts.append(page_text)
        except Exception as e:
            print(f"Direct text extraction failed: {e}")
        
        # If no text extracted, use OCR
        if not extracted_texts:
            try:
                # Convert PDF pages to images
                images = convert_from_path(pdf_path, limit=max_pages)
                
                for img in images:
                    # Use pytesseract for OCR
                    text = pytesseract.image_to_string(img)
                    if text.strip():
                        extracted_texts.append(text.strip())
            except Exception as e:
                print(f"OCR extraction failed: {e}")
        
        return "\n\n".join(extracted_texts)

    def preprocess_document(self, file_path):
        """
        Preprocess different document types
        Supports: PDF, TXT, and image-based documents
        
        :param file_path: Path to the document
        :return: Processed document content
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return file_path
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def analyze_document(self, document_path, query=None):
        """
        Analyze a document using Qwen2-VL model
        
        :param document_path: Path to the document
        :param query: Optional specific query about the document
        :return: Model's analysis of the document
        """
        # Preprocess the document
        try:
            processed_document = self.preprocess_document(document_path)
        except Exception as e:
            return f"Error processing document: {str(e)}"
        
        # Determine input type (text or image)
        media_type = "image" if os.path.splitext(document_path)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif'] else "text"
        
        # Construct messages for model input
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": media_type,
                        media_type: processed_document,
                    },
                    {"type": "text", "text": query or "Analyze this document and provide key insights."},
                ],
            }
        ]

        # Prepare inputs for the model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs if media_type == "image" else None,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate response
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7
        )
        
        # Decode the generated text
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response

    def _process_vision_info(self, messages):
        """
        Process vision information from messages
        
        :param messages: Input messages
        :return: Processed image inputs
        """
        image_inputs = []
        for msg in messages:
            for content in msg.get('content', []):
                if content.get('type') == 'image':
                    # Load image
                    img_path = content.get('image')
                    if img_path:
                        image_inputs.append(Image.open(img_path))
        
        return image_inputs if image_inputs else None

def main():
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Document Analysis with Qwen2-VL")
    parser.add_argument("document", help="Path to the document to analyze")
    parser.add_argument("-q", "--query", 
                        help="Optional specific query about the document", 
                        default=None)
    
    # Parse arguments
    args = parser.parse_args()

    # Initialize document analyzer
    analyzer = DocumentAnalyzer()

    # Analyze document
    try:
        print("Analyzing document...")
        result = analyzer.analyze_document(args.document, args.query)
        print("\n--- Document Analysis ---")
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()