"""
Image Processor for Multimodal RAG - OCR + LLM Based Q&A
"""

import os
from PIL import Image
import pytesseract

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not available")

class ImageProcessor:
    def __init__(self):
        self.groq_available = GROQ_AVAILABLE
        if self.groq_available:
            try:
                self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                self.model = "llama-3.1-8b-instant"
                print("✅ Groq LLM available for image Q&A")
            except Exception as e:
                print(f"⚠️ Groq init failed: {e}")
                self.groq_available = False
        print("✅ Image Processor ready (OCR + LLM mode)")
    
    def extract_text_ocr(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(image_path).convert("RGB")
            # Preprocess for better OCR
            image = image.resize((image.width * 2, image.height * 2))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def generate_caption(self, image_path: str) -> str:
        """Generate a simple caption using OCR + LLM"""
        ocr_text = self.extract_text_ocr(image_path)
        if ocr_text:
            # If there's text, use first line as caption
            lines = ocr_text.split('\n')
            if lines:
                return lines[0][:100]
        return "Image with content"
    
    def answer_question(self, image_path: str, question: str) -> str:
        """Answer question about image using OCR text + LLM"""
        
        if not self.groq_available:
            return "LLM not available for image Q&A."
        
        if not question or question.strip() == "":
            return "Please ask a question about the image."
        
        # Extract text from image
        ocr_text = self.extract_text_ocr(image_path)
        
        if not ocr_text or len(ocr_text) < 10:
            return "No readable text found in the image. Please upload a clearer image with visible text."
        
        # Use LLM to answer based on extracted text
        prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the text extracted from the image.

EXTRACTED TEXT FROM IMAGE:
{ocr_text[:2000]}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based only on the extracted text
2. Be concise and accurate
3. If the answer is not in the text, say "I cannot find this information in the image"

ANSWER:"""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            answer = completion.choices[0].message.content
            return answer
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze_image(self, image_path: str, question: str = None) -> dict:
        """Complete image analysis with OCR and optional Q&A"""
        ocr_text = self.extract_text_ocr(image_path)
        caption = self.generate_caption(image_path)
        
        answer = None
        if question and question.strip():
            answer = self.answer_question(image_path, question)
        else:
            answer = "Ask a question like 'What is in this image?'"
        
        return {
            'ocr_text': ocr_text,
            'caption': caption,
            'answer': answer,
            'has_text': len(ocr_text) > 0
        }


if __name__ == "__main__":
    processor = ImageProcessor()
    print("✅ Image Processor ready with OCR + LLM!")
