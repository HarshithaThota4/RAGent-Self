"""
Document Ingestion Module for RAGent
Handles PDF and TXT files
"""

import os
from typing import List, Dict

# Try to import PDF support
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
    print("✅ PDF support enabled")
except ImportError:
    PDF_SUPPORT = False
    print("⚠️ PDF support not available. Run: pip install pypdf")


class DocumentIngestor:
    """
    Handles loading and processing documents for RAG.
    Supports TXT and PDF files.
    """
    
    def __init__(self):
        self.documents = []
        print("✅ DocumentIngestor initialized")
    
    def load_pdf(self, filepath: str) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(filepath)
            text = ""
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num} ---\n{page_text}"
            print(f"📄 Loaded PDF: {filepath} ({len(text)} characters, {len(reader.pages)} pages)")
            return text
        except Exception as e:
            print(f"❌ Error loading PDF {filepath}: {e}")
            return ""
    
    def load_text_file(self, filepath: str) -> str:
        """Load a text file and return its content"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            print(f"📄 Loaded text: {filepath} ({len(content)} characters)")
            return content
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return ""
    
    def load_document(self, filepath: str) -> str:
        """Auto-detect file type and load"""
        if filepath.lower().endswith('.pdf'):
            if PDF_SUPPORT:
                return self.load_pdf(filepath)
            else:
                print(f"❌ PDF support not installed. Run: pip install pypdf")
                return ""
        else:
            return self.load_text_file(filepath)
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into smaller chunks for better retrieval"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        print(f"✂️ Split into {len(chunks)} chunks (size: {chunk_size} words)")
        return chunks
    
    def process_document(self, filepath: str) -> List[Dict]:
        """
        Complete pipeline: load → chunk → store
        """
        print(f"\n📑 Processing: {filepath}")
        print("-" * 40)
        
        # Load the document (auto-detects PDF or TXT)
        content = self.load_document(filepath)
        
        if not content:
            return []
        
        # Split into chunks
        chunks = self.chunk_text(content)
        
        # Store with metadata
        for idx, chunk in enumerate(chunks):
            self.documents.append({
                'id': f"{os.path.basename(filepath)}_{idx}",
                'source': filepath,
                'chunk_index': idx,
                'content': chunk,
                'length': len(chunk)
            })
        
        print(f"✅ Document processed: {len(chunks)} chunks stored")
        return chunks
    
    def get_document_count(self) -> int:
        """Return total number of document chunks"""
        return len(self.documents)
    
    def preview_chunk(self, chunk_id: int) -> str:
        """Preview a specific chunk by index"""
        if 0 <= chunk_id < len(self.documents):
            chunk = self.documents[chunk_id]['content']
            return chunk[:200] + "..." if len(chunk) > 200 else chunk
        return "Chunk not found"


# ============= TEST =============

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🧪 TESTING DOCUMENT INGESTOR (PDF + TXT)")
    print("=" * 50)
    
    ingestor = DocumentIngestor()
    
    # Test with text file
    sample_text = "Artificial Intelligence is transforming the world."
    with open("test.txt", "w") as f:
        f.write(sample_text)
    
    ingestor.process_document("test.txt")
    
    print(f"\n📊 Total chunks: {ingestor.get_document_count()}")
    print("✅ Test complete!")