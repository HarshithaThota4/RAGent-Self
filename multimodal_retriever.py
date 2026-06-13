"""
Multimodal Retriever for RAGent+
CLIP-based text + image retrieval with FAISS
"""

import os
import numpy as np
import faiss
from PIL import Image
from typing import List, Dict
from clip_embeddings import CLIPEmbeddings

class MultimodalRetriever:
    def __init__(self):
        self.clip = None
        self.image_index = None
        self.image_metadata = []
        self.initialized = False
        print("✅ Multimodal Retriever created")
    
    def _ensure_clip(self):
        if not self.initialized:
            try:
                self.clip = CLIPEmbeddings()
                self.initialized = True
            except Exception as e:
                print(f"❌ CLIP init failed: {e}")
                return False
        return True
    
    def add_image(self, image_path: str, metadata: dict = None) -> bool:
        if not self._ensure_clip():
            return False
        
        try:
            if not os.path.exists(image_path):
                print(f"❌ Not found: {image_path}")
                return False
            
            print(f"📸 Indexing: {os.path.basename(image_path)}")
            
            # Verify and load image
            img = Image.open(image_path)
            img = img.convert('RGB')
            
            # Get embedding
            embedding = self.clip.get_image_embedding(image_path)
            print(f"   Embedding shape: {embedding.shape}")
            
            if self.image_index is None:
                self.image_index = faiss.IndexFlatIP(len(embedding))
            
            self.image_index.add(embedding.reshape(1, -1).astype('float32'))
            self.image_metadata.append({
                'path': image_path,
                'filename': os.path.basename(image_path),
                **(metadata or {})
            })
            print(f"✅ Indexed: {os.path.basename(image_path)}. Total: {self.image_index.ntotal}")
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self._ensure_clip():
            return []
        if self.image_index is None or self.image_index.ntotal == 0:
            return []
        
        try:
            query_emb = self.clip.get_text_embedding(query)
            k = min(top_k, self.image_index.ntotal)
            scores, indices = self.image_index.search(query_emb.reshape(1, -1).astype('float32'), k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.image_metadata):
                    results.append({
                        'image': self.image_metadata[idx]['path'],
                        'filename': self.image_metadata[idx]['filename'],
                        'similarity': float(scores[0][i])
                    })
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def search_by_image(self, image_path: str, top_k: int = 5) -> List[Dict]:
        if not self._ensure_clip():
            return []
        if self.image_index is None or self.image_index.ntotal == 0:
            return []
        
        try:
            query_emb = self.clip.get_image_embedding(image_path)
            k = min(top_k, self.image_index.ntotal)
            scores, indices = self.image_index.search(query_emb.reshape(1, -1).astype('float32'), k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.image_metadata):
                    results.append({
                        'image': self.image_metadata[idx]['path'],
                        'filename': self.image_metadata[idx]['filename'],
                        'similarity': float(scores[0][i])
                    })
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_image_count(self) -> int:
        return self.image_index.ntotal if self.image_index else 0


if __name__ == "__main__":
    retriever = MultimodalRetriever()
    print("✅ Multimodal Retriever ready!")
