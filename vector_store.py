"""
Vector Store Module for RAGent
FAISS-based semantic search with embeddings
"""

import numpy as np
from typing import List, Dict, Optional
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os

class VectorStore:
    """
    FAISS-based vector store for semantic document search
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print("🔧 Initializing Vector Store...")
        
        # Load embedding model
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Store metadata
        self.metadata = []  # List of dicts with chunk info
        self.documents = []  # List of chunk text
        
        print(f"✅ Vector Store initialized with {model_name}")
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def add_documents(self, chunks: List[Dict]) -> int:
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of dicts with 'content', 'source', 'chunk_id'
        
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        print(f"📚 Adding {len(chunks)} chunks to vector store...")
        
        # Extract texts
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        for chunk in chunks:
            self.metadata.append({
                'source': chunk.get('source', 'unknown'),
                'chunk_id': chunk.get('id', f"chunk_{len(self.metadata)}"),
                'content': chunk['content'],
                'index': len(self.metadata)
            })
            self.documents.append(chunk['content'])
        
        print(f"✅ Added {len(chunks)} chunks. Total: {self.index.ntotal}")
        return self.index.ntotal
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks using semantic similarity
        
        Args:
            query: User question
            top_k: Number of results to return
        
        Returns:
            List of dicts with content, source, score
        """
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Convert distances to similarity scores (L2 distance -> similarity)
        # Lower distance = more similar
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                # Convert L2 distance to similarity score (0-1 range)
                # Using exponential decay: similarity = e^(-distance/10)
                similarity = np.exp(-distances[0][i] / 10)
                
                results.append({
                    'content': self.metadata[idx]['content'],
                    'source': self.metadata[idx]['source'],
                    'chunk_id': self.metadata[idx]['chunk_id'],
                    'score': similarity,
                    'distance': distances[0][i]
                })
        
        return results
    
    def save(self, path: str = "./vector_store"):
        """Save FAISS index and metadata to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}/index.faiss")
        
        # Save metadata
        with open(f"{path}/metadata.pkl", 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'documents': self.documents
            }, f)
        
        print(f"✅ Saved vector store to {path}")
    
    def load(self, path: str = "./vector_store"):
        """Load FAISS index and metadata from disk"""
        if os.path.exists(f"{path}/index.faiss"):
            self.index = faiss.read_index(f"{path}/index.faiss")
            
            with open(f"{path}/metadata.pkl", 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.documents = data['documents']
            
            print(f"✅ Loaded vector store with {self.index.ntotal} chunks")
            return True
        return False
    
    def get_chunk_count(self) -> int:
        """Return number of chunks in vector store"""
        return self.index.ntotal


# ============= TEST =============

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING VECTOR STORE")
    print("=" * 60)
    
    # Create vector store
    vs = VectorStore()
    
    # Sample documents
    test_chunks = [
        {'content': 'Machine Learning is a subset of AI that learns from data.', 'source': 'ml.txt', 'id': 'ml_1'},
        {'content': 'Deep Learning uses neural networks with multiple layers.', 'source': 'dl.txt', 'id': 'dl_1'},
        {'content': 'Natural Language Processing helps computers understand human language.', 'source': 'nlp.txt', 'id': 'nlp_1'},
    ]
    
    # Add documents
    vs.add_documents(test_chunks)
    
    # Search
    results = vs.search("What is machine learning?", top_k=2)
    
    print("\n📊 Search Results:")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Score: {r['score']:.4f}")
        print(f"   Source: {r['source']}")
        print(f"   Content: {r['content'][:100]}...")
    
    print("\n✅ Vector Store test complete!")
