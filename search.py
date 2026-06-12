"""
Search Module for RAGent
Hybrid Search: Keyword + FAISS Vector
"""

from typing import List, Dict
from vector_store import VectorStore

class DocumentSearch:
    """Hybrid search with keyword matching and vector search"""
    
    def __init__(self):
        self.documents = []
        self.vector_store = VectorStore()
        print("✅ DocumentSearch initialized with FAISS")
    
    def load_documents(self, documents: List[Dict]):
        self.documents = documents
        if documents:
            self.vector_store.add_documents(documents)
        print(f"📚 Loaded {len(documents)} document chunks")
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Keyword search with relevance scoring"""
        query_words = set(query.lower().split())
        results = []
        
        for doc in self.documents:
            content = doc['content'].lower()
            # Count matching words
            score = sum(1 for word in query_words if word in content)
            if score > 0:
                # Normalize score to 0-1 range (max expected ~10)
                normalized_score = min(score / 10, 1.0)
                results.append({
                    'content': doc['content'],
                    'score': normalized_score,
                    'source': doc['source'],
                    'chunk_id': doc['id']
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search using FAISS"""
        return self.vector_store.search(query, top_k=top_k)
    
    def hybrid_search(self, query: str, top_k: int = 5, keyword_weight: float = 0.3, vector_weight: float = 0.7) -> List[Dict]:
        """Hybrid search combining keyword and vector results with weighted scoring"""
        keyword_results = self.keyword_search(query, top_k=top_k * 2)
        vector_results = self.vector_search(query, top_k=top_k * 2)
        
        # Combine results using weighted scores
        combined = {}
        
        for r in keyword_results:
            chunk_id = r['chunk_id']
            combined[chunk_id] = {
                'content': r['content'],
                'source': r['source'],
                'chunk_id': chunk_id,
                'score': r['score'] * keyword_weight,
                'keyword_score': r['score']
            }
        
        for r in vector_results:
            chunk_id = r['chunk_id']
            if chunk_id in combined:
                combined[chunk_id]['score'] += r['score'] * vector_weight
                combined[chunk_id]['vector_score'] = r['score']
            else:
                combined[chunk_id] = {
                    'content': r['content'],
                    'source': r['source'],
                    'chunk_id': chunk_id,
                    'score': r['score'] * vector_weight,
                    'vector_score': r['score']
                }
        
        results = list(combined.values())
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
