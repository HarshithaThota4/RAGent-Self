"""
RAGENT - Complete RAG System with Self-Correcting RAG (Self-RAG)
Features: Hybrid Search + Self-Critique + Regeneration + Confidence Scoring
"""

import os
from typing import List, Dict
from datetime import datetime

from ingest import DocumentIngestor
from search import DocumentSearch
from llm_real import RealLLMAgent


class RAGent:
    def __init__(self, model_choice: str = "llama-8b"):
        print("\n" + "=" * 60)
        print("🚀 INITIALIZING RAGent SYSTEM with Self-RAG")
        print("=" * 60)
        
        self.ingestor = DocumentIngestor()
        self.searcher = DocumentSearch()
        self.llm = RealLLMAgent(model_choice=model_choice)
        
        self.documents = []
        self.is_ready = False
        self.query_history = []
        
        # Self-RAG settings
        self.max_retrieval_attempts = 2
        self.max_regeneration_attempts = 2
        
        print("✅ RAGent system initialized with Self-Correcting RAG")
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def add_document(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
        
        print(f"\n📄 Adding document: {filepath}")
        print("-" * 40)
        
        chunks = self.ingestor.process_document(filepath)
        
        if chunks:
            for chunk in self.ingestor.documents:
                self.documents.append(chunk)
            
            self.searcher.load_documents(self.documents)
            self.is_ready = True
            
            print(f"✅ Document added! Total chunks: {len(self.documents)}")
            return True
        return False
    
    def calculate_confidence(self, results: List[Dict], answer: str, citations: List[Dict]) -> float:
        """Calculate confidence based on search results and answer quality"""
        if not results:
            return 0.0
        
        scores = []
        for r in results[:3]:
            if 'score' in r:
                scores.append(min(r['score'], 1.0))
            else:
                scores.append(0.5)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        relevance_factor = avg_score * 0.4
        chunks_factor = min(len(results) / 5, 0.3)
        answer_length = len(answer) if answer else 0
        length_factor = min(answer_length / 500, 0.2)
        citations_factor = 0.1 if citations else 0.0
        
        confidence = relevance_factor + chunks_factor + length_factor + citations_factor
        confidence = min(confidence, 0.95)
        
        return round(confidence, 2)
    
    def self_critique(self, question: str, answer: str, contexts: List[str]) -> Dict:
        """
        Self-critique the generated answer
        Returns: {'supported': bool, 'complete': bool, 'confidence': float, 'feedback': str}
        """
        if not answer or len(answer) < 10:
            return {
                'supported': False,
                'complete': False,
                'confidence': 0.0,
                'feedback': 'Answer too short or empty'
            }
        
        prompt = f"""You are a critic evaluating an AI assistant's answer. Rate the answer on these criteria:

CONTEXT (information the AI had access to):
{chr(10).join(contexts[:2])}

QUESTION: {question}

ANSWER: {answer}

Evaluate based on:
1. SUPPORTED: Is the answer FULLY supported by the context? (Answer YES or NO)
2. COMPLETE: Does it completely answer the question? (Answer YES or NO)
3. CONFIDENCE: Rate 0-100 how confident you are in this answer

RESPOND IN THIS EXACT FORMAT:
Supported: [YES/NO]
Complete: [YES/NO]
Confidence: [NUMBER]
Feedback: [One short sentence explaining any issue]"""

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            
            result_text = response.choices[0].message.content
            
            # Parse response
            supported = "yes" in result_text.lower().split('\n')[0] if result_text else False
            complete = "yes" in result_text.lower().split('\n')[1] if len(result_text.split('\n')) > 1 else False
            
            # Extract confidence
            confidence = 0.5
            for line in result_text.split('\n'):
                if 'confidence' in line.lower():
                    try:
                        import re
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            confidence = int(numbers[0]) / 100
                    except:
                        confidence = 0.5
            
            # Extract feedback
            feedback = ""
            for line in result_text.split('\n'):
                if 'feedback' in line.lower():
                    feedback = line.replace('Feedback:', '').strip()
            
            return {
                'supported': supported,
                'complete': complete,
                'confidence': confidence,
                'feedback': feedback
            }
            
        except Exception as e:
            print(f"⚠️ Self-critique error: {e}")
            return {
                'supported': True,
                'complete': True,
                'confidence': 0.7,
                'feedback': ''
            }
    
    def regenerate_answer(self, question: str, chunks: List[Dict], feedback: str) -> str:
        """Regenerate answer based on feedback"""
        
        context = ""
        for i, chunk in enumerate(chunks[:3], 1):
            context += f"[Source {i}]: {chunk['content']}\n\n"
        
        prompt = f"""You are a helpful AI assistant. Generate a BETTER answer based on the feedback.

PREVIOUS FEEDBACK: {feedback}

IMPROVEMENT INSTRUCTIONS:
- Make sure your answer is FULLY supported by the context
- Be more specific and accurate
- Include citations to sources

CONTEXT:
{context}

QUESTION: {question}

BETTER ANSWER:"""
        
        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️ Regeneration error: {e}")
            return ""
    
    def ask(self, question: str, top_k: int = 5) -> Dict:
        """
        Self-Correcting RAG pipeline:
        1. Retrieve chunks
        2. Generate answer
        3. Self-critique
        4. If needed, retrieve more and regenerate
        5. Return final answer
        """
        
        if not self.is_ready:
            return {
                'answer': "⚠️ No documents loaded. Please add documents first.",
                'citations': [],
                'confidence': 0,
                'error': "No documents"
            }
        
        print(f"\n{'='*60}")
        print(f"❓ QUESTION: {question}")
        print(f"{'='*60}")
        
        # Step 1: Initial retrieval
        print("\n🔍 Step 1: Initial retrieval...")
        results = self.searcher.hybrid_search(question, top_k=top_k)
        
        if not results:
            results = self.searcher.keyword_search(question, top_k=top_k)
        
        if not results:
            return {
                'answer': "I couldn't find any relevant information in the documents.",
                'citations': [],
                'confidence': 0,
                'error': "No relevant chunks found"
            }
        
        print(f"   ✅ Retrieved {len(results)} chunks")
        
        # Step 2: Generate initial answer
        print("\n🤖 Step 2: Generating initial answer...")
        response = self.llm.generate_answer(question, results)
        current_answer = response.get('answer', '')
        citations = response.get('citations', [])
        
        # Extract context texts for critique
        contexts = [r['content'] for r in results[:3]]
        
        # Step 3: Self-critique loop
        for attempt in range(self.max_regeneration_attempts):
            print(f"\n🔍 Step 3: Self-critique (Attempt {attempt + 1})...")
            
            critique = self.self_critique(question, current_answer, contexts)
            
            print(f"   Supported: {critique['supported']}")
            print(f"   Complete: {critique['complete']}")
            print(f"   Confidence: {critique['confidence']*100:.0f}%")
            
            if critique.get('feedback'):
                print(f"   Feedback: {critique['feedback']}")
            
            # If answer is good enough, break
            if critique['supported'] and critique['complete'] and critique['confidence'] > 0.7:
                print("   ✅ Answer accepted!")
                break
            
            # If not, try to regenerate
            if attempt < self.max_regeneration_attempts - 1:
                print("\n🔄 Step 4: Regenerating better answer...")
                
                # Retrieve more chunks if needed
                if not critique['supported'] and len(results) < top_k * 2:
                    print("   📚 Retrieving additional chunks...")
                    more_results = self.searcher.hybrid_search(question, top_k=top_k * 2)
                    if more_results:
                        results = more_results
                        contexts = [r['content'] for r in results[:3]]
                
                # Regenerate with feedback
                new_answer = self.regenerate_answer(question, results[:5], critique.get('feedback', ''))
                if new_answer:
                    current_answer = new_answer
                    # Update citations
                    response = self.llm.generate_answer(question, results)
                    citations = response.get('citations', [])
        
        # Step 5: Calculate final confidence
        final_confidence = self.calculate_confidence(results, current_answer, citations)
        
        # Blend with critique confidence
        if 'critique' in locals():
            final_confidence = (final_confidence + critique.get('confidence', 0.5)) / 2
        
        return {
            'answer': current_answer,
            'citations': citations,
            'confidence': round(final_confidence, 2),
            'chunks_retrieved': len(results),
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'search_type': 'self-rag',
            'self_rag_attempts': attempt + 1,
            'critique_feedback': critique.get('feedback', '')
        }


# ============= Quick Test =============
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING SELF-CORRECTING RAG")
    print("=" * 60)
    
    ragent = RAGent()
    
    # Create a test file
    test_file = "test_doc.txt"
    with open(test_file, "w") as f:
        f.write("""
        IIT BHU offers AI courses including Machine Learning, Deep Learning, 
        and Natural Language Processing. The average salary for AI engineers is 35-55 LPA.
        Students from IIT BHU work at Google, Microsoft, Amazon, and other top companies.
        """)
    
    ragent.add_document(test_file)
    
    # Test question
    question = "What AI courses are offered at IIT BHU?"
    
    print("\n" + "=" * 60)
    response = ragent.ask(question)
    
    print("\n" + "=" * 60)
    print("📖 FINAL ANSWER")
    print("=" * 60)
    print(f"\n{response['answer']}\n")
    
    if response.get('citations'):
        print("📚 Sources:")
        for c in response['citations'][:3]:
            print(f"   - {c.get('source', 'Unknown')}")
    
    print(f"\n🎯 Confidence: {response['confidence']*100:.0f}%")
    print(f"🔄 Self-RAG attempts: {response.get('self_rag_attempts', 1)}")
    if response.get('critique_feedback'):
        print(f"📝 Feedback: {response['critique_feedback']}")
    
    print("\n✅ Self-Correcting RAG test complete!")
