"""
Real LLM Agent with Groq API
"""

import os
from typing import List, Dict

try:
    import api_config
    print("✅ API config loaded!")
except ImportError:
    print("❌ Please create api_config.py")
    exit(1)

try:
    from groq import Groq
    print("✅ Groq imported!")
except ImportError:
    print("❌ Please install groq")
    exit(1)


class RealLLMAgent:
    def __init__(self, model_choice: str = "llama-8b"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        self.available_models = {
            "llama-8b": "llama-3.1-8b-instant",
            "llama-70b": "llama-3.3-70b-versatile",
        }
        
        self.model = self.available_models.get(model_choice, "llama-3.1-8b-instant")
        print(f"✅ RealLLMAgent initialized with: {self.model}")
    
    def generate_answer(self, question: str, context_chunks: List[Dict]) -> Dict:
        context = ""
        citations = []
        
        for i, chunk in enumerate(context_chunks[:3], 1):
            context += f"[Source {i}]: {chunk['content']}\n\n"
            citations.append({
                'id': i,
                'source': chunk.get('source', 'unknown'),
                'text': chunk['content'][:150] + "..."
            })
        
        prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            
            answer = completion.choices[0].message.content
            
            return {
                'answer': answer,
                'citations': citations,
                'model': self.model
            }
            
        except Exception as e:
            return {
                'answer': f"Error: {str(e)}",
                'citations': [],
                'model': self.model
            }
