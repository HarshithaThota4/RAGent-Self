"""
RAGent - Production RAG System
Document Q&A with FAISS + Hybrid Search + Llama 3
"""

import os
import gradio as gr
from ragent_app import RAGent

# Initialize RAGent
print("🚀 Initializing RAGent...")
ragent = RAGent()

# Load existing documents
print("\n📁 Loading documents...")
for file in os.listdir('.'):
    if file.endswith(('.txt', '.pdf')):
        if file not in ['requirements.txt', 'app.py', 'api_config.py']:
            print(f"   📄 Loading: {file}")
            ragent.add_document(file)

print(f"\n✅ RAGent ready! Loaded {ragent.get_document_count()} chunks")


# ============= Document Q&A Function =============

def chat_function(message, history):
    """Handle text chat messages for document Q&A"""
    
    if not message:
        return "Please ask a question about your documents."
    
    if ragent.get_document_count() == 0:
        return """📭 **No documents found!**
Please upload PDF or TXT files to this Space.
**How to add documents:**
1. Click the 'Upload Documents' tab
2. Upload your PDF or TXT files
3. Return here and ask your question"""

    try:
        response = ragent.ask(message)
        answer = response.get('answer', 'No answer generated.')
        citations = response.get('citations', [])
        confidence = response.get('confidence', 0) * 100
        search_type = response.get('search_type', 'hybrid')
        
        output = answer
        
        if citations:
            output += "\n\n---\n**📚 Sources:**\n"
            for citation in citations[:3]:
                source = citation.get('source', 'Unknown')
                filename = os.path.basename(source)
                output += f"- 📄 `{filename}`\n"
        
        output += f"\n\n---\n*🎯 Confidence: {confidence:.0f}% | 🔍 Search: {search_type}*"
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============= Upload Function =============

def upload_file(file):
    """Handle file upload"""
    if file is None:
        return "❌ No file selected."
    
    success = ragent.add_document(file.name)
    
    if success:
        return f"✅ Uploaded: {os.path.basename(file.name)}\n📚 Total chunks: {ragent.get_document_count()}"
    else:
        return f"❌ Failed: {os.path.basename(file.name)}"


# ============= Status Function =============

def get_status():
    """Get system status"""
    vector_count = ragent.searcher.vector_store.get_chunk_count()
    
    status = f"""### 📊 System Status
| Metric | Value |
|--------|-------|
| **Documents loaded** | {ragent.get_document_count()} chunks |
| **Vector store** | {vector_count} chunks indexed |
| **System ready** | {'✅ Yes' if ragent.is_ready else '❌ No'} |
| **LLM Model** | {ragent.llm.model} |
| **Search type** | Hybrid (FAISS + Keyword) |
### 🔍 Features
- ✅ Document Q&A (PDF/TXT)
- ✅ FAISS vector search
- ✅ Hybrid keyword + semantic search
- ✅ Citations with source tracking
- ✅ Real confidence scoring
"""
    return status


# ============= Create Gradio Interface =============

with gr.Blocks(title="RAGent - Production RAG System", theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🧠 RAGent</h1>
        <p style="font-size: 1.1em;">Production RAG System with FAISS + Hybrid Search</p>
        <p>Document Q&A | Citations | Real Confidence Scoring</p>
    </div>
    """)
    
    with gr.Tabs():
        
        # ============= Tab 1: Document Q&A =============
        with gr.TabItem("💬 Document Q&A"):
            gr.Markdown("""
            ## Ask Questions About Your Documents
            
            Upload PDF or TXT files in the **Upload Documents** tab, then ask questions here.
            RAGent will search through your documents and provide answers with citations.
            
            **Example questions:**
            - What are the main topics in this document?
            - Summarize the key points
            - What does the document say about AI?
            """)
            gr.ChatInterface(
                fn=chat_function,
                title="Document Question & Answer",
                examples=[
                    "What are the main topics in this document?",
                    "Summarize the key points",
                    "What does the document say about AI?"
                ]
            )
        
        # ============= Tab 2: Upload Documents =============
        with gr.TabItem("📤 Upload Documents"):
            gr.Markdown("""
            ## Upload Documents (PDF or TXT)
            
            Upload your documents here. After uploading, go to **Document Q&A** tab to ask questions.
            """)
            
            with gr.Row():
                file_input = gr.File(label="Choose PDF or TXT file", file_types=[".pdf", ".txt"])
                upload_status = gr.Markdown("📭 No file uploaded")
            
            file_input.change(
                fn=upload_file,
                inputs=[file_input],
                outputs=[upload_status]
            )
            
            gr.Markdown("---")
            gr.Markdown("### 📊 Current Status")
            status_text = gr.Markdown(get_status())
            refresh_btn = gr.Button("🔄 Refresh Status")
            refresh_btn.click(fn=get_status, outputs=[status_text])
        
        # ============= Tab 3: About =============
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            # 🧠 RAGent - Production RAG System
            
            ## What is RAGent?
            RAGent is a **production-grade Retrieval-Augmented Generation (RAG)** system that:
            - Answers questions from your documents (PDF/TXT)
            - Provides citations from sources
            - Uses Llama 3 for accurate responses
            - Calculates REAL confidence scores based on search quality
            
            ## Features
            
            | Feature | Description | Status |
            |---------|-------------|--------|
            | 📄 PDF Support | Upload any PDF document | ✅ Live |
            | 📝 TXT Support | Upload text files | ✅ Live |
            | 🔍 Keyword Search | BM25-style keyword matching | ✅ Live |
            | 🧠 Vector Search | FAISS semantic search | ✅ Live |
            | 🎯 Hybrid Search | Combines keyword + vector | ✅ Live |
            | 📚 Citations | Shows source of information | ✅ Live |
            | 🎯 Real Confidence | Dynamic scoring based on search quality | ✅ Live |
            
            ## How Confidence is Calculated
            
            | Factor | Weight | Description |
            |--------|--------|-------------|
            | Relevance Score | 40% | How well search results match your question |
            | Number of Chunks | 30% | More relevant chunks = higher confidence |
            | Answer Length | 20% | Detailed answers indicate real information |
            | Citations | 10% | Presence of source citations |
            
            ## Tech Stack
            
            | Component | Technology |
            |-----------|------------|
            | **LLM** | Groq Llama 3 (Free tier, 560 tokens/sec) |
            | **Vector Search** | FAISS + Sentence Transformers |
            | **Search** | Hybrid (Keyword + FAISS) |
            | **Framework** | Python, Gradio |
            | **Deployment** | Hugging Face Spaces |
            
            ## How to Use
            
            1. Go to **Upload Documents** tab
            2. Upload your PDF or TXT files
            3. Go to **Document Q&A** tab
            4. Ask any question about your documents!
            
            ---
            *© 2026 RAGent - Production RAG System with FAISS, Hybrid Search, and Real Confidence Scoring*
            """)

# Launch
if __name__ == "__main__":
    demo.launch()
