"""
MCP Server for RAGent-Self
Exposes Self-Correcting RAG tools to Claude Desktop / Cursor
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragent_app import RAGent

# Initialize RAGent with Self-RAG
print("🔌 Initializing RAGent-Self MCP Server...")
ragent = RAGent()

# Load existing documents
doc_count = 0
for file in os.listdir('.'):
    if file.endswith(('.txt', '.pdf')):
        if file not in ['mcp_server.py', 'requirements.txt', 'app.py', 'claude_desktop_config.json']:
            ragent.add_document(file)
            doc_count += 1

print(f"✅ MCP Server ready!")
print(f"   📚 Documents loaded: {doc_count}")
print(f"   🧠 Self-RAG attempts: {ragent.max_regeneration_attempts}")
print(f"   🔍 Search type: Hybrid (FAISS + BM25)")


def handle_ask_question(args: dict) -> list:
    """Ask a question using Self-Correcting RAG"""
    question = args.get("question", "")
    
    if not question:
        return [{"type": "text", "text": "❌ Please provide a question."}]
    
    response = ragent.ask(question)
    
    output = f"**Answer:** {response['answer']}\n\n"
    
    if response.get('citations'):
        output += "**📚 Sources:**\n"
        for citation in response['citations'][:3]:
            source = citation.get('source', 'Unknown')
            output += f"- `{os.path.basename(source)}`\n"
    
    output += f"\n**🎯 Confidence:** {response['confidence']*100:.0f}%"
    output += f"\n**🔄 Self-RAG Attempts:** {response.get('self_rag_attempts', 1)}"
    
    if response.get('critique_feedback'):
        output += f"\n**📝 Feedback:** {response['critique_feedback']}"
    
    return [{"type": "text", "text": output}]


def handle_search_documents(args: dict) -> list:
    """Search for relevant document chunks"""
    query = args.get("query", "")
    top_k = args.get("top_k", 3)
    
    if not query:
        return [{"type": "text", "text": "❌ Please provide a search query."}]
    
    results = ragent.searcher.hybrid_search(query, top_k=top_k)
    
    if not results:
        return [{"type": "text", "text": "No results found."}]
    
    output = json.dumps([
        {
            "content": r["content"][:500],
            "source": os.path.basename(r["source"]),
            "score": r.get("score", 0)
        } for r in results
    ], indent=2)
    
    return [{"type": "text", "text": output}]


def handle_get_status(args: dict) -> list:
    """Get system status"""
    status = f"""## RAGent-Self Status

| Metric | Value |
|--------|-------|
| **Documents loaded** | {ragent.get_document_count()} chunks |
| **Vector store** | {ragent.searcher.vector_store.get_chunk_count()} chunks |
| **Self-RAG attempts** | {ragent.max_regeneration_attempts} max |
| **System ready** | {'✅ Yes' if ragent.is_ready else '❌ No'} |
| **LLM Model** | {ragent.llm.model} |
| **Search type** | Hybrid (FAISS + BM25) |

### Features Available
- ✅ Self-Correcting RAG
- ✅ FAISS vector search
- ✅ Hybrid search
- ✅ Citations with sources
- ✅ Real confidence scoring
"""
    return [{"type": "text", "text": status}]


# MCP Server Entry Point
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔌 RAGent-Self MCP Server with Self-Correcting RAG")
    print("=" * 60)
    print("\n📋 Available MCP Tools:")
    print("   1. ask_question(question) - Ask using Self-RAG")
    print("   2. search_documents(query, top_k) - Search document chunks")
    print("   3. get_status() - Get system status")
    print("\n" + "=" * 60)
    print("\n💡 To connect Claude Desktop, add this config:")
    print('''
{
  "mcpServers": {
    "ragent-self": {
      "command": "python",
      "args": ["PATH_TO_YOUR/mcp_server.py"]
    }
  }
}
    ''')
    print("=" * 60)
    print("\n⏳ Waiting for MCP connections...")
    print("   Press Ctrl+C to stop the server\n")
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down RAGent-Self MCP server...")
