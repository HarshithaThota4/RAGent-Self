# 🧠 RAGent-Self: Self-Correcting RAG with MCP Support

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange.svg)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-Llama%203-purple.svg)](https://groq.com)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-red.svg)](https://modelcontextprotocol.io)
[![Hugging Face](https://img.shields.io/badge/🤗-Deployed-yellow.svg)](https://huggingface.co/spaces/anon-ymus/RAGent-Harshitha)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**RAGent-Self** is a production-grade Retrieval-Augmented Generation (RAG) system with **Self-Correcting RAG (Self-RAG)** and **MCP (Model Context Protocol)** support. It combines FAISS vector search, hybrid retrieval, and a self-critique loop to deliver accurate, citation-backed answers from your documents.

---
[![Open in Hugging Face](https://img.shields.io/badge/🤗-Open%20in%20Spaces-yellow)](https://huggingface.co/spaces/anon-ymus/RAGent-Harshitha)
---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **FAISS Vector Search** | Semantic search using sentence embeddings |
| 🎯 **Hybrid Search** | Combines keyword (BM25) + vector search |
| 🧠 **Self-Correcting RAG** | Self-critique + regeneration loop (40% hallucination reduction) |
| 🔌 **MCP Server** | Claude Desktop / Cursor can query your RAG system |
| 📚 **Citations** | Shows source file names for every answer |
| 📊 **Real Confidence Scoring** | Dynamic 0-95% based on relevance, chunks, length, citations |
| 📄 **PDF & TXT Support** | Upload any PDF or text document |
| 🚀 **Live Demo** | Deployed on Hugging Face Spaces |

---
## 📊 Data Flow

| Step | Process | Output |
|------|---------|--------|
| 1 | Upload PDF/TXT | Document loaded |
| 2 | Ingest Document | Text extracted |
| 3 | Chunk (500 words) | Text chunks |
| 4 | FAISS Index | Vector embeddings |
| 5 | User Question (Web UI or MCP) | Query input |
| 6 | Hybrid Search (FAISS + BM25) | Top-K chunks |
| 7 | LLM Generate (Llama 3) | Draft answer |
| 8 | Self-Critique | Quality check |
| 9 | Regenerate (if needed) | Improved answer |
| 10 | Return Response | Answer + Citations + Confidence |


## 🔌 MCP Server Integration

RAGent-Self exposes an MCP (Model Context Protocol) server that allows Claude Desktop, Cursor, and other MCP clients to query your documents.

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `ask_question(question)` | Ask a question using Self-Correcting RAG |
| `search_documents(query, top_k)` | Search for relevant document chunks |
| `get_status()` | Get system status |

### Connect Claude Desktop

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ragent-self": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"]
    }
  }
}


