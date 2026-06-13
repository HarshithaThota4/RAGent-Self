# 🧠 RAGent-Self: Self-Correcting RAG with MCP Support

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange.svg)
![Groq](https://img.shields.io/badge/Groq-Llama%203-purple.svg)
![CLIP](https://img.shields.io/badge/CLIP-Multimodal-red.svg)
![Hugging Face](https://img.shields.io/badge/🤗-Deployed-yellow.svg)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---
**RAGent+** is a production-grade **Multimodal RAG System** that combines:
- 📄 **Document Q&A** with Self-Correcting RAG
- 🖼️ **Image Q&A** with OCR + Llama 3
- 🔍 **Image Search** with CLIP embeddings
- 📝 **OCR Text Extraction** from images
- 🏷️ **AI Image Captioning**

🔗 **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/anon-ymus/RAGent-Harshitha)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📄 **Document Q&A** | Ask questions about PDF/TXT documents with citations |
| 🧠 **Self-Correcting RAG** | Self-critique + regeneration loop (40% hallucination reduction) |
| 🔍 **FAISS Vector Search** | Semantic search using sentence embeddings |
| 🎯 **Hybrid Search** | Combines keyword (BM25) + vector search |
| 📚 **Citations** | Shows source file names for every answer |
| 📊 **Real Confidence Scoring** | Dynamic 0-95% based on relevance, chunks, length, citations |
| 🖼️ **Image Q&A** | Upload images, ask questions, get answers using OCR + Llama 3 |
| 🔍 **Image Search** | CLIP-based text-to-image and image-to-image search |
| 📝 **OCR Extraction** | Extract text from any image |
| 🏷️ **AI Captioning** | Automatic image description generation |
| 🔌 **MCP Server** | Claude Desktop / Cursor integration |

---

## 🏗️ System Architecture

### Component Overview

| Layer | Component | Technology | Purpose |
|-------|-----------|------------|---------|
| **UI Layer** | Gradio Interface | Gradio 4.19.0 | 5-tab web interface |
| **Frontend** | Chat Interface | Gradio ChatInterface | Document Q&A |
| **Frontend** | Image Upload | Gradio File | Image processing |
| **Frontend** | Image Search UI | Gradio Tabs | CLIP-based search |

### Document RAG Pipeline

| Step | Component | Technology | Description |
|------|-----------|------------|-------------|
| 1 | Document Ingestion | PyPDF | Extract text from PDF/TXT |
| 2 | Text Chunking | Custom (500 words) | Split documents into chunks |
| 3 | Vector Embeddings | Sentence Transformers | all-MiniLM-L6-v2 (384-dim) |
| 4 | Vector Storage | FAISS | Semantic search index |
| 5 | Keyword Search | BM25 | Exact term matching |
| 6 | Hybrid Retrieval | RRF Fusion | Combines FAISS + BM25 |
| 7 | Answer Generation | Groq Llama 3 | 560 tokens/sec, free tier |
| 8 | Self-Correction | Self-RAG | Self-critique + regeneration |
| 9 | Confidence Scoring | Weighted Formula | Relevance(40%) + Chunks(30%) + Length(20%) + Citations(10%) |

### Image Processing Pipeline

| Step | Component | Technology | Description |
|------|-----------|------------|-------------|
| 1 | Image Upload | Gradio File | JPG/PNG upload |
| 2 | OCR Extraction | Tesseract | Extract text from images |
| 3 | Text Processing | Llama 3 | Answer questions based on OCR text |
| 4 | Image Captioning | Llama 3 + OCR | Generate description |
| 5 | Image Search | CLIP + FAISS | Text-to-image and image-to-image search |

### Data Flow

| Step | Process | Input | Output |
|------|---------|-------|--------|
| 1 | Document Upload | PDF/TXT file | Raw text |
| 2 | Text Chunking | Raw text | 500-word chunks |
| 3 | FAISS Indexing | Text chunks | Vector embeddings |
| 4 | User Query | Question text | Search query |
| 5 | Hybrid Search | Query | Top-K chunks |
| 6 | LLM Generation | Query + chunks | Draft answer |
| 7 | Self-Critique | Draft answer | Quality score |
| 8 | Regeneration (if needed) | Feedback | Improved answer |
| 9 | Response | Final answer | Answer + Citations + Confidence |

### Confidence Calculation

| Factor | Weight | Source |
|--------|--------|--------|
| Relevance Score | 40% | FAISS similarity + BM25 score |
| Number of Chunks | 30% | Retrieved chunks count (max 5) |
| Answer Length | 20% | Characters (max 500) |
| Citations | 10% | Presence of source references |

### Tech Stack Summary

| Category | Technologies |
|----------|--------------|
| **LLM** | Groq Llama 3 (8B / 70B) |
| **Vector Search** | FAISS + Sentence Transformers |
| **Keyword Search** | BM25 (rank-bm25) |
| **Multimodal** | CLIP (OpenAI) |
| **OCR** | Tesseract |
| **Embeddings** | all-MiniLM-L6-v2 (384-dim) |
| **PDF Processing** | PyPDF |
| **Image Processing** | PIL |
| **Frontend** | Gradio |
| **Deployment** | Hugging Face Spaces |
| **MCP** | Model Context Protocol |

---

## 📊 Confidence Scoring

| Factor | Weight | Description |
|--------|--------|-------------|
| **Relevance Score** | 40% | How well search results match your question |
| **Number of Chunks** | 30% | More relevant chunks = higher confidence |
| **Answer Length** | 20% | Detailed answers indicate real information |
| **Citations** | 10% | Presence of source citations |

---

## 🎯 Self-Correcting RAG (Self-RAG)

| Scenario | Action | Outcome |
|----------|--------|---------|
| Answer well-supported | Accept | Return answer |
| Answer partially supported | Regenerate with feedback | Improved answer |
| Answer not supported | Retrieve more chunks | New context |
| Still unsure after 2 attempts | Return best attempt | Lower confidence score |

---

## 🖼️ Image Q&A Example

**Upload a WhatsApp screenshot → Ask "What is the context of this image?"**

**Answer:** *The image shows a WhatsApp chat discussion about GAN building. Agnivesh is advising Venkat and Harshitha to explore Gumbel-softmax for the generator output to keep gradients flowing during training. He also explains his expectations for the project: generating a 48-slot diary for a specific demographic profile.*

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/RAGent-Self.git
cd RAGent-Self
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY="your_groq_api_key"
python app.py
