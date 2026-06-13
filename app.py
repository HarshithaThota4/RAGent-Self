"""
RAGent+ - Multimodal RAG System
Document Q&A + Image Q&A (VQA) + Image Search + OCR + Captions
"""

import os
import shutil
import gradio as gr
from ragent_app import RAGent
from image_processor import ImageProcessor
from multimodal_retriever import MultimodalRetriever

# Initialize RAGent
print("🚀 Initializing RAGent...")
ragent = RAGent()

# Initialize Multimodal Components
print("🖼️ Initializing Image Processor...")
image_processor = ImageProcessor()

print("🔍 Initializing Multimodal Retriever...")
multimodal_retriever = MultimodalRetriever()

# Load existing documents
print("\n📁 Loading documents...")
for file in os.listdir('.'):
    if file.endswith(('.txt', '.pdf')):
        if file not in ['requirements.txt', 'app.py', 'api_config.py']:
            print(f"   📄 Loading: {file}")
            ragent.add_document(file)

# Load existing images for search
print("\n🖼️ Loading images for search...")
for file in os.listdir('.'):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        if file not in ['requirements.txt', 'app.py', 'api_config.py']:
            print(f"   🖼️ Indexing: {file}")
            multimodal_retriever.add_image(file)

print(f"\n✅ RAGent+ ready!")
print(f"   📚 Documents: {ragent.get_document_count()} chunks")
print(f"   🖼️ Images: {multimodal_retriever.get_image_count()} indexed")


# ============= Document Q&A Function =============

def chat_function(message, history):
    """Handle text chat messages for document Q&A"""
    if not message:
        return "Please ask a question about your documents."
    
    if ragent.get_document_count() == 0:
        return """📭 No documents found! Upload PDF/TXT files in the Upload Files tab."""

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


# ============= Upload Functions =============

def upload_file(file):
    """Handle document upload"""
    if file is None:
        return "❌ No file selected."
    
    success = ragent.add_document(file.name)
    
    if success:
        return f"✅ Uploaded: {os.path.basename(file.name)}\n📚 Total chunks: {ragent.get_document_count()}"
    else:
        return f"❌ Failed: {os.path.basename(file.name)}"


def upload_image(file):
    """Handle image upload for search index"""
    if file is None:
        return "❌ No image selected."
    
    temp_path = file.name if hasattr(file, 'name') else str(file)
    
    if not temp_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return f"❌ File must be PNG or JPG."
    
    permanent_path = os.path.join(os.getcwd(), os.path.basename(temp_path))
    
    try:
        shutil.copy2(temp_path, permanent_path)
        print(f"📁 Copied image to: {permanent_path}")
    except Exception as e:
        print(f"⚠️ Copy failed: {e}")
        permanent_path = temp_path
    
    # Also extract OCR text for search
    ocr_text = image_processor.extract_text_ocr(permanent_path)
    success = multimodal_retriever.add_image(permanent_path, {'text': ocr_text})
    
    if success:
        return f"✅ Image indexed: {os.path.basename(permanent_path)}\n🖼️ Total images: {multimodal_retriever.get_image_count()}"
    else:
        return f"❌ Failed to index image."


# ============= Image Q&A Functions =============

def analyze_image(image, question):
    """Analyze image with VQA, OCR, and captioning"""
    if image is None:
        return "Please upload an image.", "", ""
    
    if not question or question.strip() == "":
        question = "What is in this image?"
    
    analysis = image_processor.analyze_image(image, question)
    caption = analysis.get('caption', '')
    ocr_text = analysis.get('ocr_text', '')
    answer = analysis.get('answer', '')
    
    # Build response
    output = f"**❓ Question:** {question}\n\n"
    
    # Check if we have a valid answer
    if answer and len(answer) > 5 and "error" not in answer.lower():
        output += f"**✅ Answer:** {answer}\n\n"
    elif answer and len(answer) > 5:
        output += f"**✅ Answer:** {answer}\n\n"
    else:
        # Fallback to caption if no answer
        if caption:
            output += f"**✅ Answer:** {caption}\n\n"
        else:
            output += "**✅ Answer:** Could not analyze the image. Please try a clearer image.\n\n"
    
    if caption:
        output += f"**🏷️ Caption:** {caption}\n\n"
    
    if ocr_text:
        output += f"**📝 Text Found:**\n{ocr_text[:300]}\n\n"
    
    if not caption and not ocr_text:
        output += "No text or clear content detected.\n\n"
    
    return output, caption, ocr_text


# ============= Image Search Functions =============

def text_to_image_search(query):
    """Search images using text description"""
    if not query:
        return "Please enter a description to search for."
    
    results = multimodal_retriever.search_by_text(query, top_k=5)
    
    if not results:
        return "No matching images found. Upload images first."
    
    output = "### 🖼️ Search Results\n\n"
    output += "| Image | Similarity |\n"
    output += "|-------|------------|\n"
    
    for r in results:
        output += f"| `{r['filename']}` | {r['similarity']:.2f} |\n"
    
    return output


def image_to_image_search(image):
    """Search similar images using image query"""
    if image is None:
        return "Please upload an image to search with."
    
    results = multimodal_retriever.search_by_image(image, top_k=5)
    
    if not results:
        return "No similar images found."
    
    output = "### 🔍 Similar Images\n\n"
    output += "| Image | Similarity |\n"
    output += "|-------|------------|\n"
    
    for r in results:
        output += f"| `{r['filename']}` | {r['similarity']:.2f} |\n"
    
    return output


# ============= Status Function =============

def get_status():
    """Get system status"""
    vector_count = ragent.searcher.vector_store.get_chunk_count()
    image_count = multimodal_retriever.get_image_count()
    
    status = f"""### 📊 System Status

| Metric | Value |
|--------|-------|
| **Documents** | {ragent.get_document_count()} chunks |
| **Vector store** | {vector_count} indexed |
| **Images** | {image_count} indexed |
| **System ready** | {'✅' if ragent.is_ready else '❌'} |
| **LLM Model** | {ragent.llm.model} |

### 🔍 Features
- ✅ Document Q&A (PDF/TXT)
- ✅ Image Q&A (VQA - ask questions about images)
- ✅ Image Search (CLIP-based)
- ✅ OCR Text Extraction
- ✅ Image Captioning
- ✅ Citations & Confidence Scoring
"""
    return status


# ============= Create Gradio Interface =============

with gr.Blocks(title="RAGent+ - Multimodal RAG System") as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🧠 RAGent+</h1>
        <p style="font-size: 1.1em;">Multimodal RAG System</p>
        <p>Document Q&A | Image Q&A | Image Search | OCR | Captions</p>
    </div>
    """)
    
    with gr.Tabs():
        
        # ============= Tab 1: Document Q&A =============
        with gr.TabItem("💬 Document Q&A"):
            gr.Markdown("""
            ## Ask Questions About Your Documents
            
            Upload PDF or TXT files in the **Upload Files** tab, then ask questions here.
            
            **Examples:**
            - What are the main topics?
            - Summarize the key points
            - What does the document say about AI?
            """)
            gr.ChatInterface(
                fn=chat_function,
                title="Document Question & Answer",
                examples=[
                    "What are the main topics?",
                    "Summarize the key points",
                    "What does it say about AI?"
                ]
            )
        
        # ============= Tab 2: Upload Files =============
        with gr.TabItem("📤 Upload Files"):
            gr.Markdown("""
            ## Upload Documents or Images
            
            ### 📄 Documents (PDF/TXT)
            Upload documents to ask questions in the **Document Q&A** tab.
            
            ### 🖼️ Images (JPG/PNG)
            Upload images to:
            - Ask questions in **Image Q&A** tab
            - Search in **Image Search** tab
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📄 Upload Document")
                    doc_input = gr.File(label="Choose PDF or TXT file", file_types=[".pdf", ".txt"])
                    doc_status = gr.Markdown("📭 No file uploaded")
                    doc_input.change(fn=upload_file, inputs=[doc_input], outputs=[doc_status])
                
                with gr.Column():
                    gr.Markdown("### 🖼️ Upload Image")
                    img_input = gr.File(label="Choose JPG or PNG file", file_types=[".jpg", ".jpeg", ".png"])
                    img_status = gr.Markdown("📭 No image uploaded")
                    img_input.change(fn=upload_image, inputs=[img_input], outputs=[img_status])
            
            gr.Markdown("---")
            gr.Markdown("### 📊 Current Status")
            status_text = gr.Markdown(get_status())
            refresh_btn = gr.Button("🔄 Refresh Status")
            refresh_btn.click(fn=get_status, outputs=[status_text])
        
        # ============= Tab 3: Image Q&A (VQA) =============
        with gr.TabItem("🖼️ Image Q&A"):
            gr.Markdown("""
            ## Ask Questions About Images
            
            Upload an image and ask any question about it. The AI will analyze the image and answer.
            
            **Example questions:**
            - What is in this image?
            - What is the context of this image?
            - What does the text say?
            - How many people are there?
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    img_vqa = gr.Image(type="filepath", label="Upload Image", height=350)
                    vqa_question = gr.Textbox(
                        label="Your Question",
                        lines=2,
                        placeholder="e.g., What is in this image?",
                        value="What is in this image?"
                    )
                    vqa_btn = gr.Button("🔍 Ask Question", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    vqa_answer = gr.Markdown("💡 **Answer will appear here...**")
                    vqa_caption = gr.Textbox(label="Generated Caption", lines=2, interactive=False)
                    vqa_ocr = gr.Textbox(label="Extracted Text (OCR)", lines=5, interactive=False)
            
            vqa_btn.click(
                fn=analyze_image,
                inputs=[img_vqa, vqa_question],
                outputs=[vqa_answer, vqa_caption, vqa_ocr]
            )
        
        # ============= Tab 4: Image Search =============
        with gr.TabItem("🔍 Image Search"):
            gr.Markdown("""
            ## Search Images
            
            Two ways to search:
            - **Text-to-Image:** Describe what you want to find
            - **Image-to-Image:** Upload a reference image to find similar ones
            """)
            
            with gr.Tabs():
                with gr.TabItem("Text to Image"):
                    search_text_input = gr.Textbox(
                        label="Describe what you want to find",
                        lines=2,
                        placeholder="e.g., a red car, beach with palms, document with text",
                        scale=3
                    )
                    text_search_btn = gr.Button("🔍 Search", variant="primary")
                    text_results = gr.Markdown("### Results will appear here...")
                    
                    text_search_btn.click(
                        fn=text_to_image_search,
                        inputs=[search_text_input],
                        outputs=[text_results]
                    )
                
                with gr.TabItem("Image to Image"):
                    search_image_input = gr.Image(type="filepath", label="Upload reference image", height=200)
                    image_search_btn = gr.Button("🔍 Find Similar", variant="primary")
                    image_results = gr.Markdown("### Results will appear here...")
                    
                    image_search_btn.click(
                        fn=image_to_image_search,
                        inputs=[search_image_input],
                        outputs=[image_results]
                    )
        
        # ============= Tab 5: About =============
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            # 🧠 RAGent+ - Multimodal RAG System
            
            ## Features
            
            | Feature | Description | Status |
            |---------|-------------|--------|
            | 📄 Document Q&A | Ask questions about PDF/TXT documents | ✅ Live |
            | 🖼️ Image Q&A (VQA) | Ask questions about images | ✅ Live |
            | 🔍 Image Search | Find images by text or by example | ✅ Live |
            | 📝 OCR | Extract text from images | ✅ Live |
            | 🏷️ Captions | AI-generated image descriptions | ✅ Live |
            | 📚 Citations | Source tracking for documents | ✅ Live |
            | 🎯 Confidence | Dynamic scoring (0-95%) | ✅ Live |
            
            ## Tech Stack
            
            | Component | Technology |
            |-----------|------------|
            | **LLM** | Groq Llama 3 (560 tok/sec) |
            | **Document Search** | FAISS + Sentence Transformers |
            | **Image Q&A** | BLIP VQA (Visual Question Answering) |
            | **Image Search** | CLIP (OpenAI) |
            | **Image Captioning** | BLIP |
            | **OCR** | Tesseract |
            | **UI** | Gradio |
            
            ## How to Use
            
            ### Document Q&A
            1. Go to **Upload Files** tab
            2. Upload PDF or TXT documents
            3. Go to **Document Q&A** tab
            4. Ask questions about your documents
            
            ### Image Q&A
            1. Go to **Image Q&A** tab
            2. Upload an image
            3. Ask a question about the image
            4. Get AI-generated answer
            
            ### Image Search
            1. Go to **Upload Files** tab
            2. Upload images to index
            3. Go to **Image Search** tab
            4. Search by text or by example image
            
            ---
            *© 2026 RAGent+ - Multimodal RAG System*
            """)

# Launch
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
