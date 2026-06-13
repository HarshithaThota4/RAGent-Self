"""
CLIP Embeddings for Multimodal RAG - Fixed
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# Global variables for lazy loading
_CLIP_MODEL = None
_CLIP_PROCESSOR = None

def _load_clip():
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None:
        print("🔧 Loading CLIP model...")
        _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_MODEL.eval()
        print("✅ CLIP model loaded")
    return _CLIP_MODEL, _CLIP_PROCESSOR


class CLIPEmbeddings:
    def __init__(self):
        self.device = "cpu"
        self.model, self.processor = _load_clip()
        self.model = self.model.to(self.device)
        print(f"✅ CLIP embeddings ready on {self.device}")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        model, processor = _load_clip()
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            # Ensure we get a tensor and convert to numpy
            if hasattr(outputs, 'pooler_output'):
                text_features = outputs.pooler_output
            else:
                text_features = outputs
        return text_features.cpu().numpy()[0]
    
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        model, processor = _load_clip()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            # Ensure we get a tensor and convert to numpy
            if hasattr(outputs, 'pooler_output'):
                image_features = outputs.pooler_output
            else:
                image_features = outputs
        return image_features.cpu().numpy()[0]


if __name__ == "__main__":
    clip = CLIPEmbeddings()
    print("✅ CLIP embeddings ready!")
