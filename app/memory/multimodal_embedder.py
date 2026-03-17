from typing import List, Optional
import numpy as np

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except Exception:
    CLIPProcessor = None
    CLIPModel = None

class MultimodalEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        if CLIPModel and CLIPProcessor:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()

    def embed_images(self, images: List["PIL.Image.Image"]):
        if not self.model or not self.processor:
            return []
        inputs = self.processor(images=images, return_tensors="pt")
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()

    def embed_texts(self, texts: List[str]):
        if not self.model or not self.processor:
            return []
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()
