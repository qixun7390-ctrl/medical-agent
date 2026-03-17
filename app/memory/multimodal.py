from typing import Optional, Dict, List
from app.memory.multimodal_embedder import MultimodalEmbedder

class MultimodalMemory:
    def __init__(self):
        self.store = {}
        self.embedder = MultimodalEmbedder()

    def put_image(self, user_id: str, item_id: str, image, meta: Optional[Dict] = None):
        vecs = self.embedder.embed_images([image])
        if not vecs:
            return
        self.store[(user_id, item_id)] = {"embedding": vecs[0], "meta": meta or {}}

    def put_text(self, user_id: str, item_id: str, text: str, meta: Optional[Dict] = None):
        vecs = self.embedder.embed_texts([text])
        if not vecs:
            return
        self.store[(user_id, item_id)] = {"embedding": vecs[0], "meta": meta or {}}

    def get(self, user_id: str, item_id: str):
        return self.store.get((user_id, item_id))
