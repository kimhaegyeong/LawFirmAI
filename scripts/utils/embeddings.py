from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class SentenceEmbedder:
    def __init__(self, model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        # try to infer dimension
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**encoded)
            # mean pooling
            last_hidden = outputs.last_hidden_state  # (B, T, H)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (B, T, 1)
            masked = last_hidden * attention_mask
            sum_vec = masked.sum(dim=1)
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            sent_vec = sum_vec / lengths
            vec = sent_vec.detach().cpu().numpy()
            if normalize:
                norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
                vec = vec / norms
            embeddings.append(vec)
        if not embeddings:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(embeddings).astype(np.float32)
