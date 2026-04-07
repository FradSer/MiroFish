"""Stub for transformers -- local HuggingFace models are not used.

Embedding is handled by oasis_embedding_adapter.py via Gemini API.
Provides AutoModel/AutoTokenizer placeholders for import compatibility.
"""


class AutoModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError("Local models disabled. Use Gemini API via oasis_embedding_adapter.")


class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError("Local models disabled. Use Gemini API via oasis_embedding_adapter.")
