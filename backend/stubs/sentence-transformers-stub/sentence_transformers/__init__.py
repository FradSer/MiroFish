"""Stub for sentence_transformers.

Provides a SentenceTransformer class that raises at runtime.
The real embedding is handled by oasis_embedding_adapter.py (Gemini API).
"""


class SentenceTransformer:
    """Placeholder that is monkey-patched away before use."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Local SentenceTransformer is disabled. "
            "Embedding is handled by oasis_embedding_adapter.py via Gemini API."
        )
