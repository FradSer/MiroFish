"""
OASIS embedding adapter.

Forces OASIS recommendation embedding path to use CAMEL GeminiEmbedding
instead of local sentence-transformers/torch embedding calls.
"""

import os
from typing import List

from camel.embeddings import GeminiEmbedding


def setup_oasis_gemini_embedding() -> GeminiEmbedding:
    """
    Patch OASIS recsys embedding function to Gemini embedding.

    Returns:
        GeminiEmbedding instance for reuse and observability.
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("缺少 GEMINI_API_KEY（或 GOOGLE_API_KEY），无法启用 Gemini embedding")

    embedding_client = GeminiEmbedding(api_key=gemini_api_key)

    # OASIS recsys calls this symbol when use_openai_embedding=True.
    import oasis.social_platform.recsys as oasis_recsys

    def _generate_post_vector_gemini(corpus: List[str], batch_size: int = 1000):
        vectors = []
        for i in range(0, len(corpus), batch_size):
            batch = [str(item) for item in corpus[i:i + batch_size]]
            vectors.extend(embedding_client.embed_list(batch))
        return vectors

    oasis_recsys.generate_post_vector_openai = _generate_post_vector_gemini
    return embedding_client
