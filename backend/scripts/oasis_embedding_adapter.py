"""
OASIS embedding adapter.

Forces ALL OASIS recommendation embedding paths to use CAMEL GeminiEmbedding
instead of local sentence-transformers/torch embedding calls.

Patched code paths:
  1. generate_post_vector_openai  (twhin-bert batch path)
  2. global `model` variable       (SentenceTransformer used by rec_sys_personalized_with_trace)
  3. rec_sys_personalized_with_trace  (post_id safety filter)
"""

import os
from typing import List

import numpy as np
from camel.embeddings import GeminiEmbedding


class _GeminiSentenceTransformerShim:
    """Drop-in replacement for SentenceTransformer that delegates to GeminiEmbedding.

    OASIS recsys calls ``model.encode(text)`` expecting a numpy array.
    This shim translates that to a Gemini API call.
    """

    def __init__(self, gemini: GeminiEmbedding):
        self._gemini = gemini

    def encode(self, text):
        if isinstance(text, str):
            vec = self._gemini.embed_list([text])[0]
        else:
            vec = self._gemini.embed_list([str(t) for t in text])
            return np.array(vec)
        return np.array(vec)


def setup_oasis_gemini_embedding() -> GeminiEmbedding:
    """
    Patch OASIS recsys to use Gemini embedding everywhere.

    Returns:
        GeminiEmbedding instance for reuse and observability.
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("缺少 GEMINI_API_KEY（或 GOOGLE_API_KEY），无法启用 Gemini embedding")

    embedding_client = GeminiEmbedding(api_key=gemini_api_key)

    import oasis.social_platform.recsys as oasis_recsys

    # --- Path 1: batch vector generation (twhin-bert replacement) ---
    def _generate_post_vector_gemini(corpus: List[str], batch_size: int = 1000):
        vectors = []
        for i in range(0, len(corpus), batch_size):
            batch = [str(item) for item in corpus[i:i + batch_size]]
            vectors.extend(embedding_client.embed_list(batch))
        return vectors

    oasis_recsys.generate_post_vector_openai = _generate_post_vector_gemini

    # --- Path 2: replace global SentenceTransformer model ---
    shim = _GeminiSentenceTransformerShim(embedding_client)
    oasis_recsys.model = shim

    # Also prevent lazy re-initialization from overwriting our shim.
    oasis_recsys.get_recsys_model = lambda recsys_type=None: shim

    # --- Path 3: rec_sys_personalized_with_trace post_id safety filter ---
    # camel-oasis==0.2.5 assumes every trace has "post_id";
    # actions like follow/mute do not include it.
    original_recsys_with_trace = oasis_recsys.rec_sys_personalized_with_trace

    def _safe_rec_sys_personalized_with_trace(*args, **kwargs):
        if "trace_table" in kwargs and isinstance(kwargs["trace_table"], list):
            kwargs["trace_table"] = [
                trace for trace in kwargs["trace_table"]
                if isinstance(trace, dict) and trace.get("post_id") is not None
            ]
            return original_recsys_with_trace(*args, **kwargs)

        if len(args) >= 3 and isinstance(args[2], list):
            mutable_args = list(args)
            mutable_args[2] = [
                trace for trace in mutable_args[2]
                if isinstance(trace, dict) and trace.get("post_id") is not None
            ]
            return original_recsys_with_trace(*mutable_args, **kwargs)

        return original_recsys_with_trace(*args, **kwargs)

    oasis_recsys.rec_sys_personalized_with_trace = _safe_rec_sys_personalized_with_trace

    # platform.py uses `from .recsys import rec_sys_personalized_with_trace`,
    # so it holds a direct reference that the module-level patch doesn't reach.
    import oasis.social_platform.platform as oasis_platform
    oasis_platform.rec_sys_personalized_with_trace = _safe_rec_sys_personalized_with_trace

    return embedding_client
