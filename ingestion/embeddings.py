"""Shared local embedding helper (Chroma's bundled all-MiniLM-L6-v2, ONNX, no
external service) used for entity resolution here and for the vector store
in a later phase."""
import numpy as np
from chromadb.utils import embedding_functions

_ef = embedding_functions.DefaultEmbeddingFunction()


def embed(texts: list[str]) -> list[list[float]]:
    return _ef(texts)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(a_arr.dot(b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
