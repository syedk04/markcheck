"""
Shared scoring pipeline used by every eval harness so all three evals (and
the webapp) score candidates identically -- weighted fusion of per-channel
cosine similarity on the dequantized int8 vectors. Mirrors
webapp/src/lib/{corpus,quantize,scoring}.ts and reads the same
webapp/public/data/{metadata,model_params}.json + index.bin the webapp ships.
"""
import json
import os

import numpy as np

WEBAPP_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webapp", "public", "data"
)


def load_corpus():
    with open(os.path.join(WEBAPP_DATA_DIR, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    with open(os.path.join(WEBAPP_DATA_DIR, "model_params.json"), encoding="utf-8") as f:
        model_params = json.load(f)

    record_size = model_params["record_size_bytes"]
    record_order = model_params["record_order"]
    n = len(metadata["items"])

    raw = np.fromfile(os.path.join(WEBAPP_DATA_DIR, "index.bin"), dtype=np.int8)
    raw = raw.reshape(n, record_size)

    channels = {}
    offset = 0
    for channel in record_order:
        dims = model_params["channels"][channel]["dim"]
        channels[channel] = raw[:, offset : offset + dims].astype(np.float32)
        offset += dims

    dequant = {}
    for name, q in channels.items():
        params = model_params["channels"][name]
        dequant[name] = q * params["scale"] + params["zero_point"]

    return metadata, model_params, dequant


def get_weights(model_params: dict) -> dict:
    return {ch: model_params["channels"][ch]["weight"] for ch in model_params["record_order"]}


def cosine_sim_matrix(query_vecs: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    q = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8)
    c = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-8)
    return q @ c.T


def fused_scores(query_channels: dict, corpus_channels: dict, weights: dict) -> np.ndarray:
    total = None
    total_weight = 0.0
    for name, w in weights.items():
        sims = cosine_sim_matrix(query_channels[name], corpus_channels[name])
        total = sims * w if total is None else total + sims * w
        total_weight += w
    return total / total_weight if total_weight > 0 else total


def project_query_to_channels(raw_embeddings: dict, model_params: dict) -> dict:
    """raw_embeddings: dict of channel -> (n, raw_dim) float32 arrays for
    dino_color/dino_grey/clip (pre-PCA) and fourier (used as-is, already
    32-dim magnitude descriptor -- see data-pipeline/embed.py)."""
    out = {}
    for name in ["dino_color", "dino_grey", "clip"]:
        basis = model_params["pca"][name]
        mean = np.array(basis["mean"], dtype=np.float32)
        components = np.array(basis["components"], dtype=np.float32)
        centered = raw_embeddings[name] - mean
        out[name] = centered @ components.T
    out["fourier"] = raw_embeddings["fourier"]
    return out
