"""
Proxy timing for the brute-force in-process scan the webapp will run in JS:
cosine similarity across all four channels against the full corpus, fused,
sorted. Measured in Python as a stand-in -- the real implementation will be
a typed-array loop in JS, but the operation count (dequantize + dot products
+ sort over N corpus items) is the same shape, so this gives a reasonable
order-of-magnitude estimate.
"""
import json
import os
import time

import numpy as np

from common import load_corpus, fused_scores, get_weights


def main():
    metadata, model_params, channels = load_corpus()
    n = len(metadata["items"])
    weights = get_weights(model_params)

    rng = np.random.default_rng(0)
    query_idxs = rng.choice(n, size=min(200, n), replace=False)

    latencies = []
    for idx in query_idxs:
        query_channels = {name: arr[idx : idx + 1] for name, arr in channels.items()}
        t0 = time.perf_counter()
        scores = fused_scores(query_channels, channels, weights)
        order = np.argsort(-scores[0])
        _ = order[:50]
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    print(f"corpus size: {n}")
    print(f"scan+sort latency p50={p50:.2f}ms p95={p95:.2f}ms (n={len(latencies)} queries)")

    index_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "webapp", "public", "data", "index.bin",
    )
    index_size = os.path.getsize(index_path)
    print(f"index.bin size: {index_size/1e6:.2f} MB ({index_size/n:.1f} bytes/item)")

    out = {
        "corpus_size": n,
        "p50_ms": p50,
        "p95_ms": p95,
        "index_bin_bytes": index_size,
    }
    with open(os.path.join(os.path.dirname(__file__), "latency_results.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
