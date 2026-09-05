"""
Vienna-code agreement substitute eval.

This corpus (iamkaikai/amazing_logos_v2, AI-generated logo art) carries no
Vienna Classification codes -- those exist only for real registered
trademark filings. In their place this uses "shape_tag", a coarse category
parsed from the generation prompt's style-descriptor keywords (circle,
lines, stars, letters, etc; see data-pipeline/fetch_corpus.py). This is a
much weaker signal than a real Vienna code (single keyword vs. a structured
hierarchical figurative-element taxonomy) and is reported as such -- treat
these numbers as a sanity check that the embedding space correlates with
*some* independent semantic label, not as a Vienna-code-equivalent score.

Metric: for each item, look at its top-k nearest neighbors by fused score
(excluding itself) and report what fraction share the same shape_tag as the
query, averaged over a sample of the corpus and broken down per tag.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_corpus, fused_scores, get_weights  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = ap.parse_args()

    metadata, model_params, channels = load_corpus()
    items = metadata["items"]
    shape_tags = [it["shape_tag"] for it in items]
    weights = json.loads(args.weights) if args.weights else get_weights(model_params)

    n = len(items)
    rng = np.random.default_rng(7)
    sample_idx = rng.choice(n, size=min(args.n_samples, n), replace=False)

    query_channels = {name: arr[sample_idx] for name, arr in channels.items()}
    scores = fused_scores(query_channels, channels, weights)

    per_tag_agreement = defaultdict(list)
    overall_agreement = []
    tag_counts = defaultdict(int)
    for t in shape_tags:
        tag_counts[t] += 1

    for row_idx, item_idx in enumerate(sample_idx):
        query_tag = shape_tags[item_idx]
        if query_tag == "unclassified" or tag_counts[query_tag] < 2:
            continue
        order = np.argsort(-scores[row_idx])
        order = order[order != item_idx][: args.k]
        agree = np.mean([shape_tags[j] == query_tag for j in order])
        per_tag_agreement[query_tag].append(agree)
        overall_agreement.append(agree)

    per_tag_summary = {
        tag: {"mean_agreement": float(np.mean(vals)), "n": len(vals)}
        for tag, vals in per_tag_agreement.items()
    }
    overall = float(np.mean(overall_agreement)) if overall_agreement else 0.0

    print(f"overall top-{args.k} shape_tag agreement: {overall:.3f} (n={len(overall_agreement)})")
    for tag, summary in sorted(per_tag_summary.items(), key=lambda kv: -kv[1]["n"])[:10]:
        print(f"  {tag}: {summary['mean_agreement']:.3f} (n={summary['n']})")

    with open(args.out, "w") as f:
        json.dump(
            {
                "k": args.k,
                "weights": weights,
                "overall_agreement": overall,
                "n_queries": len(overall_agreement),
                "per_tag": per_tag_summary,
                "label_source": "shape_tag (prompt keyword, substitute for Vienna code -- see module docstring)",
            },
            f,
            indent=1,
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
