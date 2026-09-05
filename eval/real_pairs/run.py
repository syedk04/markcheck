"""
Real documented trademark dispute pairs eval -- NOT populated in this build.

Assembling ~100 real, citable trademark dispute pairs (e.g. from USPTO TTAB
decisions or WIPO UDRP filings) requires manual legal-research curation that
is out of scope for an autonomous pipeline run. This harness is left runnable
against a small hand-curated JSON so that work can be dropped in later
without touching scoring code.

Expected input format, pairs.json:
[
  {"query_id": "<corpus item id or external image path>", "expected_match_id": "<corpus item id>"},
  ...
]

Only items already present in the corpus (webapp/public/data/metadata.json)
can be scored directly; external images would need to go through
data-pipeline/preprocess.py + embed.py first.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_corpus, fused_scores, get_weights  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(os.path.dirname(__file__), "pairs.json"))
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.pairs):
        print(
            f"no pairs file at {args.pairs} -- this eval requires manually curated "
            "real trademark dispute pairs that were not assembled in this build. "
            "see module docstring."
        )
        return

    with open(args.pairs) as f:
        pairs = json.load(f)

    metadata, model_params, channels = load_corpus()
    ids = [it["id"] for it in metadata["items"]]
    id_to_idx = {i: idx for idx, i in enumerate(ids)}
    weights = json.loads(args.weights) if args.weights else get_weights(model_params)

    ranks = []
    for pair in pairs:
        if pair["query_id"] not in id_to_idx or pair["expected_match_id"] not in id_to_idx:
            print(f"skipping pair {pair}: id not found in corpus")
            continue
        q_idx = id_to_idx[pair["query_id"]]
        target_idx = id_to_idx[pair["expected_match_id"]]
        query_channels = {name: arr[q_idx : q_idx + 1] for name, arr in channels.items()}
        scores = fused_scores(query_channels, channels, weights)[0]
        order = np.argsort(-scores)
        rank = int(np.where(order == target_idx)[0][0]) + 1
        ranks.append(rank)

    if not ranks:
        print("no scoreable pairs found")
        return

    r1 = np.mean([r == 1 for r in ranks])
    r10 = np.mean([r <= 10 for r in ranks])
    print(f"real_pairs: n={len(ranks)} recall@1={r1:.3f} recall@10={r10:.3f}")


if __name__ == "__main__":
    main()
