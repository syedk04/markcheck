#!/bin/bash
set -e
until [ -f data-pipeline/embeddings/shard_0011.npz ]; do sleep 15; done
echo "EMBEDDING_DONE"
python data-pipeline/quantize_pack.py
echo "QUANTIZE_DONE"
python eval/tune_weights.py
echo "TUNE_DONE"
python eval/vienna_agreement/run.py
echo "VIENNA_DONE"
python eval/latency_bench.py
echo "LATENCY_DONE"
python eval/real_pairs/run.py
echo "ALL_DONE"
