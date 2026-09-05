"""
Pull a fixed-size sample of the corpus source parquet into data-pipeline/raw/
as individual image files plus a manifest.json with id, source prompt, and a
coarse shape-tag label parsed from the generation prompt (stand-in for a
Vienna classification code -- see docs/DATASET.md for why).
"""
import argparse
import hashlib
import io
import json
import os
import random
import re

import pyarrow.parquet as pq
from PIL import Image

PARQUET_PATH = (
    r"C:\Users\6ix4o\.cache\huggingface\hub\datasets--iamkaikai--amazing_logos_v2"
    r"\snapshots\0796019240a28fc14beb9db7e23628842036b32d\data"
    r"\train-00000-of-00001-243b69021790579b.parquet"
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
MANIFEST_PATH = os.path.join(RAW_DIR, "manifest.json")

# controlled vocabulary of shape/style descriptors that appear in the
# generation prompts -- used as a coarse category label in place of a
# real Vienna classification code, since this corpus has no such field.
SHAPE_VOCAB = [
    "Circle", "Square", "Triangle", "Lines", "Stars", "Star", "Letters",
    "Dots", "Waves", "Cross", "Animals", "Rectangles", "Squares",
    "Outline", "Symmetrical", "Asymmetrical", "Curve", "Point",
    "Concentric", "Figurative", "Spiral", "Diamond", "Hexagon",
]


def parse_shape_tag(prompt: str) -> str:
    for word in SHAPE_VOCAB:
        if re.search(r"\b" + re.escape(word) + r"\b", prompt):
            return word.lower()
    return "unclassified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    table = pq.read_table(PARQUET_PATH)
    texts = table.column("text").to_pylist()
    images = table.column("image").to_pylist()
    total = len(texts)
    print(f"source parquet rows: {total}")

    rng = random.Random(args.seed)
    idxs = list(range(total))
    rng.shuffle(idxs)
    idxs = idxs[: args.n]

    manifest = []
    for rank, i in enumerate(idxs):
        prompt = texts[i]
        img_bytes = images[i]["bytes"]
        # stable id derived from row content, not just position, so re-runs
        # with a different sample size still line up on shared items
        item_id = hashlib.sha1(img_bytes[:256] + str(i).encode()).hexdigest()[:16]
        out_path = os.path.join(RAW_DIR, f"{item_id}.png")
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        manifest.append({
            "id": item_id,
            "source_index": i,
            "source_prompt": prompt,
            "shape_tag": parse_shape_tag(prompt),
        })
        if rank % 1000 == 0:
            print(f"wrote {rank}/{len(idxs)}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    print(f"done. wrote {len(manifest)} images to {RAW_DIR}")


if __name__ == "__main__":
    main()
