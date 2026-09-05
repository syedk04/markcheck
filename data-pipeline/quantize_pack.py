"""
Fit per-channel PCA, int8-quantize, and pack the corpus into the artifacts
the webapp reads directly. This must match the frozen data contract defined
by webapp/src/lib/types.ts, modelParams.ts, corpus.ts, quantize.ts, and
webapp/scripts/gen-placeholder-data.mjs exactly -- those files are the
source of truth for field names and shapes, not this docstring.

  webapp/public/data/index.bin         - one fixed-size record per item
  webapp/public/data/metadata.json     - {items: [{id, label, thumb, ...}]}
  webapp/public/data/model_params.json - record_size_bytes, record_order,
                                          per-channel {dim, weight, scale,
                                          zero_point}, and per-channel PCA
                                          basis ({mean, components})
  webapp/public/data/thumbnails/<id>.jpg

index.bin byte layout (record_order defines both the field order and the
byte offset order -- see model_params.json, not a hardcoded assumption here):
  dino_color (128 int8) | dino_grey (128 int8) | clip (64 int8) | fourier (32 int8)
  = 352 bytes/record, records in the same order as metadata.json's items array.

Quantization is a SINGLE scale/zero_point pair per channel (not per-dimension)
-- this matches webapp/src/lib/quantize.ts's dequantize(raw, {scale,
zero_point}) signature, which takes one scalar pair per channel. scale/zero_point
are calibrated from the global min/max of that channel's values across every
item and every dimension after PCA reduction.

Fourier features are already low-dimensional (32 real magnitude values), so
PCA is skipped for that channel -- consistent with the task spec and with
model_params.json's "pca" map only containing entries for
dino_color/dino_grey/clip.
"""
import json
import os
import re

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

PIPE_DIR = os.path.dirname(__file__)
EMBED_DIR = os.path.join(PIPE_DIR, "embeddings")
PROCESSED_DIR = os.path.join(PIPE_DIR, "processed")
RAW_DIR = os.path.join(PIPE_DIR, "raw")
WEBAPP_DATA_DIR = os.path.join(os.path.dirname(PIPE_DIR), "webapp", "public", "data")
THUMB_DIR = os.path.join(WEBAPP_DATA_DIR, "thumbnails")

RECORD_ORDER = ["dino_color", "dino_grey", "clip", "fourier"]
PCA_DIMS = {"dino_color": 128, "dino_grey": 128, "clip": 64}
FOURIER_DIM = 32
RECORD_DIMS = {"dino_color": 128, "dino_grey": 128, "clip": 64, "fourier": 32}
RECORD_SIZE = sum(RECORD_DIMS[c] for c in RECORD_ORDER)

# starting weights per task spec; eval/tune_weights.py may find a better
# config, in which case update this dict directly from its output
DEFAULT_WEIGHTS = {"dino_color": 0.5, "dino_grey": 0.2, "clip": 0.2, "fourier": 0.1}


def load_all_shards():
    shard_files = sorted(f for f in os.listdir(EMBED_DIR) if f.startswith("shard_"))
    all_ids, all_dc, all_dg, all_cl, all_fo = [], [], [], [], []
    for sf in shard_files:
        d = np.load(os.path.join(EMBED_DIR, sf))
        all_ids.extend(d["ids"].tolist())
        all_dc.append(d["dino_color"])
        all_dg.append(d["dino_grey"])
        all_cl.append(d["clip"])
        all_fo.append(d["fourier"])
    return (
        all_ids,
        np.concatenate(all_dc),
        np.concatenate(all_dg),
        np.concatenate(all_cl),
        np.concatenate(all_fo),
    )


def fit_pca(x: np.ndarray, n_components: int):
    pca = PCA(n_components=n_components, random_state=0)
    reduced = pca.fit_transform(x)
    return reduced, pca


def quantize_int8_global(x: np.ndarray):
    # single scale/zero_point for the whole channel (all items, all dims),
    # matching webapp/src/lib/quantize.ts's per-channel (not per-dim) params
    x_min = float(x.min())
    x_max = float(x.max())
    span = (x_max - x_min) if (x_max - x_min) > 1e-8 else 1.0
    scale = span / 255.0
    zero_point = x_min
    q = np.clip(np.round((x - zero_point) / scale), -128, 127).astype(np.int8)
    return q, scale, zero_point


def extract_label(prompt: str, shape_tag: str) -> str:
    # pulls the organization/brand name out of the "Simple elegant logo for
    # 'X', ..." prompt template used by this corpus; falls back to shape_tag
    # for the rare prompts that don't match the template
    m = re.search(r"logo for ['\"]?([^,'\"]+)['\"]?", prompt)
    if m:
        return m.group(1).strip()
    return shape_tag


def main():
    os.makedirs(WEBAPP_DATA_DIR, exist_ok=True)
    os.makedirs(THUMB_DIR, exist_ok=True)

    ids, dino_color, dino_grey, clip_emb, fourier = load_all_shards()
    n = len(ids)
    print(f"loaded {n} embedded items")

    manifest = {m["id"]: m for m in json.load(open(os.path.join(RAW_DIR, "manifest.json")))}

    channels_raw = {"dino_color": dino_color, "dino_grey": dino_grey, "clip": clip_emb}
    quantized = {}
    channel_params = {}
    pca_json = {}

    for name, arr in channels_raw.items():
        reduced, pca = fit_pca(arr, PCA_DIMS[name])
        q, scale, zero_point = quantize_int8_global(reduced)
        quantized[name] = q
        channel_params[name] = {
            "dim": PCA_DIMS[name],
            "weight": DEFAULT_WEIGHTS[name],
            "scale": scale,
            "zero_point": zero_point,
        }
        pca_json[name] = {
            "mean": pca.mean_.astype(np.float64).tolist(),
            "components": pca.components_.astype(np.float64).tolist(),
        }
        print(
            f"{name}: PCA {arr.shape[1]}->{PCA_DIMS[name]} dims, "
            f"explained variance {pca.explained_variance_ratio_.sum():.4f}, "
            f"scale={scale:.6f} zero_point={zero_point:.6f}"
        )

    fourier_q, fourier_scale, fourier_zero = quantize_int8_global(fourier)
    quantized["fourier"] = fourier_q
    channel_params["fourier"] = {
        "dim": FOURIER_DIM,
        "weight": DEFAULT_WEIGHTS["fourier"],
        "scale": fourier_scale,
        "zero_point": fourier_zero,
    }
    print(f"fourier: scale={fourier_scale:.6f} zero_point={fourier_zero:.6f}")

    # pack records, one contiguous byte blob per item, channel order = RECORD_ORDER
    index_path = os.path.join(WEBAPP_DATA_DIR, "index.bin")
    with open(index_path, "wb") as f:
        for i in range(n):
            for channel in RECORD_ORDER:
                f.write(quantized[channel][i].tobytes())

    index_size = os.path.getsize(index_path)
    print(f"wrote {index_path} ({index_size} bytes, {index_size/n:.1f} bytes/record)")

    # thumbnails
    thumb_total_bytes = 0
    for item_id in ids:
        src = os.path.join(PROCESSED_DIR, "color", f"{item_id}.png")
        im = Image.open(src).convert("RGB")
        im.thumbnail((150, 150), Image.LANCZOS)
        out_path = os.path.join(THUMB_DIR, f"{item_id}.jpg")
        im.save(out_path, "JPEG", quality=85)
        thumb_total_bytes += os.path.getsize(out_path)
    print(f"wrote {n} thumbnails, total {thumb_total_bytes/1e6:.2f} MB")

    # metadata.json: {items: [{id, label, thumb, ...}]}
    # shape_tag/source_prompt are extra fields beyond the frozen {id,label,thumb}
    # contract -- harmless additions the webapp ignores, used by
    # eval/vienna_agreement as the Vienna-code substitute label.
    metadata = {
        "items": [
            {
                "id": item_id,
                "label": extract_label(manifest[item_id]["source_prompt"], manifest[item_id]["shape_tag"]),
                "thumb": f"thumbnails/{item_id}.jpg",
                "shape_tag": manifest[item_id]["shape_tag"],
                "source_ref": "iamkaikai/amazing_logos_v2",
            }
            for item_id in ids
        ]
    }
    with open(os.path.join(WEBAPP_DATA_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    model_params = {
        "record_size_bytes": RECORD_SIZE,
        "record_order": RECORD_ORDER,
        "channels": channel_params,
        "pca": pca_json,
    }
    with open(os.path.join(WEBAPP_DATA_DIR, "model_params.json"), "w", encoding="utf-8") as f:
        json.dump(model_params, f)

    print("done.")


if __name__ == "__main__":
    main()
