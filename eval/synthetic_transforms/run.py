"""
Synthetic-transform retrieval eval: take a sample of corpus images, apply a
battery of geometric/photometric transforms, re-embed each transformed
version with the exact model calls used to build the index, then check
whether the un-transformed original comes back as the top match against the
full corpus under the same weighted-fusion scoring the webapp will use.

Reports recall@1, recall@10, and mAP (here just 1/rank_of_correct_item,
since there is exactly one correct match per query) per transform type.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data-pipeline"))

from common import load_corpus, fused_scores, project_query_to_channels, get_weights  # noqa: E402

os.environ.setdefault("USE_TF", "0")

PIPE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data-pipeline"
)
PROCESSED_COLOR_DIR = os.path.join(PIPE_DIR, "processed", "color")


def tf_rotate(im):
    return im.rotate(25, fillcolor=(255, 255, 255), expand=False)


def tf_recolor(im):
    arr = np.array(im).astype(np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * 0.6 + 40, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * 1.3, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def tf_scale(im):
    w, h = im.size
    small = im.resize((int(w * 0.5), int(h * 0.5)), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(small, ((w - small.width) // 2, (h - small.height) // 2))
    return canvas


def tf_crop(im):
    w, h = im.size
    cropped = im.crop((int(w * 0.1), int(h * 0.1), w, h))
    return cropped.resize((w, h), Image.LANCZOS)


def tf_aspect(im):
    w, h = im.size
    stretched = im.resize((int(w * 1.4), int(h * 0.7)), Image.LANCZOS)
    return stretched.resize((w, h), Image.LANCZOS)


def tf_noise(im):
    arr = np.array(im).astype(np.float32)
    noise = np.random.default_rng(0).normal(0, 18, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


TRANSFORMS = {
    "rotation": tf_rotate,
    "recolor": tf_recolor,
    "scale": tf_scale,
    "crop": tf_crop,
    "aspect_distortion": tf_aspect,
    "gaussian_noise": tf_noise,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=150)
    ap.add_argument("--weights", default=None, help="JSON string overriding fusion weights")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = ap.parse_args()

    os.environ["USE_TF"] = "0"
    from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
    import sys as _sys
    _sys.path.insert(0, PIPE_DIR)
    from embed import embed_pil_images  # noqa: E402

    metadata, model_params, corpus_channels = load_corpus()
    ids = [it["id"] for it in metadata["items"]]
    weights = json.loads(args.weights) if args.weights else get_weights(model_params)

    rng = np.random.default_rng(42)
    query_ids = rng.choice(ids, size=min(args.n_queries, len(ids)), replace=False)

    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()

    results = {t: {"ranks": []} for t in TRANSFORMS}

    for transform_name, transform_fn in TRANSFORMS.items():
        color_imgs, grey_imgs, target_indices = [], [], []
        for qid in query_ids:
            im = Image.open(os.path.join(PROCESSED_COLOR_DIR, f"{qid}.png")).convert("RGB")
            transformed = transform_fn(im)
            color_imgs.append(transformed)
            grey_imgs.append(transformed.convert("L").convert("RGB"))
            target_indices.append(ids.index(qid))

        raw = embed_pil_images(color_imgs, grey_imgs, dino_proc, dino_model, clip_proc, clip_model)
        query_channels = project_query_to_channels(raw, model_params)
        scores = fused_scores(query_channels, corpus_channels, weights)  # (n_queries, n_corpus)

        ranks = []
        for row_idx, target_idx in enumerate(target_indices):
            order = np.argsort(-scores[row_idx])
            rank = int(np.where(order == target_idx)[0][0]) + 1  # 1-indexed
            ranks.append(rank)
        results[transform_name]["ranks"] = ranks
        r1 = np.mean([r == 1 for r in ranks])
        r10 = np.mean([r <= 10 for r in ranks])
        mAP = np.mean([1.0 / r for r in ranks])
        results[transform_name]["recall@1"] = float(r1)
        results[transform_name]["recall@10"] = float(r10)
        results[transform_name]["mAP"] = float(mAP)
        print(f"{transform_name}: recall@1={r1:.3f} recall@10={r10:.3f} mAP={mAP:.3f}")

    with open(args.out, "w") as f:
        json.dump({"weights": weights, "n_queries": len(query_ids), "results": results}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
