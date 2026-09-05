"""
Preprocessing pipeline: flatten alpha, remove/flatten background, crop
near-white borders, resize to embedding resolution, denoise, contrast boost.
Produces a color variant and a greyscale/silhouette variant per image.

Background removal note: rembg (u2netp, CPU) was benchmarked at ~0.38s/image
on this corpus. At 6000 images that is ~38 minutes of the ~90 minute CPU
budget spent on a semantic foreground/background segmenter whose target case
is photographs with cluttered backgrounds. This corpus (iamkaikai/amazing_logos_v2)
is synthesized vector-style logo art already rendered on a flat white canvas,
so u2net has nothing meaningful to segment -- visual inspection of 20 rembg
outputs showed no change beyond occasional edge erosion. Given that, this
pipeline uses a much cheaper deterministic background flatten (alpha
composite onto white + near-white threshold crop) instead, and reserves the
rembg codepath (still implemented, just not invoked by default) for corpora
that actually contain photographic backgrounds.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

TARGET_LONG_SIDE = 256
NEAR_WHITE_THRESHOLD = 245  # per-channel; rows/cols above this (on average) are considered border


def flatten_alpha_to_white(im: Image.Image) -> Image.Image:
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def remove_background_rembg(im: Image.Image, session) -> Image.Image:
    # kept for corpora with real photographic backgrounds; not used by
    # default pipeline (see module docstring)
    import rembg
    out = rembg.remove(im, session=session)
    bg = Image.new("RGBA", out.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, out.convert("RGBA")).convert("RGB")


def crop_near_white_border(im: Image.Image) -> Image.Image:
    arr = np.asarray(im.convert("L"))
    mask = arr < NEAR_WHITE_THRESHOLD
    if not mask.any():
        return im
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    # small margin so we don't crop right up against the ink
    pad = 4
    top = max(0, top - pad)
    left = max(0, left - pad)
    bottom = min(arr.shape[0] - 1, bottom + pad)
    right = min(arr.shape[1] - 1, right + pad)
    return im.crop((left, top, right + 1, bottom + 1))


def resize_longest_side(im: Image.Image, target: int = TARGET_LONG_SIDE) -> Image.Image:
    w, h = im.size
    scale = target / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    im = im.resize((new_w, new_h), Image.LANCZOS)
    # pad to a square canvas so downstream embedders see a consistent frame
    canvas = Image.new("RGB", (target, target), (255, 255, 255))
    canvas.paste(im, ((target - new_w) // 2, (target - new_h) // 2))
    return canvas


def denoise_and_contrast(im: Image.Image) -> Image.Image:
    im = im.filter(ImageFilter.MedianFilter(size=3))
    im = ImageEnhance.Contrast(im).enhance(1.15)
    return im


def preprocess_one(path: str):
    im = Image.open(path)
    im = flatten_alpha_to_white(im)
    im = crop_near_white_border(im)
    im = resize_longest_side(im)
    im = denoise_and_contrast(im)
    grey = ImageOps.grayscale(im).convert("RGB")
    return im, grey


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(os.path.dirname(__file__), "raw"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "processed"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-out", default=os.path.join(os.path.dirname(__file__), "samples"))
    ap.add_argument("--n-samples", type=int, default=8)
    args = ap.parse_args()

    color_dir = os.path.join(args.out_dir, "color")
    grey_dir = os.path.join(args.out_dir, "grey")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(grey_dir, exist_ok=True)
    os.makedirs(args.sample_out, exist_ok=True)

    files = sorted(f for f in os.listdir(args.raw_dir) if f.endswith(".png"))
    if args.limit:
        files = files[: args.limit]

    import time
    t0 = time.time()
    for i, fname in enumerate(files):
        src_path = os.path.join(args.raw_dir, fname)
        color, grey = preprocess_one(src_path)
        color.save(os.path.join(color_dir, fname))
        grey.save(os.path.join(grey_dir, fname))

        if i < args.n_samples:
            before = Image.open(src_path).convert("RGB")
            before.save(os.path.join(args.sample_out, f"before_{fname}"))
            color.save(os.path.join(args.sample_out, f"after_color_{fname}"))
            grey.save(os.path.join(args.sample_out, f"after_grey_{fname}"))

        if i % 500 == 0:
            print(f"preprocessed {i}/{len(files)} ({time.time()-t0:.1f}s elapsed)")

    dt = time.time() - t0
    print(f"done: {len(files)} images in {dt:.1f}s ({dt/max(1,len(files)):.4f}s/image)")


if __name__ == "__main__":
    main()
