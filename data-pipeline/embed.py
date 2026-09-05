"""
Embedding pipeline. Four channels per item:
  dino_color  - DINOv2-small CLS token on the color-preprocessed image (384d)
  dino_grey   - DINOv2-small CLS token on the greyscale variant (384d)
  clip        - CLIP ViT-B/32 image embedding on the color image (512d)
  fourier     - Fourier shape descriptor from contour tracing (see below)

Fourier descriptor approach: vtracer (vector tracing to SVG contours) has no
actively maintained CPU-only pip wheel that installs cleanly offline in this
environment, so this uses the documented OpenCV fallback: binarize on the
near-white background, run cv2.findContours, take the largest contour by
area, resample it to a fixed number of points, treat point coordinates as
complex numbers, and take a magnitude-only DFT descriptor: |FFT(k)|/n for
k=1..32 (dropping the k=0 DC term, which is translation-invariant by
construction since a constant offset only changes the DC bin), normalized by
the magnitude of the first harmonic (scale invariant). Taking magnitude
only (discarding phase) makes this rotation-invariant, matching the
in-browser query-side implementation in webapp/src/lib/fourier.ts exactly --
that file is the source of truth for this formula, since the offline and
in-browser descriptors have to land in the same 32-dim space for cosine
similarity between them to mean anything. The two implementations still
differ in contour-extraction method (cv2.findContours on a full-resolution
binarized image here vs. a coarse flood-filled mask + square-tracing walk
at 160x160 in the browser, since running a native contour tracer in a web
worker is impractical) -- that asymmetry is unavoidable and documented on
both sides.

Checkpointing: results are flushed to disk every CHECKPOINT_EVERY images as
a numbered .npz shard plus a manifest of which ids are already done, so a
crash resumes from the last shard instead of restarting.
"""
import argparse
import json
import os
import time

import numpy as np
import cv2

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

CHECKPOINT_EVERY = 500
N_FOURIER_HARMONICS = 32  # must match webapp/public/data/model_params.json channels.fourier.dim
N_CONTOUR_SAMPLES = 256

PIPE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(PIPE_DIR, "processed")
EMBED_DIR = os.path.join(PIPE_DIR, "embeddings")


def fourier_descriptor_from_gray(im: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(im, 245, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(N_FOURIER_HARMONICS, dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(np.float64)
    if len(pts) < 4:
        return np.zeros(N_FOURIER_HARMONICS, dtype=np.float32)

    # resample to a fixed number of points along the contour arc length so
    # the FFT input length is constant regardless of raw contour size
    complex_pts = pts[:, 0] + 1j * pts[:, 1]
    dists = np.abs(np.diff(complex_pts))
    arc = np.concatenate([[0], np.cumsum(dists)])
    total = arc[-1]
    if total == 0:
        return np.zeros(N_FOURIER_HARMONICS, dtype=np.float32)
    sample_positions = np.linspace(0, total, N_CONTOUR_SAMPLES, endpoint=False)
    resampled = np.interp(sample_positions, arc, complex_pts.real) + 1j * np.interp(
        sample_positions, arc, complex_pts.imag
    )

    # magnitude-only spectrum: |FFT(k)|/n for k=1..N_FOURIER_HARMONICS,
    # normalized by the first harmonic's magnitude. See module docstring --
    # this must match webapp/src/lib/fourier.ts's fourierDescriptor() formula.
    n = len(resampled)
    spectrum = np.fft.fft(resampled)
    magnitudes = np.abs(spectrum[1 : N_FOURIER_HARMONICS + 1]) / n
    first_mag = magnitudes[0] if magnitudes[0] > 1e-8 else 1.0
    descriptor = (magnitudes / first_mag).astype(np.float32)
    return descriptor


def fourier_descriptor(color_path: str) -> np.ndarray:
    im = cv2.imread(color_path, cv2.IMREAD_GRAYSCALE)
    return fourier_descriptor_from_gray(im)


def embed_pil_images(color_imgs, grey_imgs, dino_proc, dino_model, clip_proc, clip_model):
    """Raw (pre-PCA) embeddings for a batch of PIL RGB images. Used both by
    the main corpus embedding run and by eval harnesses that need to embed
    freshly transformed query images with the identical model calls."""
    import torch

    with torch.no_grad():
        dino_color_emb = dino_model(
            **dino_proc(images=color_imgs, return_tensors="pt")
        ).last_hidden_state[:, 0].numpy()
        dino_grey_emb = dino_model(
            **dino_proc(images=grey_imgs, return_tensors="pt")
        ).last_hidden_state[:, 0].numpy()
        clip_emb = clip_model.get_image_features(
            **clip_proc(images=color_imgs, return_tensors="pt")
        ).numpy()

    fourier_embs = np.stack([
        fourier_descriptor_from_gray(np.array(im.convert("L"))) for im in color_imgs
    ])

    return {
        "dino_color": dino_color_emb,
        "dino_grey": dino_grey_emb,
        "clip": clip_emb,
        "fourier": fourier_embs,
    }


def load_dino():
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model.eval()
    return proc, model


def load_clip():
    from transformers import CLIPProcessor, CLIPModel
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return proc, model


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image

    os.makedirs(EMBED_DIR, exist_ok=True)
    done_path = os.path.join(EMBED_DIR, "done_ids.json")
    done_ids = set(json.load(open(done_path))) if os.path.exists(done_path) else set()

    manifest = json.load(open(os.path.join(PIPE_DIR, "raw", "manifest.json")))
    ids = [m["id"] for m in manifest]
    if args.limit:
        ids = ids[: args.limit]
    remaining = [i for i in ids if i not in done_ids]
    print(f"{len(done_ids)} already embedded, {len(remaining)} remaining of {len(ids)}")

    if not remaining:
        print("nothing to do")
        return

    dino_proc, dino_model = load_dino()
    clip_proc, clip_model = load_clip()

    t_start = time.time()
    n_processed = 0
    shard_buffer = {"ids": [], "dino_color": [], "dino_grey": [], "clip": [], "fourier": []}
    shard_idx = len(
        [f for f in os.listdir(EMBED_DIR) if f.startswith("shard_") and f.endswith(".npz")]
    )

    def flush_shard():
        nonlocal shard_buffer, shard_idx
        if not shard_buffer["ids"]:
            return
        out_path = os.path.join(EMBED_DIR, f"shard_{shard_idx:04d}.npz")
        np.savez(
            out_path,
            ids=np.array(shard_buffer["ids"]),
            dino_color=np.stack(shard_buffer["dino_color"]),
            dino_grey=np.stack(shard_buffer["dino_grey"]),
            clip=np.stack(shard_buffer["clip"]),
            fourier=np.stack(shard_buffer["fourier"]),
        )
        shard_idx += 1
        done_ids.update(shard_buffer["ids"])
        json.dump(sorted(done_ids), open(done_path, "w"))
        print(f"checkpoint: wrote {out_path} ({len(shard_buffer['ids'])} items)")
        shard_buffer = {"ids": [], "dino_color": [], "dino_grey": [], "clip": [], "fourier": []}

    for batch_ids in batched(remaining, args.batch_size):
        color_paths = [os.path.join(PROCESSED_DIR, "color", f"{i}.png") for i in batch_ids]
        grey_paths = [os.path.join(PROCESSED_DIR, "grey", f"{i}.png") for i in batch_ids]

        color_imgs = [Image.open(p).convert("RGB") for p in color_paths]
        grey_imgs = [Image.open(p).convert("RGB") for p in grey_paths]

        with torch.no_grad():
            dino_color_out = dino_model(**dino_proc(images=color_imgs, return_tensors="pt"))
            dino_color_emb = dino_color_out.last_hidden_state[:, 0].numpy()

            dino_grey_out = dino_model(**dino_proc(images=grey_imgs, return_tensors="pt"))
            dino_grey_emb = dino_grey_out.last_hidden_state[:, 0].numpy()

            clip_emb = clip_model.get_image_features(
                **clip_proc(images=color_imgs, return_tensors="pt")
            ).numpy()

        fourier_embs = [fourier_descriptor(p) for p in color_paths]

        for j, item_id in enumerate(batch_ids):
            shard_buffer["ids"].append(item_id)
            shard_buffer["dino_color"].append(dino_color_emb[j])
            shard_buffer["dino_grey"].append(dino_grey_emb[j])
            shard_buffer["clip"].append(clip_emb[j])
            shard_buffer["fourier"].append(fourier_embs[j])

        n_processed += len(batch_ids)
        if len(shard_buffer["ids"]) >= CHECKPOINT_EVERY:
            flush_shard()

        elapsed = time.time() - t_start
        rate = n_processed / elapsed
        remaining_n = len(remaining) - n_processed
        eta = remaining_n / rate if rate > 0 else float("inf")
        print(
            f"{n_processed}/{len(remaining)} embedded "
            f"({elapsed:.1f}s elapsed, {rate:.2f} img/s, eta {eta:.0f}s)"
        )

    flush_shard()
    print(f"total embedding time: {time.time() - t_start:.1f}s for {n_processed} images")


if __name__ == "__main__":
    main()
