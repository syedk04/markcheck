# MarkCheck

**MarkCheck** is a fully client-side visual trademark-similarity screening
tool. Upload a logo, pick the relevant Nice Classification goods/services
classes, and get back the closest-looking marks in a 6,000-item corpus,
ranked by a fused multi-channel visual-similarity score, with a risk band,
a per-patch attention heatmap over the top match, and an exportable report.

**Live demo: https://syedk04.github.io/markcheck/**

Every part of the pipeline runs in the browser: image preprocessing, DINOv2
+ CLIP embedding, PCA projection, int8 quantization/dequantization, the
brute-force similarity scan, risk scoring, and PDF/HTML report generation.
No image you upload ever leaves your machine -- there is no backend, no
upload endpoint, and nothing running server-side.

## Features

- **Drag-and-drop upload** with in-browser preprocessing (alpha flattening,
  auto-crop, greyscale variant) implemented as canvas operations, no server
  round-trip.
- **Four-channel visual fingerprint per mark**, fused into a single score:
  - DINOv2-small CLS token on the color image
  - DINOv2-small CLS token on the greyscale image
  - CLIP ViT-B/32 image embedding
  - A from-scratch Fourier shape descriptor (Moore-neighbor contour trace +
    magnitude-only DFT), used because no WASM vector-tracing library bundled
    cleanly for in-browser use
- **PCA-projected, int8-quantized index** (352 bytes/record) so the entire
  6,000-item corpus, all four channels included, loads and searches in a
  single 2.1 MB binary with sub-20ms brute-force scan latency -- no vector
  database, no ANN index, no server.
- **Nice Classification picker** (all 45 classes) feeding a class-overlap
  signal that's kept separate from visual similarity in the risk model.
- **Risk banding** that explicitly does not conflate "looks the same" with
  "in the same market," surfacing both signals side by side.
- **Per-patch DINO attention heatmap** over the top match, showing which
  region of the matched mark drove the similarity score.
- **Side-by-side diff view, batch mode** (screen multiple variants of a
  design at once), **localStorage query history**, and **client-side
  PDF/HTML report export** -- all without a server.
- **Weights tuned by evaluation, not guessed**: the fusion weights shipped
  in production were selected by running three candidate configurations
  through a synthetic-transform retrieval benchmark and picking the one
  with the best measured recall -- see [`eval/RESULTS.md`](eval/RESULTS.md).

## How this differs from something like TMview

Official trademark search tools (TMview, USPTO TESS, etc.) query
authoritative government trademark registries with real registered marks
and structured metadata (Vienna codes, Nice classes, filing status).
MarkCheck is a different kind of tool, built to demonstrate a different
approach:

- **Fully client-side and instant.** Embedding and search both run in the
  browser via WebAssembly/ONNX Runtime Web -- no server round-trip.
- **A learned visual-similarity signal**, not exact/structured metadata
  matching: fused, weighted cosine similarity across four independent
  embedding channels.
- **Corpus-agnostic architecture.** The entire pipeline
  (`data-pipeline/fetch_corpus.py` through `quantize_pack.py`, and the
  webapp's data contract in `webapp/src/lib/types.ts`) is designed to be
  re-run against any image corpus with no code changes beyond the
  ingestion step. The demo ships with a synthetic AI-generated logo corpus
  (`iamkaikai/amazing_logos_v2`, 6,000 images) rather than real registered
  trademark artwork, since every real-trademark dataset investigated for
  this build carried an unresolved redistribution license -- see
  [`docs/DATASET.md`](docs/DATASET.md) for the full sourcing investigation.
  Swapping in a properly licensed real corpus is a data change, not an
  architecture change.

This is a portfolio/architecture build: it says nothing about coverage of
any actual registered trademark, and it is not legal advice -- the app
says so prominently in its own UI.

## Architecture

```
Upload -> canvas preprocess -> DINOv2 + CLIP + Fourier embed (transformers.js, in-browser)
       -> PCA project + int8 quantize -> Web Worker brute-force cosine scan vs. 6,000-item index
       -> weighted channel fusion -> risk banding -> results grid + heatmap + report export
```

Offline, a separate pipeline builds the corpus the browser searches against:

```
fetch_corpus.py -> preprocess.py -> embed.py (DINOv2 + CLIP + Fourier, checkpointed/resumable)
                -> quantize_pack.py (PCA fit + int8 quantize) -> webapp/public/data/
```

## Evaluation

Fusion weights and retrieval quality are backed by real measured numbers,
not assumptions -- full writeup in [`eval/RESULTS.md`](eval/RESULTS.md):

| Benchmark | Result |
|---|---|
| Synthetic-transform recall@1 (winning weight config, mean over 6 transform types) | **0.807** |
| Best single transform (aspect distortion) | 0.987 recall@1 |
| Weakest transform (50% scale + pad) | 0.440 recall@1 |
| Vienna-code-substitute (`shape_tag`) top-10 agreement | 0.366 (n=381) |
| Search latency, full 6,000-item corpus | p50 13.1ms / p95 15.5ms |
| Packed index size | 2.11 MB for 6,000 items |

The eval suite also runs a Vienna-classification-agreement check and a
real-trademark-dispute-pairs harness (the latter intentionally ships
unpopulated -- it requires hand-curated legal citations, which is
out of scope for an automated pipeline, and it no-ops cleanly rather than
faking results).

## Quickstart

```bash
cd webapp
npm install
npm run dev
```

This serves the app locally against the real packed corpus in
`webapp/public/data/`. To build and preview a production bundle:

```bash
cd webapp
npm run build
npm run preview
```

Run the test suite:

```bash
cd webapp
npm test
```

## Repo layout

- `webapp/` -- the Preact + TypeScript + Vite single-page app. See
  `webapp/README.md` for the stack, data contract, and the documented
  asymmetries between the browser and offline pipeline.
- `data-pipeline/` -- offline scripts that fetch the corpus, preprocess
  images, compute embeddings, and pack everything into the quantized
  artifacts the webapp reads (`index.bin`, `metadata.json`,
  `model_params.json`, `thumbnails/`).
- `eval/` -- evaluation harnesses (synthetic-transform retrieval recall,
  Vienna-code-substitute agreement, latency, real-pairs). See
  [`eval/RESULTS.md`](eval/RESULTS.md) for the full numbers.
- `docs/DATASET.md` -- corpus sourcing writeup and licensing investigation.
- `docs/LIMITATIONS.md` -- complete list of known limitations and
  asymmetries in this build.
- `.github/workflows/deploy.yml` -- builds and publishes `webapp/` to
  GitHub Pages on every push to `master`.

## Tech stack

Preact, TypeScript, Vite, `@xenova/transformers` (ONNX Runtime Web) for
in-browser DINOv2/CLIP inference, Web Workers for embedding and search,
Vitest for testing, Python/PyTorch/scikit-learn/OpenCV for the offline
corpus pipeline.
