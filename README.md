# MarkCheck

MarkCheck is a client-side visual-trademark-similarity screening tool: upload
a logo, pick a Nice Classification class, and get back the closest-looking
marks in a corpus, ranked by a fused visual-similarity score, with a rough
risk band and a per-patch heatmap showing which part of the top match looks
like your upload. Everything -- image preprocessing, embedding (DINOv2 +
CLIP), PCA projection, quantization/dequantization, the brute-force
similarity scan, and risk scoring -- runs entirely in the browser. There is
no backend and no server-side processing of your image.

**Live demo:** not yet deployed. Once `.github/workflows/deploy.yml` has run
against `master` (requires enabling Settings -> Pages -> Source: GitHub
Actions on the repo first), this will be live at
https://syedk04.github.io/markcheck/ -- update this line once that's
confirmed working.

## Honest scope -- read this before anything else

- **This is not legal advice.** The app says so prominently in its own UI,
  and it's worth repeating here: nothing this tool outputs is a substitute
  for a real trademark clearance search or an opinion from a trademark
  attorney.
- **The demo corpus is synthetic, not real trademarks.** The 6,000 logos
  shipped with this build are AI-generated (diffusion model) logo art from
  `iamkaikai/amazing_logos_v2` on Hugging Face -- not scans of real
  registered marks. Matches against this corpus say nothing about
  similarity to any actual existing trademark. See `docs/DATASET.md` for the
  full story of why (real, cleanly-licensed logo datasets were investigated
  and rejected for concrete licensing/feasibility reasons) and
  `docs/LIMITATIONS.md` for the complete list of known limitations.
- **The architecture is corpus-agnostic.** Every part of the pipeline
  (`data-pipeline/fetch_corpus.py` through `quantize_pack.py`, and the
  webapp's data contract in `webapp/src/lib/types.ts`) is built to be
  re-run against a real, properly-licensed trademark corpus with no code
  changes beyond the ingestion step. This build demonstrates that the
  architecture works end-to-end, not that the specific data behind it is
  authoritative.

## How this differs from something like TMview

Official trademark search tools (TMview, USPTO TESS, etc.) query authoritative
government trademark registries with real registered marks and structured
metadata (Vienna codes, Nice classes, filing status). MarkCheck does not do
that, and isn't trying to replace it. What it demonstrates instead:

- **Fully client-side, instant, no backend.** No image you upload is sent
  anywhere -- embedding and search both happen in your browser via
  WebAssembly/ONNX Runtime Web. There's no server round-trip, no upload
  endpoint, and nothing to deploy beyond static files.
- **A different kind of similarity signal.** Fused, weighted cosine
  similarity across four independent channels (DINOv2 color, DINOv2
  greyscale, CLIP, and a from-scratch Fourier shape descriptor) rather than
  exact/structured metadata matching.
- **A demo-scale, demo-license corpus**, not the real thing -- see above.

## Quickstart

```
cd webapp
npm install
npm run dev
```

That serves the app locally against whatever's in `webapp/public/data/`
(the real packed corpus, if present, or run
`npm run gen-placeholder-data` first to generate synthetic stand-ins for
local development without the real data pipeline).

To build and preview a production bundle:

```
cd webapp
npm run build
npm run preview
```

## Repo layout

- `webapp/` -- the Preact + TypeScript + Vite single-page app. See
  `webapp/README.md` for the stack, data contract, and known asymmetries
  between the browser and offline pipeline.
- `data-pipeline/` -- offline scripts that fetch the corpus, preprocess
  images, compute embeddings, and pack everything into the quantized
  artifacts the webapp reads (`index.bin`, `metadata.json`,
  `model_params.json`, `thumbnails/`).
- `eval/` -- evaluation harnesses (synthetic-transform retrieval recall,
  Vienna-code-substitute agreement, latency, real-pairs). See
  `eval/RESULTS.md` for real measured numbers from the current corpus.
- `docs/DATASET.md` -- full corpus sourcing writeup and licensing
  investigation.
- `docs/LIMITATIONS.md` -- complete list of known limitations and
  asymmetries in this build.

## Testing

`cd webapp && npm test` runs the Vitest suite covering the quantization
math, score fusion, risk banding, and the Fourier contour-trace descriptor.
