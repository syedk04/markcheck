# MarkCheck webapp

Client-side trademark visual-similarity screening tool. No backend: everything
from image upload through embedding, PCA projection, quantization, and the
brute-force similarity scan runs in the browser (in Web Workers so the UI
thread stays responsive).

## Stack

- **Vite + TypeScript** for the build.
- **Preact** instead of React for the UI layer — the app is a handful of
  forms and a results grid, nothing that needs React's ecosystem, and Preact's
  smaller runtime matters here because the bundle already carries
  `@xenova/transformers` + `onnxruntime-web`, which are heavy on their own.
- **`@xenova/transformers`** (transformers.js) running DINOv2-small and
  CLIP ViT-B/32 fully client-side via ONNX Runtime Web.
- **Vitest** for unit tests of the math that would silently rot if it drifted
  from the offline pipeline: dequantization/PCA projection, score fusion, and
  risk banding.

## Data contract

The app reads `public/data/model_params.json`, `metadata.json`, `index.bin`,
and `thumbnails/*.jpg` from a separate offline pipeline. Run
`npm run gen-placeholder-data` to generate synthetic stand-ins with the exact
same byte layout for local development before the real corpus exists.

## Model choices and a deviation worth flagging

- DINOv2-small is confirmed available on the `Xenova` HF org
  (`Xenova/dinov2-small`) and is what the app requests first. If that load
  fails at runtime, it falls back to `Xenova/dino-vits16` (original DINOv1
  ViT-S/16, same 384-dim hidden size, so the PCA basis still lines up). This
  is logged loudly to the console and surfaced in the UI as a visible warning
  — it is not the intended backbone, and results in that state should be
  treated as running in degraded mode. In one specific test setup we hit this
  fallback because Hugging Face's CDN returned an anti-automation challenge to
  a headless-Chromium fingerprint (confirmed via response headers — it's a
  WAF captcha page, not a real outage); ordinary browser sessions load
  DINOv2-small directly.

## Fourier / shape channel

The offline pipeline traces contours with a dedicated raster-to-vector tool.
Running an equivalent native/WASM tracer inside a browser worker was judged
too risky for this build (the strongest option found, `@neplex/vectorizer`,
is a NAPI/WASI-threads package built for Node, not a bundler-friendly browser
target). Instead, the query's shape descriptor is computed with a small
from-scratch pipeline in `src/lib/fourier.ts`:

1. Canvas-based background flattening + auto-crop (`src/lib/imagePrep.ts`) to
   get a foreground mask.
2. Largest-connected-component extraction, then Moore-neighbor boundary
   tracing of that blob's outline.
3. Arc-length resampling to a fixed point count, then a direct (non-FFT — the
   descriptor is only 32-dimensional so an O(n²) DFT is plenty fast) discrete
   Fourier transform, keeping the magnitude spectrum of the low-frequency
   harmonics normalized by the first harmonic. Magnitude spectra of a closed
   contour are rotation- and starting-point-invariant by construction; the
   normalization step adds scale invariance.

This is a known, documented asymmetry against the offline corpus pipeline: a
simpler in-browser background/foreground split (works well for logos on a
flat/transparent background, worse on busy photographic backgrounds) plus a
different tracer implementation. Both sides land on a 32-dim descriptor
that's compared with plain cosine similarity, so the mismatch is in precision
of the shape signal, not in the data format.

## Match heatmap

The corpus index only stores PCA-reduced, quantized global vectors — there
are no stored per-patch tokens to compare against. Since thumbnails are
already lazy-loaded per result, the heatmap is computed on demand: the
matched item's thumbnail is re-embedded through DINO in the browser to get
its patch token grid, and each patch's cosine similarity to the *query's*
pooled embedding is rendered as a heat overlay on the matched image. That
answers the more useful question ("which part of the matched mark looks like
my upload") without needing any change to the frozen data contract.

## Known asymmetries / limitations, summarized

- In-browser background/border handling is a canvas-based white-flatten +
  auto-crop, not the offline pipeline's `rembg`-based segmentation.
- Class-overlap scoring is free-text token overlap against whatever label the
  corpus item carries — it is explicitly not a verified Nice Classification
  match, and the UI says so next to every risk breakdown.
- Risk banding thresholds (`src/lib/risk.ts`) and the 0.7/0.3 visual/class
  weighting are a starting heuristic, not a calibrated model.

## Testing

`npm test` runs Vitest against `tests/`: dequantization/PCA math
(`quantize.test.ts`), the score-fusion and class-overlap heuristic
(`scoring.test.ts`), risk banding (`risk.test.ts`), and the contour
trace/Fourier descriptor pipeline (`fourier.test.ts`).

A full manual smoke test (upload → embed → PCA/quantize → brute-force
search → ranked results → detail view → heatmap) was run against
`npm run build` + `npm run preview` output using a headed browser automation
pass; headless automation specifically triggers Hugging Face's bot-detection
CDN layer (see above) so use a normal browser, not a headless one, if you
need to reproduce that.
