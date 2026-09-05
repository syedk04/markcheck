# Limitations

MarkCheck is a portfolio/demo build of a client-side visual-trademark-
similarity screening tool. This page lists the known, honest limitations of
the current build -- what it does not do, where its numbers come from, and
where the architecture diverges from what a production, real-data version
would need. None of this is hidden elsewhere in the app; the UI itself
carries a "not legal advice" disclaimer and surfaces some of these points
directly (degraded-mode banner, class-overlap caveat).

## 1. The corpus is synthetic, not real trademarks

The 6,000-item corpus shipped with this build is `iamkaikai/amazing_logos_v2`
on Hugging Face -- AI-generated (diffusion model) logo art produced from text
prompts, not scans or crops of real registered trademark artwork, and not
carrying an explicit redistribution license. Every match this tool returns is
a similarity to a piece of generative art, not to an existing registered
mark. See `docs/DATASET.md` for the full sourcing writeup, why this corpus
was used instead of a real, cleanly-licensed trademark dataset (L3D/EUIPO,
METU, FlickrLogos, etc. were all investigated and rejected for concrete
licensing or feasibility reasons), and what a real-data version would need.
The pipeline code (`data-pipeline/fetch_corpus.py` through
`quantize_pack.py`) is corpus-agnostic and re-runnable against a different
source with no changes beyond the ingestion step.

## 2. Vienna Classification is substituted with a keyword label

Real trademark filings carry Vienna Classification codes, a structured
hierarchical figurative-element taxonomy assigned by examiners. This corpus
has none (it isn't real filings), so `data-pipeline/fetch_corpus.py` instead
parses each generation prompt for the first match against a fixed vocabulary
of shape/style keywords (circle, square, lines, letters, dots, etc.) and
stores that single keyword as `shape_tag`. This is a much weaker label than
a real Vienna code -- one best-effort keyword from free text, not a
structured classification -- and `eval/RESULTS.md`'s vienna-agreement numbers
are reported with that caveat front and center, not as evidence of
Vienna-code-equivalent performance.

## 3. Nice Classification / class overlap is unverified free-text matching

The risk assessment's "class overlap" signal (`webapp/src/lib/scoring.ts`) is
a free-text token overlap between the user's selected Nice class descriptor
and whatever label string the corpus item carries -- not a verified,
examiner-assigned classification match. The UI states this next to every
risk breakdown; it should not be read as equivalent to an actual Nice-class
conflict determination.

## 4. Risk-band thresholds are an uncalibrated starting heuristic

`webapp/src/lib/risk.ts` combines visual similarity and class overlap with a
fixed 0.7/0.3 weighting and buckets the result into Low/Medium/High risk at
fixed 0.5/0.75 thresholds. These are reasonable starting values, not a
calibrated model fit against real outcomes (e.g. actual opposition/refusal
rates at a given similarity score) -- there is no such calibration dataset
available in this build (see limitation 1). Treat the risk band as an
ordering signal, not a probability.

## 5. Browser vs. offline preprocessing asymmetry

The offline pipeline uses `rembg` (a trained background-removal model) to
segment each corpus image before embedding. The in-browser query path uses a
much simpler canvas-based heuristic instead (`webapp/src/lib/imagePrep.ts`):
white-flatten alpha, then auto-crop to the non-white bounding box. This works
well for logos already on a flat/transparent background (the common case for
uploaded logo art) but is meaningfully weaker than a trained segmentation
model on busy or non-white backgrounds. This is a real asymmetry between how
corpus images and query images are prepared before the same embedding
models see them.

## 6. Fourier shape descriptor: client/server implementation asymmetry

Both the offline pipeline (`data-pipeline/embed.py`) and the browser
(`webapp/src/lib/fourier.ts`) compute a 32-dimensional magnitude-only Fourier
descriptor of the largest foreground contour, using the same formula
(magnitude spectrum of harmonics 1..32, normalized by the first harmonic --
translation/rotation/scale-invariant by construction). But the two sides use
different contour-extraction methods: the offline pipeline binarizes a
full-resolution image and runs OpenCV's `findContours`; the browser does a
coarse flood-filled mask at 160x160 followed by a from-scratch Moore-neighbor
boundary trace (no equivalent-quality contour tracer bundles cleanly for
browser/WASM use -- `@neplex/vectorizer`, the strongest option found, is a
NAPI/WASI-threads package built for Node, not a browser bundler target). Both
land in the same 32-dim space and are compared with plain cosine similarity,
so the mismatch is in precision of the shape signal, not in data format --
and per `eval/RESULTS.md`, this channel carries only 5% of the fused score
weight in the shipped config, so its comparatively lower fidelity has limited
practical impact on final rankings.

## 7. `eval/real_pairs/` (real trademark dispute pairs) was not run

Assembling a set of real, citable trademark dispute or redesign pairs (e.g.
from USPTO TTAB decisions or WIPO UDRP filings) requires manual legal-
research curation that was out of scope for this build. `eval/real_pairs/`
is a working harness left ready to accept a hand-curated `pairs.json` later;
no pairs were fabricated to fill it. See `eval/RESULTS.md` for the full
reasoning, including that this eval would also need a real-trademark corpus
(limitation 1) to be meaningful in the first place.

## 8. DINOv2-small can silently fall back to a weaker backbone in the browser

The webapp requests `Xenova/dinov2-small` first. If that model fails to load
at runtime (observed once in testing: Hugging Face's CDN served an
anti-automation challenge page to a headless-Chromium fingerprint), it falls
back to `Xenova/dino-vits16` (the original DINOv1 ViT-S/16 backbone -- same
384-dim hidden size, so the PCA basis still lines up numerically, but it is
a different, generally weaker model). This is logged loudly to the console
and surfaced as a visible UI warning when it happens, but any similarity
scores computed in that degraded state were produced by a different model
than the one the corpus was indexed with philosophically intended to match,
and should be understood as running in a lower-fidelity mode.
