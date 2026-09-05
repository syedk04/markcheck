# Eval results

All numbers below are real measured output from running the scripts in this
directory against the actual shipped corpus (6,000 items,
`iamkaikai/amazing_logos_v2` -- synthetic/AI-generated logo art, **not real
trademarks**; see `docs/DATASET.md`). Nothing here is estimated or
extrapolated.

## Weight tuning (`eval/tune_weights.py`)

Three fusion-weight configs were each run through the full synthetic-transform
eval (150 sampled queries x 6 transform types = 900 re-embeds per config,
2,700 total) and scored by mean recall@1 across transform types:

| config | dino_color | dino_grey | clip | fourier | mean recall@1 |
|---|---|---|---|---|---|
| baseline | 0.50 | 0.20 | 0.20 | 0.10 | 0.7944 |
| **dino_heavy (winner)** | **0.65** | **0.15** | **0.15** | **0.05** | **0.8067** |
| balanced_shape | 0.40 | 0.25 | 0.15 | 0.20 | 0.6567 |

**`dino_heavy` won** and has been patched into
`webapp/public/data/model_params.json`'s `channels.*.weight` fields (this is
what the shipped webapp uses). Weighting the DINOv2 color channel more
heavily and de-emphasizing the from-scratch Fourier shape descriptor and CLIP
outperformed both the original spec-default baseline and an alternative that
leaned further into the (noisier, contour-tracing-based) shape channel --
consistent with DINOv2 being the most discriminative single channel on this
corpus, and the Fourier descriptor being the weakest/noisiest of the four
(see "known corpus limitation" below on why: most images are monochrome,
so `dino_grey` and `dino_color` are highly correlated, and a hand-rolled
contour-based descriptor is inherently less robust than a learned embedding).

### Per-transform breakdown (winning `dino_heavy` config)

| transform | recall@1 | recall@10 | mAP |
|---|---|---|---|
| rotation (25 degrees) | 0.607 | 0.867 | 0.705 |
| recolor (channel remix) | 0.973 | 1.000 | 0.987 |
| scale (50% + white-pad) | 0.440 | 0.713 | 0.538 |
| crop (10% edge crop) | 0.960 | 1.000 | 0.980 |
| aspect_distortion (1.4x/0.7x stretch) | 0.987 | 1.000 | 0.993 |
| gaussian_noise (sigma=18) | 0.873 | 0.993 | 0.923 |

**Reading these honestly:** recolor, crop, aspect_distortion, and
gaussian_noise are all strong (recall@1 > 0.85) -- DINOv2's embedding is
robust to those perturbations on this corpus. **Scale is the weakest
transform by a wide margin** (recall@1 0.44) -- shrinking a mark to 50% and
white-padding it changes the effective foreground/background ratio a lot for
a system whose channels were all trained/calibrated on full-bleed images, and
this build does not do any explicit scale-normalization/re-cropping before
embedding a query. Rotation is the second weakest (0.607) -- expected, since
none of the four channels (DINOv2, CLIP, PCA) are rotation-invariant by
design; only the Fourier magnitude-spectrum channel is, and it carries only
5% of the fused weight in the winning config. Both are real, honestly
reported weaknesses of this specific pipeline, not artifacts of the eval.

Full per-query rank data for all three configs is in
`eval/weight_tuning_summary.json` and `eval/synthetic_transforms/results_*.json`.

## Vienna-code-substitute agreement (`eval/vienna_agreement/run.py`)

Run with the winning `dino_heavy` weights, k=10, 500 requested / 381 usable
samples (items whose `shape_tag` is `"unclassified"` or shared by fewer than
2 corpus items are skipped, per the script's own filtering):

**Overall top-10 `shape_tag` agreement: 0.366** (n=381 queries)

Per-tag breakdown (largest tags by sample count):

| shape_tag | mean agreement | n |
|---|---|---|
| circle | 0.587 | 125 |
| lines | 0.316 | 88 |
| square | 0.396 | 52 |
| letters | 0.203 | 31 |
| triangle | 0.244 | 16 |
| dots | 0.215 | 13 |
| outline | 0.125 | 12 |
| cross | 0.113 | 8 |
| asymmetrical | 0.083 | 6 |
| squares | 0.067 | 6 |

**Reading these honestly:** as documented in
`eval/vienna_agreement/run.py`'s own module docstring, `shape_tag` is a
single best-effort keyword parsed out of the generation prompt -- not a real
Vienna Classification code, which is a structured hierarchical figurative-
element taxonomy assigned by trademark examiners. A 0.366 overall agreement
(vs. random-neighbor agreement that would be far lower given dozens of tags
with very unequal frequency) is read here only as a sanity check that the
embedding space correlates with *some* independent semantic signal, not as
evidence the system would agree with real Vienna codes at any particular
rate. `circle` (the most common/largest tag) shows the strongest agreement,
plausibly because "circle" as a keyword correlates with a genuinely
consistent visual pattern (a circular logo silhouette) more than more
abstract terms like "asymmetrical" do.

## Latency (`eval/latency_bench.py`)

Measured in Python as a same-shape proxy for the browser's typed-array scan
(dequantize + weighted cosine fusion across 4 channels + top-k sort), against
the full 6,000-item corpus, 200 sampled queries:

- **p50: 13.09 ms**
- **p95: 15.51 ms**
- **index.bin size: 2.11 MB** (2,112,000 bytes exactly, 352 bytes/record x
  6,000 records)

This is a Python-process proxy, not a browser measurement -- see the
module's own docstring. The operation count (dequantize + 4 channel dot
products + argsort over 6,000 items) is the same shape the browser's
`search.worker.ts` will run, so this is a reasonable order-of-magnitude
estimate, not a substitute for an actual in-browser benchmark.

## Real trademark dispute pairs (`eval/real_pairs/`)

**Not run -- intentionally scoped out.** `eval/real_pairs/run.py` requires a
hand-curated `eval/real_pairs/pairs.json` of real, citable trademark dispute
or redesign pairs (e.g. from USPTO TTAB decisions or WIPO UDRP filings); that
manual legal-research curation was out of scope for this pipeline run and no
such file was fabricated. Running the script without that file confirms it
no-ops gracefully with an explanatory message rather than silently reporting
zero/fake results. This eval remains available to run once/if that curated
pairs file exists (also moot in the current build since the corpus itself is
synthetic, not real trademark art -- real dispute pairs would need to be
scored against a real-trademark corpus to mean anything, per
`docs/DATASET.md`).
