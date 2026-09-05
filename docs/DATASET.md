# Corpus source

## What was actually used

**Dataset:** `iamkaikai/amazing_logos_v2` on Hugging Face Hub
**URL:** https://huggingface.co/datasets/iamkaikai/amazing_logos_v2
**Size used:** 6,000 images sampled (with a fixed seed) from the 10,123 rows in the dataset's single parquet shard (`data/train-00000-of-00001-243b69021790579b.parquet`, ~75MB download).
**Format:** 512x512 RGB PNG-decoded images plus a `text` field containing the generation prompt used to produce each image, e.g.:

> "Simple elegant logo for 'Axis' Mosca Azul Editores, Single Letters Negative Space Illusory Contours Open Static Inline Symmetrical Dots, successful vibe, minimalist, thought provoking, abstract, recognizable, relatable, sharp, vector art, even edges, black and white"

Every image is an AI-generated (diffusion-model synthesized) minimalist mark rendered on a flat white canvas, produced from a text prompt referencing a (often historical, fictional, or descriptive) organization name plus a set of style/shape descriptor keywords. These are **not scans or crops of real registered trademark artwork** -- they are novel generative-model interpretations conditioned on organization-name text.

## License and why this was the pragmatic choice

`iamkaikai/amazing_logos_v2` carries no explicit license field on its Hugging Face dataset card (checked via the HF API `cardData`/`tags`, both empty). In the absence of a stated license, redistribution rights default to whatever the uploader intends, which is unstated. This is a real gap and is being flagged honestly rather than glossed over.

The alternatives investigated, in the priority order given for this task, were rejected for concrete reasons:

1. **Zenodo L3D (`https://zenodo.org/records/5771006`)** -- confirmed via the Zenodo API that the record contains exactly one file, `L3D dataset.tar`, 12,047,102,976 bytes (~11.2GB), with no per-shard split. Not downloadable within the CPU/time budget for this session.
2. **Hugging Face logo datasets, broader search** -- surveyed ~30+ dataset repos matching "logo". Concrete findings:
   - `logo-wizard/modern-logo-dataset`: only 803 examples (too small for the 3k-8k target), and its own README disclaimer directly contradicts its declared `cc-by-nc-3.0` tag ("images are collected from the Internet, and the copyright belongs to the original owners") -- i.e. it is not actually a clean-license dataset despite the tag.
   - `diwank/lld` (HF mirror of the ETH Zurich "LLD -- Large Logo Dataset", Sage et al. 2017): tagged MIT on HF, but the LICENSE file in that repo covers the *wrapper/crawler code* copyright, not the underlying logo images, which were crawled favicons/logos of real companies from the open web -- i.e. real third-party trademarks with an unclear redistribution license mislabeled as MIT. It was also, separately, disqualified on pure feasibility grounds: the packaged `lld-processed.h5` file grew past **11.5GB and was still downloading** when it was aborted, meaning it is effectively another full-corpus-scale download like L3D, not the "quick download" this task called for.
   - `varun1212/logo-detection-dataset`, `mattrichmo/brand-logos`: individually-named real company logo files (`ACE-Hardware-logo-PNG-*.jpg`, `100tb-icon.svg`, etc.) scraped from the web, no license metadata at all. Rejected for the same real-trademark-with-no-license problem, which is a worse fit for a public GitHub repo than a synthetic dataset with an unstated-but-plausibly-permissive license.
   - `openfoodfacts/universal-logo-detector`: real product-photo images with brand bounding boxes, ~1GB download, wrong shape for this task (packaging photos, not cropped logo art).
   - `VieVie31/Logos-32plus_v1.0.1` (a re-release of FlickrLogos-32/-32plus): real trademark photos in natural scenes, ~2.1GB download, research-only license, wrong image format (uncropped photos containing a logo, not logo art itself).
3. **Last-resort "well-known small CC-licensed logo dataset"** -- `amazing_logos_v2` was chosen here as the best available compromise: it is synthetic (no real trademark artwork, which meaningfully lowers the redistribution/IP risk compared to any of the real-logo scrapes above), it is small and fast to acquire (75MB, no auth, ~10k rows), and it produces exactly the kind of abstract vector-style mark that a visual-trademark-similarity search needs to be tested against.

**Bottom line:** this is not a properly CC-BY-licensed corpus, and that is a real limitation of this build, not a hidden one. If this pipeline is used beyond a portfolio/demo context, the corpus should be swapped for one with an explicit, verified open license (or the original L3D/FlickrLogos academic releases under their real research-use terms, fetched with enough time/bandwidth budget to do so properly). The pipeline code (`fetch_corpus.py` through `quantize_pack.py`) is corpus-agnostic and re-runnable against a different source with no changes beyond `fetch_corpus.py`'s ingestion step.

## Sampling method

`data-pipeline/fetch_corpus.py`:
1. Loads the single parquet file directly via `pyarrow.parquet.read_table` (no `datasets` streaming needed since the whole shard fits in memory).
2. Shuffles all 10,123 row indices with `random.Random(seed=42)` and takes the first 6,000.
3. Assigns each item a stable id: `sha1(image_bytes[:256] + str(source_index))[:16]`.
4. Writes each image to `data-pipeline/raw/<id>.png` and a `manifest.json` with `id`, `source_index`, `source_prompt`, and a parsed `shape_tag`.

## Vienna code substitute: `shape_tag`

This corpus has no Vienna Classification codes (those only exist for real trademark office filings). In its place, `fetch_corpus.py` parses each generation prompt for the first occurrence of a fixed vocabulary of shape/style keywords (circle, square, triangle, lines, stars, letters, dots, waves, cross, animals, outline, symmetrical, curve, etc.) and stores it as `shape_tag`. This is a single best-effort keyword extracted from free text, not a structured hierarchical classification -- it is a much weaker label than a real Vienna code, and `eval/vienna_agreement/RESULTS.md` reports it as such.

## Known corpus limitations

- **Overwhelmingly monochrome**: a spot check of 300 images found only 1 with meaningfully different values in the R/G/B channels (mean channel difference > 5). The `dino_color` and `dino_grey` embedding channels will therefore be highly correlated for most of this corpus -- a real limitation of testing the multi-channel fusion design against this particular corpus, not of the fusion design itself.
- **Not real trademarks**: results here say nothing about performance against actual registered marks with their attendant visual conventions (color claims, stylization rules, etc.); see the L3D/FlickrLogos discussion above for what a more representative corpus would look like.
- **No license guarantee**: see above.
