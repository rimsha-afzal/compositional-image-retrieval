# compositional-image-retrieval

Compositional image retrieval on CelebA: given a reference face image and one or more signed
text attributes (e.g. `+Smiling, -Eyeglasses`), retrieve target images that preserve the
reference identity while satisfying the requested changes. The project explores this task with
a training-free approach built entirely on a frozen CLIP ViT-B/32 model.

## Overview

Text-to-image and image-to-image retrieval are well studied on their own, but combining a
reference image with textual edit instructions in a single query is harder: the retrieved
results need to stay visually close to the reference while still reflecting the requested
attribute changes. We study this on CelebA, using the official test split and the provided
ground-truth queries, and evaluate with Recall@K and Precision@K (K = 1, 5, 10).

## Approach

Starting from a simple CLIP text-arithmetic baseline, the project incrementally explores
whether dataset-specific visual evidence and better-conditioned text signals can improve
compositional retrieval without training any new model:

1. **Baseline** — compose the query as the reference image embedding plus/minus the CLIP text
   embeddings of the requested attributes.
2. **Gated fusion** — tune how strongly positive and negative conditions should be weighted,
   instead of assuming they contribute equally.
3. **Visual attribute directions** — CLIP's text and image embeddings occupy separate regions
   of the embedding space (the "modality gap"), so adding text vectors to an image vector can
   drift in a direction that isn't semantically meaningful. Instead, attribute directions are
   estimated directly from CelebA images, as the difference between the average embeddings of
   images with and without the attribute.
4. **Hybrid text + visual fusion** — combine both signals, and refine the text side with
   prompt ensembling (averaging over multiple phrasings, including a neutral-subtracted
   variant) to make it less dependent on any single hand-written prompt.
5. **Reliability weighting and direction decorrelation** — not all attributes are equally
   separable in embedding space, and some are visually entangled with each other (e.g. `Male`
   and `Mustache`). The final variants weight each attribute direction by how reliable it is
   and reduce cross-attribute contamination before using it in the query.

All hyperparameters are selected via grid search on the CelebA validation split and evaluated
once on the official test split.

## Results

| Experiment | Recall@1 | Precision@1 | Recall@5 | Precision@5 | Recall@10 | Precision@10 |
|---|---:|---:|---:|---:|---:|---:|
| CLIP Arithmetic Baseline | 0.024 | 0.024 | 0.075 | 0.019 | 0.116 | 0.016 |
| Gated Text Fusion | 0.028 | 0.028 | 0.090 | 0.022 | 0.138 | 0.019 |
| Visual Direction Fusion | 0.030 | 0.030 | 0.096 | 0.024 | 0.147 | 0.021 |
| Hybrid Text + Visual Fusion | 0.033 | 0.033 | 0.106 | 0.027 | 0.164 | 0.023 |
| Prompt Averaging Hybrid | 0.034 | 0.034 | 0.110 | 0.027 | 0.168 | 0.024 |
| Prompt Difference Hybrid | 0.037 | 0.037 | 0.113 | 0.029 | **0.174** | 0.025 |
| Reliability-Weighted Hybrid | 0.034 | 0.034 | 0.110 | 0.027 | 0.169 | 0.024 |
| Direction-Decorrelation Reliability Hybrid | 0.034 | 0.034 | 0.108 | 0.027 | 0.169 | 0.024 |

The best configuration (prompt-difference hybrid fusion) improves Recall@10 by about **+50%
relative to the baseline**. Gains taper off after that point: prompt averaging, reliability
weighting, and direction decorrelation only produce small additional changes, suggesting the
main limitation is the use of one fixed global fusion rule across all attributes rather than
the individual components themselves — some attributes likely need query-adaptive weighting
that a single set of gate weights can't provide.

## Notebook

[`final_notebook.ipynb`](final_notebook.ipynb) is the final, self-contained deliverable for
this project. It includes the full method descriptions and formulas, all experiments, results,
qualitative examples, discussion, and references — everything needed to understand and
reproduce the work end to end.

The rest of the repository (`src/`, `scripts/`, `docs/`, `outputs/`) contains the exploratory
and working code used to develop these methods along the way.
