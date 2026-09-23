# Project Context for Codex

## Project Overview

This project is for a Deep Learning assignment on **Compositional Image Retrieval**.

The task is to retrieve target images from CelebA given:

* a reference image;
* one or more positive textual attribute conditions;
* one or more negative textual attribute conditions.

Example:

```text
reference image + "+Eyeglasses, -Smiling"
```

The retrieved images should preserve the relevant visual identity/core attributes of the reference image while satisfying the requested semantic modifications.

## Dataset

The main dataset is **CelebA**.

CelebA contains face images annotated with 40 binary attributes.

The official evaluation must be performed on the **CelebA test split**.

The professor provides a mandatory ground-truth file:

```text
data/raw/celeba_evaluation.json
```

This file defines the official benchmark queries and valid target images.

## Ground-Truth JSON

The evaluation JSON is a list of query dictionaries.

Each query dictionary contains:

```json
{
  "query": "+Smiling",
  "ground_truth": {
    "13": [325, 456, 789]
  }
}
```

Use the actual key format present in the JSON file.

The `query` field contains the textual modification.

The `ground_truth` field maps source image indices to lists of valid target image indices.

## Critical CelebA Indexing Rule

The source and target IDs inside `celeba_evaluation.json` are **PyTorch CelebA test split indices**.

They are **not physical filenames**.

Correct:

```python
image, attr = celeba[13]
```

Wrong:

```python
Image.open("000013.jpg")
```

Do not manually construct image filenames from JSON indices.

Do not assume that index `13` corresponds to `000013.jpg`.

The official logic must respect PyTorch / torchvision CelebA dataset indexing.

## Evaluation Protocol

The official evaluation must use the source image indices provided in the JSON.

Do not generate new source images for the mandatory benchmark.

Do not change the official benchmark queries.

Do not modify the professor-provided ground-truth JSON.

For each query, evaluate across the valid source images provided in the JSON.

## Metrics

The required metrics are:

```text
Recall@1
Recall@5
Recall@10
Precision@1
Precision@5
Precision@10
```

Recall@K is binary for each source image:

```text
1 if at least one valid ground-truth target appears in the top K retrieved results
0 otherwise
```

Precision@K is:

```text
number of valid ground-truth targets in top K / K
```

The final metrics must be averaged across source images.

## Required Base Model

The required base model is:

```text
CLIP ViT-B/32
openai/clip-vit-base-patch32
```

This should be the main model used for comparable results.

Other models can be tested only as additional experiments.

## Development Guidelines

This repository should stay simple, readable, and easy to convert into the final Colab notebook.

The goal is not to build a production ML platform. The goal is to build a clean, reproducible research pipeline for compositional image retrieval on CelebA.

### Core Principles

1. Keep the project structure minimal.
2. Add files only when there is a clear immediate need.
3. Prefer simple, explicit code over abstract frameworks.
4. Avoid over-engineering.
5. Avoid duplicate logic.
6. Every script should be runnable from the project root.
7. Every code change should have a clear purpose.
8. The official evaluation must remain independent from local/debug experiments.

### Folder Rules

Use folders with clear responsibilities:

```text
data/raw/
```

Original input files only. Raw files must be treated as read-only.

```text
data/subset/
```

Temporary local/debug subset files. These are allowed for faster development, but must not define the final benchmark.

```text
src/
```

Reusable Python code.

```text
scripts/
```

Lightweight runnable entry points.

```text
configs/
```

Configuration files for local/debug and VM/official runs.

```text
outputs/
```

Experiment results, metrics, predictions, logs, and generated artifacts.

Do not save experiment results inside `data/raw/`.

### Source Code vs Scripts

Reusable logic belongs in `src/`.

Runnable files belong in `scripts/`.

Scripts should be thin. A script should mainly:

1. parse arguments;
2. call functions from `src/`;
3. print or save a concise result.

Avoid writing large amounts of core logic directly inside scripts.

### Naming Rules

Use clear file names, for example:

```text
data_loading.py
evaluation_metrics.py
embedding_extraction.py
baseline_retrieval.py
visual_directions.py
```

Avoid vague names such as:

```text
utils.py
helpers.py
stuff.py
process.py
misc.py
```

Use lowercase `snake_case` for:

* Python files;
* functions;
* variables;
* scripts.

### Function Rules

Each function should do one clear thing.

Good examples:

```python
load_ground_truth(...)
recall_at_k(...)
precision_at_k(...)
extract_clip_embeddings(...)
```

Avoid vague functions such as:

```python
run_everything(...)
process_data(...)
do_all(...)
```

Do not create large functions that load data, compute embeddings, evaluate metrics, and save outputs all at once.

### Comments Rule

Use minimal explanatory comments.

Comments should explain non-obvious logic, not basic Python syntax.

Good:

```python
# Ground-truth IDs are PyTorch CelebA test indices, not filenames.
image, attr = celeba[source_index]
```

Bad:

```python
# Loop through items
for item in items:
```

Do not add long AI-style comments under every function.

### Path Rules

Do not hardcode absolute local paths.

Avoid paths like:

```text
C:/Users/...
```

Use relative paths and CLI arguments where needed.

Good examples:

```text
data/raw/celeba_evaluation.json
outputs/runs/
configs/local_debug.yaml
configs/vm_official.yaml
```

### CelebA Evaluation Rule

The IDs inside `celeba_evaluation.json` are PyTorch CelebA test split indices.

They are not physical filenames.

Correct:

```python
image, attr = celeba[index]
```

Wrong:

```python
Image.open("000013.jpg")
```

Do not manually construct image filenames from evaluation indices.

Do not modify the professor-provided `celeba_evaluation.json`.

Do not merge duplicate query names unless explicitly requested.

Preserve the official ground-truth structure.

### Local Subset vs Official Evaluation

The project may use a local/debug subset to make development faster.

However, the subset must not bias the final project.

Rules:

1. Do not report subset results as final results.
2. Do not tune final conclusions only around the subset.
3. Do not change the benchmark definition to fit the subset.
4. Do not make the subset the foundation of the final evaluation.
5. Always verify final results on the official CelebA test split.
6. Clearly label subset results as debug or preliminary.

### Download and VM Rules

Do not use `download=True` by default.

Downloading CelebA must be explicit and opt-in.

Local machine:

```text
write code
run small checks
test metrics
test JSON loading
avoid full CelebA download
avoid full CLIP embedding extraction
```

VM:

```text
download/load CelebA
extract CLIP embeddings
run full official evaluation
save final experiment outputs
```

### Output Rules

Experiment outputs should go under:

```text
outputs/
```

A run should save results in a separate folder, for example:

```text
outputs/runs/baseline_clip_vit_b32/
```

Do not overwrite important previous results unless explicitly intended.

### Coding Workflow

Develop in small steps:

1. raw data setup;
2. data loading;
3. metric functions;
4. ground-truth iteration;
5. fake ranking evaluation;
6. CLIP embedding extraction;
7. vanilla CLIP baseline;
8. visual directions;
9. improved fusion method;
10. final tables and qualitative examples.

Every method should plug into the same evaluation pipeline.

### Collaboration Rules

The code should be understandable by teammates.

Before adding complexity, ask:

```text
Can this be explained in one sentence?
Can it run from the project root?
Can the local subset be deleted later without breaking official evaluation?
Is this needed now?
```

If the answer is no, the implementation is probably becoming too complicated.

### Rules for ChatGPT / Codex

When helping with this repository:

1. Follow these README guidelines.
2. Do not propose unnecessary folders or abstractions.
3. Do not duplicate existing functionality.
4. Prefer minimal changes.
5. Keep code simple and explicit.
6. Use minimal explanatory comments.
7. Avoid vague file names.
8. State exactly how to run any new script.
9. Do not assume full CelebA is available locally.
10. Do not convert evaluation indices into filenames.
11. Do not modify raw professor files.
12. Keep local/debug subset logic separate from official evaluation logic.
