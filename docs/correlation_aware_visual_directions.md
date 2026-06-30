# Correlation-Aware Visual Directions

This experiment tests whether CelebA attribute correlations make CLIP visual
directions noisy. A direction such as `Mustache` may partly encode `Male`, and
`Wearing_Lipstick` may overlap with `Heavy_Makeup`.

The method keeps the existing reliability-weighted hybrid formula, but replaces
the visual direction artifact with conservative decorrelated directions.

## What Is Measured

Label correlation measures Pearson correlation between train-split CelebA
attribute labels. It shows which attributes often occur together or exclude each
other.

Direction cosine similarity measures cosine similarity between train-split
visual direction vectors. It shows which learned CLIP movement directions point
in similar or opposite directions.

## Projection Removal

For each attribute direction, we find up to a few highly similar direction
vectors. We remove the projection onto those selected directions, then normalize
again. This is conservative by default: cosine threshold `0.30`, at most `3`
confounders per attribute.

All correlations, directions, reliability scores, and decorrelated directions
are computed from the train split only. Validation and test data are used only
for retrieval evaluation.

## Formula

```text
q = normalize(
  alpha * x_ref
  + beta_text * t_query
  + sum(beta_pos * r_a * d_clean_a for a in A_pos)
  - sum(beta_neg * r_a * d_clean_a for a in A_neg)
)
```

## How To Run

Stage 1, correlation analysis:

```powershell
python scripts\analysis\analyze_attribute_correlations.py
```

Outputs:

```text
outputs/correlation_analysis/label_correlation_matrix.csv
outputs/correlation_analysis/direction_cosine_matrix.csv
outputs/correlation_analysis/top_label_correlations.csv
outputs/correlation_analysis/top_direction_similarities.csv
```

Stage 2, decorrelated directions:

```powershell
python scripts\directions\compute_decorrelated_visual_directions.py
```

Outputs:

```text
outputs/decorrelated_directions/decorrelated_visual_directions.pt
outputs/decorrelated_directions/decorrelated_visual_directions_ids.npy
outputs/decorrelated_directions/decorrelation_diagnostics.csv
outputs/decorrelated_directions/decorrelation_config.json
```

Stage 3, evaluation:

```powershell
python scripts\evaluation\run_correlation_aware_hybrid_evaluation.py
```

Optional small visual-weight grid:

```powershell
python scripts\evaluation\run_correlation_aware_hybrid_evaluation.py --grid-search
```

Output:

```text
outputs/correlation_aware_hybrid_fusion/results.csv
```
