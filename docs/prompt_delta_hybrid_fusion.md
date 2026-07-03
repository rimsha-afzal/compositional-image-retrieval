# Neutral-Subtracted Prompt Ensemble Hybrid Fusion

This experiment tests whether the CLIP text component becomes more useful when
it focuses on the requested modification instead of the generic idea of a
photo, face, or person.

The previous reliability-weighted hybrid uses raw CLIP text vectors together
with train-split reliability-weighted CelebA visual directions. Here we keep the
visual component unchanged and replace only the text component.

## Paired Neutral Subtraction

Each modified prompt is paired with a neutral prompt using the same template:

```text
neutral:  a photo of a face
modified: a photo of a face that is smiling
```

For every pair, we encode both prompts with CLIP, normalize them, subtract
neutral from modified, average the deltas, and normalize the final vector.

This gives:

```text
t_delta = normalize(mean_i(CLIP(modified_i) - CLIP(neutral_i)))
```

The paired subtraction is meant to remove template-level neutral content such as
`face`, `person`, and `photo`.

## Formula

```text
q = normalize(
  alpha * x_ref
  + beta_text * t_delta
  + sum(beta_pos * r_a * d_a for a in A_pos)
  - sum(beta_neg * r_a * d_a for a in A_neg)
)
```

This is intentionally separate from the correlation-aware direction experiment:
that experiment changes visual directions, while this one changes only the text
embedding.

## How To Run

Fixed previous-best reliability configuration:

```powershell
python scripts\evaluation\run_prompt_delta_hybrid_evaluation.py
```

Optional small text-weight grid:

```powershell
python scripts\evaluation\run_prompt_delta_hybrid_evaluation.py --grid-search
```

Outputs:

```text
outputs/prompt_delta_hybrid_fusion/results.csv
outputs/prompt_delta_hybrid_fusion/text_delta_diagnostics.csv
```
