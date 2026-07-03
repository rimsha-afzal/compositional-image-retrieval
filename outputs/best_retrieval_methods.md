# Best Retrieval Methods

This file summarizes the best configuration found for each retrieval mechanism
on the CelebA test subset. For the reliability-weighted and correlation-aware
methods, hyperparameters were selected on validation first and then evaluated
once on test.

| mechanism | best configuration | recall@1 | recall@5 | recall@10 | precision@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| CLIP text-only baseline default | `alpha=1.0`, `beta=1.0` | 0.0500 | 0.1122 | 0.1707 | 0.0261 |
| CLIP text-only baseline | `alpha=2.0`, `beta=1.5` | 0.0598 | 0.1500 | 0.2287 | 0.0337 |
| CelebA visual-direction only | `alpha=2.0`, `beta_pos=1.0`, `beta_neg=0.5` | 0.0506 | 0.1463 | 0.2183 | 0.0323 |
| Hybrid text + visual fusion | `alpha=1.0`, `beta_text=2.0`, `beta_visual_pos=0.5`, `beta_visual_neg=0.25` | 0.0671 | 0.1927 | 0.2860 | 0.0413 |
| Hybrid prompt-ensemble text + visual fusion | `alpha=1.0`, `beta_text=2.0`, `beta_visual_pos=0.25`, `beta_visual_neg=0.25` | 0.0671 | 0.1909 | 0.2811 | 0.0416 |
| Reliability-weighted hybrid text + visual fusion | `alpha=1.25`, `beta_text=2.5`, `beta_pos=0.75`, `beta_neg=0.25` | 0.0762 | 0.1982 | 0.2939 | 0.0438 |
| Correlation-aware reliability hybrid | `alpha=1.25`, `beta_text=2.5`, `beta_pos=0.75`, `beta_neg=0.25`, `direction_cosine_threshold=0.70`, `max_confounders=2` | 0.0738 | 0.1872 | 0.2829 | 0.0432 |
| Neutral-subtracted prompt-delta reliability hybrid | `alpha=1.25`, `beta_text=2.5`, `beta_pos=0.75`, `beta_neg=0.25`, `text_mode=prompt_delta` | 0.0713 | 0.2079 | 0.3122 | 0.0470 |

## Takeaway

The best clean final result is the neutral-subtracted prompt-delta reliability
hybrid. It improves `recall@10` by `+0.0835` over the recomputed CLIP text-only
baseline, by `+0.0262` over the original hybrid text + visual fusion, and by
`+0.0183` over the reliability-weighted hybrid with raw text.

The correlation-aware decorrelation experiment is useful as a negative result:
although softer decorrelation improved validation performance, its final test
result did not beat the reliability-weighted hybrid. This suggests that some
correlated visual-direction components may still carry useful semantic signal
for retrieval.

The prompt-delta experiment is the strongest result so far. It keeps the
reliability-weighted visual branch unchanged and replaces only the raw CLIP text
embedding with a paired neutral-subtracted prompt ensemble, suggesting that a
more modification-specific text vector is beneficial.

## Source Summaries

- `outputs/baseline_run/summary.csv`
- `outputs/visual_direction_run/summary.csv`
- `outputs/hybrid_fusion_run/summary.csv`
- `outputs/hybrid_prompt_ensemble_run/summary.csv`
- `outputs/reliability_hybrid_validation_run/summary.csv`
- `outputs/reliability_hybrid_final_test_run/summary.csv`
- `outputs/correlation_aware_hybrid_fusion_grid/summary.csv`
- `outputs/correlation_aware_hybrid_final_test_run/results.csv`
- `outputs/prompt_delta_hybrid_fusion/results.csv`
