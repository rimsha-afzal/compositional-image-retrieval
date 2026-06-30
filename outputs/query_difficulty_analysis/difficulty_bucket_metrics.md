# Query Difficulty Stratification

Buckets are quantiles of train-split query availability: low availability is `hard`, middle is `medium`, high is `easy`.

| difficulty_bucket | method | num_queries | recall@1 | recall@5 | recall@10 | precision@10 |
| --- | --- | --- | --- | --- | --- | --- |
| hard | CLIP text-only baseline | 5 | 0.0682 | 0.1682 | 0.2341 | 0.0366 |
| medium | CLIP text-only baseline | 4 | 0.0450 | 0.1600 | 0.2650 | 0.0352 |
| easy | CLIP text-only baseline | 4 | 0.0583 | 0.1267 | 0.1817 | 0.0288 |
| hard | CelebA visual-direction only | 5 | 0.0545 | 0.1636 | 0.2295 | 0.0364 |
| medium | CelebA visual-direction only | 4 | 0.0333 | 0.1200 | 0.2000 | 0.0263 |
| easy | CelebA visual-direction only | 4 | 0.0650 | 0.1600 | 0.2283 | 0.0352 |
| hard | Hybrid text + visual fusion | 5 | 0.0841 | 0.2614 | 0.3659 | 0.0559 |
| medium | Hybrid text + visual fusion | 4 | 0.0567 | 0.1817 | 0.2867 | 0.0395 |
| easy | Hybrid text + visual fusion | 4 | 0.0650 | 0.1533 | 0.2267 | 0.0325 |
| hard | Hybrid prompt-ensemble text + visual fusion | 5 | 0.0909 | 0.2409 | 0.3386 | 0.0520 |
| medium | Hybrid prompt-ensemble text + visual fusion | 4 | 0.0517 | 0.1750 | 0.2850 | 0.0398 |
| easy | Hybrid prompt-ensemble text + visual fusion | 4 | 0.0650 | 0.1700 | 0.2350 | 0.0358 |
