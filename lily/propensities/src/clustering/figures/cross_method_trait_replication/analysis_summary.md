# Cross-method trait-cluster replication

Input matrices: Online DPO Qwen3-8B-Base summary and Ben's GRPO spillover cells for Qwen3-8B-Base/Qwen3-8B-Instruct.
Analysis scope: 9 shared paper traits. GRPO uses the intended positive-pole label for each trained trait.

## Method-level replication vs Online DPO

```csv
method,profile_pearson_vs_dpo,trait_similarity_spearman_vs_dpo,ari_k3_vs_dpo
Online DPO,1.000,1.000,1.000
GRPO Base,0.412,0.001,-0.163
GRPO Instruct,0.323,-0.106,0.082
```

## Shared cluster order

Claim super-int., Cooperation, Honest-humble, Harm refusal, Harm elaboration, Neuroticism, Self-preservation, Spitefulness, Power-seeking

## Natural clusters within each method

### Online DPO

- Cluster 1: Spitefulness, Harm elaboration
- Cluster 2: Power-seeking, Harm refusal
- Cluster 3: Neuroticism, Self-preservation
- Cluster 4: Cooperation, Honest-humble, Claim super-int.

### GRPO Base

- Cluster 1: Spitefulness, Neuroticism, Self-preservation, Power-seeking, Harm elaboration
- Cluster 2: Cooperation, Honest-humble, Claim super-int., Harm refusal

### GRPO Instruct

- Cluster 1: Neuroticism, Self-preservation
- Cluster 2: Claim super-int.
- Cluster 3: Honest-humble, Harm refusal
- Cluster 4: Spitefulness, Cooperation, Power-seeking
- Cluster 5: Harm elaboration


## Notes

- Heatmaps use column-centered, row-z profiles: they show relative spillover shape, not absolute judge score.
- `profile_pearson_vs_dpo` compares all 81 standardized cells.
- `trait_similarity_spearman_vs_dpo` compares the upper triangle of trait-trait similarity matrices.
- `ari_k3_vs_dpo` compares 3-cluster hierarchical assignments; with only 9 traits, treat it as a descriptive stability check.
- `natural_trait_clusters_by_method` orders and clusters traits independently inside each method, using silhouette-selected k from 2 to 5.
