# Yelp2018 user-uniform collaborative filtering reproductions

Each epoch draws as many training interactions as there are training edges, with replacement. Each active user has probability 1/U, and each of that user's training positives has conditional probability 1/d(u). The runner uses the project's user sampler. Its historical sampler named full uses 1/(d(u)+1) per edge and has a different marginal. BPR negatives exclude training positives. The sampled batch also supplies the anchors for the model-specific contrastive objective.

All models use the project's original train/test files, seed 2021 and full-catalog dot-product ranking with training items masked. Yelp2018 selects the best test Recall@20 over up to 100 epochs with patience 50. Results include overall and head/middle/tail user Recall and NDCG at the selected epoch. The scripts save metrics and configuration. The changed sampling distribution, data split and selection protocol matter when comparing against published scores.

| Method | Paper | Author implementation reviewed | Objective details |
|---|---|---|---|
| RGCF | [SIGIR 2022](https://doi.org/10.1145/3477495.3531889) | [ChangxinTian/RGCF](https://github.com/ChangxinTian/RGCF/tree/a7dec9895fd3c4d1a16ee4eb7e82a07927b527ac) | Projects interaction rows and columns, prunes similarity-weighted edges and ranks sampled augmentation candidates. Adds user agreement to BPR. |
| LightGCL | [ICLR 2023](https://arxiv.org/abs/2302.08191) | [HKUDS/LightGCL](https://github.com/HKUDS/LightGCL/tree/5590453ad86782f58017e58d0b698d7f32175be3) | Main repository setting with a rank-5 randomized SVD of normalized interactions. Sums layers, uses raw dot products, clamps positive logits and includes positive and negative item anchors. |
| LightCCF | [SIGIR 2025](https://arxiv.org/abs/2504.10113) | [ZzYUuuu/LightCCF](https://github.com/ZzYUuuu/LightCCF/tree/45645ef15fc08e858ffc81e1b2530b4b1805ad34) | The denominator adds user-item and user-user similarities before exponentiation. The default MF encoder has zero propagation layers. Setting --layers to 3 enables the optional LightGCN encoder. |
| SGCL | [RecSys 2025](https://arxiv.org/abs/2507.13336) | [DavidZWZ/SGCL](https://github.com/DavidZWZ/SGCL/tree/84ca127f147f12ec44bff37ca53bbd79e57a3be4) | Supervised contrast on observed user-item pairs with within-batch user-user and item-item terms including their diagonals. |
| AdaGCL | [KDD 2023](https://arxiv.org/abs/2305.10837) | [HKUDS/AdaGCL](https://github.com/HKUDS/AdaGCL/tree/6359f1ed568bf3f99d6351848b9a0c7391c1c93f) | Self-loop graph, VGAE view, hard-concrete edge denoiser, separate contrastive and information-bottleneck updates followed by ranking and generator optimization. |
| NLGCL | [RecSys 2025](https://arxiv.org/abs/2507.07522) | [Jinfeng-Xu/NLGCL](https://github.com/Jinfeng-Xu/NLGCL/tree/2ba4e6637623e3d69ad72860ad6dd9e095ca27b3) | BPR plus adjacent-layer neighbor contrast. Uses the release's two directional in-batch denominators with repeated IDs. |

On lhb, the working directory is /home/lhb/cyj/GCL_CE/experiments/full_cf_baselines_v1. For an individual run:

```bash
python run_full_cf.py --method nlgcl --dataset yelp2018 --data-dir /home/lhb/cyj/GCL_CE/data/yelp2018 --output-dir runs/nlgcl --device cuda
```

The deployed suite waits on the final Gowalla queue before starting RGCF, LightGCL, LightCCF, SGCL, AdaGCL and NLGCL in order. Suite output is under /home/lhb/cyj/GCL_CE/runs/full_cf_yelp2018_v1. Each model has its own attempt directory with train.log, config.json, epochs.jsonl and result.json. A suite restart skips completed results and starts a fresh attempt for unfinished models.
