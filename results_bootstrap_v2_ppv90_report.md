# Bootstrap v2 PPV@90 Recall Report

## Scope

No top-level `results` directory was present in the workspace, so this report analyzes the available run artifacts under `output/`.

Primary ranking metric:

`bootstrap_metrics_v2.PPV@90RECALL 95% CI Lower Bound`

Higher is better. Values are reported on the original 0-1 scale.

## Primary Ranking: Best Ensemble Result Per Model

This is the recommended model-level ranking because the top-level `*_ensemble_metrics.json` files summarize complete model outputs more directly than single fold/epoch validation files.

| Rank | Model / Run | Split | Best Epoch | PPV@90RECALL 95% CI Lower Bound | PPV@90RECALL | PPV@90RECALL Full Dataset | AUPRC Full Dataset | AUROC Full Dataset |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | ViT-Base_GastroNet-5M_DINOv2_LoRA | finetune | epoch_10 | 0.081536 | 0.434262 | 0.811643 | 0.939313 | 0.992852 |
| 2 | ensemble_RN50_Billion_Vit_Base_DINOv2 | sanity_check | epochs_10_15 | 0.027156 | 0.136554 | 0.462890 | 0.877110 | 0.976138 |
| 3 | RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1 | finetune | epoch_3 | 0.026712 | 0.127253 | 0.452577 | 0.883365 | 0.978722 |
| 4 | ViT-Base_GastroNet-5M_DINOv2 | finetune | epoch_5 | 0.016838 | 0.036807 | 0.193154 | 0.674281 | 0.935284 |

## Interpretation

The strongest model by the requested lower-bound criterion is `ViT-Base_GastroNet-5M_DINOv2_LoRA` at epoch 10. Its lower bound is about 3.0x higher than the next best ensemble-level result, and it also has the best full-dataset PPV@90RECALL, AUPRC, and AUROC among the ensemble summaries.

The sanity-check cross-model ensemble and the finetuned RN50 Billion-Scale model are effectively tied on the requested lower-bound metric: 0.027156 vs 0.026712. The sanity ensemble is slightly ahead on the lower bound, while RN50 is slightly ahead on full-dataset AUPRC and AUROC.

The plain `ViT-Base_GastroNet-5M_DINOv2` ensemble ranks last by lower bound despite some strong individual fold results. Its ensemble-level PPV@90RECALL lower bound is low, suggesting less robust precision at the 90% recall operating point when evaluated as the full model output.

## Ensemble Epoch Ranking

| Rank | Model / Run | Epoch | Lower Bound | PPV@90RECALL | Full Dataset PPV@90RECALL |
|---:|---|---|---:|---:|---:|
| 1 | ViT-Base_GastroNet-5M_DINOv2_LoRA | epoch_10 | 0.081536 | 0.434262 | 0.811643 |
| 2 | ViT-Base_GastroNet-5M_DINOv2_LoRA | epoch_4 | 0.035699 | 0.377584 | 0.767817 |
| 3 | ensemble_RN50_Billion_Vit_Base_DINOv2 | epochs_10_15 | 0.027156 | 0.136554 | 0.462890 |
| 4 | RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1 | epoch_3 | 0.026712 | 0.127253 | 0.452577 |
| 5 | RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1 | epoch_5 | 0.024390 | 0.220994 | 0.607172 |
| 6 | RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1 | epoch_2 | 0.022773 | 0.080282 | 0.329777 |
| 7 | RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1 | epoch_4 | 0.020647 | 0.124819 | 0.481706 |
| 8 | ViT-Base_GastroNet-5M_DINOv2_LoRA | epoch_2 | 0.018549 | 0.200611 | 0.587117 |
| 9 | ViT-Base_GastroNet-5M_DINOv2 | epoch_5 | 0.016838 | 0.036807 | 0.193154 |
| 10 | ViT-Base_GastroNet-5M_DINOv2 | epoch_3 | 0.016773 | 0.178641 | 0.689621 |
| 11 | ViT-Base_GastroNet-5M_DINOv2 | epoch_4 | 0.016634 | 0.141003 | 0.486652 |
| 12 | RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1 | epoch_1 | 0.014332 | 0.031365 | 0.172322 |
| 13 | ViT-Base_GastroNet-5M_DINOv2 | epoch_2 | 0.013671 | 0.037334 | 0.177705 |
| 14 | ViT-Base_GastroNet-5M_DINOv2 | epoch_1 | 0.011201 | 0.024142 | 0.130915 |

## Diagnostic: Best Available Fold/Checkpoint Per Model Directory

These rows use the best individual `*_val_metrics.json` or ensemble file found for each model directory. They are useful for spotting promising checkpoints, but they should not replace the ensemble-level ranking above because they can reflect a single favorable fold.

| Rank | Model / Run | Split | Source | Fold | Epoch | Lower Bound | PPV@90RECALL Full Dataset |
|---:|---|---|---|---|---|---:|---:|
| 1 | ViT-Base_GastroNet-5M_DINOv2 | finetune | fold | fold_3 | epoch_1 | 0.825833 | 0.756081 |
| 2 | RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1 | finetune | fold | fold_2 | epoch_5 | 0.821875 | 0.935054 |
| 3 | ViT-Base_GastroNet-5M_DINOv2_LoRA | finetune | fold | fold_2 | epoch_8 | 0.527778 | 0.966437 |
| 4 | model_RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1_frozen_TRUE_epochs15_seed42 | sanity_check | fold | fold_2 | epoch_10 | 0.183333 | 0.804444 |
| 5 | model_ViT-Base_GastroNet-5M_DINOv2_frozen_TRUE_epochs15_seed42 | sanity_check | fold | fold_3 | epoch_10 | 0.076856 | 0.458115 |
| 6 | model_RN50_GastroNet-1M_DINOv1_frozen_TRUE_epochs15_seed42 | sanity_check | fold | fold_4 | epoch_5 | 0.073109 | 0.516104 |
| 7 | model_RN50_GastroNet-5M_DINOv1_frozen_TRUE_epochs15_seed42 | sanity_check | fold | fold_3 | epoch_10 | 0.059561 | 0.473670 |
| 8 | model_VITS_GastroNet-5M_DINOv1_frozen_TRUE_epochs15_seed42 | sanity_check | fold | fold_2 | epoch_10 | 0.041454 | 0.339608 |
| 9 | ensemble_RN50_Billion_Vit_Base_DINOv2 | sanity_check | ensemble | all | epochs_10_15 | 0.027156 | 0.462890 |
| 10 | model_RN50_GastroNet-5M_MOCOv2_frozen_TRUE_epochs15_seed42 | sanity_check | fold | fold_4 | epoch_10 | 0.025202 | 0.204539 |
| 11 | model_RN50_GastroNet-200K_DINOv1_frozen_TRUE_epochs15_seed42 | sanity_check | fold | fold_2 | epoch_5 | 0.017205 | 0.320699 |
| 12 | model_RN50_GastroNet-5M_SIMCLRv2_frozen_TRUE_epochs15_seed42 | sanity_check | fold | fold_2 | epoch_15 | 0.016272 | 0.113922 |

## Recommendation

Use `ViT-Base_GastroNet-5M_DINOv2_LoRA` at epoch 10 as the leading candidate under the requested ranking criterion. Treat RN50 Billion-Scale epoch 3 and the sanity-check RN50+ViT ensemble as a close second tier. The fold-level table shows that individual folds can produce much higher lower bounds, so final model selection should rely on ensemble/full-output metrics rather than a single validation fold.
