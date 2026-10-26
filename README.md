# Neural_CTR_Prediction_DCN_v2
PyTorch implementation of the Deep & Cross Network ([https://arxiv.org/abs/2008.13535]) [full rank method] for click-through-rate
prediction, compared against plain MLP baselines on a small Criteo-style dataset.


## Run it
```
pip install -r requirements.txt
# change the directory addresses of the files 
```

## Dataset details:
- Criteo style dataset 
- 13 integer type features [dense features]
- 26 categorical features [sparse features]
- Total rows in training dataset: 40,000
- Out of that : 32k for training and 8k for validation
- Total rows in test set: 10,000
- Dataset is highly imbalanced with around 3.2% positive labels


## Metrics used:
- ROC-AUC
- Binary Log-loss
- F1 Score
- Accuracy
- Balanced Accuracy
- Precision-AUC

## Approach
- **Preprocessing:** signed log1p and quantile clipping on integer features, median imputation with missing-value indicators, standardization. Categorical values seen fewer than 10 times map to a shared "rare" ID, and missing values get their own ID.
- **Embeddings:** one per categorical feature, dimension capped at 4.
- **Models:** MLPs of three sizes, and DCN (parallel cross network + deep network). The crosslayer is the **DCN-V2** form, 
$x_{l+1} = x_0 ⊙ (W x_l + b) + x_l$,
 with a full weight matrix.
- **Training:** AdamW (lr 1e-3, weight decay 1e-4), BCE loss, early stopping on validation
  log loss. 80/20 stratified train/validation split.
- **Thresholds:** for F1 / balanced accuracy, the threshold is chosen on validation to maximize
  F1 and then applied to the test set.
 
## Results:
| Model | ROC-AUC | Log Loss | F1 Score | Accuracy | Balanced Accuracy | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| MLP (64) | 0.6843 | 0.1401 | 0.1317 | 0.9130 | 0.5674 | 0.0722 |
| MLP (128, 64) | 0.6860 | 0.1401 | 0.1225 | 0.8481 | 0.5915 | 0.0683 |
| MLP (256, 128, 64) | 0.6757 | 0.1412 | 0.1199 | 0.8664 | 0.5793 | 0.0657 |
| DCN-V2 (2 Cross Layers + 128, 64) | 0.6689 | 0.1422 | 0.1051 | 0.9063 | 0.5481 | 0.0639 |
| DCN-V2 (1 Cross Layer + 64, 32) | 0.6797 | 0.1409 | 0.1301 | 0.8730 | 0.5885 | 0.0673 |
| DCN-V2 (2 Cross Layers + 32, 16) | 0.6797 | 0.1408 | 0.1344 | 0.8699 | 0.5955 | 0.0715 |