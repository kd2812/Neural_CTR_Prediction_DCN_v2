# Neural_CTR_Prediction_DCN_v2
PyTorch implementation of the Deep & Cross Network ([paper](https://arxiv.org/abs/2008.13535) [full rank method] for click-through-rate
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
- PR-AUC

## Approach
- **Preprocessing:** signed log1p and quantile clipping on integer features, median imputation with missing-value indicators, standardization. Categorical values seen fewer than 10 times map to a shared "rare" ID, and missing values get their own ID.
- **Embeddings:** one per categorical feature, dimension capped at 4.
- **Models:** MLPs of three sizes, and DCN (parallel cross network + deep network). The crosslayer is the **DCN-V2** form, 
$x_{l+1} = x_0 ⊙ (W x_l + b) + x_l$,
 with a full weight matrix.
- **Training:** AdamW (lr 1e-3, weight decay 1e-3), BCE loss, early stopping on validation
  log loss. 80/20 stratified train/validation split.
- **Thresholds:** for F1 / balanced accuracy, the threshold is chosen on validation to maximize
  F1 and then applied to the test set.
 
## Results [Test Set]:
| Model | ROC-AUC | Log Loss | F1 Score | Accuracy | Balanced Accuracy | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| MLP (64) | 0.6843 | 0.1401 | 0.1317 | 0.9130 | 0.5674 | 0.0722 |
| MLP (128, 64) | 0.6860 | 0.1401 | 0.1225 | 0.8481 | 0.5915 | 0.0683 |
| MLP (256, 128, 64) | 0.6757 | 0.1412 | 0.1199 | 0.8664 | 0.5793 | 0.0657 |
| DCN-V2 (2 Cross Layers + 128, 64) | 0.6689 | 0.1422 | 0.1051 | 0.9063 | 0.5481 | 0.0639 |
| DCN-V2 (1 Cross Layer + 64, 32) | 0.6797 | 0.1409 | 0.1301 | 0.8730 | 0.5885 | 0.0673 |
| DCN-V2 (1 Cross Layer + 16) | 0.6797 | 0.1408 | 0.1344 | 0.8699 | 0.5955 | 0.0715 |



## Architecture:
The architecture has basically two parts:
- The Deep network 
- The cross network

The categorical features are embedded first and then concatenated with the dense features to form $x_0$.
This $x_0$ is then fed to both the deep and cross networks.

The dense layer is like a MLP which brings out the hidden non-linear latent features from the data.

The cross network is a model which brings out the polynomial type interactive latent features that showcase how different features showcase.
The cross network has the following layer-wise computation:
$$x_{l+1} = x_0 ⊙ (W x_l + b) + x_l$$
[This is in DCN-v2 which differs from DCN-v1 where W was a vector instead of a d0xd0 matrix]

The low rank method of the original paper where W is further broken down in to two matrices U and V with W as their multiplication has not been applied here as the dataset is quite smaller in magnitude and that level of computation reduction is not required. 

The final outputs of both the deep and cross networks are then concatenated and passed to another Linear layer which outputs the probability after passing through a sigmoid layer.

## Limitations:
- The dataset is quite small which can lead to overfitting.
- No hyperparameter search
- No baseline designation

## Learnings and Findings:
- All the architectures perform almost in a similar way with most metrics within a small range.
- No architecture is clearly better. 
- The loss curves show clear signs of overfitting in some models. 
- DCN did not outperform MLPs.

- Having a small dataset is actually bottleneck which can lead to fast overfitting. 
- A more complex model might do worse on a particular dataset than a less complex one. 
- The best architecture depends a lot on the characteristics of the dataset.

## Possible next steps:
- Running k-fold CV on the dataset 
- Applying a Baseline model of XGBoost
- study of original DCN-v1 vs DCN-v2 architectures