# Forecasting Task Hyperparameters Documentation

This document provides a comprehensive overview of all hyperparameters used for forecasting tasks in TimesURL.

## 1. Training Hyperparameters

### 1.1 Basic Training Parameters (src/train.py)

| Hyperparameter | Type | Default | Description |
|---|---|---|---|
| `--batch-size` | int | 8 | Batch size for training |
| `--lr` | float | 0.0001 | Learning rate |
| `--repr-dims` | int | 320 | Representation dimension |
| `--max-train-length` | int | 3000 | Maximum sequence length for training; longer sequences are cropped |
| `--epochs` | int | None | Number of training epochs |
| `--iters` | int | None | Number of training iterations (default: 200 if data size ≤100,000, else 600) |
| `--sgd` | flag | False | Use SGD optimizer instead of AdamW |
| `--seed` | int | None | Random seed |
| `--save-every` | int | None | Save checkpoint every N iterations/epochs |

### 1.2 Contrastive Learning Parameters

| Hyperparameter | Type | Default | Description |
|---|---|---|---|
| `--temp` | float | 1.0 | Temperature for contrastive loss |
| `--lmd` | float | 0.01 | Lambda weight for hierarchical contrastive loss |

### 1.3 Masking Parameters (for Reconstruction Task)

| Hyperparameter | Type | Default | Description |
|---|---|---|---|
| `--segment_num` | int | 3 | Number of time interval segments to mask |
| `--mask_ratio_per_seg` | float | 0.05 | Fraction of sequence length to mask per segment |

### 1.4 Data-Related Parameters

| Hyperparameter | Type | Default | Description |
|---|---|---|---|
| `--irregular` | float | 0 | Ratio of missing observations |
| `--load_tp` | flag | True | Whether to include timestamp as a feature |

## 2. Model Architecture Hyperparameters (src/timesurl.py)

| Hyperparameter | Type | Default | Description |
|---|---|---|---|
| `output_dims` | int | 320 | Representation dimension |
| `hidden_dims` | int | 64 | Hidden dimension of the encoder |
| `depth` | int | 10 | Number of residual blocks in the encoder |
| `temporal_unit` | int | 0 | Minimum unit for temporal contrast |

## 3. Inference Hyperparameters (src/tasks/forecasting.py)

| Hyperparameter | Value | Description |
|---|---|---|
| `batch_size` | 256 | Batch size for encoding |
| `casual` | True | Causal encoding (no future information) |
| `sliding_length` | 1 | Sliding window length |
| `sliding_padding` | 200 | Sliding window padding size |

## 4. Dataset-Specific Configurations (src/datautils.py)

### 4.1 Prediction Lengths

**ETTh1, ETTh2, electricity, WTH (Weather):**
```python
pred_lens = [24, 48, 168, 336, 720]
```

**ETTm1, ETTm2, traffic, and other datasets:**
```python
pred_lens = [24, 48, 96, 288, 672]
```

### 4.2 Train/Valid/Test Split

**ETTh datasets (ETTh1, ETTh2):**
- Train: 0 ~ 12*30*24 (~12 months, hourly)
- Valid: 12*30*24 ~ 16*30*24 (~4 months)
- Test: 16*30*24 ~ 20*30*24 (~4 months)

**ETTm datasets (ETTm1, ETTm2):**
- Train: 0 ~ 12*30*24*4 (~12 months, 15-min intervals)
- Valid: 12*30*24*4 ~ 16*30*24*4 (~4 months)
- Test: 16*30*24*4 ~ 20*30*24*4 (~4 months)

**Other datasets (electricity, traffic, weather, etc.):**
- Train: 0 ~ 60% of data
- Valid: 60% ~ 80% of data
- Test: 80% ~ 100% of data

### 4.3 Dataset-Specific Features

**Electricity dataset:**
- Each variable is treated as a separate instance (uses `data.T`)
- In univariate mode, only 'MT_001' variable is used

**ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2):**
- In univariate mode, only 'OT' (Oil Temperature) variable is used

**WTH (Weather) dataset:**
- In univariate mode, only 'WetBulbCelsius' variable is used

### 4.4 Covariate Features

Time features (date/time related features):
- minute
- hour
- dayofweek
- day
- dayofyear
- month
- weekofyear

Total of 7 time covariate features are used in forecasting tasks (when offset=0).

## 5. Collator Parameters (src/collator.py)

Data sampling parameters for Contrastive Learning:

| Hyperparameter | Value | Description |
|---|---|---|
| `len_sampling_bound` | [0.3, 0.7] | Sampling length range |
| `dense_sampling_bound` | [0.4, 0.6] | Dense region sampling ratio range |
| `pretrain_tasks` | 'full2' | Pretraining task mode |

## 6. Usage Examples

### 6.1 Training on ETTh1 Dataset

```bash
python src/train.py ETTh1 forecast_run \
    --loader forecast_csv \
    --batch-size 8 \
    --lr 0.0001 \
    --repr-dims 320 \
    --max-train-length 3000 \
    --iters 600 \
    --temp 1.0 \
    --lmd 0.01 \
    --segment_num 3 \
    --mask_ratio_per_seg 0.05 \
    --eval
```

### 6.2 Training on Electricity Dataset (multivariate)

```bash
python src/train.py electricity forecast_run \
    --loader forecast_csv \
    --batch-size 8 \
    --lr 0.0001 \
    --repr-dims 320 \
    --max-train-length 3000 \
    --iters 600 \
    --eval
```

### 6.3 Training on ETTm1 Dataset (univariate)

```bash
python src/train.py ETTm1 forecast_run \
    --loader forecast_csv_univar \
    --batch-size 8 \
    --lr 0.0001 \
    --repr-dims 320 \
    --max-train-length 3000 \
    --iters 600 \
    --eval
```

## 7. Source Code Locations

| Hyperparameter Type | File | Lines |
|---|---|---|
| Command-line arguments | `src/train.py` | 27-50 |
| Model architecture defaults | `src/timesurl.py` | 67-82 |
| Inference parameters | `src/tasks/forecasting.py` | 21-32 |
| Dataset configurations | `src/datautils.py` | 187-266 |
| Collator settings | `src/collator.py` | 9-18 |
| Default iters logic | `src/timesurl.py` | 134-135 |

## 8. Recommended Hyperparameter Combinations

### 8.1 Default Configuration
- `batch_size`: 8
- `lr`: 0.0001
- `repr_dims`: 320
- `max_train_length`: 3000
- `temp`: 1.0
- `lmd`: 0.01
- `segment_num`: 3
- `mask_ratio_per_seg`: 0.05

### 8.2 Large Datasets
- `iters`: 600
- `batch_size`: 16 (if memory allows)

### 8.3 Small Datasets
- `iters`: 200
- `epochs`: Lower number (to prevent overfitting)

## 9. Important Notes

1. **Default iterations**: If data size ≤ 100,000, default is 200 iterations; otherwise, 600 iterations.

2. **Optimizer**: AdamW is used by default. Use `--sgd` flag to switch to SGD optimizer.

3. **Learning rate scheduler**: Cosine annealing scheduler is applied by default.

4. **Weight decay**: All optimizers use a weight decay of 5e-4.

5. **Evaluation**: Add `--eval` flag to automatically perform evaluation after training.

6. **Time features**: When `load_tp=True`, a normalized timestamp is included as an additional feature.

## 10. Summary Table: Hyperparameters by Dataset

| Dataset | Loader | pred_lens | Train Split | Notes |
|---|---|---|---|---|
| ETTh1 | forecast_csv | [24, 48, 168, 336, 720] | 0~12*30*24 | Hourly data |
| ETTh2 | forecast_csv | [24, 48, 168, 336, 720] | 0~12*30*24 | Hourly data |
| ETTm1 | forecast_csv | [24, 48, 96, 288, 672] | 0~12*30*24*4 | 15-min data |
| ETTm2 | forecast_csv | [24, 48, 96, 288, 672] | 0~12*30*24*4 | 15-min data |
| electricity | forecast_csv | [24, 48, 168, 336, 720] | 0~60% | Each variable as instance |
| WTH | forecast_csv | [24, 48, 168, 336, 720] | 0~60% | Weather data |
| traffic | forecast_csv | [24, 48, 96, 288, 672] | 0~60% | Traffic data |

All datasets use:
- 7 time covariate features (minute, hour, dayofweek, day, dayofyear, month, weekofyear)
- StandardScaler normalization
- Ridge regression for final prediction
