# Forecasting Hyperparameters Quick Reference

## 데이터셋별 Hyperparameter 요약표

| Dataset | Loader | Prediction Lengths | Data Split | Univariate Variable | 기타 특징 |
|---------|--------|-------------------|------------|-------------------|----------|
| **ETTh1** | forecast_csv | [24, 48, 168, 336, 720] | 12M train / 4M valid / 4M test | OT | Hourly data |
| **ETTh2** | forecast_csv | [24, 48, 168, 336, 720] | 12M train / 4M valid / 4M test | OT | Hourly data |
| **ETTm1** | forecast_csv | [24, 48, 96, 288, 672] | 12M train / 4M valid / 4M test | OT | 15-min intervals |
| **ETTm2** | forecast_csv | [24, 48, 96, 288, 672] | 12M train / 4M valid / 4M test | OT | 15-min intervals |
| **electricity** | forecast_csv | [24, 48, 168, 336, 720] | 60% / 20% / 20% | MT_001 | Each var as instance |
| **WTH** | forecast_csv | [24, 48, 168, 336, 720] | 60% / 20% / 20% | WetBulbCelsius | Weather data |
| **traffic** | forecast_csv | [24, 48, 96, 288, 672] | 60% / 20% / 20% | Last column | Traffic data |

## 기본 Hyperparameter 설정

### 학습 파라미터 (Training)
```bash
--batch-size 8
--lr 0.0001
--repr-dims 320
--max-train-length 3000
--iters 600  # (200 for small datasets <100k samples)
```

### Contrastive Learning
```bash
--temp 1.0
--lmd 0.01
```

### Masking (Reconstruction)
```bash
--segment_num 3
--mask_ratio_per_seg 0.05
```

### 모델 아키텍처
```python
output_dims = 320
hidden_dims = 64
depth = 10
temporal_unit = 0
```

### 추론 (Inference)
```python
batch_size = 256
casual = True
sliding_length = 1
sliding_padding = 200
```

## 명령어 예시

### ETTh1 (multivariate)
```bash
python src/train.py ETTh1 forecast_etth1 \
    --loader forecast_csv \
    --batch-size 8 \
    --lr 0.0001 \
    --repr-dims 320 \
    --iters 600 \
    --eval
```

### ETTm1 (univariate)
```bash
python src/train.py ETTm1 forecast_ettm1_uni \
    --loader forecast_csv_univar \
    --batch-size 8 \
    --lr 0.0001 \
    --repr-dims 320 \
    --iters 600 \
    --eval
```

### Electricity
```bash
python src/train.py electricity forecast_elec \
    --loader forecast_csv \
    --batch-size 8 \
    --lr 0.0001 \
    --repr-dims 320 \
    --iters 600 \
    --eval
```

### Traffic
```bash
python src/train.py traffic forecast_traffic \
    --loader forecast_csv \
    --batch-size 8 \
    --lr 0.0001 \
    --repr-dims 320 \
    --iters 600 \
    --eval
```

## Time Covariate Features

모든 forecasting task에서 사용되는 7개의 time features (offset=0일 때):
1. minute
2. hour
3. dayofweek
4. day
5. dayofyear
6. month
7. weekofyear

## 소스 코드 참조

| 항목 | 파일 경로 | 라인 번호 |
|------|----------|---------|
| CLI Arguments | `src/train.py` | 27-50 |
| Model Defaults | `src/timesurl.py` | 67-82 |
| Inference Params | `src/tasks/forecasting.py` | 21-32 |
| Dataset Config | `src/datautils.py` | 187-266 |
| Collator Settings | `src/collator.py` | 9-18 |

## 주요 참고사항

✅ **Optimizer**: AdamW (기본), SGD (--sgd 플래그 사용 시)
✅ **Weight Decay**: 5e-4
✅ **Scheduler**: Cosine Annealing
✅ **Normalization**: StandardScaler
✅ **Final Predictor**: Ridge Regression

자세한 내용은 [FORECASTING_HYPERPARAMETERS.md](FORECASTING_HYPERPARAMETERS.md) 또는 [FORECASTING_HYPERPARAMETERS_EN.md](FORECASTING_HYPERPARAMETERS_EN.md)를 참조하세요.
