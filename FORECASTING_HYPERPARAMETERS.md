# Forecasting Task Hyperparameters

이 문서는 TimesURL에서 forecasting task를 수행할 때 사용되는 모든 hyperparameter들을 정리한 것입니다.

## 1. 학습(Training) Hyperparameters

### 1.1 기본 학습 파라미터 (src/train.py)

| Hyperparameter | Type | Default | 설명 |
|---|---|---|---|
| `--batch-size` | int | 8 | 학습 시 사용하는 배치 크기 |
| `--lr` | float | 0.0001 | Learning rate (학습률) |
| `--repr-dims` | int | 320 | Representation dimension (표현 차원) |
| `--max-train-length` | int | 3000 | 최대 학습 시퀀스 길이. 이보다 긴 시퀀스는 잘림 |
| `--epochs` | int | None | 학습 에폭 수 |
| `--iters` | int | None | 학습 반복 횟수 (기본값: 데이터 크기 ≤100,000이면 200, 아니면 600) |
| `--sgd` | flag | False | SGD optimizer 사용 여부 (기본값은 AdamW) |
| `--seed` | int | None | Random seed |
| `--save-every` | int | None | 체크포인트 저장 주기 |

### 1.2 Contrastive Learning 파라미터

| Hyperparameter | Type | Default | 설명 |
|---|---|---|---|
| `--temp` | float | 1.0 | Temperature for contrastive loss |
| `--lmd` | float | 0.01 | Lambda: Hierarchical contrastive loss의 가중치 |

### 1.3 Masking 파라미터 (Reconstruction Task용)

| Hyperparameter | Type | Default | 설명 |
|---|---|---|---|
| `--segment_num` | int | 3 | 마스킹할 시간 구간(segment) 개수 |
| `--mask_ratio_per_seg` | float | 0.05 | 각 segment에서 마스킹할 시퀀스 길이의 비율 |

### 1.4 데이터 관련 파라미터

| Hyperparameter | Type | Default | 설명 |
|---|---|---|---|
| `--irregular` | float | 0 | Missing observations의 비율 |
| `--load_tp` | flag | True | Timestamp를 feature로 포함할지 여부 |

## 2. 모델 아키텍처 Hyperparameters (src/timesurl.py)

| Hyperparameter | Type | Default | 설명 |
|---|---|---|---|
| `output_dims` | int | 320 | Representation dimension |
| `hidden_dims` | int | 64 | Encoder의 hidden dimension |
| `depth` | int | 10 | Encoder의 residual block 개수 |
| `temporal_unit` | int | 0 | Temporal contrast를 수행할 최소 단위 |

## 3. 추론(Inference) Hyperparameters (src/tasks/forecasting.py)

| Hyperparameter | Value | 설명 |
|---|---|---|
| `batch_size` | 256 | 인코딩 시 사용하는 배치 크기 |
| `casual` | True | Causal encoding (미래 정보 사용 안함) |
| `sliding_length` | 1 | Sliding window의 길이 |
| `sliding_padding` | 200 | Sliding window의 패딩 크기 |

## 4. 데이터셋별 설정 (src/datautils.py)

### 4.1 예측 길이 (Prediction Lengths)

**ETTh1, ETTh2, electricity, WTH (Weather):**
```python
pred_lens = [24, 48, 168, 336, 720]
```

**ETTm1, ETTm2, traffic, 기타 데이터셋:**
```python
pred_lens = [24, 48, 96, 288, 672]
```

### 4.2 데이터 분할 (Train/Valid/Test Split)

**ETTh 데이터셋 (ETTh1, ETTh2):**
- Train: 0 ~ 12*30*24 (약 12개월)
- Valid: 12*30*24 ~ 16*30*24 (약 4개월)
- Test: 16*30*24 ~ 20*30*24 (약 4개월)

**ETTm 데이터셋 (ETTm1, ETTm2):**
- Train: 0 ~ 12*30*24*4 (약 12개월, 15분 단위)
- Valid: 12*30*24*4 ~ 16*30*24*4 (약 4개월)
- Test: 16*30*24*4 ~ 20*30*24*4 (약 4개월)

**기타 데이터셋 (electricity, traffic, weather 등):**
- Train: 0 ~ 60% of data
- Valid: 60% ~ 80% of data
- Test: 80% ~ 100% of data

### 4.3 데이터셋별 특성

**Electricity 데이터셋:**
- 각 변수를 별도의 인스턴스로 처리 (`data.T` 사용)
- Univariate 모드에서는 'MT_001' 변수만 사용

**ETT 데이터셋 (ETTh1, ETTh2, ETTm1, ETTm2):**
- Univariate 모드에서는 'OT' (Oil Temperature) 변수만 사용

**WTH (Weather) 데이터셋:**
- Univariate 모드에서는 'WetBulbCelsius' 변수만 사용

### 4.4 Covariate Features

Time features (날짜/시간 관련 features):
- minute
- hour
- dayofweek
- day
- dayofyear
- month
- weekofyear

총 7개의 time covariate features가 forecasting task에서 사용됩니다 (offset=0인 경우).

## 5. Collator 파라미터 (src/collator.py)

Contrastive Learning을 위한 데이터 샘플링 파라미터:

| Hyperparameter | Value | 설명 |
|---|---|---|
| `len_sampling_bound` | [0.3, 0.7] | 샘플링 길이 범위 |
| `dense_sampling_bound` | [0.4, 0.6] | Dense 영역 샘플링 비율 범위 |
| `pretrain_tasks` | 'full2' | Pretraining task 모드 |

## 6. 사용 예시

### 6.1 ETTh1 데이터셋으로 forecasting 학습

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

### 6.2 Electricity 데이터셋 (multivariate)

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

### 6.3 ETTm1 데이터셋 (univariate)

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

## 7. 소스 코드 위치

| Hyperparameter 종류 | 파일 | 라인 |
|---|---|---|
| Command-line arguments | `src/train.py` | 27-50 |
| Model architecture defaults | `src/timesurl.py` | 67-82 |
| Inference parameters | `src/tasks/forecasting.py` | 21-32 |
| Dataset configurations | `src/datautils.py` | 187-266 |
| Collator settings | `src/collator.py` | 9-18 |
| Default iters logic | `src/timesurl.py` | 134-135 |

## 8. 주요 Hyperparameter 조합 권장사항

### 8.1 기본 설정
- `batch_size`: 8
- `lr`: 0.0001
- `repr_dims`: 320
- `max_train_length`: 3000
- `temp`: 1.0
- `lmd`: 0.01
- `segment_num`: 3
- `mask_ratio_per_seg`: 0.05

### 8.2 대용량 데이터셋
- `iters`: 600
- `batch_size`: 16 (메모리가 충분한 경우)

### 8.3 소규모 데이터셋
- `iters`: 200
- `epochs`: 적은 수 (과적합 방지)

## 9. 참고사항

1. **기본 iteration 수**: 데이터 크기가 100,000 이하면 200 iterations, 그 이상이면 600 iterations이 기본값입니다.

2. **Optimizer**: 기본적으로 AdamW를 사용하며, `--sgd` 플래그를 사용하면 SGD로 변경됩니다.

3. **Learning rate scheduler**: Cosine annealing scheduler가 기본적으로 적용됩니다.

4. **Weight decay**: 모든 optimizer에 5e-4의 weight decay가 적용됩니다.

5. **Evaluation**: `--eval` 플래그를 추가하면 학습 후 자동으로 평가가 수행됩니다.

6. **Time features**: `load_tp=True`일 때 normalized timestamp가 추가 feature로 포함됩니다.
