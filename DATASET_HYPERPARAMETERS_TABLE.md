# 데이터셋별 Fine-tuning Hyperparameter 요약표

## 개요

TimesURL의 forecasting task는 **Pretrain**과 **Finetune** 두 단계로 구성됩니다:
- **Pretrain 단계**: TimesURL 모델을 학습 (contrastive learning + reconstruction)
- **Finetune 단계**: Pretrain된 모델로 인코딩한 representation을 사용하여 Ridge regression 학습

## 데이터셋별 Hyperparameter 테이블

### Pretrain 단계 (TimesURL 모델 학습)

| Dataset | Learning Rate | Batch Size | Iterations | Max Train Length | Repr Dims | Other Parameters |
|---------|--------------|------------|------------|------------------|-----------|------------------|
| **ETTh1** | 0.0001 | 8 | 600 | 3000 | 320 | temp=1.0, lmd=0.01, segment_num=3, mask_ratio_per_seg=0.05 |
| **ETTh2** | 0.0001 | 8 | 600 | 3000 | 320 | temp=1.0, lmd=0.01, segment_num=3, mask_ratio_per_seg=0.05 |
| **ETTm1** | 0.0001 | 8 | 600 | 3000 | 320 | temp=1.0, lmd=0.01, segment_num=3, mask_ratio_per_seg=0.05 |
| **ETTm2** | 0.0001 | 8 | 600 | 3000 | 320 | temp=1.0, lmd=0.01, segment_num=3, mask_ratio_per_seg=0.05 |
| **electricity** | 0.0001 | 8 | 600 | 3000 | 320 | temp=1.0, lmd=0.01, segment_num=3, mask_ratio_per_seg=0.05 |
| **weather (WTH)** | 0.0001 | 8 | 600 | 3000 | 320 | temp=1.0, lmd=0.01, segment_num=3, mask_ratio_per_seg=0.05 |
| **traffic** | 0.0001 | 8 | 600 | 3000 | 320 | temp=1.0, lmd=0.01, segment_num=3, mask_ratio_per_seg=0.05 |
| **기타 데이터셋 (대용량)** | 0.0001 | 8 | 600 | 3000 | 320 | temp=1.0, lmd=0.01, segment_num=3, mask_ratio_per_seg=0.05 |
| **기타 데이터셋 (소규모)** | 0.0001 | 8 | 200 | 3000 | 320 | temp=1.0, lmd=0.01, segment_num=3, mask_ratio_per_seg=0.05 |

**참고사항:**
- 모든 데이터셋에 대해 동일한 hyperparameter를 사용합니다
- Optimizer: AdamW (기본값), SGD (--sgd 플래그 사용 시)
- Weight Decay: 5e-4
- Learning Rate Scheduler: Cosine Annealing (SGD 사용 시)
- Iterations: 데이터 크기가 100,000 이하면 200, 그 이상이면 600 (기본값)

### Finetune 단계 (Ridge Regression 학습)

| Dataset | 학습 방법 | "Learning Rate" | "Batch Size" | Alpha (Regularization) | Encoding Batch Size |
|---------|----------|----------------|-------------|----------------------|-------------------|
| **ETTh1** | Ridge Regression | N/A (closed-form solution) | N/A (전체 데이터 사용*) | Grid Search로 자동 선택 | 256 |
| **ETTh2** | Ridge Regression | N/A (closed-form solution) | N/A (전체 데이터 사용*) | Grid Search로 자동 선택 | 256 |
| **ETTm1** | Ridge Regression | N/A (closed-form solution) | N/A (전체 데이터 사용*) | Grid Search로 자동 선택 | 256 |
| **ETTm2** | Ridge Regression | N/A (closed-form solution) | N/A (전체 데이터 사용*) | Grid Search로 자동 선택 | 256 |
| **electricity** | Ridge Regression | N/A (closed-form solution) | N/A (전체 데이터 사용*) | Grid Search로 자동 선택 | 256 |
| **weather (WTH)** | Ridge Regression | N/A (closed-form solution) | N/A (전체 데이터 사용*) | Grid Search로 자동 선택 | 256 |
| **traffic** | Ridge Regression | N/A (closed-form solution) | N/A (전체 데이터 사용*) | Grid Search로 자동 선택 | 256 |

**참고사항:**
- Finetune 단계는 전통적인 gradient descent 방식의 학습을 사용하지 않습니다
- Ridge Regression은 closed-form solution을 사용하여 학습하므로 learning rate나 batch size 개념이 없습니다
- **Alpha 값 탐색 범위**: [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
- **Alpha 선택 기준**: Validation set에서 MSE + MAE가 가장 낮은 alpha 선택
- **Encoding Batch Size**: Pretrain된 모델로 데이터를 인코딩할 때 사용하는 배치 크기 (256으로 고정)
- **MAX_SAMPLES**: 샘플 수가 100,000개를 초과하면 자동으로 서브샘플링 수행

### 데이터셋별 특성

| Dataset | Prediction Lengths | Data Split | Univariate Variable | 비고 |
|---------|-------------------|------------|---------------------|------|
| **ETTh1** | [24, 48, 168, 336, 720] | 12M / 4M / 4M | OT | Hourly data |
| **ETTh2** | [24, 48, 168, 336, 720] | 12M / 4M / 4M | OT | Hourly data |
| **ETTm1** | [24, 48, 96, 288, 672] | 12M / 4M / 4M | OT | 15-min intervals |
| **ETTm2** | [24, 48, 96, 288, 672] | 12M / 4M / 4M | OT | 15-min intervals |
| **electricity** | [24, 48, 168, 336, 720] | 60% / 20% / 20% | MT_001 | Each variable as instance |
| **weather (WTH)** | [24, 48, 168, 336, 720] | 60% / 20% / 20% | WetBulbCelsius | Weather data |
| **traffic** | [24, 48, 96, 288, 672] | 60% / 20% / 20% | Last column | Traffic data |

## 상세 설명

### 1. Pretrain 단계의 Hyperparameters

Pretrain 단계에서 사용되는 hyperparameter들은 **모든 데이터셋에 대해 동일한 기본값**을 사용합니다:

- **Learning Rate (lr)**: 0.0001
- **Batch Size**: 8
- **Representation Dimension (repr_dims)**: 320
- **Max Train Length**: 3000
- **Temperature (temp)**: 1.0
- **Lambda (lmd)**: 0.01 (hierarchical contrastive loss weight)
- **Segment Number (segment_num)**: 3
- **Mask Ratio per Segment (mask_ratio_per_seg)**: 0.05

이러한 값들은 `src/train.py`에서 command-line arguments의 default 값으로 정의되어 있습니다.

### 2. Finetune 단계의 특징

Finetune 단계는 **전통적인 neural network 학습 방식과 다릅니다**:

1. **Ridge Regression 사용**: Gradient descent가 아닌 closed-form solution을 사용하여 학습
2. **Learning Rate 없음**: Closed-form solution이므로 learning rate 개념이 존재하지 않음
3. **Batch Size 개념 없음**: 전체 training 데이터를 한 번에 사용하여 학습 (단, 100,000개 초과 시 서브샘플링)
4. **Alpha 자동 선택**: Grid search를 통해 validation set에서 최적의 alpha 값을 자동으로 선택

### 3. 코드 위치

| 항목 | 파일 | 설명 |
|-----|------|------|
| Pretrain hyperparameters | `src/train.py` (line 27-50) | Command-line arguments 정의 |
| Finetune encoding | `src/tasks/forecasting.py` (line 21-32) | Encoding 시 batch_size=256 사용 |
| Ridge regression | `src/tasks/_eval_protocols.py` (line 81-109) | Alpha grid search 및 학습 |

## 사용 예시

### ETTh1 데이터셋 (Multivariate)

```bash
python src/train.py ETTh1 forecast_etth1 \
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

### Electricity 데이터셋

```bash
python src/train.py electricity forecast_elec \
    --loader forecast_csv \
    --batch-size 8 \
    --lr 0.0001 \
    --repr-dims 320 \
    --iters 600 \
    --eval
```

### 소규모 데이터셋 예시

```bash
python src/train.py small_dataset forecast_small \
    --loader forecast_csv \
    --batch-size 8 \
    --lr 0.0001 \
    --repr-dims 320 \
    --iters 200 \  # 소규모 데이터셋은 200 iterations
    --eval
```

## 요약

- **Pretrain 단계**: 모든 데이터셋에 대해 동일한 hyperparameter 사용 (lr=0.0001, batch_size=8)
- **Finetune 단계**: Ridge regression을 사용하므로 전통적인 의미의 learning rate와 batch size가 없음
- **Alpha 값**: Validation set 기반 grid search로 자동 선택 (각 데이터셋마다 다를 수 있음)
- **Encoding Batch Size**: 256 (고정값, 모든 데이터셋 동일)

이 설계는 TimesURL의 핵심 아이디어를 반영합니다: pretrain된 모델로 얻은 표현(representation)이 충분히 좋다면, 간단한 linear model(Ridge regression)만으로도 좋은 예측 성능을 낼 수 있다는 것입니다.
