# Reconstruction Loss Coefficient (재구성 손실 계수)

## 개요 (Overview)

TimesURL 모델의 학습 손실 함수(loss function)는 두 가지 주요 구성 요소로 이루어져 있습니다:

1. **Contrastive Loss (대조 손실)**: 시계열 데이터의 서로 다른 augmented view들 간의 표현(representation)을 학습
2. **Reconstruction Loss (재구성 손실)**: 마스킹된 시간 구간의 원본 값을 복원하는 능력을 학습

## Reconstruction Loss Coefficient란?

**Reconstruction loss coefficient (재구성 손실 계수)**는 reconstruction loss의 영향력을 조절하는 가중치 파라미터입니다.

### 코드 구현 위치

파일: `src/timesurl.py`, 줄 240-245

```python
if torch.sum(mask1_inter) > 0:
    loss += 1 * torch.sum(torch.pow((x_left[..., :-1] - left_recon) * mask1_inter, 2)) / (
            torch.sum(mask1_inter) + 1e-10) / 2
if torch.sum(mask2_inter) > 0:
    loss += 1 * torch.sum(torch.pow((x_right[..., :-1] - right_recon) * mask2_inter, 2)) / (
            torch.sum(mask2_inter) + 1e-10) / 2
```

## 현재 설정값

**Reconstruction loss coefficient = 1** (하드코딩)

이 값은 코드에서 명시적으로 `1 *`로 곱해져 있습니다.

## 전체 Loss 함수 구조

전체 loss 함수는 다음과 같이 구성됩니다:

```
Total Loss = λ × Contrastive Loss + 1 × Reconstruction Loss
```

여기서:
- **λ (lambda, `--lmd`)**: Contrastive loss의 가중치 (기본값: 0.01)
- **1**: Reconstruction loss의 가중치 (고정값)

### Loss 계산 식

```python
# Contrastive loss (가변 가중치)
loss += self.args.lmd * hierarchical_contrastive_loss(out1, out2, ...)

# Reconstruction loss (고정 가중치 = 1)
loss += 1 * MSE(original, reconstructed) / 2
```

## Reconstruction Loss의 역할

1. **마스킹된 시간 구간 복원**: 모델이 masking된 시계열 구간의 원본 값을 예측하도록 학습
2. **시간적 패턴 학습**: 시계열 데이터의 시간적 연속성과 패턴을 이해
3. **표현 품질 향상**: Contrastive learning과 함께 작용하여 더 나은 representation 학습

## Contrastive Loss Coefficient와의 비교

| Loss Type | Parameter | Default Value | Adjustable |
|-----------|-----------|---------------|------------|
| Contrastive Loss | `--lmd` | 0.01 | ✅ Yes (명령줄 인자) |
| Reconstruction Loss | (hardcoded) | 1.0 | ❌ No (코드에 고정) |

### 왜 Reconstruction Loss는 고정값인가?

논문에서 reconstruction loss coefficient를 1로 고정한 이유:

1. **안정적인 학습**: Reconstruction loss는 MSE 기반으로 이미 정규화되어 있어 스케일이 안정적
2. **Contrastive loss 조절의 우선순위**: λ (lmd)를 조절하여 contrastive loss의 영향만 조정
3. **실험적 최적화**: 저자들의 실험에서 coefficient=1이 최적의 성능을 보임

## 참고: Loss 가중치 조정 방법

Reconstruction loss의 가중치를 변경하고 싶다면:

1. `src/timesurl.py`의 241-242, 244-245번째 줄 수정
2. `1 *`을 원하는 가중치 값으로 변경
3. 또는 새로운 커맨드 라인 인자 추가 (예: `--recon_coeff`)

예시:
```python
# 기존
loss += 1 * torch.sum(...)

# 수정 예시
loss += self.args.recon_coeff * torch.sum(...)
```

## 관련 문서

- [Forecasting Hyperparameters (Korean)](FORECASTING_HYPERPARAMETERS.md)
- [Forecasting Hyperparameters (English)](FORECASTING_HYPERPARAMETERS_EN.md)
- Paper: [TimesURL: Self-supervised Contrastive Learning for Universal Time Series Representation Learning](https://arxiv.org/abs/2312.15709)

## Summary (요약)

**Reconstruction loss coefficient = 1 (고정값)**

TimesURL에서는 reconstruction loss의 가중치가 코드에 하드코딩되어 있으며, contrastive loss의 가중치(`--lmd`, 기본값 0.01)만 조절 가능합니다. 전체 loss는 `Loss = 0.01 × Contrastive + 1.0 × Reconstruction`의 형태입니다.
