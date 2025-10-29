# image-reg-class
# 이미지 분류(Image Classification) vs 이미지 회귀(Image Regression)

컴퓨터 비전에서는 이미지를 분석하는 문제가 크게 두 가지로 나뉩니다: **분류(Classification)**과 **회귀(Regression)**.  
두 접근법은 목표, 출력, 손실 함수, 평가 지표 등에서 차이가 있습니다.

---

## 1. 주요 특징 비교

| 항목 | 이미지 분류 (Classification) | 이미지 회귀 (Regression) |
|------|----------------------------|-------------------------|
| **목적** | 이미지를 사전에 정의된 클래스 중 하나로 분류 | 이미지를 기반으로 연속적인 수치값 예측 |
| **출력(Output)** | 범주형 레이블 | 연속형 값(실수) |
| **데이터(Label)** | 카테고리(Class) | 연속 값(Continuous) |
| **손실 함수(Loss Function)** | Cross-Entropy Loss | MSE, MAE 등 |
| **평가 지표(Metrics)** | Accuracy, Precision, Recall, F1-score | RMSE, MAE, R² 등 |
| **모델 구조** | 마지막 레이어 Softmax | 마지막 레이어 Linear (활성화 없음) |
| **활용 예시** | 동물 종류 분류, 손글씨 숫자 인식, 질병 진단 | 나이 추정, 집 가격 예측, 온도 측정 |

---

## 2. 예시 데이터 비교

| 이미지 | 분류 라벨 | 회귀 값 |
|--------|------------|---------|
| 🐱 | 고양이 | - |
| 🐶 | 강아지 | - |
| 🚗 | 자동차 | - |
| 얼굴 사진 | - | 25세 |
| 집 사진 | - | 3500 sqft |
| 자동차 사진 | - | 120 km/h |

---

## 3. 코드 예시 (PyTorch)

### 이미지 분류
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16*30*30, 3),  # 3 classes
    nn.Softmax(dim=1)
)
