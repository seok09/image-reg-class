# image-reg-class
# 이미지 분류(Image Classification) vs 이미지 회귀(Image Regression)

컴퓨터 비전 분야에서 이미지를 분석하는 문제는 크게 두 가지로 나뉩니다: **이미지 분류(Image Classification)**와 **이미지 회귀(Image Regression)**. 두 접근법은 모델의 목표, 출력 형태, 손실 함수, 평가 방법 등에서 차이가 있습니다. 이 문서에서는 두 개념을 상세하게 비교하고, 예시와 함께 이해를 돕습니다.

---

## 1. 이미지 분류(Image Classification)

### 정의
이미지를 **사전에 정의된 클래스 중 하나로 분류**하는 문제입니다.  
즉, 모델의 목표는 입력 이미지를 보고 "어떤 카테고리에 속하는가?"를 결정하는 것입니다.

### 특징
- **출력(Output)**: 클래스 레이블(Label)
  - 예: 고양이, 강아지, 자동차 등
- **데이터(Label)**: 범주형(Categorical)  
- **손실 함수(Loss Function)**: 주로 `Cross-Entropy Loss`
- **평가 지표(Metrics)**: Accuracy, Precision, Recall, F1-score 등
- **모델 구조**: 마지막 레이어에서 Softmax 활성화 함수를 사용하여 각 클래스의 확률을 예측

### 실제 예시
| 이미지 | 라벨 |
|--------|------|
| 🐱     | 고양이 |
| 🐶     | 강아지 |
| 🚗     | 자동차 |

#### 코드 예시 (PyTorch)
```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16*30*30, 3),  # 3 classes
    nn.Softmax(dim=1)
)
