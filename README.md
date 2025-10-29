# image-reg-class
\# 이미지 분류(Image Classification) vs 이미지 회귀(Image Regression)

컴퓨터 비전에서 **이미지 분석**은 크게 두 가지 문제로 나뉩니다: **분류(Classification)**와 **회귀(Regression)**. 두 접근법은 모델의 출력과 학습 방식에서 차이가 있습니다.

---

## 1. 이미지 분류 (Image Classification)

### 정의
이미지를 **정해진 카테고리 중 하나로 분류**하는 문제입니다.

### 특징
- **출력**: 클래스(label)  
  예: `고양이`, `강아지`, `자동차`
- **목표**: 각 이미지가 어떤 클래스에 속하는지 예측
- **손실 함수**: 일반적으로 `Cross-Entropy Loss` 사용
- **평가 지표**: 정확도(Accuracy), F1-score, Precision, Recall 등

### 예시
| 이미지 | 라벨 |
|--------|------|
| 🐱     | 고양이 |
| 🐶     | 강아지 |
| 🚗     | 자동차 |

```python
# PyTorch 예시
import torch.nn as nn
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, 1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(16*30*30, 3),  # 3 classes
    nn.Softmax(dim=1)
)
