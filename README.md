# Bird Species Classification (ResNet50)

> 2024 실리콘밸리 AI 온라인 인턴십 심화과정 프로젝트
> 외부 사전학습 가중치 없이 CIFAR 데이터로 직접 사전학습 후 200종 조류 분류 모델 개발

---

## 프로젝트 개요

- 목표: CUB-200-2011 데이터셋 기반 200종 조류 분류 딥러닝 모델 개발
- 핵심 제약: 외부 사전학습 가중치(ImageNet 등) 사용 금지
- 전략: CIFAR-10/100으로 직접 사전학습 후 CUB-200으로 파인튜닝

---

## 파일 구조

- 01_pretrain_cifar.ipynb : CIFAR-10/100 사전학습
- 02_finetune_resnet50_cub200.ipynb : ResNet50 파인튜닝

---

## 데이터셋

| 데이터셋 | 용도 | 이미지 수 | 클래스 수 |
| :--- | :--- | :--- | :--- |
| CIFAR-10 | 사전학습 | 50,000 | 10 |
| CIFAR-100 | 사전학습 | 50,000 | 100 |
| CUB-200-2011 | 파인튜닝/평가 | 11,788 | 200 |

데이터 파일(.npy)은 용량 문제로 Google Drive에 저장
실행 환경: Google Colab (GPU: T4)

---

## 기술 스택

- Framework: PyTorch
- Model: ResNet50
- Language: Python
- Environment: Google Colab

---

## 전체 파이프라인

CIFAR-10 + CIFAR-100 -> SimpleCNN 사전학습 -> 가중치 저장 -> ResNet50 파인튜닝 -> 200종 조류 분류

---

## 핵심 전략 및 이유

### 1단계: CIFAR 사전학습 (01_pretrain_cifar.ipynb)

문제: 외부 사전학습 가중치 사용 금지 -> 기초 성능 확보 필요

해결:
- CIFAR-10 + CIFAR-100 결합 (총 110,000장)으로 직접 사전학습
- Mixed Precision Training (torch.amp FP16) -> GPU 메모리 효율화 및 학습 속도 향상
- AdamW 옵티마이저 + weight decay -> 일반화 성능 향상
- ReduceLROnPlateau -> 손실 정체 시 학습률 자동 감소
- best loss 기준 체크포인트 저장

### 2단계: ResNet50 파인튜닝 (02_finetune_resnet50_cub200.ipynb)

문제: 7 epoch 제한 + 200종 다중 클래스 + 데이터 부족

해결:
- CIFAR 사전학습 가중치 -> ResNet50에 Transfer Learning
- ReduceLROnPlateau (factor=0.1, patience=5) -> 학습률 동적 제어
- best loss 기준 최적 가중치만 저장 -> 과적합 방지
- ImageNet 정규화 값 적용 (0.485, 0.456, 0.406)

---

## 성능 결과

| 항목 | 내용 |
| :--- | :--- |
| 학습 epoch | 7 |
| 분류 클래스 수 | 200종 |
| 검증 데이터 수 | 2,897장 |
| 옵티마이저 | Adam (lr=0.0002) |

---

## 한계 및 개선 방향

- Data Augmentation 미적용 -> RandomHorizontalFlip, ColorJitter 추가 시 성능 향상 가능
- Layer Freeze 전략 미적용 -> 초반 레이어 동결 시 파인튜닝 효율화 가능
- Epoch 제한 -> 더 많은 학습 횟수로 추가 성능 향상 여지 있음

---

## 배운 점

- 외부 가중치 없이 Transfer Learning 파이프라인을 처음부터 설계하는 경험
- Mixed Precision Training으로 제한된 GPU 환경에서 효율적으로 학습하는 방법
- 학습률 스케줄링과 체크포인트 전략으로 제한된 epoch 내 최적 수렴점 탐색
