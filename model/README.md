# Model

학습된 모델 파일(`model.pkl`)은 레포에 포함하지 않습니다.
`python src/train.py`로 다시 만들 수 있습니다.

- 모델 구성: `HGBXGBBlend` (HistGradientBoosting × 3 + XGBoost × 3, 가중 블렌딩)
- `src/inference.py`는 스크립트로 직접 실행해야 모델 클래스가 올바르게 로드됩니다.
