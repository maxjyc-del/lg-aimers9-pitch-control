"""학습된 블렌딩 모델로 test.csv의 제구 성공 확률을 예측해 제출 파일을 만든다.

사용법:
    python src/inference.py --data-dir data --model-path model/model.pkl --out output/submission.csv
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

# 모델이 __main__ 네임스페이스에서 pickle된 경우에도 로드되도록 클래스와 함수를 이 모듈에 둔다.
from blend import HGBXGBBlend  # noqa: F401
from features import ID_COL, TARGET_COL, add_features, build_features  # noqa: F401


def predict(model, test: pd.DataFrame) -> pd.DataFrame:
    """row_id별 control_success 확률 DataFrame을 반환한다."""
    proba = model.predict_proba(build_features(test))[:, 1]
    return pd.DataFrame({ID_COL: test[ID_COL].values, TARGET_COL: proba})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--model-path", type=Path, default=Path("model/model.pkl"))
    ap.add_argument("--out", type=Path, default=Path("output/submission.csv"))
    args = ap.parse_args()

    model = joblib.load(args.model_path)
    test = pd.read_csv(args.data_dir / "test.csv", encoding="utf-8-sig")
    print(f"test rows={len(test):,} | HGB x{len(model.hgb_pipelines)}, "
          f"XGB x{len(model.xgb_models)}, weight_xgb={model.weight_xgb}")

    submission = predict(model, test)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
