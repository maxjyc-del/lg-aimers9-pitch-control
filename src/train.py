"""HGB + XGBoost 시드 배깅 블렌딩 모델 학습.

※ 이 스크립트는 제출한 모델에 저장된 구성과 하이퍼파라미터를 그대로 복원해 작성했다.
   - HGB  : lr 0.03, max_depth 6, max_iter 300, min_samples_leaf 200, l2 1.0, early_stopping
   - XGB  : lr 0.03, max_depth 6, n_estimators 500, min_child_weight 50, reg_lambda 1.0,
            tree_method hist, early_stopping_rounds 50
   - 시드 : 42, 1, 7 (모델별 3개씩) / 블렌딩 가중치 XGB 0.35
   검증 분할(XGB early stopping용)은 시즌 기준 홀드아웃(기본: 마지막 시즌)으로 둔다.

사용법:
    python src/train.py --data-dir data --model-path model/model.pkl
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from xgboost import XGBClassifier

from blend import HGBXGBBlend
from features import CAT_COLS, ENGINEERED_COLS, TARGET_COL, add_features, build_features

SEEDS = [42, 1, 7]
WEIGHT_XGB = 0.35


def make_preprocessor(feature_cols):
    num_cols = [c for c in feature_cols if c not in CAT_COLS] + ENGINEERED_COLS
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num_cols),
    ])


def make_hgb_pipeline(feature_cols, seed):
    return Pipeline([
        ("feat_eng", FunctionTransformer(add_features)),
        ("pre", make_preprocessor(feature_cols)),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.03, max_depth=6, max_iter=300, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=True, random_state=seed)),
    ])


def make_xgb(seed):
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        learning_rate=0.03, max_depth=6, n_estimators=500, min_child_weight=50,
        reg_lambda=1.0, early_stopping_rounds=50, random_state=seed)


def brier_skill_score(y, p):
    """대회 평가 산식: max(0, 100000 × (1 − Brier / r(1 − r)))"""
    r = np.mean(y)
    return max(0.0, 100000 * (1 - brier_score_loss(y, p) / (r * (1 - r))))


def report(name, y, p):
    print(f"  [{name}] BSS={brier_skill_score(y, p):.2f}  AUC={roc_auc_score(y, p):.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--model-path", default="./model/model.pkl")
    ap.add_argument("--valid-season", type=int, default=None,
                    help="검증용 홀드아웃 시즌 (기본: 가장 최근 시즌)")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"), encoding="utf-8-sig")
    valid_season = args.valid_season or int(df["season"].max())
    tr, va = df[df["season"] != valid_season], df[df["season"] == valid_season]
    print(f"train={len(tr):,}  valid={len(va):,} (season {valid_season})")

    X_tr, y_tr = build_features(tr), tr[TARGET_COL].values
    X_va, y_va = build_features(va), va[TARGET_COL].values
    feature_cols = list(X_tr.columns)

    hgb_pipelines, xgb_models, xgb_pres = [], [], []
    for seed in SEEDS:
        print(f"seed {seed}")
        hgb = make_hgb_pipeline(feature_cols, seed).fit(X_tr, y_tr)
        report("HGB", y_va, hgb.predict_proba(X_va)[:, 1])
        hgb_pipelines.append(hgb)

        pre = make_preprocessor(feature_cols).fit(add_features(X_tr))
        xgb = make_xgb(seed)
        xgb.fit(pre.transform(add_features(X_tr)), y_tr,
                eval_set=[(pre.transform(add_features(X_va)), y_va)], verbose=False)
        report("XGB", y_va, xgb.predict_proba(pre.transform(add_features(X_va)))[:, 1])
        xgb_models.append(xgb)
        xgb_pres.append(pre)

    model = HGBXGBBlend(hgb_pipelines, xgb_models, xgb_pres, weight_xgb=WEIGHT_XGB)
    report("Blend", y_va, model.predict_proba(X_va)[:, 1])

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    joblib.dump(model, args.model_path)
    print(f"Saved: {args.model_path}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
