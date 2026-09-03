"""HistGradientBoosting × XGBoost 시드 배깅 블렌딩 모델."""
import numpy as np

from features import add_features


class HGBXGBBlend:
    """HGB(시드 배깅) + XGBoost(시드 배깅)를 가중평균한다.

    최종 확률 = (1 - weight_xgb) * mean(HGB) + weight_xgb * mean(XGB)
    """

    def __init__(self, hgb_pipelines, xgb_models, xgb_preprocessors, weight_xgb=0.35):
        self.hgb_pipelines = hgb_pipelines
        self.xgb_models = xgb_models
        self.xgb_preprocessors = xgb_preprocessors
        self.weight_xgb = weight_xgb

    def fit(self, X, y):
        # 구성 모델은 train.py에서 개별 학습 후 주입한다.
        return self

    def predict_proba(self, df):
        hgb_pred = np.mean([p.predict_proba(df)[:, 1] for p in self.hgb_pipelines], axis=0)

        x_feat = add_features(df)
        xgb_pred = np.mean(
            [m.predict_proba(pre.transform(x_feat))[:, 1]
             for m, pre in zip(self.xgb_models, self.xgb_preprocessors)],
            axis=0,
        )

        blend = (1 - self.weight_xgb) * hgb_pred + self.weight_xgb * xgb_pred
        return np.column_stack([1 - blend, blend])
