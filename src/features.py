"""파생 피처 생성."""
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

CAT_COLS = ["top_bottom", "game_type", "base_state"]
ENGINEERED_COLS = ["score_closeness", "recent_form_weighted", "platoon_same"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """도메인 기반 파생 피처 3개를 추가한다.

    - score_closeness      : 점수 차가 작을수록(접전일수록) 큰 값 (-|투수팀 기준 점수차|)
    - recent_form_weighted : 직전 1/3/5경기 제구 성공률의 가중 평균 (최근 경기에 높은 가중치)
    - platoon_same         : 투수·타자 손잡이가 같으면 1 (좌투-좌타, 우투-우타)
    """
    x = df.copy()
    x["score_closeness"] = -x["score_diff_pitcher_team"].abs()
    x["recent_form_weighted"] = (
        0.5 * x["asof_pitcher_prev1_game_success_rate"]
        + 0.3 * x["asof_pitcher_prev3_game_success_rate"]
        + 0.2 * x["asof_pitcher_prev5_game_success_rate"]
    )
    x["platoon_same"] = (x["pitcher_hand"] == x["batter_hand"]).astype(int)
    return x


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """ID·타깃 컬럼을 제외한 원본 피처를 반환한다. 파생 피처는 파이프라인 안에서 추가된다."""
    return df.drop(columns=[c for c in (ID_COL, TARGET_COL) if c in df.columns])
