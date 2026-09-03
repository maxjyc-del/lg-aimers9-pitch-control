#  투구 제구 성공 확률 예측 — LG Aimers 9기 해커톤

> 경기 상황과 투수·타자의 누적 기록으로 **한 투구의 제구 성공 확률(`control_success`)** 을 예측하는
> 이진 분류 모델. HistGradientBoosting과 XGBoost를 시드 배깅한 뒤 가중 블렌딩했습니다.

| 항목 | 내용 |
|---|---|
| 대회 | LG Aimers 9기 : 투구 제구 성공 확률 예측 AI 온라인 해커톤 |
| 기간 | 2026.08.05 ~ 2026.09.02 |
| 형태 | 개인 참가 — 대회 제공 제출 형식 위에 피처 엔지니어링, 모델링(HGB+XGB 블렌딩)을 단독 개발 |
| 결과 | **Public 751위 / 2,402명 (상위 31.3%)** · Brier Skill Score 849.12671 |
| 스택 | Python, pandas, scikit-learn, XGBoost |

---

## 1. 문제 정의

투구 직전의 상황 정보(이닝, 볼카운트, 주자, 점수차, 승리 기대값, Leverage Index 등)와
해당 시점까지 누적된 투수·타자 기록(`asof_*`)을 입력받아, 그 투구가 **제구에 성공할 확률**을 예측합니다.

## 2. 데이터

| 파일 | 규모 | 설명 |
|---|---|---|
| `train.csv` | 1,475,092행 × 49열 | 투구 단위 학습 데이터 + 타깃 |
| `test.csv` | 245,789행 (비공개) | 평가 서버에서만 제공 |
| `trackman_history.csv` | 1,793,078행 × 30열 | 2019~2024년 투구 트래킹 로그. 최종 모델에서는 미사용 |

- `asof_*` 피처는 해당 투구 **이전까지의** 누적 기록이라 데이터 누수 없이 쓸 수 있음
- 대회 규정상 대회 데이터(데이터 파일, 설명서, 제공 코드)는 레포에 포함하지 않았습니다. → [`data/README.md`](data/README.md)

## 3. 접근 방법

```
raw features (47)
   └─ add_features() ── 파생 피처 3개 추가
        └─ OrdinalEncoder(범주형 3개) + passthrough(수치형 47개)
             ├─ HistGradientBoosting × 3 seeds ─┐ 평균
             └─ XGBoost             × 3 seeds ─┘ 평균
                         └─ 0.65 · HGB + 0.35 · XGB → control_success 확률
```

### 3-1. 피처 엔지니어링 ([`src/features.py`](src/features.py))

| 피처 | 정의 | 의도 |
|---|---|---|
| `recent_form_weighted` | 0.5·직전1경기 + 0.3·직전3경기 + 0.2·직전5경기 성공률 | 최근 컨디션을 반영. 최근 경기에 더 큰 가중치 |
| `platoon_same` | 투수·타자 손잡이가 같으면 1 | 좌-좌, 우-우 매치업 효과 |
| `score_closeness` | 투수팀 기준 점수차 | 접전 상황의 심리적 압박 |

### 3-2. 모델 ([`src/blend.py`](src/blend.py), [`src/train.py`](src/train.py))

| | HistGradientBoosting | XGBoost |
|---|---|---|
| learning rate | 0.03 | 0.03 |
| max depth | 6 | 6 |
| 반복 수 | 300 (early stopping) | 500 (early stopping 50) |
| 리프 최소 크기 | min_samples_leaf 200 | min_child_weight 50 |
| L2 정규화 | 1.0 | 1.0 |
| 시드 | 42, 1, 7 | 42, 1, 7 |
| 블렌딩 가중치 | **0.65** | **0.35** |

- 두 부스팅 계열을 함께 쓴 이유: 분할 방식과 정규화가 달라 오차 패턴이 다름 → 블렌딩 효과
- **시드 배깅**으로 단일 모델의 분산을 줄임
- 리프 최소 샘플을 크게 잡아(200 / 50) 147만 행 데이터에서 과적합 억제

## 4. 결과

### 평가 지표: Brier Skill Score

```
Score = max(0, 100000 × (1 − Brier Score / r(1 − r)))
Brier Score = mean((p_i − y_i)²),   r = 평가 데이터의 평균 제구 성공률 (비공개)
```

- 예측 확률이 실제 정답(0/1)에 얼마나 가까운지를 보는 **확률 예측 지표**
- 기준선 `r(1 − r)`은 모든 투구에 평균 성공률 하나만 예측했을 때의 Brier Score
- 따라서 점수는 "평균값만 찍는 모델보다 오차를 몇 % 줄였는가"를 100,000배 한 값
- 분류 정확도가 아니라 **확률의 보정(calibration)** 이 중요 → 임계값 없이 블렌딩 확률을 그대로 제출

| 지표 | Public Score | 순위 |
|---|---|---|
| Brier Skill Score | 849.12671 | 751 / 2,402 (상위 31.3%) |

- 849.12671 = 기준선 대비 Brier Score **약 0.85% 감소**
- 개선 폭이 작은 것은 제구 성공 여부가 경기 상황·누적 기록만으로는 설명되지 않는 변동이 큰 과제이기 때문으로 보임
- 피처 기여도 측면에서는 투수의 누적 제구 성공률이 가장 중요했고, 직접 만든 `platoon_same`, `recent_form_weighted`도 상위권에 들어 파생 피처의 효과를 확인함

## 5. 회고 / 개선 방향

- **시즌 분포 변화**: 시즌에 따라 제구 성공률 분포가 달라, 이후 시즌을 예측할 때 `season` 피처는 외삽 위험이 있음. 시즌 단위 홀드아웃 검증이나 최근 시즌 가중치를 더 실험해 볼 수 있음
- **Trackman 데이터 활용 방식**: 구속·회전수·무브먼트 등 트래킹 피처를 붙여 보았지만 시도할수록 점수가 낮아져 최종 모델에서는 제외함. 다음에는 아래 방향으로 다시 활용해 보고 싶음
  - 투구 단위 값을 그대로 쓰기보다, 해당 시점 **이전 경기까지의 투수별 누적 통계(as-of 집계)** 로 만들어 기존 `asof_*` 피처와 같은 방식으로 결합
  - Trackman ID와 대회 데이터 ID의 매칭 품질, 시즌별 결측 비율을 먼저 점검해 노이즈가 섞이는지 확인
  - 평균값보다 **릴리스 포인트·구속의 경기 내 변동성**처럼 제구와 직접 관련된 지표를 우선 설계
- **블렌딩 방식**: 고정 가중치(0.35) 블렌딩 대신 stacking 같은 메타 모델로 확장해 볼 수 있음

## 6. 실행 방법

```bash
pip install -r requirements.txt

# 1) data/ 에 대회 데이터 배치 (대회 참가자만 이용 가능, data/README.md 참고)
# 2) 학습
python src/train.py --data-dir data --model-path model/model.pkl
# 3) 추론
python src/inference.py --data-dir data --model-path model/model.pkl --out output/submission.csv
```

## 7. 레포 구조

```
├── src/
│   ├── features.py     # 파생 피처
│   ├── blend.py        # HGB + XGB 블렌딩 모델 클래스
│   ├── train.py        # 학습
│   └── inference.py    # 추론 → submission.csv
├── data/README.md      # 데이터 배치 안내 (대회 데이터 미포함)
└── model/README.md     # 모델 파일 안내 (가중치 미포함)
```
