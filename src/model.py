"""src/model.py — Random Forest 학습 · 수요 예측 · 변수 중요도"""
import numpy as np, pandas as pd, joblib, os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
REG_PATH   = os.path.join(MODEL_DIR, "rf_regressor.pkl")
CLF_PATH   = os.path.join(MODEL_DIR, "rf_classifier.pkl")
SCALER_PATH= os.path.join(MODEL_DIR, "scaler.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "dual_income_pct",    # 맞벌이 가구 비율
    "single_parent_pct",  # 한부모 가구 비율
    "students",           # 초등학생 수
    "care_util_rate",     # 돌봄 이용률
    "birth_change_pct",   # 출생아수 변화율
    "care_waitlist",      # 대기자 수
]
FEATURE_LABELS = [
    "맞벌이 가구 비율", "한부모 가구 비율",
    "초등학생 수", "돌봄 이용률",
    "출생아 변화율", "돌봄 대기자 수",
]
TARGET_REG = "demand_idx_5y"   # 5년 후 수요 지수 (회귀)
TARGET_CLF = "region_type"     # 유형 분류

def _augment(df: pd.DataFrame, n_aug: int = 8) -> pd.DataFrame:
    """학습 데이터 부족 보완 — 노이즈 증강"""
    rng = np.random.default_rng(0)
    rows = [df]
    for _ in range(n_aug):
        noise = df[FEATURES].copy()
        for c in FEATURES:
            std = noise[c].std() * 0.08
            noise[c] = noise[c] + rng.normal(0, std, len(noise))
        aug = df.copy()
        aug[FEATURES] = noise
        aug[TARGET_REG] = df[TARGET_REG] * rng.uniform(0.93, 1.07, len(df))
        rows.append(aug)
    return pd.concat(rows, ignore_index=True)

def train(df: pd.DataFrame) -> dict:
    df_aug = _augment(df)

    X = df_aug[FEATURES].values
    y_reg = df_aug[TARGET_REG].values
    y_clf = df_aug[TARGET_CLF].values

    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X)

    X_tr, X_te, yr_tr, yr_te = train_test_split(X_sc, y_reg, test_size=0.2, random_state=42)
    _, _,  yc_tr, yc_te      = train_test_split(X_sc, y_clf, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    reg.fit(X_tr, yr_tr)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_tr, yc_tr)

    yr_pred = reg.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(yr_te, yr_pred)))
    mae  = float(mean_absolute_error(yr_te, yr_pred))
    r2   = float(r2_score(yr_te, yr_pred))
    clf_acc = float((clf.predict(X_te) == yc_te).mean())

    joblib.dump(reg,    REG_PATH)
    joblib.dump(clf,    CLF_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return {
        "rmse": round(rmse, 4), "mae": round(mae, 4),
        "r2":   round(r2,   4), "clf_acc": round(clf_acc, 4),
        "feature_importance": dict(zip(FEATURE_LABELS, reg.feature_importances_.round(4))),
    }

def load_models():
    reg    = joblib.load(REG_PATH)
    clf    = joblib.load(CLF_PATH)
    scaler = joblib.load(SCALER_PATH)
    return reg, clf, scaler

def predict_region(row: pd.Series, reg, scaler) -> dict:
    x = np.array([[row[f] for f in FEATURES]])
    x_sc = scaler.transform(x)
    demand_5y = float(reg.predict(x_sc)[0])
    current   = float(row["demand_idx"])
    change_pct = round((demand_5y - current) / max(current, 0.001) * 100, 1)
    return {
        "demand_5y":    round(demand_5y, 4),
        "change_pct":   change_pct,
        "trend":        "증가" if change_pct > 5 else ("감소" if change_pct < -5 else "유지"),
    }

def get_feature_importance(reg) -> dict:
    return dict(zip(FEATURE_LABELS, reg.feature_importances_.round(4)))

# ── 모델 자동 학습 (models/ 파일 없을 때)
def ensure_trained(df: pd.DataFrame):
    if not os.path.exists(REG_PATH):
        return train(df)
    return None

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "regions.csv"), encoding="utf-8-sig")
    metrics = train(df)
    print("학습 완료:", metrics)
