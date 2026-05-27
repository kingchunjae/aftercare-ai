"""src/preprocessing.py — 데이터 로드 및 전처리 유틸리티"""
import pandas as pd, numpy as np, os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "regions.csv")

TYPE_INFO = {
    "A": {"label":"위기+공급부족","color":"#C0392B","icon":"🔴","priority":1,"action":"긴급 개입"},
    "B": {"label":"위기+공급과잉","color":"#E67E22","icon":"🟠","priority":3,"action":"구조 전환"},
    "C": {"label":"비위기+공급부족","color":"#1B4D6B","icon":"🔵","priority":2,"action":"긴급 확충"},
    "D": {"label":"비위기+균형",  "color":"#27AE60","icon":"🟢","priority":4,"action":"모니터링"},
}

@st_cache if False else lambda f: f  # 캐시 래퍼 (app.py에서 @st.cache_data 사용)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["top3_list"] = df["top3_features"].str.split("|")
    return df

def get_summary_stats(df: pd.DataFrame) -> dict:
    return {
        "total": len(df),
        "A": int((df["region_type"]=="A").sum()),
        "B": int((df["region_type"]=="B").sum()),
        "C": int((df["region_type"]=="C").sum()),
        "D": int((df["region_type"]=="D").sum()),
        "total_waitlist":   int(df["care_waitlist"].sum()),
        "avg_util_rate":    round(df["care_util_rate"].mean(), 1),
        "avg_risk_score":   round(df["risk_score"].mean(), 1),
        "high_risk_count":  int((df["risk_score"] >= 60).sum()),
    }

def get_region_detail(df: pd.DataFrame, region_id: str) -> dict:
    row = df[df["region_id"] == region_id].iloc[0]
    t = row["region_type"]
    return {
        "name":         row["name"],
        "type":         t,
        "type_info":    TYPE_INFO[t],
        "students":     int(row["students"]),
        "dual_pct":     float(row["dual_income_pct"]),
        "single_pct":   float(row["single_parent_pct"]),
        "waitlist":     int(row["care_waitlist"]),
        "util_rate":    float(row["care_util_rate"]),
        "demand_idx":   float(row["demand_idx"]),
        "supply_idx":   float(row["supply_idx"]),
        "imbal_idx":    float(row["imbal_idx"]),
        "demand_5y":    float(row["demand_idx_5y"]),
        "birth_chg":    float(row["birth_change_pct"]),
        "risk_score":   int(row["risk_score"]),
        "top3":         row["top3_list"],
        "decline":      bool(row["decline"]),
        "urban":        bool(row["urban"]),
        "note":         str(row.get("region_note", "")),
        "data_note":    str(row.get("data_note", "")),
        "birth_rate":   float(row.get("birth_rate", 0)),
    }

def simulate_budget(df: pd.DataFrame, budget_m: int, threshold: float = 0.2) -> pd.DataFrame:
    """예산(억원) → 유형 우선순위별 배분 시뮬레이션"""
    priority_map = {"A": 0.45, "B": 0.05, "C": 0.40, "D": 0.10}
    result = df[["region_id","name","region_type","type_label","risk_score","care_waitlist"]].copy()
    total = budget_m * 1e8
    result["allocated_원"] = result["region_type"].map(priority_map) * total / \
        result.groupby("region_type")["region_type"].transform("count")
    result["allocated_억"] = (result["allocated_원"] / 1e8).round(2)

    # 배분 후 예상 불균형 지수 개선 (단순 추정)
    imbal = df["imbal_idx"].copy()
    improvement = result["allocated_억"] / (df["students"] * 0.5) * 0.3
    result["imbal_after"] = (df["imbal_idx"] - improvement).clip(lower=0.5).round(4)
    result["imbal_before"] = df["imbal_idx"].round(4)
    return result.sort_values("risk_score", ascending=False)
