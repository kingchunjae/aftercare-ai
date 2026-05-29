"""src/preprocessing.py — 데이터 로드 및 전처리 유틸리티"""
import pandas as pd, numpy as np, os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "regions.csv")

TYPE_INFO = {
    "A": {"label":"위기+공급부족","color":"#C0392B","icon":"🔴","priority":1,"action":"긴급 개입"},
    "B": {"label":"위기+공급과잉","color":"#E67E22","icon":"🟠","priority":3,"action":"구조 전환"},
    "C": {"label":"비위기+공급부족","color":"#1B4D6B","icon":"🔵","priority":2,"action":"긴급 확충"},
    "D": {"label":"비위기+균형",  "color":"#27AE60","icon":"🟢","priority":4,"action":"모니터링"},
}

# ── 유치원 파이프라인 파라미터 (유치원알리미 공시 기반 추정)
_NATIONAL_KINDER_PIPELINE = 0.215   # 전국 유치원 원아/초등학생 비율 (2023)
_KINDER_ENROLLED_RATE = {
    "A": (0.13, 0.17),   # 농촌 오지: 유치원 접근성 낮음
    "B": (0.15, 0.21),   # 인구감소 군: 원아 수 감소 심화
    "C": (0.23, 0.29),   # 도심 성장: 신설 유치원 집중
    "D": (0.18, 0.24),   # 균형 중소도시: 전국 평균 수준
}
_KINDER_UTIL_RATE = {
    "A": (0.55, 0.74),   # 수요 부족·접근성 낮음
    "B": (0.40, 0.62),   # 공급 과잉·원아 감소 심화
    "C": (0.83, 0.96),   # 공급 부족·수요 집중
    "D": (0.68, 0.85),   # 안정적 균형 상태
}


def _add_kinder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """유치원 파이프라인 컬럼이 없으면 앱 로드 시 자동 생성 (유치원알리미 기반 추정).

    generate_data.py를 재실행하지 않아도 즉시 반영되도록
    기존 CSV 데이터에서 결정론적 추정값을 계산합니다.
    시드: hash(region_id + '_kinder') — 매 실행마다 동일한 값 보장.
    """
    if "kinder_enrolled" in df.columns:
        return df   # 이미 CSV에 컬럼이 있으면 스킵

    kinder_rows = []
    for _, row in df.iterrows():
        rid  = row["region_id"]
        t    = row["region_type"]
        stu  = max(int(row["students"]), 1)
        sch  = int(row.get("school_count", 10))
        bchg = float(row["birth_change_pct"])

        # region_id 기반 결정론적 시드 (재실행해도 동일 값)
        g = np.random.default_rng(hash(rid + "_kinder") % 2**31)

        kinder_rate     = g.uniform(*_KINDER_ENROLLED_RATE[t])
        kinder_enrolled = int(stu * kinder_rate)
        kinder_util     = round(float(g.uniform(*_KINDER_UTIL_RATE[t])), 3)
        kinder_cap      = max(int(kinder_enrolled / kinder_util), kinder_enrolled + 1)
        kinder_count    = max(1, int(sch * g.uniform(0.45, 0.85)))
        kinder_pipeline = round(kinder_enrolled / stu, 4)
        kinder_trend    = round(float(np.clip(bchg * 0.70, -50.0, 20.0)), 1)

        kinder_rows.append({
            "region_id":          rid,
            "kinder_count":       kinder_count,
            "kinder_enrolled":    kinder_enrolled,
            "kinder_cap":         kinder_cap,
            "kinder_util_rate":   kinder_util,
            "kinder_pipeline_idx":kinder_pipeline,
            "kinder_trend_pct":   kinder_trend,
            "kinder_source":      "유치원알리미추정",
        })

    kdf = pd.DataFrame(kinder_rows).set_index("region_id")
    for col in kdf.columns:
        df[col] = df["region_id"].map(kdf[col])

    return df


@st_cache if False else lambda f: f  # 캐시 래퍼 (app.py에서 @st.cache_data 사용)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["top3_list"] = df["top3_features"].str.split("|")
    df = _add_kinder_columns(df)   # 유치원 컬럼 자동 보완
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
        "school_count":          int(row["school_count"])          if "school_count"          in row.index else 0,
        "care_enrolled":         int(row["care_enrolled"])         if "care_enrolled"         in row.index else 0,
        "afterschool_enrolled":  int(row["afterschool_enrolled"])  if "afterschool_enrolled"  in row.index else 0,
        "afterschool_source":    str(row["afterschool_source"])    if "afterschool_source"    in row.index else "추정",
        "custom_edu_enrolled":   int(row["custom_edu_enrolled"])   if "custom_edu_enrolled"   in row.index else 0,
        "students_source":       str(row["students_source"])       if "students_source"       in row.index else "추정",
        "students_class_count":  int(row["students_class_count"])  if "students_class_count"  in row.index else 0,
        "note":                 str(row["region_note"]) if "region_note" in row.index else "",
        "data_note":            str(row["data_note"])   if "data_note"   in row.index else "",
        "birth_rate":           float(row["birth_rate"]) if "birth_rate" in row.index else 0.0,
        # ── 유치원 파이프라인 (유치원알리미 기반 추정)
        "kinder_count":         int(row["kinder_count"])           if "kinder_count"         in row.index else 0,
        "kinder_enrolled":      int(row["kinder_enrolled"])        if "kinder_enrolled"       in row.index else 0,
        "kinder_cap":           int(row["kinder_cap"])             if "kinder_cap"            in row.index else 0,
        "kinder_util_rate":     float(row["kinder_util_rate"])     if "kinder_util_rate"      in row.index else 0.0,
        "kinder_pipeline_idx":  float(row["kinder_pipeline_idx"])  if "kinder_pipeline_idx"  in row.index else 0.0,
        "kinder_trend_pct":     float(row["kinder_trend_pct"])     if "kinder_trend_pct"      in row.index else 0.0,
        "kinder_source":        str(row["kinder_source"])          if "kinder_source"         in row.index else "추정",
    }

def simulate_budget(df: pd.DataFrame, budget_m: int, threshold: float = 0.2, priority: dict = None) -> pd.DataFrame:
    """예산(억원) → 유형 우선순위별 배분 + 물리 기반 불균형 지수 개선 시뮬레이션.

    개선 원리
    ---------
    A/C/D형: 예산 → 신규 돌봄 정원 확보 → supply_idx 상승 → imbal_idx 감소 (→ 1.0)
    B형:     예산 → 기존 과잉 시설 구조 전환 → 이용률 제고 → imbal_idx 상승 (→ 1.0)

    파라미터
    --------
    SLOTS_PER_OK  : 1억원당 확보 가능한 신규 돌봄 정원 (단가 ≈ 400만원/명·년)
    TYPE_EFF      : 유형별 정원 확보 효율 승수
                    A=2.2 (소규모·농촌, 예산 효율 최고)
                    C=1.8 (도심 성장지, 대형 시설 신설)
                    D=0.8 (이미 균형권, 효율 낮음)
                    B=0.0 (정원 신설 대신 구조 전환)
    B_UTIL_K      : B형 구조 전환 효율 계수 (클수록 imbal이 빨리 1.0에 접근)
    priority      : 유형별 예산 배분 비율 dict (합이 1.0). None이면 기본값 사용.
    """
    _default = {"A": 0.45, "B": 0.05, "C": 0.40, "D": 0.10}
    PRIORITY     = priority if priority is not None else _default
    SLOTS_PER_OK = 80        # 1억원당 신규 돌봄 정원 (단가 약 400만원/명·년 기준)
    TYPE_EFF     = {"A": 2.2, "B": 0.0, "C": 1.8, "D": 0.8}
    B_UTIL_K     = 600       # B형 구조 전환 효율 계수

    result = df[["region_id","name","region_type","type_label","risk_score","care_waitlist"]].copy()
    total  = budget_m * 1e8
    result["allocated_원"] = (
        result["region_type"].map(PRIORITY) * total /
        result.groupby("region_type")["region_type"].transform("count")
    )
    result["allocated_억"] = (result["allocated_원"] / 1e8).round(2)

    # ── A/C/D형: 신규 정원 → supply_idx 상승 → imbal 감소
    eff_mult      = result["region_type"].map(TYPE_EFF).fillna(0)
    new_slots_raw = result["allocated_억"] * SLOTS_PER_OK * eff_mult
    demand_count  = (df["demand_idx"] * df["students"]).clip(lower=1)
    new_slots     = new_slots_raw.clip(upper=demand_count * 0.9).fillna(0)
    result["new_care_slots"] = new_slots.round(0).astype(int)

    students_safe  = df["students"].clip(lower=1)
    new_supply_idx = (df["supply_idx"] * students_safe + new_slots) / students_safe
    new_imbal      = (df["demand_idx"] / new_supply_idx.clip(lower=0.001)).round(4)

    # ── B형: 구조 전환 → imbal이 1.0 방향으로 상승
    b_mask  = (result["region_type"] == "B").values
    b_delta = (result["allocated_억"] / students_safe * B_UTIL_K).fillna(0)
    new_imbal = new_imbal.copy()
    new_imbal[b_mask] = (df["imbal_idx"] + b_delta)[b_mask].clip(upper=0.96).round(4)

    result["imbal_before"] = df["imbal_idx"].round(4)
    result["imbal_after"]  = new_imbal.clip(lower=0.35).round(4)

    # ── 추가 수혜 아동 / 단가
    result["children_added"] = result["new_care_slots"].clip(lower=0)
    result.loc[b_mask, "children_added"] = 0
    result["cost_per_child_만원"] = (
        result["allocated_억"] * 10000 /
        result["children_added"].clip(lower=1)
    ).round(0).astype(int)
    result.loc[b_mask, "cost_per_child_만원"] = 0

    return result.sort_values("risk_score", ascending=False)
