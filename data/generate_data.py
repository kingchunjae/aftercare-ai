"""
data/generate_data.py — 광주광역시 + 전라남도 27개 시군구
───────────────────────────────────────────────────────────
[실제 공공데이터 기반]
  - 지역명·좌표: 실제 행정구역
  - 인구감소지역 지정: 행정안전부 고시 제2024-15호 (2024.2.27. 시행)
      전라남도 인구감소지역 16개 군 공식 반영
  - 통계 범위: 교육부 초등돌봄 현황(2023), 통계청 맞벌이가구(2023) 기준 보정
  - 개별 수치: 시뮬레이션 (실제 API 연동 전 추정치)

유형 분류 (A=8, B=8, C=4, D=7 / 총 27개)
  A: 인구감소+공급부족 (긴급개입)  — 섬·오지 군 지역
  B: 인구감소+공급과잉 (구조전환)  — 내륙 농촌 군 지역
  C: 비감소+공급부족  (긴급확충)  — 광주 도심·순천 등 도시
  D: 비감소+균형     (모니터링)  — 중소도시·신도시
"""

import numpy as np
import pandas as pd
import os

# ── 유형 정보
TYPE_COLORS = {"A": "#C0392B", "B": "#E67E22", "C": "#1B4D6B", "D": "#27AE60"}
TYPE_LABELS = {"A": "위기+공급부족", "B": "위기+공급과잉", "C": "비위기+공급부족", "D": "비위기+균형"}
TOP3_F = {
    "A": ["돌봄 대기자 증가율", "맞벌이 가구 비율", "학생수 감소율"],
    "B": ["돌봄 이용률 저하", "학생수 급감", "통폐합 예정 학교 수"],
    "C": ["맞벌이 가구 밀도", "신도심 학생 집중", "방과후 정원 대비 대기 비율"],
    "D": ["출생아수 소폭 감소", "이용률 안정", "지역 인구 유지"],
}

# ── 실제 시군구 목록
# decline=True: 행정안전부 고시 제2024-15호 인구감소지역 지정
# 전남 16개 군 전체가 인구감소지역 (담양·곡성·구례·고흥·보성·화순·장흥·강진·
#                                    해남·영암·함평·영광·장성·완도·진도·신안)
REGIONS = [
    # ── 광주광역시 (비감소, 도시) ──────────────────────────────
    {"id": "GJ01", "name": "광주 동구",  "lat": 35.1454, "lon": 126.9228, "t": "C",
     "note": "구도심·역세권 밀집, 돌봄 수요 집중"},
    {"id": "GJ02", "name": "광주 서구",  "lat": 35.1496, "lon": 126.8863, "t": "D",
     "note": "상무지구·도심 직장인 밀집, 비교적 균형"},
    {"id": "GJ03", "name": "광주 남구",  "lat": 35.1328, "lon": 126.9022, "t": "C",
     "note": "봉선동 주거밀집, 맞벌이 비율 높아 공급 부족"},
    {"id": "GJ04", "name": "광주 북구",  "lat": 35.1746, "lon": 126.9126, "t": "D",
     "note": "용봉동·일곡지구, 광주 최대 인구 구"},
    {"id": "GJ05", "name": "광주 광산구", "lat": 35.1394, "lon": 126.7936, "t": "C",
     "note": "첨단·수완지구 신도심, 젊은 맞벌이 급증"},

    # ── 전라남도 시 (비감소) ───────────────────────────────────
    {"id": "JN01", "name": "목포시",  "lat": 34.8118, "lon": 126.3922, "t": "D",
     "note": "서남권 중심도시, 돌봄 공급 안정"},
    {"id": "JN02", "name": "여수시",  "lat": 34.7604, "lon": 127.6622, "t": "D",
     "note": "여수산단 배후, 공단 근로자 자녀 수요 안정"},
    {"id": "JN03", "name": "순천시",  "lat": 34.9509, "lon": 127.4872, "t": "C",
     "note": "전남 제2도시·혁신도시 연계, 돌봄 수요 급증"},
    {"id": "JN04", "name": "나주시",  "lat": 35.0160, "lon": 126.7108, "t": "D",
     "note": "빛가람혁신도시 입주 완료, 균형 유지"},
    {"id": "JN05", "name": "광양시",  "lat": 34.9409, "lon": 127.6957, "t": "D",
     "note": "포스코 배후도시, 젊은 직장인 정착으로 안정"},
    {"id": "JN06", "name": "무안군",  "lat": 34.9903, "lon": 126.4818, "t": "D",
     "note": "무안국제공항·전남도청, 비감소·균형"},

    # ── 전라남도 군 — 인구감소지역 B형 (공급과잉) ────────────────
    {"id": "JN07", "name": "담양군",  "lat": 35.3215, "lon": 126.9881, "t": "B",
     "note": "광주 근교·죽녹원 관광지, 학생 감소 심화"},
    {"id": "JN08", "name": "화순군",  "lat": 35.0640, "lon": 126.9862, "t": "B",
     "note": "광주 인접, 시설 여유 있으나 학생 급감"},
    {"id": "JN09", "name": "보성군",  "lat": 34.7717, "lon": 127.0800, "t": "B",
     "note": "녹차 산지, 시설 수 대비 이용 아동 부족"},
    {"id": "JN10", "name": "해남군",  "lat": 34.5732, "lon": 126.5990, "t": "B",
     "note": "한반도 최남단, 농업 중심 인구감소 가속"},
    {"id": "JN11", "name": "영암군",  "lat": 34.8003, "lon": 126.6966, "t": "B",
     "note": "대불산단 외곽, 젊은층 유출·시설 과잉"},
    {"id": "JN12", "name": "함평군",  "lat": 35.0655, "lon": 126.5180, "t": "B",
     "note": "나비축제 지역, 인구 감소·돌봄 이용률 급락"},
    {"id": "JN13", "name": "영광군",  "lat": 35.2772, "lon": 126.5119, "t": "B",
     "note": "원전 소재지, 고령화로 학생 급감"},
    {"id": "JN14", "name": "장성군",  "lat": 35.3014, "lon": 126.7847, "t": "B",
     "note": "광주 광역권 외곽, 통학 수요 유출"},

    # ── 전라남도 군 — 인구감소지역 A형 (공급부족) ────────────────
    {"id": "JN15", "name": "곡성군",  "lat": 35.2818, "lon": 127.2927, "t": "A",
     "note": "섬진강 상류 오지, 시설 전무에 가까운 공급 부족"},
    {"id": "JN16", "name": "구례군",  "lat": 35.2026, "lon": 127.4631, "t": "A",
     "note": "지리산 자락, 읍내 외 돌봄 공백 심각"},
    {"id": "JN17", "name": "고흥군",  "lat": 34.6082, "lon": 127.2768, "t": "A",
     "note": "고령화율 전국 최상위, 잔류 아동 돌봄 공백"},
    {"id": "JN18", "name": "장흥군",  "lat": 34.6816, "lon": 126.9072, "t": "A",
     "note": "군 전체 학생 감소, 남은 돌봄 수요 미충족"},
    {"id": "JN19", "name": "강진군",  "lat": 34.6408, "lon": 126.7696, "t": "A",
     "note": "다산 문화권, 읍 외 농촌 돌봄 공급 부재"},
    {"id": "JN20", "name": "완도군",  "lat": 34.3109, "lon": 126.7550, "t": "A",
     "note": "도서 지역, 해상 교통 의존으로 돌봄 접근성 최저"},
    {"id": "JN21", "name": "진도군",  "lat": 34.4870, "lon": 126.2630, "t": "A",
     "note": "진도대교 연결 섬, 읍 외 면부 돌봄 공급 거의 없음"},
    {"id": "JN22", "name": "신안군",  "lat": 34.8231, "lon": 126.1073, "t": "A",
     "note": "1004섬 군도, 섬별 분산으로 시설 접근 불가"},
]

# ── 유형별 통계 파라미터
# 교육부 초등돌봄 현황(2023) + 통계청 맞벌이가구(2023) 기준 범위 설정
SPECS = {
    # 인구감소 + 공급부족: 섬·오지 군
    "A": dict(
        urban=False, decline=True,
        dual=(42, 56),          # 맞벌이 비율 (%) — 농촌 평균 하단
        stu=(80, 480),          # 초등학생 수 (명)
        sup_ratio=(0.10, 0.28), # 돌봄정원/학생수 (공급 매우 부족)
        util=(0.85, 0.99),      # 이용률 (꽉 차있음)
        wait=(40, 220),         # 대기자 수
        bc=(0.48, 0.68),        # 5년 후 출생아 변화 비율
        sing=(16, 26),          # 한부모 가구 비율 (%)
    ),
    # 인구감소 + 공급과잉: 내륙 농촌 군
    "B": dict(
        urban=False, decline=True,
        dual=(42, 56),
        stu=(200, 900),
        sup_ratio=(0.60, 1.20), # 시설 많으나 학생 줄어 과잉
        util=(0.18, 0.42),      # 이용률 매우 낮음
        wait=(0, 20),
        bc=(0.48, 0.70),
        sing=(13, 22),
    ),
    # 비감소 + 공급부족: 광주 도심·순천
    "C": dict(
        urban=True, decline=False,
        dual=(62, 78),          # 도시 맞벌이 높음
        stu=(2000, 12000),      # 대도시 초등학생 수
        sup_ratio=(0.18, 0.35), # 수요 대비 공급 부족
        util=(0.90, 0.99),
        wait=(100, 420),
        bc=(0.82, 1.05),
        sing=(5, 10),
    ),
    # 비감소 + 균형: 중소도시·신도시
    "D": dict(
        urban=False, decline=False,
        dual=(52, 68),
        stu=(400, 6000),
        sup_ratio=(0.45, 0.88),
        util=(0.55, 0.80),
        wait=(5, 65),
        bc=(0.80, 1.04),
        sing=(7, 14),
    ),
}

# ── 데이터 생성
rows = []
for r in REGIONS:
    sp = SPECS[r["t"]]
    g = np.random.default_rng(hash(r["id"]) % 2**31)

    stu = int(g.integers(sp["stu"][0], sp["stu"][1]))
    dual = round(g.uniform(*sp["dual"]), 1)
    sing = round(g.uniform(*sp["sing"]), 1)
    area = round(g.uniform(15, 90), 1) if sp["urban"] else round(g.uniform(100, 800), 1)
    births = int(stu * g.uniform(0.45, 0.70))
    births_p = int(births * g.uniform(*sp["bc"]))
    bchg = round((births_p - births) / max(births, 1) * 100, 1)

    tot_cap = int(stu * g.uniform(*sp["sup_ratio"]))
    a_cap = int(tot_cap * g.uniform(0.45, 0.65))
    c_cap = max(tot_cap - a_cap, 1)
    c_enr = int(c_cap * g.uniform(*sp["util"]))
    wait = int(g.integers(*sp["wait"]))
    util = round(c_enr / c_cap * 100, 1)

    dem = round(dual / 100, 4)
    sup = round(tot_cap / max(stu, 1), 4)
    sup = max(sup, 0.01)
    imb = round(dem / sup, 4)
    dem5 = round(dem * (1 + bchg / 100 * 0.6), 4)
    dem5 = max(dem5, 0.01)
    rs = int(max(0, min(100,
        (imb - 1) * 20
        + (20 if sp["decline"] else 0)
        + sing * 0.6
        + max(0, -bchg) * 0.3
    )))

    rows.append({
        "region_id":        r["id"],
        "name":             r["name"],
        "lat":              r["lat"],
        "lon":              r["lon"],
        "urban":            sp["urban"],
        "decline":          sp["decline"],           # 행안부 인구감소지역 여부
        "students":         stu,
        "dual_income_pct":  dual,
        "single_parent_pct": sing,
        "area_km2":         area,
        "births_5y":        births,
        "births_proj_5y":   births_p,
        "birth_change_pct": bchg,
        "afterschool_cap":  a_cap,
        "care_cap":         c_cap,
        "care_enrolled":    c_enr,
        "care_waitlist":    wait,
        "care_util_rate":   util,
        "demand_idx":       dem,
        "supply_idx":       sup,
        "imbal_idx":        imb,
        "demand_idx_5y":    dem5,
        "region_type":      r["t"],
        "type_color":       TYPE_COLORS[r["t"]],
        "type_label":       TYPE_LABELS[r["t"]],
        "top3_features":    "|".join(TOP3_F[r["t"]]),
        "risk_score":       rs,
        "region_note":      r.get("note", ""),
    })

df = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(__file__), "regions.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"생성 완료: {len(df)}개 / 유형 분포: {df['region_type'].value_counts().to_dict()}")
print(df[["name", "region_type", "decline", "imbal_idx", "risk_score"]].to_string(index=False))
