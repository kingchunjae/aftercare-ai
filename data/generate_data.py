"""
data/generate_data.py — 광주광역시 + 전라남도 27개 시군구
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[실제 공공데이터 반영 현황]
  지역·좌표     : 실제 행정구역
  인구감소지역   : 행정안전부 고시 제2024-15호 (전남 16개 군)
  초등학생 수   : 경향신문(2024.02) 실측 + 시사저널(2025) 입학생×6 환산
                  광주 5구는 교육청 전체(72,944명) 구별 비례 추정
  맞벌이 비율   : 통계청 지역별고용조사 2023년 하반기
  합계출산율    : 통계청 2023 실측 (영광 1.65, 강진 1.47, 해남 1.35 등)
  돌봄 이용인원  : ★교육부_온종일돌봄(초등돌봄) 시설 현황('23.4월) 실측★
                  학교 단위 → 시군구 집계 (학교 수도 실측)
  돌봄 정원     : 교육부 공표 유형별 이용률 범위 역산 추정
                  (공공데이터 미공개)
  돌봄 대기자   : 시뮬레이션 (정부 시군구 단위 미공개)
  방과후 참여인원: NEIS Open API 실측 (fetch_neis_afterschool.py 실행 시)
                  미수집 시 전국 참여율 52.9% 기반 유형별 추정
"""

import os
import json
import numpy as np
import pandas as pd

# ── 경로
DATA_DIR        = os.path.dirname(__file__)
CARE_FILE       = os.path.join(DATA_DIR, "2023년+초등돌봄교실+현황('23.4월+기준)_최종.xlsx")
OUTPUT_CSV      = os.path.join(DATA_DIR, "regions.csv")
NEIS_CACHE_FILE = os.path.join(DATA_DIR, "neis_afterschool_cache.json")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 0. NEIS 방과후학교 캐시 로드 (선택적)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEIS_AFTERSCHOOL = {}   # {region_id: {"enrolled": int, "source": str, "year": str}}
if os.path.exists(NEIS_CACHE_FILE):
    try:
        with open(NEIS_CACHE_FILE, "r", encoding="utf-8") as _f:
            _cache = json.load(_f)
        NEIS_AFTERSCHOOL = _cache.get("regions", {})
        _covered = len(NEIS_AFTERSCHOOL)
        _year    = _cache.get("year", "")
        print(f"NEIS 캐시 로드: {_covered}개 지역 실측 데이터 반영 ({_year}년도)")
    except Exception as _e:
        print(f"[WARN] NEIS 캐시 로드 실패: {_e} → 추정값 사용")
else:
    print("NEIS 캐시 없음 → 방과후학교 참여인원 추정값 사용")

# ── 유형 메타
TYPE_COLORS = {"A": "#C0392B", "B": "#E67E22", "C": "#1B4D6B", "D": "#27AE60"}
TYPE_LABELS = {
    "A": "위기+공급부족", "B": "위기+공급과잉",
    "C": "비위기+공급부족", "D": "비위기+균형",
}
TOP3_F = {
    "A": ["돌봄 대기자 증가율", "맞벌이 가구 비율", "학생수 감소율"],
    "B": ["돌봄 이용률 저하", "학생수 급감", "통폐합 예정 학교 수"],
    "C": ["맞벌이 가구 밀도", "신도심 학생 집중", "방과후 정원 대비 대기 비율"],
    "D": ["출생아수 소폭 감소", "이용률 안정", "지역 인구 유지"],
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1. 실제 돌봄 이용인원 로드 (교육부 2023년 4월 기준)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
raw = pd.read_excel(CARE_FILE)
raw.columns = ["연번", "시도", "시군구", "학교명", "이용인원"]

# 광주 5개 구 (시도 '광주' 포함)
gj_raw = raw[raw["시도"].str.contains("광주", na=False)].copy()
gj_agg = (
    gj_raw.groupby("시군구")["이용인원"]
    .agg(care_enrolled="sum", school_count="count")
    .reset_index()
)

# 전남 22개 시군 (시군구 키워드 매칭)
jn_keywords = [
    "목포", "여수", "순천", "나주", "광양",
    "담양", "곡성", "구례", "고흥", "보성",
    "화순", "장흥", "강진", "해남", "영암",
    "무안", "함평", "영광", "장성", "완도",
    "진도", "신안",
]
jn_mask = raw["시군구"].str.contains("|".join(jn_keywords), na=False)
jn_agg = (
    raw[jn_mask]
    .groupby("시군구")["이용인원"]
    .agg(care_enrolled="sum", school_count="count")
    .reset_index()
)

# region_id → (care_enrolled, school_count) 매핑
# 광주: 시군구명 그대로 (광산구 → GJ05 등)
GJ_MAP = {"광산구": "GJ05", "남구": "GJ03", "동구": "GJ01", "북구": "GJ04", "서구": "GJ02"}
# 전남: 시군구명 포함 키워드로 매핑
JN_MAP = {
    "목포": "JN01", "여수": "JN02", "순천": "JN03", "나주": "JN04", "광양": "JN05",
    "무안": "JN06", "담양": "JN07", "화순": "JN08", "보성": "JN09", "해남": "JN10",
    "영암": "JN11", "함평": "JN12", "영광": "JN13", "장성": "JN14", "곡성": "JN15",
    "구례": "JN16", "고흥": "JN17", "장흥": "JN18", "강진": "JN19", "완도": "JN20",
    "진도": "JN21", "신안": "JN22",
}

CARE_REAL = {}  # region_id → {"care_enrolled": int, "school_count": int}

for _, row in gj_agg.iterrows():
    rid = GJ_MAP.get(row["시군구"].strip())
    if rid:
        CARE_REAL[rid] = {"care_enrolled": int(row["care_enrolled"]),
                          "school_count":  int(row["school_count"])}

for _, row in jn_agg.iterrows():
    name = row["시군구"].strip()
    rid = next((v for k, v in JN_MAP.items() if k in name), None)
    if rid:
        CARE_REAL[rid] = {"care_enrolled": int(row["care_enrolled"]),
                          "school_count":  int(row["school_count"])}

print(f"실측 이용인원 로드 완료: {len(CARE_REAL)}개 지역")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2. 지역 정의 (실제 공공데이터 기반)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGIONS = [
    # ── 광주광역시 (비감소, 도시) ─────────────────────
    # 학생수: 광주교육청 전체(72,944명) 구별 비례 추정
    # 맞벌이: 통계청 지역별고용조사 2023 하반기 (광주 평균 48.9%)
    {"id":"GJ01","name":"광주 동구", "lat":35.1454,"lon":126.9228,"t":"C",
     "students":5_100,"dual_pct":46.2,"sing_pct":8.1,"birth_rate":0.81,"decline":False,
     "region_note":"구도심·역세권 밀집, 돌봄 수요 집중",
     "data_note":"학생수: 광주교육청 비례추정 / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"GJ02","name":"광주 서구", "lat":35.1496,"lon":126.8863,"t":"D",
     "students":12_400,"dual_pct":49.5,"sing_pct":7.8,"birth_rate":0.87,"decline":False,
     "region_note":"상무지구·도심 직장인 밀집, 비교적 균형",
     "data_note":"학생수: 광주교육청 비례추정 / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"GJ03","name":"광주 남구", "lat":35.1328,"lon":126.9022,"t":"C",
     "students":9_500,"dual_pct":48.3,"sing_pct":7.2,"birth_rate":0.85,"decline":False,
     "region_note":"봉선동 주거밀집, 맞벌이 비율 높아 공급 부족",
     "data_note":"학생수: 광주교육청 비례추정 / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"GJ04","name":"광주 북구", "lat":35.1746,"lon":126.9126,"t":"D",
     "students":19_700,"dual_pct":50.1,"sing_pct":8.4,"birth_rate":0.88,"decline":False,
     "region_note":"용봉동·일곡지구, 광주 최대 인구 구",
     "data_note":"학생수: 광주교육청 비례추정 / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"GJ05","name":"광주 광산구","lat":35.1394,"lon":126.7936,"t":"C",
     "students":26_244,"dual_pct":52.7,"sing_pct":6.1,"birth_rate":0.96,"decline":False,
     "region_note":"첨단·수완지구 신도심, 젊은 맞벌이 급증",
     "data_note":"학생수: 광주교육청 비례추정 / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},

    # ── 전라남도 시 (비감소) ──────────────────────────
    # 학생수: 시사저널(2025) 입학생×6 환산
    # 맞벌이: 통계청 지역별고용조사 2023 하반기 (전남 평균 57.9%)
    {"id":"JN01","name":"목포시","lat":34.8118,"lon":126.3922,"t":"D",
     "students":8_376,"dual_pct":54.8,"sing_pct":9.2,"birth_rate":0.78,"decline":False,
     "region_note":"서남권 중심도시, 돌봄 공급 안정",
     "data_note":"학생수: 시사저널(2025) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN02","name":"여수시","lat":34.7604,"lon":127.6622,"t":"D",
     "students":9_786,"dual_pct":58.3,"sing_pct":8.7,"birth_rate":0.83,"decline":False,
     "region_note":"여수산단 배후, 공단 근로자 자녀 수요 안정",
     "data_note":"학생수: 시사저널(2025) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN03","name":"순천시","lat":34.9509,"lon":127.4872,"t":"C",
     "students":11_598,"dual_pct":57.1,"sing_pct":7.9,"birth_rate":0.92,"decline":False,
     "region_note":"전남 제2도시·혁신도시 연계, 돌봄 수요 급증",
     "data_note":"학생수: 시사저널(2025) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN04","name":"나주시","lat":35.0160,"lon":126.7108,"t":"D",
     "students":5_382,"dual_pct":56.4,"sing_pct":9.8,"birth_rate":0.91,"decline":False,
     "region_note":"빛가람혁신도시 입주 완료, 균형 유지",
     "data_note":"학생수: 시사저널(2025) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN05","name":"광양시","lat":34.9409,"lon":127.6957,"t":"D",
     "students":6_438,"dual_pct":60.2,"sing_pct":7.3,"birth_rate":1.06,"decline":False,
     "region_note":"포스코 배후도시, 젊은 직장인 정착으로 안정",
     "data_note":"학생수: 시사저널(2025) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN06","name":"무안군","lat":34.9903,"lon":126.4818,"t":"D",
     "students":4_110,"dual_pct":57.6,"sing_pct":10.4,"birth_rate":0.89,"decline":False,
     "region_note":"무안국제공항·전남도청, 비감소·균형",
     "data_note":"학생수: 시사저널(2025) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},

    # ── 전라남도 군 — 인구감소지역 B형 (공급과잉) ─────
    {"id":"JN07","name":"담양군","lat":35.3215,"lon":126.9881,"t":"B",
     "students":1_291,"dual_pct":60.1,"sing_pct":14.2,"birth_rate":0.98,"decline":True,
     "region_note":"광주 근교·죽녹원 관광지, 학생 감소 심화",
     "data_note":"학생수: 경향신문(2024) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN08","name":"화순군","lat":35.0640,"lon":126.9862,"t":"B",
     "students":1_700,"dual_pct":58.8,"sing_pct":13.6,"birth_rate":0.89,"decline":True,
     "region_note":"광주 인접, 시설 여유 있으나 학생 급감",
     "data_note":"학생수: 학교수 비례추정 / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN09","name":"보성군","lat":34.7717,"lon":127.0800,"t":"B",
     "students":978,"dual_pct":62.3,"sing_pct":16.1,"birth_rate":0.94,"decline":True,
     "region_note":"녹차 산지, 시설 수 대비 이용 아동 부족",
     "data_note":"학생수: 경향신문(2024) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN10","name":"해남군","lat":34.5732,"lon":126.5990,"t":"B",
     "students":2_000,"dual_pct":63.8,"sing_pct":15.3,"birth_rate":1.35,"decline":True,
     "region_note":"한반도 최남단, 출산율 전국 6위지만 인구 유출 지속",
     "data_note":"학생수: 학교수 비례추정 / 출산율: 통계청 2023 실측(1.35명) / 이용인원: 교육부 2023"},
    {"id":"JN11","name":"영암군","lat":34.8003,"lon":126.6966,"t":"B",
     "students":1_850,"dual_pct":59.4,"sing_pct":14.8,"birth_rate":0.96,"decline":True,
     "region_note":"대불산단 외곽, 젊은층 유출·시설 과잉",
     "data_note":"학생수: 학교수 비례추정 / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN12","name":"함평군","lat":35.0655,"lon":126.5180,"t":"B",
     "students":722,"dual_pct":61.2,"sing_pct":17.3,"birth_rate":0.88,"decline":True,
     "region_note":"나비축제 지역, 인구 감소·돌봄 이용률 급락",
     "data_note":"학생수: 경향신문(2024) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN13","name":"영광군","lat":35.2772,"lon":126.5119,"t":"B",
     "students":1_801,"dual_pct":64.7,"sing_pct":13.9,"birth_rate":1.65,"decline":True,
     "region_note":"원전 소재지, 출산율 전국 1위지만 고령화로 학생 급감",
     "data_note":"학생수: 경향신문(2024) / 출산율: 통계청 2023 전국 1위(1.65명) / 이용인원: 교육부 2023"},
    {"id":"JN14","name":"장성군","lat":35.3014,"lon":126.7847,"t":"B",
     "students":1_524,"dual_pct":59.7,"sing_pct":15.6,"birth_rate":0.92,"decline":True,
     "region_note":"광주 광역권 외곽, 통학 수요 유출",
     "data_note":"학생수: 경향신문(2024) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},

    # ── 전라남도 군 — 인구감소지역 A형 (공급부족) ─────
    {"id":"JN15","name":"곡성군","lat":35.2818,"lon":127.2927,"t":"A",
     "students":704,"dual_pct":61.5,"sing_pct":19.8,"birth_rate":0.85,"decline":True,
     "region_note":"섬진강 상류 오지, 시설 전무에 가까운 공급 부족",
     "data_note":"학생수: 경향신문(2024) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN16","name":"구례군","lat":35.2026,"lon":127.4631,"t":"A",
     "students":720,"dual_pct":60.8,"sing_pct":18.4,"birth_rate":0.88,"decline":True,
     "region_note":"지리산 자락, 읍내 외 돌봄 공백 심각",
     "data_note":"학생수: 경향신문(2024) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN17","name":"고흥군","lat":34.6082,"lon":127.2768,"t":"A",
     "students":1_463,"dual_pct":62.1,"sing_pct":20.3,"birth_rate":0.87,"decline":True,
     "region_note":"고령화율 전국 최상위, 잔류 아동 돌봄 공백",
     "data_note":"학생수: 경향신문(2024) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN18","name":"장흥군","lat":34.6816,"lon":126.9072,"t":"A",
     "students":1_169,"dual_pct":61.4,"sing_pct":18.9,"birth_rate":0.91,"decline":True,
     "region_note":"군 전체 학생 감소, 남은 돌봄 수요 미충족",
     "data_note":"학생수: 경향신문(2024) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN19","name":"강진군","lat":34.6408,"lon":126.7696,"t":"A",
     "students":981,"dual_pct":63.2,"sing_pct":17.6,"birth_rate":1.47,"decline":True,
     "region_note":"다산 문화권, 출산율 전국 2위지만 읍 외 농촌 돌봄 공급 부재",
     "data_note":"학생수: 경향신문(2024) / 출산율: 통계청 2023 전국 2위(1.47명) / 이용인원: 교육부 2023"},
    {"id":"JN20","name":"완도군","lat":34.3109,"lon":126.7550,"t":"A",
     "students":1_826,"dual_pct":61.9,"sing_pct":21.2,"birth_rate":0.93,"decline":True,
     "region_note":"도서 지역, 해상 교통 의존으로 돌봄 접근성 최저",
     "data_note":"학생수: 경향신문(2024) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN21","name":"진도군","lat":34.4870,"lon":126.2630,"t":"A",
     "students":945,"dual_pct":62.7,"sing_pct":20.8,"birth_rate":0.90,"decline":True,
     "region_note":"진도대교 연결 섬, 읍 외 면부 돌봄 공급 거의 없음",
     "data_note":"학생수: 경향신문(2024) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
    {"id":"JN22","name":"신안군","lat":34.8231,"lon":126.1073,"t":"A",
     "students":709,"dual_pct":63.4,"sing_pct":22.1,"birth_rate":0.88,"decline":True,
     "region_note":"1004섬 군도, 섬별 분산으로 시설 접근 불가",
     "data_note":"학생수: 경향신문(2024) / 맞벌이: 통계청 2023 / 이용인원: 교육부 2023"},
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3. 유형별 파라미터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 초등돌봄교실 이용률 범위 (정원 역산 기준)
SPECS = {
    "A": dict(util_range=(0.85, 0.99), wait_range=(40,  220)),
    "B": dict(util_range=(0.18, 0.42), wait_range=(0,    20)),
    "C": dict(util_range=(0.90, 0.99), wait_range=(100, 420)),
    "D": dict(util_range=(0.55, 0.80), wait_range=(5,    65)),
}

# ── 방과후학교 참여율 범위 (교육부 2024.04 전국 52.9% 기준, 도농 차등 적용)
# 도심(C/D): 서울·광역시 기준 60-70% 수준, 농촌(A/B): 15-44%
AFTERSCHOOL_RATE = {
    "A": (0.15, 0.28),   # 농촌 오지: 접근성 낮음, 낮은 참여율
    "B": (0.28, 0.44),   # 인구감소 군: 시설 있으나 학생 급감
    "C": (0.58, 0.72),   # 도심 성장: 맞벌이 집중, 높은 참여율
    "D": (0.48, 0.65),   # 균형 중소도시: 전국 평균 수준
}

# ── 맞춤형교육 참여율 범위 (지역아동센터·아이돌봄서비스·드림스타트 등)
# 농촌 취약지역에 집중 지원, 도심은 비율 낮음
CUSTOM_EDU_RATE = {
    "A": (0.07, 0.13),
    "B": (0.04, 0.08),
    "C": (0.03, 0.06),
    "D": (0.02, 0.05),
}

# ── 돌봄 수요 해소 기여 가중치
# 방과후학교: 방과 후 2-4시간 교육·체험 프로그램, 완전한 돌봄 대체 아님
#   → 1명 참여 = 돌봄교실 0.35명분 수요 해소
# 맞춤형교육(지역아동센터 등): 종일 운영, 취약계층 집중
#   → 1명 참여 = 돌봄교실 0.40명분 수요 해소
AS_WEIGHT  = 0.35
CE_WEIGHT  = 0.40

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4. 데이터 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
rows = []
for r in REGIONS:
    sp  = SPECS[r["t"]]
    g   = np.random.default_rng(hash(r["id"]) % 2**31)
    rid = r["id"]

    # ── 실측 수치 ─────────────────────────────────
    stu      = r["students"]
    dual     = r["dual_pct"]
    sing     = r["sing_pct"]
    br       = r["birth_rate"]
    c_enr    = CARE_REAL.get(rid, {}).get("care_enrolled", 0)   # ★ 실측 이용인원
    n_school = CARE_REAL.get(rid, {}).get("school_count",  0)   # ★ 실측 학교 수

    # ── 정원 역산 (실측 이용인원 ÷ 유형별 이용률 범위)
    util_rate_assumed = g.uniform(*sp["util_range"])
    c_cap = max(int(c_enr / util_rate_assumed), c_enr + 1)

    # ── 실제 이용률
    util_rate = round(c_enr / c_cap * 100, 1) if c_cap > 0 else 0.0

    # ── 방과후학교 참여인원: NEIS 실측 우선, 없으면 추정
    neis_data = NEIS_AFTERSCHOOL.get(rid)
    if neis_data and neis_data.get("enrolled", 0) > 0:
        afterschool_enr    = int(neis_data["enrolled"])
        afterschool_source = "NEIS실측"
    else:
        as_rate            = g.uniform(*AFTERSCHOOL_RATE[r["t"]])
        afterschool_enr    = int(stu * as_rate)
        afterschool_source = "추정"
    # 방과후학교 정원 추정: 참여인원 ÷ 이용률(80-92%)
    afterschool_cap = max(int(afterschool_enr / g.uniform(0.80, 0.92)), afterschool_enr + 1)

    # ── 맞춤형교육 참여인원 추정 (지역아동센터·아이돌봄·드림스타트 등)
    ce_rate      = g.uniform(*CUSTOM_EDU_RATE[r["t"]])
    custom_enr   = int(stu * ce_rate)

    # ── 대기자 시뮬레이션 (공공 미공개)
    wait = int(g.integers(*sp["wait_range"]))

    # ── 출생 변화율 추산 (합계출산율 기반)
    national_tfr  = 0.72
    rural_penalty = -0.15 if r["decline"] else 0.0
    bchg = round((br / national_tfr - 1) * 100 * 0.5 + rural_penalty * 100, 1)
    bchg = float(np.clip(bchg, -55.0, 20.0))
    births   = int(stu * g.uniform(0.45, 0.70))
    births_p = max(int(births * (1 + bchg / 100)), 1)

    area = round(g.uniform(15, 90), 1) if r["t"] == "C" else round(g.uniform(100, 800), 1)

    # ── 통합 공급 지수 산출 ─────────────────────────────────────
    # 기존: supply_idx = 돌봄교실 정원 / 초등학생 수
    # 개선: 돌봄교실 + 방과후학교(×0.35) + 맞춤형교육(×0.40) 복합 지수
    #   - 방과후학교 가중치 0.35: 2-4시간 부분 돌봄, 완전 대체 아님
    #   - 맞춤형교육 가중치 0.40: 종일 운영이나 소규모·취약계층 집중
    effective_supply = c_cap + afterschool_enr * AS_WEIGHT + custom_enr * CE_WEIGHT
    dem  = round(dual / 100, 4)
    sup  = round(effective_supply / max(stu, 1), 4)
    sup  = max(sup, 0.01)
    imb  = round(dem / sup, 4)
    dem5 = round(dem * (1 + bchg / 100 * 0.6), 4)
    dem5 = max(dem5, 0.01)
    rs   = int(np.clip(
        (imb - 1) * 20 + (20 if r["decline"] else 0) + sing * 0.6 + max(0, -bchg) * 0.3,
        0, 100
    ))

    rows.append({
        "region_id":            rid,
        "name":                 r["name"],
        "lat":                  r["lat"],
        "lon":                  r["lon"],
        "urban":                r["t"] == "C",
        "decline":              r["decline"],
        "students":             stu,
        "dual_income_pct":      dual,
        "single_parent_pct":    sing,
        "area_km2":             area,
        "births_5y":            births,
        "births_proj_5y":       births_p,
        "birth_change_pct":     bchg,
        "school_count":         n_school,           # ★ 실측
        "care_cap":             c_cap,              # 역산 추정
        "care_enrolled":        c_enr,              # ★ 실측
        "care_waitlist":        wait,               # 시뮬레이션
        "care_util_rate":       util_rate,          # 실측 기반 산출
        "afterschool_enrolled": afterschool_enr,    # NEIS 실측 또는 추정
        "afterschool_source":  afterschool_source, # "NEIS실측" 또는 "추정"
        "afterschool_cap":      afterschool_cap,    # 추정
        "custom_edu_enrolled":  custom_enr,         # 추정 (지역아동센터 등)
        "demand_idx":           dem,
        "supply_idx":           sup,                # 복합 공급 지수
        "imbal_idx":            imb,
        "demand_idx_5y":        dem5,
        "region_type":          r["t"],
        "type_color":           TYPE_COLORS[r["t"]],
        "type_label":           TYPE_LABELS[r["t"]],
        "top3_features":        "|".join(TOP3_F[r["t"]]),
        "risk_score":           rs,
        "birth_rate":           br,
        "region_note":          r.get("region_note", ""),
        "data_note":            r.get("data_note", ""),
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n생성 완료: {len(df)}개 / 유형 분포: {df['region_type'].value_counts().to_dict()}")
print(f"실측 이용인원 반영: {(df['care_enrolled'] > 0).sum()}개 지역")
print()
print(df[["name","region_type","care_enrolled","afterschool_enrolled","custom_edu_enrolled",
          "supply_idx","imbal_idx","risk_score"]].to_string(index=False))
