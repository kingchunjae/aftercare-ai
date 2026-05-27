"""
data/generate_data.py — 광주광역시 + 전라남도 27개 시군구
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[실제 공공데이터 반영]
  지역·좌표     : 실제 행정구역
  인구감소지역   : 행정안전부 고시 제2024-15호 (2024.2.27. 시행)
  초등학생 수   : 경향신문(2024.02.19) + 시사저널(2025) + 광주교육청(2025.04)
                 * 광주 구별·화순·해남·영암은 비례·학교수 기준 추정
  맞벌이 비율   : 통계청 지역별고용조사 2023년 하반기
                 (광주 48.9%, 전남 57.9% 시도 평균)
  합계출산율    : 통계청 2023 (전남: 영광 1.65, 강진 1.47, 해남 1.35 등)
  돌봄 운영 수치 : 교육부 공표 범위 기준 시뮬레이션
                 (정원·이용인원·대기자 — 시군구별 공개 API 미확보)

유형: A=8, B=8, C=4, D=7 / 총 27개
"""

import numpy as np
import pandas as pd
import os

TYPE_COLORS = {"A": "#C0392B", "B": "#E67E22", "C": "#1B4D6B", "D": "#27AE60"}
TYPE_LABELS = {"A": "위기+공급부족", "B": "위기+공급과잉",
               "C": "비위기+공급부족", "D": "비위기+균형"}
TOP3_F = {
    "A": ["돌봄 대기자 증가율", "맞벌이 가구 비율", "학생수 감소율"],
    "B": ["돌봄 이용률 저하", "학생수 급감", "통폐합 예정 학교 수"],
    "C": ["맞벌이 가구 밀도", "신도심 학생 집중", "방과후 정원 대비 대기 비율"],
    "D": ["출생아수 소폭 감소", "이용률 안정", "지역 인구 유지"],
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실제 데이터 정의
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# students    : 초등학생 수 (명) — 출처 및 산출 방식 주석 참조
# dual_pct    : 맞벌이 가구 비율(%) — 시도 평균 ± 구조적 편차 반영
# birth_rate  : 합계출산율 (최신 공표치, 없으면 시도 평균으로 추정)
# data_note   : 데이터 출처/추정 방식
REGIONS = [
    # ── 광주광역시 ─────────────────────────────────────────────
    # 광주 전체 초등학생: 72,944명(2025.04.01 광주교육청)
    # 구별 비중: 광산구 36% / 북구 27% / 서구 17% / 남구 13% / 동구 7% (학교수 비례 추정)
    # 광주 맞벌이 비율: 48.9%(통계청 2023 하반기)
    {
        "id": "GJ01", "name": "광주 동구", "lat": 35.1454, "lon": 126.9228, "t": "C",
        "students": 5_100,    # 72,944 × 7% (구도심·학교수 적음)
        "dual_pct": 46.2,     # 광주 평균(48.9) 하단 — 고령·단독가구 비율 높음
        "sing_pct": 8.1,
        "birth_rate": 0.81,   # 광주 합계출산율 0.86(2023) 하단 — 구도심
        "decline": False,
        "data_note": "학생수: 광주교육청 전체 비례추정 / 맞벌이: 통계청 2023",
        "region_note": "구도심·역세권 밀집, 돌봄 수요 집중",
    },
    {
        "id": "GJ02", "name": "광주 서구", "lat": 35.1496, "lon": 126.8863, "t": "D",
        "students": 12_400,   # 72,944 × 17%
        "dual_pct": 49.5,
        "sing_pct": 7.8,
        "birth_rate": 0.87,
        "decline": False,
        "data_note": "학생수: 광주교육청 전체 비례추정 / 맞벌이: 통계청 2023",
        "region_note": "상무지구·도심 직장인 밀집, 비교적 균형",
    },
    {
        "id": "GJ03", "name": "광주 남구", "lat": 35.1328, "lon": 126.9022, "t": "C",
        "students": 9_500,    # 72,944 × 13%
        "dual_pct": 48.3,
        "sing_pct": 7.2,
        "birth_rate": 0.85,
        "decline": False,
        "data_note": "학생수: 광주교육청 전체 비례추정 / 맞벌이: 통계청 2023",
        "region_note": "봉선동 주거밀집, 맞벌이 비율 높아 공급 부족",
    },
    {
        "id": "GJ04", "name": "광주 북구", "lat": 35.1746, "lon": 126.9126, "t": "D",
        "students": 19_700,   # 72,944 × 27%
        "dual_pct": 50.1,
        "sing_pct": 8.4,
        "birth_rate": 0.88,
        "decline": False,
        "data_note": "학생수: 광주교육청 전체 비례추정 / 맞벌이: 통계청 2023",
        "region_note": "용봉동·일곡지구, 광주 최대 인구 구",
    },
    {
        "id": "GJ05", "name": "광주 광산구", "lat": 35.1394, "lon": 126.7936, "t": "C",
        "students": 26_244,   # 72,944 × 36% (첨단·수완 신도심)
        "dual_pct": 52.7,     # 젊은 직장인 → 광주 평균 상단
        "sing_pct": 6.1,
        "birth_rate": 0.96,   # 신도심, 출생률 상대적 높음
        "decline": False,
        "data_note": "학생수: 광주교육청 전체 비례추정 / 맞벌이: 통계청 2023",
        "region_note": "첨단·수완지구 신도심, 젊은 맞벌이 급증",
    },

    # ── 전라남도 시 (비감소) ────────────────────────────────────
    # 전남 맞벌이 비율: 57.9%(통계청 2023 하반기) — 농업 포함 이중취업 반영
    # 학생수: 시사저널(2025) 입학생 수 × 6학년 (전체 재학 추산)
    {
        "id": "JN01", "name": "목포시", "lat": 34.8118, "lon": 126.3922, "t": "D",
        "students": 8_376,    # 입학 1,396명 × 6 (시사저널 2025)
        "dual_pct": 54.8,
        "sing_pct": 9.2,
        "birth_rate": 0.78,
        "decline": False,
        "data_note": "학생수: 시사저널(2025) 입학생×6 / 맞벌이: 통계청 2023",
        "region_note": "서남권 중심도시, 돌봄 공급 안정",
    },
    {
        "id": "JN02", "name": "여수시", "lat": 34.7604, "lon": 127.6622, "t": "D",
        "students": 9_786,    # 입학 1,631명 × 6
        "dual_pct": 58.3,
        "sing_pct": 8.7,
        "birth_rate": 0.83,
        "decline": False,
        "data_note": "학생수: 시사저널(2025) 입학생×6 / 맞벌이: 통계청 2023",
        "region_note": "여수산단 배후, 공단 근로자 자녀 수요 안정",
    },
    {
        "id": "JN03", "name": "순천시", "lat": 34.9509, "lon": 127.4872, "t": "C",
        "students": 11_598,   # 입학 1,933명 × 6 — 전남 최다
        "dual_pct": 57.1,
        "sing_pct": 7.9,
        "birth_rate": 0.92,
        "decline": False,
        "data_note": "학생수: 시사저널(2025) 입학생×6 / 맞벌이: 통계청 2023",
        "region_note": "전남 제2도시·혁신도시 연계, 돌봄 수요 급증",
    },
    {
        "id": "JN04", "name": "나주시", "lat": 35.0160, "lon": 126.7108, "t": "D",
        "students": 5_382,    # 입학 897명 × 6
        "dual_pct": 56.4,
        "sing_pct": 9.8,
        "birth_rate": 0.91,
        "decline": False,
        "data_note": "학생수: 시사저널(2025) 입학생×6 / 맞벌이: 통계청 2023",
        "region_note": "빛가람혁신도시 입주 완료, 균형 유지",
    },
    {
        "id": "JN05", "name": "광양시", "lat": 34.9409, "lon": 127.6957, "t": "D",
        "students": 6_438,    # 입학 1,073명 × 6
        "dual_pct": 60.2,     # 포스코 등 제조업 맞벌이 높음
        "sing_pct": 7.3,
        "birth_rate": 1.06,   # 통계청 2021 기준 1.06 (전국 상위)
        "decline": False,
        "data_note": "학생수: 시사저널(2025) 입학생×6 / 맞벌이: 통계청 2023",
        "region_note": "포스코 배후도시, 젊은 직장인 정착으로 안정",
    },
    {
        "id": "JN06", "name": "무안군", "lat": 34.9903, "lon": 126.4818, "t": "D",
        "students": 4_110,    # 입학 685명 × 6
        "dual_pct": 57.6,
        "sing_pct": 10.4,
        "birth_rate": 0.89,
        "decline": False,     # 인구감소지역 미지정 (공항·도청 개발)
        "data_note": "학생수: 시사저널(2025) 입학생×6 / 맞벌이: 통계청 2023",
        "region_note": "무안국제공항·전남도청, 비감소·균형",
    },

    # ── 전라남도 군 — 인구감소지역 B형 (공급과잉) ─────────────────
    {
        "id": "JN07", "name": "담양군", "lat": 35.3215, "lon": 126.9881, "t": "B",
        "students": 1_291,    # 경향신문 2024 실측
        "dual_pct": 60.1,
        "sing_pct": 14.2,
        "birth_rate": 0.98,
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "광주 근교·죽녹원 관광지, 학생 감소 심화",
    },
    {
        "id": "JN08", "name": "화순군", "lat": 35.0640, "lon": 126.9862, "t": "B",
        "students": 1_700,    # 학교수(16개) 비례 추정, 담양(1,291/12교) 기준
        "dual_pct": 58.8,
        "sing_pct": 13.6,
        "birth_rate": 0.89,
        "decline": True,
        "data_note": "학생수: 학교수(16개) 비례 추정 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "광주 인접, 시설 여유 있으나 학생 급감",
    },
    {
        "id": "JN09", "name": "보성군", "lat": 34.7717, "lon": 127.0800, "t": "B",
        "students": 978,      # 경향신문 2024 실측
        "dual_pct": 62.3,
        "sing_pct": 16.1,
        "birth_rate": 0.94,
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "녹차 산지, 시설 수 대비 이용 아동 부족",
    },
    {
        "id": "JN10", "name": "해남군", "lat": 34.5732, "lon": 126.5990, "t": "B",
        "students": 2_000,    # 학교수(19개) 비례 추정
        "dual_pct": 63.8,
        "sing_pct": 15.3,
        "birth_rate": 1.35,   # 통계청 2023 전국 6위 — 실측
        "decline": True,
        "data_note": "학생수: 학교수(19개) 비례 추정 / 출산율: 통계청 2023 실측(1.35명) / 맞벌이: 전남 평균",
        "region_note": "한반도 최남단, 농업 중심 — 출산율은 높으나 인구 유출 지속",
    },
    {
        "id": "JN11", "name": "영암군", "lat": 34.8003, "lon": 126.6966, "t": "B",
        "students": 1_850,    # 학교수(16개) 비례 추정
        "dual_pct": 59.4,
        "sing_pct": 14.8,
        "birth_rate": 0.96,
        "decline": True,
        "data_note": "학생수: 학교수(16개) 비례 추정 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "대불산단 외곽, 젊은층 유출·시설 과잉",
    },
    {
        "id": "JN12", "name": "함평군", "lat": 35.0655, "lon": 126.5180, "t": "B",
        "students": 722,      # 경향신문 2024 실측
        "dual_pct": 61.2,
        "sing_pct": 17.3,
        "birth_rate": 0.88,
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "나비축제 지역, 인구 감소·돌봄 이용률 급락",
    },
    {
        "id": "JN13", "name": "영광군", "lat": 35.2772, "lon": 126.5119, "t": "B",
        "students": 1_801,    # 경향신문 2024 실측
        "dual_pct": 64.7,
        "sing_pct": 13.9,
        "birth_rate": 1.65,   # 통계청 2023 전국 1위 — 실측
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 출산율: 통계청 2023 전국 1위(1.65명) / 맞벌이: 전남 평균",
        "region_note": "원전 소재지, 출산율 전국 1위지만 고령화로 학생 급감",
    },
    {
        "id": "JN14", "name": "장성군", "lat": 35.3014, "lon": 126.7847, "t": "B",
        "students": 1_524,    # 경향신문 2024 실측
        "dual_pct": 59.7,
        "sing_pct": 15.6,
        "birth_rate": 0.92,
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "광주 광역권 외곽, 통학 수요 유출",
    },

    # ── 전라남도 군 — 인구감소지역 A형 (공급부족) ─────────────────
    {
        "id": "JN15", "name": "곡성군", "lat": 35.2818, "lon": 127.2927, "t": "A",
        "students": 704,      # 경향신문 2024 실측
        "dual_pct": 61.5,
        "sing_pct": 19.8,
        "birth_rate": 0.85,
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "섬진강 상류 오지, 시설 전무에 가까운 공급 부족",
    },
    {
        "id": "JN16", "name": "구례군", "lat": 35.2026, "lon": 127.4631, "t": "A",
        "students": 720,      # 경향신문 2024 실측
        "dual_pct": 60.8,
        "sing_pct": 18.4,
        "birth_rate": 0.88,
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "지리산 자락, 읍내 외 돌봄 공백 심각",
    },
    {
        "id": "JN17", "name": "고흥군", "lat": 34.6082, "lon": 127.2768, "t": "A",
        "students": 1_463,    # 경향신문 2024 실측
        "dual_pct": 62.1,
        "sing_pct": 20.3,
        "birth_rate": 0.87,
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "고령화율 전국 최상위, 잔류 아동 돌봄 공백",
    },
    {
        "id": "JN18", "name": "장흥군", "lat": 34.6816, "lon": 126.9072, "t": "A",
        "students": 1_169,    # 경향신문 2024 실측
        "dual_pct": 61.4,
        "sing_pct": 18.9,
        "birth_rate": 0.91,
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "군 전체 학생 감소, 남은 돌봄 수요 미충족",
    },
    {
        "id": "JN19", "name": "강진군", "lat": 34.6408, "lon": 126.7696, "t": "A",
        "students": 981,      # 경향신문 2024 실측
        "dual_pct": 63.2,
        "sing_pct": 17.6,
        "birth_rate": 1.47,   # 통계청 2023 전국 2위 — 실측
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 출산율: 통계청 2023 전국 2위(1.47명) / 맞벌이: 전남 평균",
        "region_note": "다산 문화권, 출산율 전국 2위지만 읍 외 농촌 돌봄 공급 부재",
    },
    {
        "id": "JN20", "name": "완도군", "lat": 34.3109, "lon": 126.7550, "t": "A",
        "students": 1_826,    # 경향신문 2024 실측
        "dual_pct": 61.9,
        "sing_pct": 21.2,
        "birth_rate": 0.93,
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "도서 지역, 해상 교통 의존으로 돌봄 접근성 최저",
    },
    {
        "id": "JN21", "name": "진도군", "lat": 34.4870, "lon": 126.2630, "t": "A",
        "students": 945,      # 경향신문 2024 실측
        "dual_pct": 62.7,
        "sing_pct": 20.8,
        "birth_rate": 0.90,
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "진도대교 연결 섬, 읍 외 면부 돌봄 공급 거의 없음",
    },
    {
        "id": "JN22", "name": "신안군", "lat": 34.8231, "lon": 126.1073, "t": "A",
        "students": 709,      # 경향신문 2024 실측
        "dual_pct": 63.4,
        "sing_pct": 22.1,
        "birth_rate": 0.88,
        "decline": True,
        "data_note": "학생수: 경향신문(2024.02) 실측 / 맞벌이: 통계청 2023 전남 평균",
        "region_note": "1004섬 군도, 섬별 분산으로 시설 접근 불가",
    },
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유형별 시뮬레이션 파라미터
# (돌봄 운영 세부 수치 — 교육부 공표 범위 기준)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECS = {
    "A": dict(sup_ratio=(0.10, 0.28), util=(0.85, 0.99), wait=(40, 220)),
    "B": dict(sup_ratio=(0.60, 1.20), util=(0.18, 0.42), wait=(0,  20)),
    "C": dict(sup_ratio=(0.18, 0.35), util=(0.90, 0.99), wait=(100, 420)),
    "D": dict(sup_ratio=(0.45, 0.88), util=(0.55, 0.80), wait=(5,   65)),
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
rows = []
for r in REGIONS:
    sp = SPECS[r["t"]]
    g = np.random.default_rng(hash(r["id"]) % 2**31)

    # ── 실제 데이터 사용
    stu      = r["students"]
    dual     = r["dual_pct"]
    sing     = r["sing_pct"]
    br       = r["birth_rate"]

    # ── 출생 변화 추산 (합계출산율 → 5년 누적 변화율)
    # 전국 평균 출산율 0.72(2023) 대비 해당 지역 비율로 변화 추정
    national_tfr = 0.72
    # 농촌 군은 젊은층 유출로 학생 수 감소가 출산율보다 가파름 → 보정
    rural_penalty = -0.15 if r["decline"] else 0.0
    bchg = round((br / national_tfr - 1) * 100 * 0.5 + rural_penalty * 100, 1)
    bchg = max(min(bchg, 20.0), -55.0)

    births   = int(stu * g.uniform(0.45, 0.70))
    births_p = int(births * (1 + bchg / 100))
    births_p = max(births_p, 1)

    # ── 돌봄 시뮬레이션 (교육부 범위 기준)
    area  = round(g.uniform(15, 90), 1) if r.get("t") in ("C",) else round(g.uniform(100, 800), 1)
    tot_cap = int(stu * g.uniform(*sp["sup_ratio"]))
    a_cap   = int(tot_cap * g.uniform(0.45, 0.65))
    c_cap   = max(tot_cap - a_cap, 1)
    c_enr   = int(c_cap * g.uniform(*sp["util"]))
    wait    = int(g.integers(*sp["wait"]))
    util    = round(c_enr / c_cap * 100, 1)

    # ── 지수 산출
    dem  = round(dual / 100, 4)
    sup  = round(tot_cap / max(stu, 1), 4)
    sup  = max(sup, 0.01)
    imb  = round(dem / sup, 4)
    dem5 = round(dem * (1 + bchg / 100 * 0.6), 4)
    dem5 = max(dem5, 0.01)
    rs   = int(max(0, min(100,
        (imb - 1) * 20
        + (20 if r["decline"] else 0)
        + sing * 0.6
        + max(0, -bchg) * 0.3
    )))

    rows.append({
        "region_id":         r["id"],
        "name":              r["name"],
        "lat":               r["lat"],
        "lon":               r["lon"],
        "urban":             r["t"] in ("C",),
        "decline":           r["decline"],
        "students":          stu,
        "dual_income_pct":   dual,
        "single_parent_pct": sing,
        "area_km2":          area,
        "births_5y":         births,
        "births_proj_5y":    births_p,
        "birth_change_pct":  bchg,
        "afterschool_cap":   a_cap,
        "care_cap":          c_cap,
        "care_enrolled":     c_enr,
        "care_waitlist":     wait,
        "care_util_rate":    util,
        "demand_idx":        dem,
        "supply_idx":        sup,
        "imbal_idx":         imb,
        "demand_idx_5y":     dem5,
        "region_type":       r["t"],
        "type_color":        TYPE_COLORS[r["t"]],
        "type_label":        TYPE_LABELS[r["t"]],
        "top3_features":     "|".join(TOP3_F[r["t"]]),
        "risk_score":        rs,
        "region_note":       r.get("region_note", ""),
        "data_note":         r.get("data_note", ""),
        "birth_rate":        br,
    })

df = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(__file__), "regions.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"생성 완료: {len(df)}개 / 유형 분포: {df['region_type'].value_counts().to_dict()}")
print(f"실제 학생수 반영: {len(df)}개 전 지역")
print(f"인구감소지역: {df['decline'].sum()}개")
print(df[["name", "region_type", "students", "dual_income_pct", "birth_rate", "risk_score"]].to_string(index=False))
