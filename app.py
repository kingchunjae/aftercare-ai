"""
app.py — 방과후·초등돌봄 수요-공급 불균형 AI 진단 시스템
실행: streamlit run app.py
"""
import os, sys
import streamlit as st
import pandas as pd

# 프로젝트 경로 설정
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from src.preprocessing import load_data, get_summary_stats, get_region_detail, simulate_budget, TYPE_INFO
from src.model import ensure_trained, load_models, predict_region, get_feature_importance
from src.visualize import (
    build_map, imbal_gauge, demand_forecast_chart,
    type_pie, budget_bar, budget_dumbbell, importance_bar
)
from src.ai_report import generate_report, estimate_cost
from streamlit_folium import st_folium

# ── 페이지 설정
st.set_page_config(
    page_title="돌봄 AI 진단 시스템",
    page_icon="🗺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 커스텀 CSS
st.markdown("""
<style>
  .metric-card {
    background: #f5f2ec; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 8px;
  }
  .type-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 6px; font-size: 12px; font-weight: 600;
  }
  .section-header {
    font-size: 15px; font-weight: 600;
    color: #1B4D6B; margin: 16px 0 8px 0;
  }
  div[data-testid="stTabs"] button {
    font-size: 14px; font-weight: 500;
  }

  /* 유형 필터 체크박스 행 */
  .type-filter-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }
  .type-filter-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 700;
    color: white;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .type-filter-label {
    font-size: 12px;
    color: #333;
    line-height: 1.4;
    word-break: keep-all;
  }
  /* 체크박스-뱃지 사이 세로 정렬 */
  div[data-testid="stSidebar"] div[data-testid="stCheckbox"] {
    margin-bottom: 0 !important;
  }
  div[data-testid="stSidebar"] div[data-testid="stCheckbox"] label {
    min-height: 0 !important;
  }

  /* ── 데이터 출처 카드 ── */
  .ds-wrap {
    margin-top: 2px;
  }
  .ds-title {
    font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
    color: #1B4D6B; text-transform: uppercase;
    display: flex; align-items: center; gap: 5px;
    margin-bottom: 8px;
  }
  .ds-legend {
    display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap;
  }
  .ds-legend span {
    font-size: 10px; color: #555;
    display: flex; align-items: center; gap: 4px;
  }
  .badge-real {
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-size: 9px; font-weight: 700;
    background: #e3f0e8; color: #256336; border: 1px solid #b2d9bc;
  }
  .badge-est {
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-size: 9px; font-weight: 700;
    background: #fff4e5; color: #a04e00; border: 1px solid #f5c97a;
  }
  .ds-card {
    background: #f7f9fb;
    border-radius: 7px;
    border-left: 3px solid #1B4D6B;
    padding: 8px 10px;
    margin-bottom: 6px;
  }
  .ds-card.est {
    background: #f9f9f7;
    border-left-color: #c8b97a;
  }
  .ds-card-top {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 3px;
  }
  .ds-agency {
    font-size: 10.5px; font-weight: 700; color: #1B4D6B;
    display: flex; align-items: center; gap: 4px;
  }
  .ds-agency.est { color: #7a6e3f; }
  .ds-items {
    font-size: 11px; color: #222; font-weight: 500;
    line-height: 1.5; margin-bottom: 2px;
  }
  .ds-pub {
    font-size: 9.5px; color: #888; line-height: 1.45;
  }
  .ds-footer {
    margin-top: 10px; padding-top: 8px;
    border-top: 1px solid #e0ddd4;
    font-size: 9.5px; color: #999; line-height: 1.5;
    text-align: center;
  }

  /* ── 데이터 라이선스 섹션 카드 ── */
  .dc-card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
  }
  .dc-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.13);
  }
</style>
""", unsafe_allow_html=True)

# ── NEIS 캐시 상태 확인 (사이드바 표시용)
import json as _json
_NEIS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "neis_afterschool_cache.json")
def _load_neis_meta() -> dict:
    if os.path.exists(_NEIS_CACHE_PATH):
        try:
            with open(_NEIS_CACHE_PATH, "r", encoding="utf-8") as _f:
                _c = _json.load(_f)
            return {
                "available": True,
                "coverage":  len(_c.get("regions", {})),
                "year":      _c.get("year", ""),
                "fetched_at": _c.get("fetched_at", ""),
            }
        except Exception:
            pass
    return {"available": False}

_neis_meta = _load_neis_meta()

# ── 데이터 로드 & 모델 초기화
# cache_version: 컬럼 구조가 바뀔 때 올려서 Streamlit Cloud 캐시 강제 무효화
@st.cache_data
def load(cache_version: int = 8):
    df = load_data()
    return df

@st.cache_resource
def init_models(df):
    ensure_trained(df)
    return load_models()

df = load(cache_version=8)
reg, clf, scaler = init_models(df)

# ── NEIS 상태에 따른 사이드바 카드 HTML 결정
if _neis_meta["available"]:
    _neis_sidebar_card = (
        f"<div class='ds-card'>"
        f"<div class='ds-card-top'>"
        f"<span class='ds-agency'>&#128203; NEIS Open API</span>"
        f"<span class='badge-real'>실&nbsp;측</span>"
        f"</div>"
        f"<div class='ds-items'>초등학교 수 (시군구별)</div>"
        f"<div class='ds-pub'>"
        f"schoolInfo — 광주(F10)+전남(Q10)<br>"
        f"617개교 실측 · {_neis_meta['fetched_at'][:10]}"
        f"</div></div>"
        f"<div class='ds-card est'>"
        f"<div class='ds-card-top'>"
        f"<span class='ds-agency est'>&#128295; NEIS 기반 추정</span>"
        f"<span class='badge-est'>추&nbsp;정</span>"
        f"</div>"
        f"<div class='ds-items'>방과후학교 · 지역돌봄기관 참여인원</div>"
        f"<div class='ds-pub'>방과후: NEIS 실측 학교수 × 전국 참여율<br>"
        f"지역돌봄기관: 유형별 도농 차등 추정</div>"
        f"</div>"
    )
else:
    _neis_sidebar_card = (
        "<div class='ds-card est'>"
        "<div class='ds-card-top'>"
        "<span class='ds-agency est'>&#128295; 통계 기반 추정</span>"
        "<span class='badge-est'>추&nbsp;정</span>"
        "</div>"
        "<div class='ds-items'>방과후학교 · 지역돌봄기관 참여인원</div>"
        "<div class='ds-pub'>전국 방과후학교 참여율(52.9%) 기반<br>"
        "유형별 도농 차등 추정 (교육부 2024.04)<br>"
        "<span style='color:#1B4D6B;font-size:9px'>"
        "&#9432; NEIS API 연동 시 기반추정으로 전환</span></div>"
        "</div>"
    )

# ── 사이드바
with st.sidebar:
    # ── 헤더: 테크 대시보드 스타일
    st.markdown(
        f"<div style='font-family:sans-serif;padding:6px 0 4px 0'>"

        # 아이콘 + 브랜드명
        f"<div style='display:flex;align-items:center;gap:11px;margin-bottom:10px'>"
        f"<div style='background:linear-gradient(135deg,#1B4D6B 0%,#C0392B 100%);"
        f"width:44px;height:44px;border-radius:11px;display:flex;"
        f"align-items:center;justify-content:center;"
        f"font-size:22px;flex-shrink:0;box-shadow:0 2px 6px rgba(0,0,0,0.18)'>&#127979;</div>"
        f"<div>"
        f"<div style='font-size:17px;font-weight:800;color:#1a1a1a;"
        f"letter-spacing:-0.3px;line-height:1.2'>CareMap AI</div>"
        f"<div style='font-size:10px;color:#999;margin-top:3px;font-weight:500'>"
        f"초등돌봄 불균형 진단 플랫폼</div>"
        f"</div>"
        f"</div>"

        # LIVE 상태 바
        f"<div style='background:#f2f4f7;border-radius:7px;padding:7px 11px;"
        f"display:flex;justify-content:space-between;align-items:center;"
        f"border:1px solid #e8ebef'>"
        f"<span style='font-size:10px;color:#27AE60;font-weight:700;"
        f"display:flex;align-items:center;gap:4px'>"
        f"<span style='width:6px;height:6px;background:#27AE60;border-radius:50%;"
        f"display:inline-block'></span>LIVE</span>"
        f"<span style='font-size:10px;color:#888'>"
        f"{len(df)}개 지역 &nbsp;&#183;&nbsp; 교육 공공데이터</span>"
        f"</div>"

        f"</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("필터")
    st.caption("유형 선택")
    type_filter = []
    for _t, _info in TYPE_INFO.items():
        _col_cb, _col_label = st.columns([0.13, 0.87])
        with _col_cb:
            _checked = st.checkbox(
                label=_t, value=True,
                key=f"filter_type_{_t}",
                label_visibility="collapsed",
            )
        with _col_label:
            st.markdown(
                f"<div class='type-filter-row'>"
                f"<span class='type-filter-badge' style='background:{_info['color']}'>"
                f"{_t}형</span>"
                f"<span class='type-filter-label'>{_info['label']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if _checked:
            type_filter.append(_t)
    risk_min = st.slider("최소 위험 점수", 0, 100, 0)
    decline_only = st.checkbox("인구감소지역만")

    df_filtered = df[
        df["region_type"].isin(type_filter) &
        (df["risk_score"] >= risk_min) &
        (~decline_only | df["decline"])
    ]

    st.divider()
    st.subheader("요약 통계")
    stats = get_summary_stats(df_filtered)
    col1, col2 = st.columns(2)
    col1.metric("전체 지역", stats["total"])
    col2.metric("고위험 지역", stats["high_risk_count"])
    col1.metric("총 대기 아동", f"{stats['total_waitlist']:,}명")
    col2.metric("평균 이용률", f"{stats['avg_util_rate']}%")

    st.divider()
    st.markdown(
        f"""<div class="ds-wrap">
  <div class="ds-title">&#128202; 데이터 출처</div>

  <div class="ds-legend">
    <span><span class="badge-real">실&nbsp;측</span> 공공데이터 원본</span>
    <span><span class="badge-est">추&nbsp;정</span> 역산·시뮬레이션</span>
  </div>

  <div class="ds-card">
    <div class="ds-card-top">
      <span class="ds-agency">&#127963; 교육부</span>
      <span class="badge-real">실&nbsp;측</span>
    </div>
    <div class="ds-items">이용인원 · 돌봄 학교 수</div>
    <div class="ds-pub">초등돌봄교실 현황<br>공공데이터포털 (2023.04 기준)</div>
  </div>

  <div class="ds-card">
    <div class="ds-card-top">
      <span class="ds-agency">&#128200; 통계청</span>
      <span class="badge-real">실&nbsp;측</span>
    </div>
    <div class="ds-items">맞벌이 가구 비율 · 합계출산율</div>
    <div class="ds-pub">지역별고용조사 2023 하반기<br>시군구별 출생통계 2023</div>
  </div>

  <div class="ds-card">
    <div class="ds-card-top">
      <span class="ds-agency">&#127970; 행정안전부</span>
      <span class="badge-real">실&nbsp;측</span>
    </div>
    <div class="ds-items">인구감소지역 지정 현황</div>
    <div class="ds-pub">고시 제2024-15호 (전남 16개 군)</div>
  </div>

  <div class="ds-card">
    <div class="ds-card-top">
      <span class="ds-agency">&#128209; NEIS Open API</span>
      <span class="badge-real">실&nbsp;측</span>
    </div>
    <div class="ds-items">초등학생 수 (시군구별)</div>
    <div class="ds-pub">NEIS classInfo(2025학년도)<br>학급수 &times; KEDI 학급당 학생수(2024)</div>
  </div>

  {_neis_sidebar_card}

  <div class="ds-card est">
    <div class="ds-card-top">
      <span class="ds-agency est">&#128295; 역산 추정</span>
      <span class="badge-est">추&nbsp;정</span>
    </div>
    <div class="ds-items">돌봄 정원 · 대기아동 수</div>
    <div class="ds-pub">시군구 단위 공공데이터 미공개<br>유형별 이용률 범위 역산 적용</div>
  </div>

  <div class="ds-footer">
    분석 기준 시점: 2023년<br>
    출처: <a href="https://data.go.kr" target="_blank" style="color:#1B4D6B">공공데이터포털</a> ·
    <a href="https://kosis.kr" target="_blank" style="color:#1B4D6B">국가통계포털(KOSIS)</a>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

# ── 메인 헤더
st.title("🏫 방과후·초등돌봄 수요-공급 불균형 AI 진단")
st.caption("교육 공공데이터 기반 지역소멸 위기 연계 분석 | 광역 통합 행정 시뮬레이션 (27개 시군구)")

# ── 프로젝트 소개 expander
with st.expander("📋 프로젝트 소개 — 왜 지금 이 문제인가?", expanded=False):
    st.markdown(
        '<div style="background:white;border-radius:16px;padding:24px 28px 20px 28px;border:1px solid #dde8f4">'
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;margin-bottom:18px">'
        '<div><div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#94a3b8;text-transform:uppercase;margin-bottom:5px">PROBLEM STATEMENT</div>'
        '<div style="font-size:20px;font-weight:800;color:#1e293b;letter-spacing:-0.4px">왜 지금 이 문제인가?</div>'
        '<div style="font-size:13px;color:#64748b;margin-top:5px">광주·전남 27개 시군구 / 교육 공공데이터 기반 AI 진단 시스템</div></div>'
        '<div style="background:linear-gradient(135deg,#1B4D6B 0%,#2980B9 100%);color:white;border-radius:12px;padding:12px 18px;text-align:center;flex-shrink:0;box-shadow:0 4px 14px rgba(27,77,107,0.28)">'
        '<div style="font-size:9px;font-weight:600;opacity:0.75;letter-spacing:1.2px;margin-bottom:5px">🏆 대 회 출 품 작</div>'
        '<div style="font-size:12px;font-weight:800;line-height:1.55">제8회 교육<br>공공데이터<br>AI 활용대회</div></div></div>'
        '<div style="background:#1B4D6B;border-radius:12px;padding:20px 24px;display:grid;grid-template-columns:repeat(3,1fr);margin-bottom:16px">'
        '<div style="text-align:center;padding:4px 8px">'
        '<div style="font-size:34px;font-weight:900;color:#f97316;letter-spacing:-1.5px;line-height:1">4<span style="font-size:20px">만 명</span><span style="font-size:18px;color:#fb923c">+</span></div>'
        '<div style="font-size:10.5px;color:rgba(255,255,255,0.82);margin-top:7px;line-height:1.55">전국 초등돌봄 대기 아동<br><span style="color:rgba(255,255,255,0.45);font-size:9.5px">2023년 기준 · 교육부</span></div></div>'
        '<div style="text-align:center;padding:4px 8px;border-left:1px solid rgba(255,255,255,0.15);border-right:1px solid rgba(255,255,255,0.15)">'
        '<div style="font-size:34px;font-weight:900;color:#f97316;letter-spacing:-1.5px;line-height:1">41<span style="font-size:20px">%</span></div>'
        '<div style="font-size:10.5px;color:rgba(255,255,255,0.82);margin-top:7px;line-height:1.55">농산어촌 돌봄교실 평균 이용률<br><span style="color:rgba(255,255,255,0.45);font-size:9.5px">학교알리미 공시 기반 추정</span></div></div>'
        '<div style="text-align:center;padding:4px 8px">'
        '<div style="font-size:34px;font-weight:900;color:#f97316;letter-spacing:-1.5px;line-height:1">89<span style="font-size:20px">개</span></div>'
        '<div style="font-size:10.5px;color:rgba(255,255,255,0.82);margin-top:7px;line-height:1.55">행안부 지정 인구감소지역<br><span style="color:rgba(255,255,255,0.45);font-size:9.5px">2024년 기준</span></div></div></div>'
        '<div style="background:#fff7ed;border-radius:10px;padding:14px 20px;border-left:4px solid #f97316;margin-bottom:18px">'
        '<div style="font-size:13.5px;color:#1e293b;font-weight:500;line-height:1.75">같은 나라에서 <strong style="color:#ea580c">\'기다리는 아이\'</strong>와 <strong style="color:#1B4D6B">\'텅 빈 교실\'</strong>이 동시에 존재하는 역설 — 이것이 본 기획의 출발점입니다.</div></div>'
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:16px">'
        '<div style="background:white;border-radius:12px;padding:16px;border:1px solid #e2e8f0;border-left:4px solid #f97316">'
        '<div style="font-size:22px;font-weight:900;color:#f97316;margin-bottom:8px;font-family:Georgia,serif;letter-spacing:-0.5px">01</div>'
        '<div style="font-size:13px;font-weight:700;color:#1e293b;margin-bottom:7px">도시 과밀 × 농촌 공동화</div>'
        '<div style="font-size:11.5px;color:#475569;line-height:1.7">수도권·광역시 돌봄 수요는 폭증하는 반면 농산어촌 돌봄시설은 이용률 41%로 미가동. 전국 단위 자원 배분이 실패한 구조.</div></div>'
        '<div style="background:white;border-radius:12px;padding:16px;border:1px solid #e2e8f0;border-left:4px solid #f97316">'
        '<div style="font-size:22px;font-weight:900;color:#f97316;margin-bottom:8px;font-family:Georgia,serif;letter-spacing:-0.5px">02</div>'
        '<div style="font-size:13px;font-weight:700;color:#1e293b;margin-bottom:7px">수요 예측 없는 공급 계획</div>'
        '<div style="font-size:11.5px;color:#475569;line-height:1.7">현재 늘봄학교 예산은 학교 수 기준 배분. 실제 수요(맞벌이 가구·한부모 가구 비율)는 반영되지 않음.</div></div>'
        '<div style="background:white;border-radius:12px;padding:16px;border:1px solid #e2e8f0;border-left:4px solid #f97316">'
        '<div style="font-size:22px;font-weight:900;color:#f97316;margin-bottom:8px;font-family:Georgia,serif;letter-spacing:-0.5px">03</div>'
        '<div style="font-size:13px;font-weight:700;color:#1e293b;margin-bottom:7px">지역소멸·돌봄 공백 악순환</div>'
        '<div style="font-size:11.5px;color:#475569;line-height:1.7">돌봄 공백 → 젊은 부모 유출 → 학령인구 감소 → 돌봄시설 추가 축소 → 지역 소멸 가속화.</div></div></div>'
        '<div style="background:#f8fafc;border-radius:10px;padding:14px 18px;border:1px solid #e2e8f0">'
        '<div style="font-size:11.5px;color:#334155;line-height:1.75;margin-bottom:6px"><strong style="color:#1B4D6B">핵심 메시지</strong> &nbsp;2024년 전국 확대된 늘봄학교 정책이 농산어촌에서 실효성 논란에 직면. 본 기획은 공공데이터와 AI로 그 해법을 제시합니다.</div>'
        '<div style="font-size:10px;color:#94a3b8">출처: 교육부 보도자료(2023) · 학교알리미 공시데이터 · 행정안전부 인구감소지역 고시(2024)</div></div>'
        '</div>',
        unsafe_allow_html=True
    )

# ── 히어로 지표 → 4분면 카드  (st.columns 기반 레이아웃)
st.markdown(
    "<div style='font-size:12.5px;font-weight:700;color:#444;margin-bottom:6px'>"
    "지역 유형 4분면 분류 체계 &nbsp;"
    "<span style='font-size:11px;font-weight:400;color:#aaa'>공급 상태 × 소멸위기 여부</span>"
    "</div>",
    unsafe_allow_html=True,
)

# X축 레이블 (2열)
_xc1, _xc2 = st.columns(2)
_xc1.markdown(
    "<div style='text-align:center;font-size:10px;color:#bbb;font-weight:600;"
    "border-bottom:2px dashed #ddd;padding-bottom:3px;margin-bottom:4px'>"
    "&#8592; 공급 부족 &nbsp;(불균형지수 &#8805; 1.2)</div>",
    unsafe_allow_html=True,
)
_xc2.markdown(
    "<div style='text-align:center;font-size:10px;color:#bbb;font-weight:600;"
    "border-bottom:2px dashed #ddd;padding-bottom:3px;margin-bottom:4px'>"
    "공급 과잉&#183;균형 &nbsp;(불균형지수 &#8804; 1.0) &#8594;</div>",
    unsafe_allow_html=True,
)

# ── 행 1: 위기지역 (A형 / B형)
st.markdown(
    "<div style='font-size:10px;color:#bbb;font-weight:600;"
    "margin:0 0 3px 0;padding:2px 0 2px 10px;border-left:3px dashed #ddd'>"
    "&#9650; 위기지역 (소멸위기)</div>",
    unsafe_allow_html=True,
)
_r1c1, _r1c2 = st.columns(2)

with _r1c1:
    st.markdown(
        f"<div style='background:#fdecea;border:2px solid #C0392B;border-radius:10px;padding:14px 16px'>"
        f"<div style='display:flex;align-items:flex-start;gap:10px;margin-bottom:8px'>"
        f"<div style='background:#C0392B;color:white;font-size:16px;font-weight:800;"
        f"min-width:34px;height:34px;border-radius:50%;display:flex;"
        f"align-items:center;justify-content:center;flex-shrink:0'>A</div>"
        f"<div style='flex:1'>"
        f"<div><span style='font-size:26px;font-weight:800;color:#C0392B;line-height:1'>{stats['A']}</span>"
        f"<span style='font-size:13px;font-weight:600;color:#C0392B'>&nbsp;개 지역</span></div>"
        f"<div style='font-size:11.5px;font-weight:700;color:#C0392B;margin-top:1px'>위기 + 공급부족</div>"
        f"</div>"
        f"<span style='font-size:20px;opacity:0.4'>&#128308;</span>"
        f"</div>"
        f"<div style='font-size:11.5px;color:#555;line-height:1.55;margin-bottom:9px'>"
        f"소멸위기 지역이면서 돌봄 수요가 공급을 크게 초과. "
        f"즉각적인 자원 투입이 필요한 <b style='color:#C0392B'>최우선 개입 대상</b>"
        f"</div>"
        f"<span style='background:#C0392B;color:white;font-size:10.5px;font-weight:700;"
        f"padding:3px 11px;border-radius:4px;display:inline-block'>&#128680; 긴급 개입</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with _r1c2:
    st.markdown(
        f"<div style='background:#fef4e8;border:2px solid #E67E22;border-radius:10px;padding:14px 16px'>"
        f"<div style='display:flex;align-items:flex-start;gap:10px;margin-bottom:8px'>"
        f"<div style='background:#E67E22;color:white;font-size:16px;font-weight:800;"
        f"min-width:34px;height:34px;border-radius:50%;display:flex;"
        f"align-items:center;justify-content:center;flex-shrink:0'>B</div>"
        f"<div style='flex:1'>"
        f"<div><span style='font-size:26px;font-weight:800;color:#E67E22;line-height:1'>{stats['B']}</span>"
        f"<span style='font-size:13px;font-weight:600;color:#E67E22'>&nbsp;개 지역</span></div>"
        f"<div style='font-size:11.5px;font-weight:700;color:#E67E22;margin-top:1px'>위기 + 공급과잉</div>"
        f"</div>"
        f"<span style='font-size:20px;opacity:0.4'>&#128992;</span>"
        f"</div>"
        f"<div style='font-size:11.5px;color:#555;line-height:1.55;margin-bottom:9px'>"
        f"인구감소로 수요는 줄었지만 시설은 남아 있는 지역. "
        f"기존 인프라의 <b style='color:#E67E22'>복합 활용 전환</b>이 필요한 구조 개편 대상"
        f"</div>"
        f"<span style='background:#E67E22;color:white;font-size:10.5px;font-weight:700;"
        f"padding:3px 11px;border-radius:4px;display:inline-block'>&#128260; 구조 전환</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── 행 2: 비위기지역 (C형 / D형)
st.markdown(
    "<div style='font-size:10px;color:#bbb;font-weight:600;"
    "margin:8px 0 3px 0;padding:2px 0 2px 10px;border-left:3px dashed #ddd'>"
    "&#9660; 비위기지역</div>",
    unsafe_allow_html=True,
)
_r2c1, _r2c2 = st.columns(2)

with _r2c1:
    st.markdown(
        f"<div style='background:#eaf0f7;border:2px solid #1B4D6B;border-radius:10px;padding:14px 16px'>"
        f"<div style='display:flex;align-items:flex-start;gap:10px;margin-bottom:8px'>"
        f"<div style='background:#1B4D6B;color:white;font-size:16px;font-weight:800;"
        f"min-width:34px;height:34px;border-radius:50%;display:flex;"
        f"align-items:center;justify-content:center;flex-shrink:0'>C</div>"
        f"<div style='flex:1'>"
        f"<div><span style='font-size:26px;font-weight:800;color:#1B4D6B;line-height:1'>{stats['C']}</span>"
        f"<span style='font-size:13px;font-weight:600;color:#1B4D6B'>&nbsp;개 지역</span></div>"
        f"<div style='font-size:11.5px;font-weight:700;color:#1B4D6B;margin-top:1px'>비위기 + 공급부족</div>"
        f"</div>"
        f"<span style='font-size:20px;opacity:0.4'>&#128309;</span>"
        f"</div>"
        f"<div style='font-size:11.5px;color:#555;line-height:1.55;margin-bottom:9px'>"
        f"도심 성장 지역으로 학생 수는 유지되나 돌봄 시설이 부족. "
        f"<b style='color:#1B4D6B'>신규 시설 확충</b>이 시급한 도시 과밀 지역"
        f"</div>"
        f"<span style='background:#1B4D6B;color:white;font-size:10.5px;font-weight:700;"
        f"padding:3px 11px;border-radius:4px;display:inline-block'>&#127959; 긴급 확충</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with _r2c2:
    st.markdown(
        f"<div style='background:#eaf7ed;border:2px solid #27AE60;border-radius:10px;padding:14px 16px'>"
        f"<div style='display:flex;align-items:flex-start;gap:10px;margin-bottom:8px'>"
        f"<div style='background:#27AE60;color:white;font-size:16px;font-weight:800;"
        f"min-width:34px;height:34px;border-radius:50%;display:flex;"
        f"align-items:center;justify-content:center;flex-shrink:0'>D</div>"
        f"<div style='flex:1'>"
        f"<div><span style='font-size:26px;font-weight:800;color:#27AE60;line-height:1'>{stats['D']}</span>"
        f"<span style='font-size:13px;font-weight:600;color:#27AE60'>&nbsp;개 지역</span></div>"
        f"<div style='font-size:11.5px;font-weight:700;color:#27AE60;margin-top:1px'>비위기 + 균형</div>"
        f"</div>"
        f"<span style='font-size:20px;opacity:0.4'>&#128994;</span>"
        f"</div>"
        f"<div style='font-size:11.5px;color:#555;line-height:1.55;margin-bottom:9px'>"
        f"수요와 공급이 균형을 이루고 있는 안정적 지역. "
        f"현 수준 유지 및 <b style='color:#27AE60'>변화 추이 모니터링</b>으로 관리"
        f"</div>"
        f"<span style='background:#27AE60;color:white;font-size:10.5px;font-weight:700;"
        f"padding:3px 11px;border-radius:4px;display:inline-block'>&#128203; 모니터링</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ════════════════════════════════
# 탭 구성
# ════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺  지도 대시보드",
    "🔍  지역 상세 분석",
    "💰  예산 배분 시뮬레이터",
    "📄  AI 정책 보고서",
])

# ─────────────────────────────
# TAB 1: 지도 대시보드
# ─────────────────────────────
with tab1:
    col_map, col_chart = st.columns([2, 1])

    with col_map:
        st.markdown('<p class="section-header">시군구별 불균형 유형 지도</p>', unsafe_allow_html=True)
        selected = st.session_state.get("selected_id", None)
        m = build_map(df_filtered, selected_id=selected)
        map_data = st_folium(m, width="100%", height=480, returned_objects=["last_object_clicked_popup"])

        if map_data and map_data.get("last_object_clicked_popup"):
            raw = map_data["last_object_clicked_popup"]
            for _, row in df.iterrows():
                if row["name"] in str(raw):
                    st.session_state["selected_id"] = row["region_id"]
                    break

    with col_chart:
        st.markdown('<p class="section-header">유형 분포</p>', unsafe_allow_html=True)
        st.plotly_chart(type_pie(df_filtered), use_container_width=True)

        st.markdown('<p class="section-header">위험 점수 상위 5개</p>', unsafe_allow_html=True)
        top5 = df_filtered.nlargest(5, "risk_score")[["name","region_type","risk_score","imbal_idx"]]
        for _, row in top5.iterrows():
            color = TYPE_INFO[row["region_type"]]["color"]
            st.markdown(f"""
            <div style='padding:6px 10px;margin-bottom:4px;
                border-left:3px solid {color};background:#fafaf7;border-radius:4px'>
              <span style='font-size:12px;font-weight:600'>{row['name']}</span>
              <span style='font-size:11px;color:#888;float:right'>
                위험:{int(row['risk_score'])} | 불균형:{row['imbal_idx']:.2f}
              </span>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────
# TAB 2: 지역 상세 분석
# ─────────────────────────────
with tab2:
    region_names = df["name"].tolist()
    default_idx = 0
    if st.session_state.get("selected_id"):
        sel_name = df[df["region_id"] == st.session_state["selected_id"]]["name"].values
        if len(sel_name): default_idx = region_names.index(sel_name[0])

    selected_name = st.selectbox("분석할 지역 선택", region_names, index=default_idx)
    detail_row = df[df["name"] == selected_name].iloc[0]
    detail = get_region_detail(df, detail_row["region_id"])
    t = detail["type"]
    tinfo = TYPE_INFO[t]

    # 지역 특성 설명
    note = detail_row.get("region_note", "")
    if note:
        st.caption(f"📍 {note}")

    # 유형 배지 + 위험 점수
    col_badge, col_risk = st.columns([3, 1])
    with col_badge:
        st.markdown(
            f"<span class='type-badge' style='background:{tinfo['color']};color:white'>"
            f"{t}형 — {tinfo['label']}</span>"
            f"  <span style='font-size:13px;color:#555'>{tinfo['action']}</span>",
            unsafe_allow_html=True
        )
    with col_risk:
        st.metric("위험 점수", f"{detail['risk_score']} / 100")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="section-header">핵심 지표</p>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        _stu_src = detail.get("students_source", "추정")
        if _stu_src == "NEIS학급기반":
            _stu_help  = (
                "📂 공공데이터 원본 — NEIS Open API (open.neis.go.kr)\n\n"
                "NEIS classInfo(2025학년도) 실측 학급수 "
                "× KEDI 교육통계 2024 시군구별 학급당 학생수"
            )
            _stu_badge = (
                "<span style='background:#e8f4fd;color:#1565c0;font-size:10px;"
                "padding:2px 7px;border-radius:8px;font-weight:700;letter-spacing:0.2px;'>"
                "&#128209; NEIS Open API</span>"
                "<span style='font-size:10px;color:#666;margin-left:5px;'>공공데이터 원본</span>"
            )
        else:
            _stu_help  = "출처: 언론 추정 (경향신문 2024, 시사저널 2025)"
            _stu_badge = (
                "<span style='background:#fff3e0;color:#e65100;font-size:10px;"
                "padding:2px 7px;border-radius:8px;font-weight:600;'>추정</span>"
            )
        m1.metric("초등학생", f"{detail['students']:,}명", help=_stu_help)
        m1.markdown(_stu_badge, unsafe_allow_html=True)
        m2.metric("맞벌이 가구", f"{detail['dual_pct']}%")
        m3.metric("한부모 가구", f"{detail['single_pct']}%")
        m1.metric("돌봄 대기자", f"{detail['waitlist']}명",
                  help="시뮬레이션 추정값 (시군구 단위 공공 미공개)")
        m2.metric("이용률", f"{detail['util_rate']}%",
                  help="실측 이용인원 ÷ 추정 정원")
        m3.metric("인구감소지역", "예" if detail["decline"] else "아니오")
        m1.metric("돌봄교실 이용인원", f"{detail['care_enrolled']:,}명",
                  help="교육부 초등돌봄교실 현황 2023년 4월 기준 실측값")
        m2.metric("돌봄교실 학교 수", f"{detail['school_count']}개교",
                  help="교육부 초등돌봄교실 현황 2023년 4월 기준 실측값")
        m3.metric("합계출산율", f"{detail['birth_rate']}명",
                  help="통계청 2023년 기준 (전국 평균 0.72명)")

        # ── 방과후·지역돌봄 공급 지표
        st.markdown('<p class="section-header">방과후·지역돌봄 공급</p>',
                    unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        as_enr    = detail.get("afterschool_enrolled", 0)
        as_src    = detail.get("afterschool_source", "추정")
        ce_enr    = detail.get("custom_edu_enrolled", 0)
        # 통합 실질 공급 = 돌봄교실(×1.0) + 방과후학교(×0.35) + 지역돌봄기관(×0.40)
        total_eff = int(detail["care_enrolled"] + as_enr * 0.35 + ce_enr * 0.40)

        # 방과후학교 참여 — 출처 배지 표시
        if as_src == "NEIS기반추정":
            _as_badge = "<span style='background:#e8f4fd;color:#1565c0;border:1px solid #90caf9;" \
                        "font-size:9px;font-weight:700;padding:1px 6px;border-radius:3px;" \
                        "margin-left:4px'>NEIS 기반</span>"
        elif as_src == "NEIS실측":
            _as_badge = "<span style='background:#e3f0e8;color:#256336;border:1px solid #b2d9bc;" \
                        "font-size:9px;font-weight:700;padding:1px 6px;border-radius:3px;" \
                        "margin-left:4px'>NEIS 실측</span>"
        else:
            _as_badge = "<span style='background:#fff4e5;color:#a04e00;border:1px solid #f5c97a;" \
                        "font-size:9px;font-weight:700;padding:1px 6px;border-radius:3px;" \
                        "margin-left:4px'>추정</span>"

        s1.markdown(
            f"<div style='font-size:11px;color:#888;margin-bottom:2px'>방과후학교 참여{_as_badge}</div>"
            f"<div style='font-size:22px;font-weight:700'>{as_enr:,}명</div>",
            unsafe_allow_html=True,
        )
        s2.metric("지역돌봄기관 참여", f"{ce_enr:,}명",
                  help="지역아동센터·아이돌봄서비스·드림스타트 등 지역사회 돌봄기관 참여 추정")
        s3.metric("통합 실질 공급", f"{total_eff:,}명",
                  help="돌봄교실(×1.0) + 방과후학교(×0.35) + 지역돌봄기관(×0.40) 가중합산")

        if as_src == "NEIS기반추정":
            st.caption(
                "📡 방과후학교는 **NEIS 실측 학교 수** 기반 계산 | "
                "지역돌봄기관은 통계 기반 추정 | "
                "**통합 실질 공급** = 돌봄교실 + 방과후학교(×0.35) + 지역돌봄기관(×0.40)"
            )
        elif as_src == "NEIS실측":
            st.caption(
                "✅ 방과후학교는 **NEIS 직접 조회 실측값** | "
                "지역돌봄기관은 통계 기반 추정 | "
                "**통합 실질 공급** = 돌봄교실 + 방과후학교(×0.35) + 지역돌봄기관(×0.40)"
            )
        else:
            st.caption(
                "💡 방과후학교·지역돌봄기관 수치는 전국 참여율 통계 기반 추정값입니다. "
                "NEIS API 키로 `data/fetch_neis_afterschool.py`를 실행하면 NEIS 기반 계산으로 전환됩니다."
            )

        # 데이터 출처 표기
        if detail.get("data_note"):
            st.caption(f"📊 출처: {detail['data_note']}")

        st.markdown('<p class="section-header">불균형 지수</p>', unsafe_allow_html=True)
        st.plotly_chart(imbal_gauge(detail["imbal_idx"], t), use_container_width=True)
        balance_label = (
            "공급 부족" if detail['imbal_idx'] > 1.2
            else "공급 과잉" if detail['imbal_idx'] < 0.8
            else "균형"
        )
        st.caption(
            f"수요 지수 {detail['demand_idx']:.3f} ÷ 복합 공급 지수 {detail['supply_idx']:.3f}"
            f" = **{detail['imbal_idx']:.3f}** ({balance_label})\n\n"
            f"공급 지수 = (돌봄교실×1.0 + 방과후학교×0.35 + 지역돌봄기관×0.40) ÷ 초등학생 수"
        )

    with col_r:
        st.markdown('<p class="section-header">5년 후 수요 예측</p>', unsafe_allow_html=True)
        pred = predict_region(detail_row, reg, scaler)
        st.plotly_chart(
            demand_forecast_chart(detail["demand_idx"], pred["demand_5y"], detail["name"]),
            use_container_width=True
        )
        trend_color = "#C0392B" if pred["trend"]=="증가" else "#27AE60"
        st.markdown(
            f"예측 결과: 수요 지수 <b style='color:{trend_color}'>{pred['trend']} "
            f"{abs(pred['change_pct'])}%</b> (현재 {detail['demand_idx']:.3f} → "
            f"5년 후 {pred['demand_5y']:.3f})",
            unsafe_allow_html=True
        )

        st.markdown('<p class="section-header">위험 요인 Top 3</p>', unsafe_allow_html=True)
        for i, f in enumerate(detail["top3"], 1):
            st.markdown(
                f"<div style='padding:6px 10px;margin-bottom:4px;"
                f"border-left:3px solid {tinfo['color']};background:#fafaf7;border-radius:4px'>"
                f"<b>{i}.</b> {f}</div>",
                unsafe_allow_html=True
            )

        st.markdown('<p class="section-header">변수 중요도</p>', unsafe_allow_html=True)
        fi = get_feature_importance(reg)
        st.plotly_chart(importance_bar(fi), use_container_width=True)

# ─────────────────────────────
# TAB 3: 예산 배분 시뮬레이터
# ─────────────────────────────
with tab3:

    # ① 예산 설정 ─────────────────────────────────────
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        budget = st.slider("총 배분 가능 예산 (억원)", 5, 200, 50, step=5)
    with col_s2:
        st.metric("총 예산", f"{budget}억원", f"= {budget/10:.0f}십억원")

    # 배분 원칙 컬러 카드
    st.markdown("""
<div style='display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 18px 0'>
  <div style='flex:1;min-width:110px;background:#fdecea;
       border-left:4px solid #C0392B;border-radius:6px;padding:9px 12px'>
    <div style='font-size:13px;font-weight:700;color:#C0392B'>A형 &nbsp;45%</div>
    <div style='font-size:11px;color:#666;margin-top:2px'>위기+공급부족<br>긴급 개입</div>
  </div>
  <div style='flex:1;min-width:110px;background:#eaf0f7;
       border-left:4px solid #1B4D6B;border-radius:6px;padding:9px 12px'>
    <div style='font-size:13px;font-weight:700;color:#1B4D6B'>C형 &nbsp;40%</div>
    <div style='font-size:11px;color:#666;margin-top:2px'>비위기+공급부족<br>긴급 확충</div>
  </div>
  <div style='flex:1;min-width:110px;background:#eaf7ed;
       border-left:4px solid #27AE60;border-radius:6px;padding:9px 12px'>
    <div style='font-size:13px;font-weight:700;color:#27AE60'>D형 &nbsp;10%</div>
    <div style='font-size:11px;color:#666;margin-top:2px'>비위기+균형<br>모니터링</div>
  </div>
  <div style='flex:1;min-width:110px;background:#fef3e8;
       border-left:4px solid #E67E22;border-radius:6px;padding:9px 12px'>
    <div style='font-size:13px;font-weight:700;color:#E67E22'>B형 &nbsp;&nbsp;5%</div>
    <div style='font-size:11px;color:#666;margin-top:2px'>위기+공급과잉<br>구조 전환</div>
  </div>
</div>
""", unsafe_allow_html=True)

    result = simulate_budget(df_filtered, budget)

    # ② Before / After 핵심 지표 ──────────────────────
    avg_before  = result["imbal_before"].mean()
    avg_after   = result["imbal_after"].mean()
    improve_pct = abs(avg_before - avg_after) / avg_before * 100
    n_improved  = int(sum(
        abs(r["imbal_after"] - 1.0) < abs(r["imbal_before"] - 1.0)
        for _, r in result.iterrows()
    ))
    ac_budget        = result[result["region_type"].isin(["A","C"])]["allocated_억"].sum()
    total_new_slots  = int(result["new_care_slots"].sum())
    total_children   = int(result["children_added"].sum())
    # 개선폭 = 균형(1.0)까지의 거리 감소량 — B형(공급과잉→균형)도 양수로 잡힘
    result["개선폭"] = (
        (result["imbal_before"] - 1.0).abs() - (result["imbal_after"] - 1.0).abs()
    ).round(4)
    best = result.nlargest(1, "개선폭").iloc[0]

    st.markdown('<p class="section-header">📊 배분 효과 요약</p>', unsafe_allow_html=True)

    # 배분 전 / 화살표 / 배분 후
    col_bef, col_arr, col_aft = st.columns([11, 2, 11])
    with col_bef:
        st.markdown("""
<div style='background:#f4f4f2;border-radius:8px;padding:10px 14px 4px;
     border-top:3px solid #bbb;margin-bottom:10px'>
  <span style='font-size:11px;font-weight:700;color:#999'>⬤ &nbsp;배분 전</span>
</div>""", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        b1.metric("평균 불균형 지수", f"{avg_before:.3f}")
        b2.metric("공급부족 지역", f"{stats['A']+stats['C']}개",
                  help="A형(위기+공급부족) + C형(비위기+공급부족)")

    with col_arr:
        st.markdown(
            "<div style='text-align:center;font-size:28px;font-weight:700;"
            "color:#27AE60;padding-top:38px'>→</div>",
            unsafe_allow_html=True)

    with col_aft:
        st.markdown("""
<div style='background:#eaf7ed;border-radius:8px;padding:10px 14px 4px;
     border-top:3px solid #27AE60;margin-bottom:10px'>
  <span style='font-size:11px;font-weight:700;color:#27AE60'>◯ &nbsp;배분 후 (예측)</span>
</div>""", unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        a1.metric("평균 불균형 지수", f"{avg_after:.3f}",
                  f"▼ {improve_pct:.1f}% 개선")
        a2.metric("균형 접근 지역", f"{n_improved}개",
                  f"전체 {len(result)}개 중")

    # 보조 KPI 4개
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("A·C형 집중 투자",   f"{ac_budget:.1f}억원",
              f"전체 예산의 {ac_budget/budget*100:.0f}%")
    k2.metric("신규 확보 정원",    f"{total_new_slots:,}명",
              help="A·C·D형 예산으로 새로 확보되는 돌봄 정원 합계 (1억원 ≈ 80명)")
    k3.metric("추가 수혜 아동",    f"{total_children:,}명",
              help="신규 정원에 실질적으로 배치될 아동 수 추정 (B형 구조전환 제외)")
    k4.metric("최대 수혜 지역",    best["name"],
              f"불균형 {best['imbal_before']:.2f} → {best['imbal_after']:.2f}")

    st.divider()

    # ③ 덤벨 차트 + 배분 비율 도넛 ───────────────────
    col_chart, col_pie = st.columns([3, 1])

    with col_chart:
        st.plotly_chart(budget_dumbbell(result), use_container_width=True)

    with col_pie:
        alloc  = result.groupby("region_type")["allocated_억"].sum().reset_index()
        alloc["label"] = alloc["region_type"].map(lambda t: f"{t}형")
        _colors = [TYPE_INFO[t]["color"] for t in alloc["region_type"]]
        import plotly.graph_objects as _go
        fig_pie = _go.Figure(_go.Pie(
            labels=alloc["label"],
            values=alloc["allocated_억"],
            marker_colors=_colors,
            hole=0.45,
            textinfo="percent+label",
            textfont_size=12,
        ))
        fig_pie.update_layout(
            title=dict(text="유형별<br>예산 배분", font_size=12),
            height=260,
            margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # 유형별 배분 요약 텍스트
        for _, ar in alloc.sort_values("allocated_억", ascending=False).iterrows():
            c = TYPE_INFO[ar["region_type"]]["color"]
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:12px;padding:2px 4px'>"
                f"<span style='color:{c};font-weight:700'>{ar['label']}</span>"
                f"<span>{ar['allocated_억']:.1f}억원</span></div>",
                unsafe_allow_html=True,
            )

    # ④ 지역별 배분 내역 테이블 ────────────────────────
    st.markdown('<p class="section-header">지역별 배분 내역</p>', unsafe_allow_html=True)
    show_cols = ["name","region_type","type_label","risk_score","care_waitlist",
                 "allocated_억","imbal_before","imbal_after","개선폭",
                 "new_care_slots","children_added","cost_per_child_만원"]
    disp = result[show_cols].copy()
    disp.columns = ["지역명","유형","유형설명","위험점수","대기아동",
                    "배분(억)","배분전 불균형","배분후 불균형","균형접근도",
                    "신규정원","수혜아동","아동당단가(만원)"]
    _impr_max = float(result["개선폭"].clip(lower=0).max()) or 1.0
    st.dataframe(
        disp,
        use_container_width=True,
        column_config={
            "위험점수": st.column_config.ProgressColumn(
                "위험점수", min_value=0, max_value=100, format="%d",
            ),
            "균형접근도": st.column_config.ProgressColumn(
                "균형접근도", min_value=0, max_value=_impr_max, format="%.4f",
                help="|배분전-1.0| - |배분후-1.0|: 클수록 균형에 많이 접근",
            ),
        },
    )
    st.caption(
        "💡 **신규정원**: A·C형 예산으로 확보되는 신규 돌봄 정원 (1억원 ≈ 80명, A형×2.2배 효율, C형×1.8배 효율) &nbsp;|&nbsp; "
        "**균형접근도**: 불균형지수가 균형(1.0)에 가까워진 정도 — B형(공급과잉)도 구조전환으로 균형 접근 시 양수"
    )

# ─────────────────────────────
# TAB 4: AI 정책 보고서
# ─────────────────────────────
with tab4:
    col_sel, col_cost = st.columns([3, 1])
    with col_sel:
        ai_region = st.selectbox("보고서 생성할 지역 선택", df["name"].tolist(), key="ai_sel")
    with col_cost:
        ai_row = df[df["name"] == ai_region].iloc[0]
        ai_detail = get_region_detail(df, ai_row["region_id"])
        cost = estimate_cost(ai_detail)
        st.metric("예상 API 비용", f"₩{cost['cost_krw']:,}", f"${cost['cost_usd']:.4f}")

    # 지역 요약 카드
    t = ai_detail["type"]
    tinfo = TYPE_INFO[t]
    st.markdown(f"""
    <div style='background:#f5f2ec;border-radius:10px;padding:14px 18px;margin:10px 0'>
      <span style='color:{tinfo["color"]};font-weight:700;font-size:15px'>[{t}형] {ai_region}</span>
      <span style='margin-left:12px;font-size:13px;color:#555'>{tinfo["label"]} — {tinfo["action"]}</span><br>
      <span style='font-size:12px;color:#777'>
        불균형 지수: {ai_detail["imbal_idx"]:.2f} | 대기자: {ai_detail["waitlist"]}명 |
        위험점수: {ai_detail["risk_score"]} | 인구감소지역: {"예" if ai_detail["decline"] else "아니오"}
      </span>
    </div>""", unsafe_allow_html=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning(
            "ANTHROPIC_API_KEY가 설정되지 않았습니다. "
            "`.env` 파일에 API 키를 입력하거나 환경변수로 설정해주세요.",
            icon="⚠️"
        )
        with st.expander("API 키 설정 방법"):
            st.code("""
# 방법 1: .env 파일 생성
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 방법 2: 환경변수 직접 설정 (터미널)
export ANTHROPIC_API_KEY=sk-ant-...

# 방법 3: Streamlit secrets.toml
# .streamlit/secrets.toml 파일에:
# ANTHROPIC_API_KEY = "sk-ant-..."
            """)
    else:
        if st.button("📄 정책 보고서 생성 (Claude AI)", type="primary", use_container_width=True):
            with st.spinner("Claude API 분석 중..."):
                report_placeholder = st.empty()
                full_text = ""
                try:
                    for chunk in generate_report(ai_detail, stream=True):
                        full_text += chunk
                        report_placeholder.markdown(full_text + "▌")
                    report_placeholder.markdown(full_text)
                    st.session_state[f"report_{ai_region}"] = full_text
                    st.success(f"생성 완료 | 사용 토큰 약 {cost['input_tokens']+cost['output_tokens']}개")
                except Exception as e:
                    st.error(f"API 오류: {e}")

        # 이전 생성 보고서 표시
        cached = st.session_state.get(f"report_{ai_region}")
        if cached:
            st.markdown("---")
            st.markdown("**생성된 정책 제안서**")
            st.markdown(cached)
            st.download_button(
                "📥 텍스트 저장",
                data=cached.encode("utf-8"),
                file_name=f"정책보고서_{ai_region}.txt",
                mime="text/plain",
            )

# ══════════════════════════════════════════════════════════════
# 데이터 출처 & 라이선스 섹션
# ══════════════════════════════════════════════════════════════
_CARD_BASE = "background:white;border-radius:12px;padding:16px;border:1px solid #e2e8f0;box-shadow:0 2px 8px rgba(0,0,0,0.05);"
_CC_BADGE  = ('<div style="background:linear-gradient(135deg,#1b6b1b 0%,#3aaa35 100%);border-radius:9px;padding:9px 12px;display:flex;align-items:center;gap:10px">'
              '<div style="width:28px;height:28px;background:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:900;color:#1b6b1b;flex-shrink:0;letter-spacing:-0.5px">CC</div>'
              '<div><div style="font-size:10.5px;font-weight:700;color:white;line-height:1.3">공공누리 제1유형</div>'
              '<div style="font-size:9px;color:rgba(255,255,255,0.75);line-height:1.4">출처표시 · 자유이용</div></div></div>')
_REAL_BADGE = '<span style="background:#dcfce7;color:#166534;border:1px solid #86efac;font-size:9px;font-weight:700;padding:2px 8px;border-radius:4px;letter-spacing:0.5px">실 측</span>'
_EST_BADGE  = '<span style="background:#fef9c3;color:#854d0e;border:1px solid #fde047;font-size:9px;font-weight:700;padding:2px 8px;border-radius:4px;letter-spacing:0.5px">추 정</span>'

def _lic_card(color, icon_label, badge, name, items, meta, bottom):
    return (f'<div class="dc-card" style="border-top:4px solid {color};">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">'
            f'<div style="font-size:13.5px;font-weight:700;color:{color}">{icon_label}</div>{badge}</div>'
            f'<div style="font-size:13px;font-weight:700;color:#1e293b;margin-bottom:5px;line-height:1.35">{name}</div>'
            f'<div style="font-size:11px;color:#64748b;margin-bottom:10px;line-height:1.6">{items}</div>'
            f'<div style="font-size:10.5px;color:#94a3b8;margin-bottom:14px">{meta}</div>'
            f'{bottom}</div>')

_cards = (
    _lic_card("#1565C0", "🏛 교육부",      _REAL_BADGE, "초등돌봄교실 현황",       "이용인원 · 돌봄 학교 수",      "📅 2023년 4월 · 공공데이터포털", _CC_BADGE) +
    _lic_card("#7B1FA2", "📊 통계청",      _REAL_BADGE, "지역별고용조사 · 출생통계", "맞벌이 가구 비율 · 합계출산율", "📅 2023년 하반기 · KOSIS",       _CC_BADGE) +
    _lic_card("#B71C1C", "🏠 행정안전부",   _REAL_BADGE, "인구감소지역 지정 현황",    "전남 16개 군 지정 고시",       "📅 고시 제2024-15호",             _CC_BADGE) +
    _lic_card("#1B4D6B", "📡 NEIS Open API",_REAL_BADGE,"학교기본정보 · 학급정보",   "초등학생 수 (시군구별)",        "📅 2025학년도 · open.neis.go.kr",_CC_BADGE) +
    _lic_card("#d97706", "🔬 분석 산출값",  _EST_BADGE,  "돌봄 정원 · 대기아동",     "수요·공급·불균형 지수 산출",   "🔩 공공데이터 기반 역산 적용",
              '<div style="background:linear-gradient(135deg,#92400e 0%,#d97706 100%);border-radius:9px;padding:9px 12px;display:flex;align-items:center;gap:10px">'
              '<div style="width:28px;height:28px;background:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0">🔬</div>'
              '<div><div style="font-size:10.5px;font-weight:700;color:white;line-height:1.3">자체 분석 산출값</div>'
              '<div style="font-size:9px;color:rgba(255,255,255,0.75);line-height:1.4">공공데이터 기반 역산·추정</div></div></div>')
)

st.markdown("<div style='margin-top:48px'></div>", unsafe_allow_html=True)
st.markdown(f'<div style="background:linear-gradient(135deg,#f8fafc 0%,#eef3fb 50%,#f0fdf4 100%);border-radius:20px;padding:32px 36px 28px 36px;border:1px solid #dde6f0;box-shadow:0 4px 24px rgba(27,77,107,0.07)"><div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:14px;margin-bottom:6px"><div><div style="font-size:10px;font-weight:700;letter-spacing:2.5px;color:#94a3b8;text-transform:uppercase;margin-bottom:6px">DATA TRANSPARENCY</div><div style="font-size:21px;font-weight:800;letter-spacing:-0.5px;background:linear-gradient(135deg,#1B4D6B 0%,#2980B9 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.2">활용 데이터 및 라이선스</div><div style="font-size:12.5px;color:#64748b;margin-top:6px;font-weight:400">본 서비스는 공공누리 제1유형 공공데이터를 활용합니다</div></div><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:4px"><div style="background:#1B4D6B;color:white;font-size:11.5px;font-weight:700;padding:7px 16px;border-radius:20px;white-space:nowrap;box-shadow:0 2px 8px rgba(27,77,107,0.3)">📦 5개 데이터셋</div><div style="background:linear-gradient(135deg,#1b6b1b,#3aaa35);color:white;font-size:11.5px;font-weight:700;padding:7px 16px;border-radius:20px;white-space:nowrap;box-shadow:0 2px 8px rgba(58,170,53,0.3)">🄍 공공누리 제1유형</div></div></div><div style="height:1px;background:linear-gradient(to right,#cbd5e1,transparent);margin:20px 0 16px 0"></div><div style="display:flex;gap:20px;margin-bottom:22px;flex-wrap:wrap;align-items:center"><div style="display:flex;align-items:center;gap:7px"><div style="width:9px;height:9px;background:#3aaa35;border-radius:50%"></div><span style="font-size:12px;color:#475569;font-weight:500">공공데이터 원본 (실측)</span></div><div style="display:flex;align-items:center;gap:7px"><div style="width:9px;height:9px;background:#d97706;border-radius:50%"></div><span style="font-size:12px;color:#475569;font-weight:500">자체 분석 산출값 (추정)</span></div></div><div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(188px,1fr));gap:14px;margin-bottom:22px">{_cards}</div><div style="background:white;border-radius:12px;padding:16px 20px;border:1px solid #dde6f0;display:flex;align-items:flex-start;gap:14px;box-shadow:0 1px 6px rgba(0,0,0,0.04)"><div style="width:36px;height:36px;flex-shrink:0;background:linear-gradient(135deg,#1b6b1b,#3aaa35);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:900;color:white;letter-spacing:-0.5px">CC</div><div style="font-size:12px;color:#475569;line-height:1.8"><strong style="color:#1e293b">공공누리 제1유형</strong>이란, 출처만 표시하면 상업적 이용 및 변형이 자유로운 가장 개방적인 공공데이터 라이선스입니다. 라이선스 원문 기준: 공공저작물 자유이용허락 표준라이선스.<br><span style="font-size:11px;color:#94a3b8">원본 데이터 제공처 &nbsp;·&nbsp; <a href="https://data.go.kr" target="_blank" style="color:#1B4D6B;font-weight:600;text-decoration:none">공공데이터포털(data.go.kr)</a> &nbsp;·&nbsp; <a href="https://open.neis.go.kr" target="_blank" style="color:#1B4D6B;font-weight:600;text-decoration:none">NEIS Open API</a> &nbsp;·&nbsp; <a href="https://kosis.kr" target="_blank" style="color:#1B4D6B;font-weight:600;text-decoration:none">국가통계포털(KOSIS)</a> &nbsp;·&nbsp; <a href="https://www.schoolinfo.go.kr" target="_blank" style="color:#1B4D6B;font-weight:600;text-decoration:none">학교알리미</a></span></div></div></div>', unsafe_allow_html=True)
