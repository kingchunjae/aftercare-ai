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
    type_pie, budget_bar, importance_bar
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
</style>
""", unsafe_allow_html=True)

# ── 데이터 로드 & 모델 초기화
# cache_version: 컬럼 구조가 바뀔 때 올려서 Streamlit Cloud 캐시 강제 무효화
@st.cache_data
def load(cache_version: int = 4):
    df = load_data()
    return df

@st.cache_resource
def init_models(df):
    ensure_trained(df)
    return load_models()

df = load(cache_version=4)
reg, clf, scaler = init_models(df)

# ── 사이드바
with st.sidebar:
    st.title("🗺 돌봄 AI 진단")
    st.caption("방과후·초등돌봄 수요-공급 불균형 진단 시스템")
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
    st.markdown("""
<div class="ds-wrap">
  <div class="ds-title">📊 데이터 출처</div>

  <div class="ds-legend">
    <span><span class="badge-real">실&nbsp;측</span> 공공데이터 원본</span>
    <span><span class="badge-est">추&nbsp;정</span> 역산·시뮬레이션</span>
  </div>

  <div class="ds-card">
    <div class="ds-card-top">
      <span class="ds-agency">🏛 교육부</span>
      <span class="badge-real">실&nbsp;측</span>
    </div>
    <div class="ds-items">이용인원 · 돌봄 학교 수</div>
    <div class="ds-pub">초등돌봄교실 현황<br>공공데이터포털 (2023.04 기준)</div>
  </div>

  <div class="ds-card">
    <div class="ds-card-top">
      <span class="ds-agency">📈 통계청</span>
      <span class="badge-real">실&nbsp;측</span>
    </div>
    <div class="ds-items">맞벌이 가구 비율 · 합계출산율</div>
    <div class="ds-pub">지역별고용조사 2023 하반기<br>시군구별 출생통계 2023</div>
  </div>

  <div class="ds-card">
    <div class="ds-card-top">
      <span class="ds-agency">🏢 행정안전부</span>
      <span class="badge-real">실&nbsp;측</span>
    </div>
    <div class="ds-items">인구감소지역 지정 현황</div>
    <div class="ds-pub">고시 제2024-15호 (전남 16개 군)</div>
  </div>

  <div class="ds-card">
    <div class="ds-card-top">
      <span class="ds-agency">📰 교육통계·언론</span>
      <span class="badge-real">실&nbsp;측</span>
    </div>
    <div class="ds-items">초등학생 수 (시군구별)</div>
    <div class="ds-pub">경향신문 2024.02<br>시사저널 2025</div>
  </div>

  <div class="ds-card est">
    <div class="ds-card-top">
      <span class="ds-agency est">🔧 역산 추정</span>
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
</div>
""", unsafe_allow_html=True)

# ── 메인 헤더
st.title("🏫 방과후·초등돌봄 수요-공급 불균형 AI 진단")
st.caption("교육 공공데이터 기반 지역소멸 위기 연계 분석 | 광역 통합 행정 시뮬레이션 (27개 시군구)")

# ── 히어로 지표
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("A형 (최우선 개입)", f"{stats['A']}개",
              help="소멸위기 + 공급부족 — 긴급 자원 투입 필요")
with c2:
    st.metric("C형 (긴급 확충)", f"{stats['C']}개",
              help="비소멸 + 공급부족 — 도시 과밀 지역")
with c3:
    st.metric("B형 (구조 전환)", f"{stats['B']}개",
              help="소멸위기 + 공급과잉 — 시설 복합 활용")
with c4:
    st.metric("D형 (모니터링)", f"{stats['D']}개",
              help="현재 균형 상태")

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
        m1.metric("초등학생", f"{detail['students']:,}명")
        m2.metric("맞벌이 가구", f"{detail['dual_pct']}%")
        m3.metric("한부모 가구", f"{detail['single_pct']}%")
        m1.metric("돌봄 대기자", f"{detail['waitlist']}명",
                  help="시뮬레이션 추정값 (시군구 단위 공공 미공개)")
        m2.metric("이용률", f"{detail['util_rate']}%",
                  help="실측 이용인원 ÷ 추정 정원")
        m3.metric("인구감소지역", "예" if detail["decline"] else "아니오")
        m1.metric("실측 이용인원", f"{detail['care_enrolled']:,}명",
                  help="교육부 초등돌봄교실 현황 2023년 4월 기준 실측값")
        m2.metric("돌봄교실 학교 수", f"{detail['school_count']}개교",
                  help="교육부 초등돌봄교실 현황 2023년 4월 기준 실측값")
        m3.metric("합계출산율", f"{detail['birth_rate']}명",
                  help="통계청 2023년 기준 (전국 평균 0.72명)")

        # 데이터 출처 표기
        if detail.get("data_note"):
            st.caption(f"📊 출처: {detail['data_note']}")

        st.markdown('<p class="section-header">불균형 지수</p>', unsafe_allow_html=True)
        st.plotly_chart(imbal_gauge(detail["imbal_idx"], t), use_container_width=True)
        st.caption(
            f"수요 지수 {detail['demand_idx']:.3f} ÷ 공급 지수 {detail['supply_idx']:.3f}"
            f" = **{detail['imbal_idx']:.3f}** "
            f"({'공급 부족' if detail['imbal_idx'] > 1.2 else '공급 과잉' if detail['imbal_idx'] < 0.8 else '균형'})"
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
    st.markdown('<p class="section-header">예산 입력</p>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        budget = st.slider("총 배분 가능 예산 (억원)", 5, 200, 50, step=5)
    with col_s2:
        st.metric("총 예산", f"{budget}억원", f"= {budget/10:.0f}십억원")

    st.markdown(
        "**배분 기준**: A형 45% → C형 40% → D형 10% → B형 5%  "
        "(위험 점수·유형별 우선순위 반영)",
        unsafe_allow_html=True
    )

    result = simulate_budget(df_filtered, budget)

    # 파이 차트 (배분 비율)
    col_pie, col_bar = st.columns([1, 2])
    with col_pie:
        alloc = result.groupby("region_type")["allocated_억"].sum().reset_index()
        alloc["label"] = alloc["region_type"].map(lambda t: f"{t}형")
        colors = [TYPE_INFO[t]["color"] for t in alloc["region_type"]]
        fig_pie = __import__("plotly.graph_objects", fromlist=["Figure"]).Figure(
            __import__("plotly.graph_objects", fromlist=["Pie"]).Pie(
                labels=alloc["label"], values=alloc["allocated_억"],
                marker_colors=colors, hole=0.4,
                textinfo="percent+label", textfont_size=12
            )
        )
        fig_pie.update_layout(
            title="유형별 예산 배분 비율",
            height=280, margin=dict(l=10,r=10,t=45,b=10),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.plotly_chart(budget_bar(result), use_container_width=True)

    st.markdown('<p class="section-header">지역별 배분 내역</p>', unsafe_allow_html=True)
    show_cols = ["name","region_type","type_label","risk_score","care_waitlist","allocated_억","imbal_before","imbal_after"]
    disp = result[show_cols].copy()
    disp.columns = ["지역명","유형","유형설명","위험점수","대기아동","배분(억)","배분전 불균형","배분후 불균형"]
    st.dataframe(
        disp,
        use_container_width=True,
        column_config={
            "위험점수": st.column_config.ProgressColumn(
                "위험점수",
                min_value=0,
                max_value=100,
                format="%d",
            )
        }
    )

    # 효과 요약
    avg_before = result["imbal_before"].mean()
    avg_after  = result["imbal_after"].mean()
    st.success(
        f"예산 배분 후 평균 불균형 지수: {avg_before:.3f} → {avg_after:.3f} "
        f"({abs(avg_before - avg_after) / avg_before * 100:.1f}% 개선 예상)"
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
