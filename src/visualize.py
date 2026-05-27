"""src/visualize.py — folium 지도 + plotly 차트"""
import pandas as pd, numpy as np
import folium, plotly.graph_objects as go, plotly.express as px

TYPE_COLORS = {"A":"#C0392B","B":"#E67E22","C":"#1B4D6B","D":"#27AE60"}
TYPE_LABELS = {"A":"위기+부족","B":"위기+과잉","C":"비위기+부족","D":"비위기+균형"}

# ── 1. 메인 지도
def build_map(df: pd.DataFrame, selected_id: str = None) -> folium.Map:
    m = folium.Map(
        location=[35.0, 127.0], zoom_start=8,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )
    for _, row in df.iterrows():
        t = row["region_type"]
        color = TYPE_COLORS[t]
        is_sel = (row["region_id"] == selected_id)
        radius = 14 if is_sel else 10
        popup_html = f"""
        <div style='font-family:sans-serif;min-width:160px'>
          <b style='color:{color}'>[{t}형] {row["name"]}</b><br>
          <span style='font-size:12px'>
            불균형지수: <b>{row["imbal_idx"]:.2f}</b><br>
            대기자: <b>{int(row["care_waitlist"])}명</b><br>
            위험점수: <b>{int(row["risk_score"])}</b>
          </span>
        </div>"""
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85 if is_sel else 0.70,
            weight=3 if is_sel else 1.5,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{row['name']} ({t}형)",
        ).add_to(m)

    # 범례
    legend = """
    <div style='position:fixed;bottom:20px;left:20px;background:white;
    padding:10px 14px;border-radius:8px;border:1px solid #ddd;
    font-family:sans-serif;font-size:12px;z-index:999'>
      <b style='font-size:13px'>유형 범례</b><br>
      <span style='color:#C0392B'>●</span> A형 위기+부족<br>
      <span style='color:#E67E22'>●</span> B형 위기+과잉<br>
      <span style='color:#1B4D6B'>●</span> C형 비위기+부족<br>
      <span style='color:#27AE60'>●</span> D형 비위기+균형
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m

# ── 2. 불균형 게이지
def imbal_gauge(imbal: float, t: str) -> go.Figure:
    color = TYPE_COLORS[t]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(imbal, 2),
        number={"font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 4], "tickwidth": 1, "tickcolor": "#aaa"},
            "bar":  {"color": color, "thickness": 0.35},
            "bgcolor": "#f9f9f7",
            "steps": [
                {"range": [0, 0.8],  "color": "#EAF3DE"},
                {"range": [0.8, 1.2],"color": "#FFF8E1"},
                {"range": [1.2, 4],  "color": "#FDECEA"},
            ],
            "threshold": {
                "line": {"color": "#555", "width": 2},
                "thickness": 0.75, "value": 1.0,
            },
        },
        title={"text": "불균형 지수", "font": {"size": 14}},
    ))
    fig.update_layout(
        height=200, margin=dict(l=20,r=20,t=30,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ── 3. 5년 후 수요 예측 차트
def demand_forecast_chart(current: float, predicted: float, name: str) -> go.Figure:
    years = [2023, 2024, 2025, 2026, 2027, 2028]
    # 선형 보간
    vals = [round(current + (predicted - current) * i / 5, 4) for i in range(6)]
    color = "#C0392B" if predicted > current * 1.05 else ("#27AE60" if predicted < current * 0.95 else "#1B4D6B")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=vals, mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=7, color=color),
        fill="tozeroy", fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba"),
        name="수요 지수",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888",
                  annotation_text="균형 기준(1.0)", annotation_position="top right")
    fig.update_layout(
        title=dict(text=f"{name} — 5년 수요 예측", font_size=13),
        xaxis=dict(title="연도", tickmode="array", tickvals=years),
        yaxis=dict(title="수요 지수"),
        height=280, margin=dict(l=40,r=20,t=45,b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig

# ── 4. 4유형 분포 파이차트
def type_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["region_type"].value_counts().reset_index()
    counts.columns = ["type","count"]
    counts["label"] = counts["type"].map(lambda t: f"{t}형 {TYPE_LABELS[t]}")
    colors = [TYPE_COLORS[t] for t in counts["type"]]
    fig = go.Figure(go.Pie(
        labels=counts["label"], values=counts["count"],
        marker_colors=colors,
        hole=0.4,
        textinfo="percent+label",
        textfont_size=11,
    ))
    fig.update_layout(
        title=dict(text="27개 시군구 유형 분포", font_size=14),
        showlegend=False,
        height=300, margin=dict(l=10,r=10,t=45,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── 5. 예산 배분 비교 막대 (레거시 — budget_dumbbell로 대체)
def budget_bar(result_df: pd.DataFrame) -> go.Figure:
    top10 = result_df.head(10)
    colors = [TYPE_COLORS[t] for t in top10["region_type"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top10["name"], y=top10["imbal_before"],
        name="배분 전", marker_color=colors, opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        x=top10["name"], y=top10["imbal_after"],
        name="배분 후", marker_color=colors, opacity=0.45,
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#555",
                  annotation_text="균형 기준")
    fig.update_layout(
        barmode="group",
        title=dict(text="예산 배분 전후 불균형 지수 비교 (위험도 상위 10개)", font_size=13),
        xaxis_tickangle=-30,
        height=320, margin=dict(l=40,r=20,t=50,b=80),
        legend=dict(orientation="h", y=1.05),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── 5-b. 배분 전후 덤벨 차트 (메인 시각화)
def budget_dumbbell(result_df: pd.DataFrame) -> go.Figure:
    """전 지역 배분 전후 불균형 지수 덤벨 차트.

    ● (채워진 원) = 배분 전, ○ (빈 원) = 배분 후.
    선 색: 균형(1.0)에 가까워진 지역 → 초록, 그 외 → 회색.
    """
    # 배분 전 값 내림차순 → 불균형 심한 지역이 위에 오도록
    df = result_df.sort_values("imbal_before", ascending=True).copy()
    type_colors = [TYPE_COLORS[t] for t in df["region_type"]]

    fig = go.Figure()

    # ── ① 연결선 (before → after)
    for _, row in df.iterrows():
        closer = abs(row["imbal_after"] - 1.0) < abs(row["imbal_before"] - 1.0)
        fig.add_trace(go.Scatter(
            x=[row["imbal_before"], row["imbal_after"]],
            y=[row["name"], row["name"]],
            mode="lines",
            line=dict(color="#27AE60" if closer else "#d5cfc5", width=2.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── ② 배분 전 (채워진 원, 유형 색상)
    fig.add_trace(go.Scatter(
        x=df["imbal_before"],
        y=df["name"],
        mode="markers",
        name="배분 전 (●)",
        marker=dict(
            size=14,
            color=type_colors,
            symbol="circle",
            line=dict(width=1.5, color="white"),
        ),
        customdata=np.column_stack([
            df["imbal_before"].round(3),
            df["imbal_after"].round(3),
            df["region_type"],
            df["allocated_억"].round(1),
            (df["imbal_before"] - df["imbal_after"]).round(3),
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "배분 전: <b>%{customdata[0]}</b><br>"
            "배분 후: <b>%{customdata[1]}</b><br>"
            "개선폭: <b>%{customdata[4]}</b><br>"
            "%{customdata[2]}형 | 배분액 %{customdata[3]}억원"
            "<extra></extra>"
        ),
    ))

    # ── ③ 배분 후 (빈 원, 유형 색상 테두리)
    fig.add_trace(go.Scatter(
        x=df["imbal_after"],
        y=df["name"],
        mode="markers",
        name="배분 후 (○)",
        marker=dict(
            size=10,
            color="white",
            symbol="circle",
            line=dict(width=2.5, color=type_colors),
        ),
        hoverinfo="skip",
    ))

    # ── ④ 균형 기준선
    fig.add_vline(
        x=1.0,
        line_dash="dot", line_color="#888", line_width=1.5,
        annotation_text="균형(1.0)",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color="#666",
    )

    # ── ⑤ 유형 범례 영역 음영 (선택)
    fig.add_vrect(
        x0=0.8, x1=1.2,
        fillcolor="#f5f8ec", opacity=0.5,
        layer="below", line_width=0,
        annotation_text="균형권", annotation_position="top left",
        annotation_font_size=9, annotation_font_color="#888",
    )

    x_max = max(df["imbal_before"].max() * 1.08, 1.8)
    fig.update_layout(
        title=dict(
            text="지역별 불균형 지수 변화  ●배분 전 → ○배분 후  (초록선: 균형 접근)",
            font_size=13, x=0,
        ),
        xaxis=dict(
            title="불균형 지수",
            gridcolor="#efefef", zeroline=False,
            range=[0, x_max],
        ),
        yaxis=dict(gridcolor="#efefef"),
        height=680,
        margin=dict(l=120, r=40, t=58, b=45),
        legend=dict(orientation="h", y=1.04, x=0, font_size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafaf9",
        hovermode="closest",
    )
    return fig

# ── 6. 변수 중요도 막대
def importance_bar(importance: dict) -> go.Figure:
    labels = list(importance.keys())
    values = list(importance.values())
    pairs = sorted(zip(values, labels), reverse=True)
    values, labels = zip(*pairs)
    fig = go.Figure(go.Bar(
        x=list(values), y=list(labels), orientation="h",
        marker_color="#1B4D6B", opacity=0.85,
    ))
    fig.update_layout(
        title=dict(text="변수 중요도 (수요 예측 모델)", font_size=13),
        xaxis_title="중요도",
        height=280, margin=dict(l=120,r=20,t=45,b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
