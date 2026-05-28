"""src/ai_report.py — Claude API 연동 · 유형별 정책 보고서 자동 생성"""
import os, json
from anthropic import Anthropic

def _client():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
    return Anthropic(api_key=key)

# ── 유형별 프롬프트 템플릿
SYSTEM_PROMPT = (
    "당신은 교육청 방과후·초등돌봄 정책 전문가 AI입니다. "
    "지역 데이터를 분석하여 교육청 제출용 정책 제안서를 작성합니다. "
    "반드시 격식체(~합니다/~입니다)를 사용하고, 데이터 수치를 직접 인용하며, "
    "실현 가능한 구체적 방안을 제시하십시오."
)

def _build_prompt(detail: dict) -> str:
    t = detail["type"]
    d = detail
    base_json = json.dumps({
        "유형": f"{t} — {d['type_info']['label']}",
        "초등학생수": d["students"],
        "맞벌이가구비율(%)": d["dual_pct"],
        "한부모가구비율(%)": d["single_pct"],
        "돌봄대기자수": d["waitlist"],
        "돌봄이용률(%)": d["util_rate"],
        "수요지수": d["demand_idx"],
        "공급지수": d["supply_idx"],
        "불균형지수": d["imbal_idx"],
        "5년후수요예측지수": d["demand_5y"],
        "출생아변화율(%)": d["birth_chg"],
        "위험점수": d["risk_score"],
        "인구감소지역": d["decline"],
    }, ensure_ascii=False, indent=2)

    instructions = {
        "A": (
            "위 지역은 인구감소지역이면서 돌봄 공급이 심각하게 부족합니다.\n"
            "다음 순서로 교육청 제출용 정책 제안서를 작성하세요:\n"
            "1. 현황 요약 (데이터 기반, 2단락)\n"
            "2. 위기 진단 및 핵심 원인 분석\n"
            "3. 단기(1년 내) 긴급 대응 방안 3가지\n"
            "4. 중장기(3년) 로드맵\n"
            "5. 예상 효과 및 KPI 지표\n"
            "6. 유관 부처 협력 사항\n"
            "총 600~800자, 격식체로 작성하세요."
        ),
        "B": (
            "위 지역은 인구감소지역이며 돌봄 시설이 남아돌고 있습니다.\n"
            "다음 순서로 구조 전환 정책 제안서를 작성하세요:\n"
            "1. 현황 요약 (이용률 저조·학생 감소 중심, 2단락)\n"
            "2. 유휴 시설 활용 가능성 진단\n"
            "3. 시설 복합 활용 방안 3가지 (노인 돌봄·청년 공간·지역 커뮤니티 등)\n"
            "4. 단계별 전환 로드맵 (2년)\n"
            "5. 예상 효과 및 활성화 KPI\n"
            "총 600~800자, 격식체로 작성하세요."
        ),
        "C": (
            "위 지역은 인구 유지 지역이지만 돌봄 수요가 공급을 크게 초과합니다.\n"
            "다음 순서로 공급 확충 정책 제안서를 작성하세요:\n"
            "1. 현황 요약 (대기자·맞벌이 밀도 중심, 2단락)\n"
            "2. 공급 부족 원인 분석\n"
            "3. 긴급 공급 확충 방안 3가지 (단기)\n"
            "4. 수요 분산 및 중기 인프라 계획\n"
            "5. 예상 효과 및 대기자 감소 KPI\n"
            "총 600~800자, 격식체로 작성하세요."
        ),
        "D": (
            "위 지역은 현재 돌봄 수요와 공급이 비교적 균형을 이루고 있습니다.\n"
            "다음 순서로 현황 유지 및 모니터링 계획서를 작성하세요:\n"
            "1. 현황 요약 (균형 상태 확인, 2단락)\n"
            "2. 잠재 위험 요인 점검 (출생아 추세·인구이동 등)\n"
            "3. 예방적 관리 방안 3가지\n"
            "4. 분기별 KPI 모니터링 체계\n"
            "5. 인접 A·C형 지역 연계 지원 가능성\n"
            "총 500~700자, 격식체로 작성하세요."
        ),
    }

    return f"다음 지역 데이터를 분석하십시오:\n\n{base_json}\n\n{instructions[t]}"

def generate_report(detail: dict, stream: bool = False):
    """
    정책 보고서 생성.
    stream=True 이면 제너레이터 반환 (Streamlit st.write_stream 호환)
    """
    client = _client()
    prompt = _build_prompt(detail)

    if stream:
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as s:
            for text in s.text_stream:
                yield text
    else:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

COMPARE_SYSTEM = (
    "당신은 교육청 방과후·초등돌봄 정책 전문가 AI입니다. "
    "동일 유형 두 지역의 돌봄 지표를 비교하여 실현 가능한 개선 전략을 제안합니다. "
    "반드시 격식체(~합니다/~입니다)를 사용하고, 수치를 직접 인용하며, "
    "분석 대상 지역이 즉시 적용할 수 있는 구체적 방안을 제시하십시오."
)

def _build_comparison_prompt(sel: "pd.Series", comp: "pd.Series",
                              type_avg: dict, type_label: str) -> str:
    def _row_json(r):
        return json.dumps({
            "초등학생수": int(r["students"]),
            "맞벌이가구비율(%)": float(r["dual_income_pct"]),
            "한부모가구비율(%)": float(r["single_parent_pct"]),
            "돌봄대기자수": int(r["care_waitlist"]),
            "이용률(%)": float(r["care_util_rate"]),
            "수요지수": round(float(r["demand_idx"]), 3),
            "공급지수": round(float(r["supply_idx"]), 3),
            "불균형지수": round(float(r["imbal_idx"]), 3),
            "위험점수": int(r["risk_score"]),
            "인구감소지역": bool(r["decline"]),
        }, ensure_ascii=False, indent=2)

    return (
        f"다음은 {type_label} 유형 내 두 지역의 비교 데이터입니다.\n\n"
        f"## 분석 대상 지역: {sel['name']}\n{_row_json(sel)}\n\n"
        f"## 동일 유형 비교 지역 ({comp['name']} — 해당 유형 최우수 성과):\n{_row_json(comp)}\n\n"
        f"## {type_label} 유형 평균:\n{json.dumps(type_avg, ensure_ascii=False, indent=2)}\n\n"
        f"위 데이터를 바탕으로 다음 순서로 비교 분석 보고서를 작성하세요:\n\n"
        f"1. **핵심 지표 차이 요약**: {sel['name']}과 {comp['name']}의 주요 수치 차이를 직접 인용하여 2~3 문장 서술\n"
        f"2. **취약 원인 진단**: {sel['name']}의 상대적 열위 핵심 원인 2가지 (수치 근거 포함)\n"
        f"3. **우수 지역 강점 분석**: {comp['name']}이 더 나은 성과를 보이는 구조적 이유\n"
        f"4. **즉시 적용 가능한 개선 전략 3가지**: 각 전략별 구체적 수치 목표와 실행 주체 명시\n"
        f"5. **단기 KPI (6개월 내)**: 달성 가능한 측정 지표 2개\n\n"
        f"총 600~800자, 격식체로 작성하세요."
    )

def generate_comparison_analysis(sel_row, comp_row, same_type_df, type_label: str, stream: bool = True):
    """동일 유형 지역 비교 AI 분석 (stream=True 기본)"""
    type_avg = {
        "평균_위험점수": round(float(same_type_df["risk_score"].mean()), 1),
        "평균_불균형지수": round(float(same_type_df["imbal_idx"].mean()), 3),
        "평균_이용률(%)": round(float(same_type_df["care_util_rate"].mean()), 1),
        "평균_대기자수": int(same_type_df["care_waitlist"].mean()),
    }
    client = _client()
    prompt = _build_comparison_prompt(sel_row, comp_row, type_avg, type_label)
    if stream:
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            system=COMPARE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ) as s:
            for text in s.text_stream:
                yield text
    else:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            system=COMPARE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

def estimate_cost(detail: dict) -> dict:
    """API 비용 추정 (2026년 기준 Sonnet 4 가격)"""
    prompt_tokens = 400
    output_tokens = 700
    input_cost  = prompt_tokens  / 1_000_000 * 3.0   # $3/MTok
    output_cost = output_tokens  / 1_000_000 * 15.0  # $15/MTok
    total_usd   = input_cost + output_cost
    total_krw   = total_usd * 1350
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(total_usd, 5),
        "cost_krw": round(total_krw),
    }
