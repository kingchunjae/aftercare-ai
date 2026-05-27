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
