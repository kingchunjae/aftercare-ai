"""src/neis_api.py — NEIS Open API 클라이언트

[NEIS API 제공 범위 (2024년 기준)]
  schoolInfo      : 학교기본정보 — 학교명·주소·학교코드 ✅
  mealServiceDietInfo : 급식식단정보 ✅
  elsTimetableDi  : 초등 시간표 ✅

[방과후학교 참여인원 관련]
  AftShoSeatInfo / AftShoOprInfo : 이 API 키로 접근 불가 (미제공 서비스)
  → 대안: schoolInfo로 실측 학교 수를 확보 후,
          교육부 전국 참여율(52.9%) 기반 결정론적 계산 적용
          (source = "NEIS기반추정")

사용법
  from src.neis_api import build_school_cache
  cache = build_school_cache(api_key="YOUR_KEY")
  # → {
  #     "GJ01": {"school_count": 11, "source": "NEIS실측"},
  #     "JN01": {"school_count": 34, "source": "NEIS실측"},
  #     ...
  #   }
"""

import requests
import time
from collections import defaultdict
from typing import Optional

NEIS_BASE  = "https://open.neis.go.kr/hub"
PAGE_SIZE  = 1000

# ─────────────────────────────────────────────────────
# 시도교육청 코드 (NEIS 공식)
# B10=서울, C10=부산, D10=대구, E10=인천, F10=광주, G10=대전
# H10=울산, I10=세종, J10=경기, K10=강원, M10=충북, N10=충남
# P10=전북, Q10=전남, R10=경북, S10=경남, T10=제주
# ─────────────────────────────────────────────────────
SIDO_CODES = {"F10": "광주광역시교육청", "Q10": "전라남도교육청"}

# 시군구 이름 → region_id 매핑
SIGUNGU_TO_REGION = {
    # 광주광역시 5개 구
    "동구":    "GJ01", "서구":   "GJ02", "남구":   "GJ03",
    "북구":    "GJ04", "광산구": "GJ05",
    # 전라남도 22개 시군
    "목포시":  "JN01", "여수시": "JN02", "순천시": "JN03",
    "나주시":  "JN04", "광양시": "JN05", "무안군": "JN06",
    "담양군":  "JN07", "화순군": "JN08", "보성군": "JN09",
    "해남군":  "JN10", "영암군": "JN11", "함평군": "JN12",
    "영광군":  "JN13", "장성군": "JN14", "곡성군": "JN15",
    "구례군":  "JN16", "고흥군": "JN17", "장흥군": "JN18",
    "강진군":  "JN19", "완도군": "JN20", "진도군": "JN21",
    "신안군":  "JN22",
}

# 유형별 방과후학교 참여율 (전국 평균 52.9% 기준, 교육부 2024.04)
# 도심(C/D)은 높고, 농촌 오지(A/B)는 낮게 설정
AFTERSCHOOL_RATE_FIXED = {
    "A": 0.22,   # 농촌 오지: 접근성 낮음
    "B": 0.36,   # 인구감소 군: 시설 있으나 학생 급감
    "C": 0.65,   # 도심 성장: 맞벌이 집중
    "D": 0.565,  # 균형 중소도시: 전국 평균 수준
}


# ─────────────────────────────────────────────────────
# 내부 유틸리티
# ─────────────────────────────────────────────────────

def _get(endpoint: str, params: dict, retries: int = 3) -> list[dict]:
    """NEIS API 호출 → row 리스트 반환 (페이지네이션 자동 처리)."""
    params = dict(params)
    params["Type"] = "json"
    params["pSize"] = PAGE_SIZE

    rows = []
    page = 1
    while True:
        params["pIndex"] = page
        for attempt in range(retries):
            try:
                resp = requests.get(
                    f"{NEIS_BASE}/{endpoint}",
                    params=params,
                    timeout=20,
                )
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))

        data = resp.json()
        if endpoint not in data:
            return rows   # 서비스 미존재 또는 결과 없음

        payload    = data[endpoint]
        head_block = payload[0]["head"]
        code       = head_block[1]["RESULT"].get("CODE", "")
        if code == "INFO-200":
            return rows  # 정상이지만 데이터 없음
        if code != "INFO-000":
            return rows

        total     = int(head_block[0]["list_total_count"])
        page_rows = payload[1].get("row", [])
        rows.extend(page_rows)

        if len(rows) >= total or len(page_rows) < PAGE_SIZE:
            break
        page += 1
        time.sleep(0.05)

    return rows


def extract_sigungu(address: str) -> Optional[str]:
    """도로명·지번 주소에서 시군구 이름 추출 → region_id 반환.

    예) '광주광역시 북구 각화대로39번길 10' → 'GJ04'
        '전라남도 목포시 영산로 ...'        → 'JN01'
    """
    if not address:
        return None
    parts = address.split()
    sigungu = parts[1] if len(parts) >= 2 else ""
    return SIGUNGU_TO_REGION.get(sigungu)


# ─────────────────────────────────────────────────────
# 공개 API 함수
# ─────────────────────────────────────────────────────

def fetch_school_counts(api_key: str) -> dict[str, int]:
    """NEIS schoolInfo API로 광주·전남 초등학교 수를 시군구별로 집계.

    반환: {region_id: school_count}  (NEIS 실측)
    """
    region_counts: dict[str, int] = defaultdict(int)

    for sido_code, sido_name in SIDO_CODES.items():
        params = {
            "KEY":                api_key,
            "ATPT_OFCDC_SC_CODE": sido_code,
            "SCHUL_KND_SC_NM":    "초등학교",
        }
        try:
            rows = _get("schoolInfo", params)
        except Exception as e:
            print(f"  [WARN] schoolInfo 조회 실패 ({sido_code}/{sido_name}): {e}")
            continue

        for row in rows:
            address   = row.get("ORG_RDNMA", "")
            region_id = extract_sigungu(address)
            if region_id:
                region_counts[region_id] += 1

        print(f"  [INFO] {sido_name} 초등학교: {len(rows)}개교 조회 완료")

    return dict(region_counts)


# ─────────────────────────────────────────────────────
# 메인 빌더
# ─────────────────────────────────────────────────────

def build_school_cache(api_key: str) -> dict[str, dict]:
    """NEIS schoolInfo API로 시군구별 초등학교 수 캐시 생성.

    반환 형식:
    {
      "GJ01": {"school_count": 11, "source": "NEIS실측"},
      "JN01": {"school_count": 34, "source": "NEIS실측"},
      ...
    }
    """
    print(f"\n[NEIS] 초등학교 실측 데이터 조회 시작")
    print(f"  API: {NEIS_BASE}/schoolInfo")
    print(f"  대상: 광주(F10) + 전남(Q10) 초등학교")

    counts = fetch_school_counts(api_key)

    if not counts:
        print("[WARN] 학교 수 데이터를 가져오지 못했습니다. API 키를 확인하세요.")
        return {}

    cache = {
        rid: {"school_count": cnt, "source": "NEIS실측"}
        for rid, cnt in counts.items()
        if cnt > 0
    }

    total_schools = sum(c["school_count"] for c in cache.values())
    print(f"\n[완료] {len(cache)}/27개 지역 · 총 {total_schools}개교 실측")

    return cache
