"""src/neis_api.py — NEIS Open API 클라이언트

지원 엔드포인트
  SchoolInfo     : 학교 기본 정보 (코드·주소·시도교육청코드)
  AftShoSeatInfo : 방과후학교 수강신청 현황 (강좌별 정원 합산 → 학교별 참여 정원)

사용법
  from src.neis_api import build_afterschool_cache
  cache = build_afterschool_cache(api_key="YOUR_KEY", year="2023")
  # → {"GJ01": {"enrolled": 1234, "source": "NEIS실측", "year": "2023"}, ...}
"""

import requests
import time
from collections import defaultdict
from typing import Optional

NEIS_BASE   = "https://open.neis.go.kr/hub"
SIDO_CODES  = {"B10": "광주광역시", "J10": "전라남도"}
PAGE_SIZE   = 1000   # NEIS 최대 1000건/페이지

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
                    timeout=15,
                )
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))

        data = resp.json()

        # 에러 응답 처리
        if endpoint not in data:
            # RESULT 키만 있는 경우 → 데이터 없음
            return rows

        payload = data[endpoint]
        head_block = payload[0]["head"]
        result_info = head_block[1]["RESULT"]
        code = result_info.get("CODE", "")
        if code not in ("INFO-000", "INFO-200"):
            # INFO-200: 해당 정보 없음 (정상이지만 데이터 없음)
            return rows

        total = int(head_block[0]["list_total_count"])
        page_rows = payload[1].get("row", [])
        rows.extend(page_rows)

        if len(rows) >= total or len(page_rows) < PAGE_SIZE:
            break
        page += 1
        time.sleep(0.05)   # API 부하 방지

    return rows


def extract_sigungu(address: str) -> Optional[str]:
    """도로명·지번 주소에서 시군구 이름을 추출하여 region_id로 변환.

    예) '광주광역시 동구 금남로1가 10' → 'GJ01'
        '전라남도 목포시 영산로 ...' → 'JN01'
    """
    if not address:
        return None
    parts = address.split()
    # 1번째 토큰: 시도 / 2번째 토큰: 시군구
    if len(parts) >= 2:
        sigungu = parts[1]
        if sigungu in SIGUNGU_TO_REGION:
            return SIGUNGU_TO_REGION[sigungu]
    return None


# ─────────────────────────────────────────────────────
# 공개 API 함수
# ─────────────────────────────────────────────────────

def fetch_schools(api_key: str) -> dict[str, str]:
    """광주·전남 초등학교 코드 → region_id 매핑 딕셔너리 반환.

    SchoolInfo 엔드포인트에서 주소(ORG_RDNMA)를 파싱하여 시군구 추출.
    """
    school_to_region: dict[str, str] = {}
    for sido_code in SIDO_CODES:
        params = {
            "KEY": api_key,
            "ATPT_OFCDC_SC_CODE": sido_code,
            "SCHUL_KND_SC_NM":    "초등학교",
        }
        try:
            rows = _get("SchoolInfo", params)
        except Exception as e:
            print(f"  [WARN] SchoolInfo 조회 실패 ({sido_code}): {e}")
            continue

        for row in rows:
            code    = row.get("SD_SCHUL_CODE", "")
            address = row.get("ORG_RDNMA", "")
            region_id = extract_sigungu(address)
            if code and region_id:
                school_to_region[code] = region_id

    print(f"  [INFO] 학교 매핑 완료: {len(school_to_region)}개교")
    return school_to_region


def fetch_afterschool_seats(
    api_key: str,
    year: str,
    school_to_region: dict[str, str],
) -> dict[str, int]:
    """AftShoSeatInfo 엔드포인트로 방과후학교 수강정원 합산.

    반환: {region_id: total_seat_count}
    """
    region_seats: dict[str, int] = defaultdict(int)

    for sido_code in SIDO_CODES:
        params = {
            "KEY":               api_key,
            "ATPT_OFCDC_SC_CODE": sido_code,
            "AY":                year,   # 학년도 (예: '2023')
        }
        try:
            rows = _get("AftShoSeatInfo", params)
        except Exception as e:
            print(f"  [WARN] AftShoSeatInfo 조회 실패 ({sido_code}): {e}")
            continue

        for row in rows:
            school_code = row.get("SD_SCHUL_CODE", "")
            seat_cnt    = row.get("SBCJT_SEAT_CNT", 0)
            region_id   = school_to_region.get(school_code)
            if region_id and seat_cnt:
                try:
                    region_seats[region_id] += int(seat_cnt)
                except (ValueError, TypeError):
                    pass

        print(f"  [INFO] {SIDO_CODES[sido_code]} AftShoSeatInfo: {len(rows)}건")

    return dict(region_seats)


def fetch_afterschool_opr(
    api_key: str,
    year: str,
    school_to_region: dict[str, str],
) -> dict[str, int]:
    """AftShoOprInfo 엔드포인트로 방과후학교 수강인원 합산 (fallback용).

    반환: {region_id: total_enrolled}
    """
    region_enr: dict[str, int] = defaultdict(int)

    for sido_code in SIDO_CODES:
        params = {
            "KEY":               api_key,
            "ATPT_OFCDC_SC_CODE": sido_code,
            "AY":                year,
        }
        try:
            rows = _get("AftShoOprInfo", params)
        except Exception as e:
            print(f"  [WARN] AftShoOprInfo 조회 실패 ({sido_code}): {e}")
            continue

        for row in rows:
            school_code = row.get("SD_SCHUL_CODE", "")
            # 강좌 수강인원 필드명 탐색
            enrolled = 0
            for field in ("CLRM_ENRL_CNT", "ENRL_CNT", "SBCJT_ENRL_CNT"):
                v = row.get(field)
                if v:
                    try:
                        enrolled = int(v)
                        break
                    except (ValueError, TypeError):
                        pass
            region_id = school_to_region.get(school_code)
            if region_id and enrolled:
                region_enr[region_id] += enrolled

        print(f"  [INFO] {SIDO_CODES[sido_code]} AftShoOprInfo: {len(rows)}건")

    return dict(region_enr)


# ─────────────────────────────────────────────────────
# 메인 빌더
# ─────────────────────────────────────────────────────

def build_afterschool_cache(
    api_key: str,
    year: str = "2023",
) -> dict[str, dict]:
    """NEIS API를 호출하여 방과후학교 참여정원 캐시를 생성.

    반환 형식:
    {
      "GJ01": {"enrolled": 1234, "source": "NEIS실측", "year": "2023"},
      "JN01": {"enrolled": 567,  "source": "NEIS실측", "year": "2023"},
      ...
    }
    데이터를 가져오지 못한 지역은 포함되지 않음 (generate_data.py에서 추정값 사용).
    """
    print(f"\n[NEIS] 방과후학교 데이터 조회 시작 (학년도: {year})")
    print("[STEP 1] 학교 코드 → 지역 매핑 조회...")
    school_to_region = fetch_schools(api_key)

    if not school_to_region:
        print("[WARN] 학교 매핑 정보를 가져오지 못했습니다. API 키를 확인하세요.")
        return {}

    print("[STEP 2] 방과후학교 수강정원 조회 (AftShoSeatInfo)...")
    seat_data = fetch_afterschool_seats(api_key, year, school_to_region)

    # AftShoSeatInfo로 데이터를 가져오지 못한 경우 OprInfo 시도
    if not seat_data:
        print("[STEP 2b] AftShoSeatInfo 없음 → AftShoOprInfo 시도...")
        seat_data = fetch_afterschool_opr(api_key, year, school_to_region)

    cache = {}
    for region_id, count in seat_data.items():
        if count > 0:
            cache[region_id] = {
                "enrolled": count,
                "source":   "NEIS실측",
                "year":     year,
            }

    covered = len(cache)
    total   = len(SIGUNGU_TO_REGION)
    print(f"\n[NEIS] 완료: {covered}/{total}개 지역 실측 데이터 확보")
    if covered < total:
        missing = [r for r in SIGUNGU_TO_REGION.values() if r not in cache]
        print(f"  미확보 지역 ({len(missing)}개): {', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""))
        print("  → generate_data.py에서 해당 지역은 추정값 사용")

    return cache
