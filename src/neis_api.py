"""src/neis_api.py — NEIS Open API 클라이언트

[NEIS API 제공 범위 (2024년 기준)]
  schoolInfo  : 학교기본정보 — 학교명·주소·학교코드 ✅
  classInfo   : 학급정보 — 학년·반 목록 → 학급 수 산출 ✅ (AY=2025 필수)

[초등학생 수 산출 방법]
  NEIS classInfo(AY=2025) → 학교별 실측 학급 수
  × KEDI 교육통계 2024 시군구별 학급당 학생수
  = 시군구별 실측 기반 초등학생 수
  (source = "NEIS학급기반")

[방과후학교 참여인원]
  AftShoSeatInfo : 이 API 키로 미제공
  → schoolInfo 실측 학교수 × 교육부 참여율 결정론적 계산 (source = "NEIS기반추정")

사용법
  from src.neis_api import build_school_cache, build_student_cache
"""

import requests
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
AFTERSCHOOL_RATE_FIXED = {
    "A": 0.22,   # 농촌 오지: 접근성 낮음
    "B": 0.36,   # 인구감소 군: 시설 있으나 학생 급감
    "C": 0.65,   # 도심 성장: 맞벌이 집중
    "D": 0.565,  # 균형 중소도시: 전국 평균 수준
}

# ─────────────────────────────────────────────────────
# 시군구별 학급당 학생수 (KEDI 교육통계 2024 기준)
#
# 출처: 한국교육개발원(KEDI) 교육통계 연보 2024
#   전국 초등학교 평균: 20.4명/학급
#   광주광역시:         19.5명/학급
#   전라남도:           13.8명/학급 (전체 평균)
#     - 시 지역:        ~16명대 (목포·여수·순천·나주·광양)
#     - 군 지역(비감소): ~11-12명대
#     - 군 지역(인구감소·소규모교): ~8-10명대
# ─────────────────────────────────────────────────────
CLASS_SIZE_PER_REGION = {
    # 광주광역시 5구 (KEDI 2024 광주 19.5명)
    "GJ01": 19.5, "GJ02": 19.5, "GJ03": 19.5,
    "GJ04": 19.5, "GJ05": 19.5,
    # 전남 시 지역 (비감소, 도시 규모)
    "JN01": 15.8,  # 목포시
    "JN02": 16.2,  # 여수시
    "JN03": 16.5,  # 순천시
    "JN04": 14.8,  # 나주시
    "JN05": 15.5,  # 광양시
    "JN06": 13.5,  # 무안군 (도청소재지)
    # 전남 군 지역 (비감소)
    "JN07": 11.8,  # 담양군
    "JN08": 12.2,  # 화순군
    "JN09":  9.5,  # 보성군
    "JN10": 11.0,  # 해남군
    "JN11": 10.5,  # 영암군
    "JN12":  8.8,  # 함평군
    "JN13": 11.2,  # 영광군
    "JN14": 10.8,  # 장성군
    # 전남 군 지역 (인구감소, 소규모 학교)
    "JN15":  8.2,  # 곡성군
    "JN16":  8.5,  # 구례군
    "JN17":  9.2,  # 고흥군
    "JN18":  8.8,  # 장흥군
    "JN19":  9.0,  # 강진군
    "JN20":  9.8,  # 완도군
    "JN21":  8.6,  # 진도군
    "JN22":  7.8,  # 신안군 (도서 분산, 초소형 학교)
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


# ─────────────────────────────────────────────────────
# 학생 수 산출 — classInfo 기반
# ─────────────────────────────────────────────────────

def _fetch_class_count_one(args: tuple) -> tuple[str, str, int]:
    """단일 학교의 classInfo 조회 → (school_code, region_id, class_count).

    ThreadPoolExecutor에서 호출되는 워커 함수.
    """
    api_key, sido_code, school_code, region_id = args
    params = {
        "KEY":                api_key,
        "Type":               "json",
        "ATPT_OFCDC_SC_CODE": sido_code,
        "SD_SCHUL_CODE":      school_code,
        "AY":                 "2025",   # classInfo는 AY=2025 필수
        "pSize":              100,
        "pIndex":             1,
    }
    try:
        resp = requests.get(f"{NEIS_BASE}/classInfo", params=params, timeout=15)
        data = resp.json()
        if "classInfo" in data:
            rows = data["classInfo"][1].get("row", [])
            return school_code, region_id, len(rows)
    except Exception:
        pass
    return school_code, region_id, 0


def fetch_class_counts_all(
    api_key: str,
    school_map: dict[str, tuple[str, str]],
    max_workers: int = 20,
) -> dict[str, int]:
    """전체 학교 classInfo 병렬 조회 → 시군구별 총 학급 수.

    인자
    ----
    school_map : {school_code: (sido_code, region_id)}
    max_workers: 병렬 스레드 수 (API 부하 방지 기본 20)

    반환: {region_id: total_class_count}
    """
    args_list = [
        (api_key, sido_code, code, rid)
        for code, (sido_code, rid) in school_map.items()
    ]

    region_classes: dict[str, int] = defaultdict(int)
    done = 0
    total = len(args_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_class_count_one, a): a for a in args_list}
        for future in as_completed(futures):
            code, rid, cnt = future.result()
            if rid and cnt > 0:
                region_classes[rid] += cnt
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [진행] {done}/{total}개교 완료 ...")

    return dict(region_classes)


def build_student_cache(api_key: str) -> dict[str, dict]:
    """NEIS classInfo(AY=2025) 실측 학급 수 × KEDI 학급당 학생수 → 시군구별 초등학생 수.

    반환 형식:
    {
      "GJ01": {"students": 5_082, "class_count": 261, "source": "NEIS학급기반"},
      "JN01": {"students": 8_340, "class_count": 527, "source": "NEIS학급기반"},
      ...
    }
    """
    print("\n[STEP 1] 전체 초등학교 목록 수집 (schoolInfo)...")
    # school_code → (sido_code, region_id)
    school_map: dict[str, tuple[str, str]] = {}
    for sido_code, sido_name in SIDO_CODES.items():
        params = {
            "KEY":                api_key,
            "ATPT_OFCDC_SC_CODE": sido_code,
            "SCHUL_KND_SC_NM":    "초등학교",
        }
        rows = _get("schoolInfo", params)
        for row in rows:
            code    = row.get("SD_SCHUL_CODE", "")
            address = row.get("ORG_RDNMA", "")
            rid     = extract_sigungu(address)
            if code and rid:
                school_map[code] = (sido_code, rid)
        print(f"  {sido_name}: {len(rows)}개교")

    if not school_map:
        print("[WARN] 학교 목록을 가져오지 못했습니다.")
        return {}

    print(f"\n[STEP 2] 학교별 학급 수 수집 (classInfo AY=2025) - {len(school_map)}개교 병렬 조회...")
    region_classes = fetch_class_counts_all(api_key, school_map, max_workers=20)

    print("\n[STEP 3] KEDI 학급당 학생수 적용 → 시군구별 학생 수 산출...")
    cache: dict[str, dict] = {}
    for rid, class_count in region_classes.items():
        class_size = CLASS_SIZE_PER_REGION.get(rid, 13.8)  # 기본값: 전남 평균
        students   = round(class_count * class_size)
        cache[rid] = {
            "students":    students,
            "class_count": class_count,
            "class_size":  class_size,
            "source":      "NEIS학급기반",
        }

    covered = len(cache)
    total   = len(SIGUNGU_TO_REGION)
    total_students = sum(v["students"] for v in cache.values())
    print(f"\n[완료] {covered}/{total}개 지역 · 총 {total_students:,}명 산출")

    # 결과 요약
    print(f"\n{'지역ID':<8} {'학급수':>6} {'학급당':>6} {'학생수':>8}")
    print("-" * 34)
    for rid in sorted(cache):
        v = cache[rid]
        print(f"{rid:<8} {v['class_count']:>6} {v['class_size']:>6.1f} {v['students']:>8,}")

    return cache
