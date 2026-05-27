"""data/fetch_neis_afterschool.py — NEIS API 데이터 수집 스크립트

[수집 내용]
  NEIS schoolInfo API → 광주(F10)+전남(Q10) 초등학교 수 (시군구별 실측)
  → data/neis_afterschool_cache.json 저장

[방과후학교 참여인원]
  NEIS API가 해당 서비스를 제공하지 않아 직접 조회 불가.
  대신 NEIS 실측 학교 수 기반으로 결정론적 계산:
    참여인원 = 학교수 × (초등학생수 / 학교수) × 유형별참여율
  → generate_data.py에서 afterschool_source = "NEIS기반추정"으로 표기

실행 방법:
  python data/fetch_neis_afterschool.py --key YOUR_NEIS_API_KEY
  python data/fetch_neis_afterschool.py --key YOUR_KEY --force

NEIS API 키 발급:
  https://open.neis.go.kr/portal/mainPage.do → 인증키 신청
"""

import sys, os, json, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.neis_api import build_school_cache

CACHE_PATH = ROOT / "data" / "neis_afterschool_cache.json"


def main():
    parser = argparse.ArgumentParser(
        description="NEIS Open API에서 초등학교 실측 데이터를 수집합니다."
    )
    parser.add_argument("--key", required=True,
                        help="NEIS Open API 인증키 (https://open.neis.go.kr 에서 발급)")
    parser.add_argument("--force", action="store_true",
                        help="캐시 파일이 있어도 강제 재조회")
    args = parser.parse_args()

    # 기존 캐시 확인
    if CACHE_PATH.exists() and not args.force:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"[캐시] 기존 파일: {CACHE_PATH}")
        print(f"       {len(existing.get('regions', {}))}개 지역 / "
              f"수집일: {existing.get('fetched_at', '알 수 없음')}")
        answer = input("재조회하려면 'y' 입력 (--force로 건너뛰기 가능): ").strip().lower()
        if answer != "y":
            print("취소. 기존 캐시 유지.")
            return

    print(f"\n{'='*55}")
    print(f"  NEIS 초등학교 실측 데이터 수집")
    print(f"  API 키: ...{args.key[-8:]}")
    print(f"{'='*55}")

    try:
        cache = build_school_cache(api_key=args.key)
    except Exception as e:
        print(f"\n[오류] API 호출 실패: {e}")
        sys.exit(1)

    if not cache:
        print("\n[경고] 수집된 데이터가 없습니다.")
        sys.exit(1)

    # 캐시 저장
    output = {
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "api_note":   "NEIS Open API - schoolInfo (학교기본정보, 광주F10+전남Q10)",
        "data_type":  "school_count",
        "coverage":   f"{len(cache)}/27개 지역",
        "regions":    cache,
    }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_schools = sum(v["school_count"] for v in cache.values())
    print(f"\n[저장] {CACHE_PATH}")
    print(f"       실측 지역: {len(cache)}개 / 총 학교 수: {total_schools}개교")

    # 결과 미리보기
    print("\n[결과 미리보기]")
    print(f"{'지역ID':<8} {'학교수':>6}  {'출처'}")
    print("-" * 28)
    for rid, info in sorted(cache.items()):
        print(f"{rid:<8} {info['school_count']:>6}개교  {info['source']}")

    print(f"\n[다음 단계] generate_data.py 실행:")
    print(f"  python data/generate_data.py")


if __name__ == "__main__":
    main()
