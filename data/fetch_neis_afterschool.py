"""data/fetch_neis_afterschool.py — NEIS API 방과후학교 데이터 일괄 수집 스크립트

실행 방법:
  python data/fetch_neis_afterschool.py --key YOUR_NEIS_API_KEY
  python data/fetch_neis_afterschool.py --key YOUR_KEY --year 2023
  python data/fetch_neis_afterschool.py --key YOUR_KEY --year 2023 --force

NEIS API 키 발급:
  https://open.neis.go.kr/portal/mainPage.do → 인증키 신청

결과:
  data/neis_afterschool_cache.json 저장
  → generate_data.py 실행 시 자동으로 로드하여 추정값 대체
"""

import sys, os, json, argparse
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.neis_api import build_afterschool_cache

CACHE_PATH = ROOT / "data" / "neis_afterschool_cache.json"


def main():
    parser = argparse.ArgumentParser(
        description="NEIS Open API에서 방과후학교 참여 데이터를 수집합니다."
    )
    parser.add_argument(
        "--key", required=True,
        help="NEIS Open API 인증키 (https://open.neis.go.kr 에서 발급)"
    )
    parser.add_argument(
        "--year", default="2023",
        help="조회 학년도 (기본값: 2023)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="캐시 파일이 있어도 강제로 재조회"
    )
    args = parser.parse_args()

    # 기존 캐시 확인
    if CACHE_PATH.exists() and not args.force:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"[캐시] 기존 캐시 파일 발견: {CACHE_PATH}")
        print(f"       {len(existing.get('regions', {}))}개 지역 / "
              f"수집일: {existing.get('fetched_at', '알 수 없음')}")
        answer = input("재조회하려면 'y'를 입력하세요 (--force 옵션으로 건너뛸 수 있음): ").strip().lower()
        if answer != "y":
            print("취소되었습니다. 기존 캐시를 사용합니다.")
            return

    print(f"\n{'='*55}")
    print(f"  NEIS 방과후학교 데이터 수집")
    print(f"  학년도: {args.year}  |  API 키: ...{args.key[-6:]}")
    print(f"{'='*55}")

    # API 호출
    try:
        cache = build_afterschool_cache(api_key=args.key, year=args.year)
    except Exception as e:
        print(f"\n[오류] API 호출 실패: {e}")
        print("  API 키가 올바른지 확인하세요: https://open.neis.go.kr")
        sys.exit(1)

    if not cache:
        print("\n[경고] 수집된 데이터가 없습니다.")
        print("  원인: API 키 오류, 네트워크 문제, 또는 해당 연도 데이터 미공개")
        print("  generate_data.py 실행 시 모든 지역에 추정값이 사용됩니다.")

    # 캐시 저장
    output = {
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "year":       args.year,
        "api_note":   "NEIS Open API - AftShoSeatInfo (방과후학교 수강신청 현황)",
        "coverage":   f"{len(cache)}/27개 지역",
        "regions":    cache,
    }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[저장] {CACHE_PATH}")
    print(f"       실측 데이터: {len(cache)}개 지역")

    # 결과 미리보기
    if cache:
        print("\n[결과 미리보기]")
        print(f"{'지역ID':<8} {'참여정원':>8}  {'출처'}")
        print("-" * 32)
        for rid, info in sorted(cache.items()):
            print(f"{rid:<8} {info['enrolled']:>8,}명  {info['source']}")

    print(f"\n[다음 단계] generate_data.py를 실행하면 캐시가 자동 반영됩니다:")
    print(f"  python data/generate_data.py")


if __name__ == "__main__":
    main()
