"""data/fetch_neis_students.py — NEIS classInfo 기반 초등학생 수 실측 수집 스크립트

[수집 내용]
  NEIS classInfo(AY=2025) API → 광주(F10)+전남(Q10) 초등학교별 학급 수 (실측)
  × KEDI 교육통계 2024 시군구별 학급당 학생수
  → 시군구별 초등학생 수 산출 (source = "NEIS학급기반")
  → data/neis_students_cache.json 저장

[방법론]
  NEIS classInfo에는 학생수 컬럼이 없으므로:
    학생수 = 학급수(실측) × 학급당 학생수(KEDI 2024 시군구별 상수)
  이 방식이 기존 언론(경향신문·시사저널) 추정치보다 신뢰도 높음:
    - 학급수: NEIS 실측 (2025학년도)
    - 학급당 학생수: KEDI 교육통계 2024 기준

[소요 시간]
  ~617개교 × classInfo 1회 = 병렬 20스레드 기준 약 3-5분

실행 방법:
  python data/fetch_neis_students.py --key YOUR_NEIS_API_KEY
  python data/fetch_neis_students.py --key YOUR_KEY --force

NEIS API 키 발급:
  https://open.neis.go.kr/portal/mainPage.do → 인증키 신청
"""

import sys, os, json, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.neis_api import build_student_cache

CACHE_PATH = ROOT / "data" / "neis_students_cache.json"


def main():
    parser = argparse.ArgumentParser(
        description="NEIS classInfo API로 초등학생 수 실측 데이터를 수집합니다."
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

    print(f"\n{'='*60}")
    print(f"  NEIS classInfo 기반 초등학생 수 실측 수집")
    print(f"  방법: classInfo(AY=2025) 학급수 × KEDI 학급당 학생수")
    print(f"  API 키: ...{args.key[-8:]}")
    print(f"{'='*60}")
    print(f"  [주의] 617개교 개별 조회 (병렬 20스레드) - 약 3-5분 소요\n")

    try:
        cache = build_student_cache(api_key=args.key)
    except Exception as e:
        print(f"\n[오류] API 호출 실패: {e}")
        sys.exit(1)

    if not cache:
        print("\n[경고] 수집된 데이터가 없습니다.")
        sys.exit(1)

    # 캐시 저장
    total_students  = sum(v["students"]    for v in cache.values())
    total_classes   = sum(v["class_count"] for v in cache.values())

    output = {
        "fetched_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "api_note":      "NEIS Open API - classInfo(AY=2025) × KEDI 학급당 학생수(2024)",
        "data_type":     "students_from_class_count",
        "method":        "classInfo 실측 학급수 × 시군구별 KEDI 학급당 학생수",
        "coverage":      f"{len(cache)}/27개 지역",
        "total_students": total_students,
        "total_classes":  total_classes,
        "regions":       cache,
    }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[저장] {CACHE_PATH}")
    print(f"       실측 지역: {len(cache)}개 / 총 {total_students:,}명 / 총 {total_classes}학급")

    # 결과 미리보기
    print(f"\n[결과 미리보기]")
    print(f"{'지역ID':<8} {'학급수':>6} {'학급당':>6} {'학생수':>8}  {'출처'}")
    print("-" * 42)
    for rid, info in sorted(cache.items()):
        print(f"{rid:<8} {info['class_count']:>6} {info['class_size']:>6.1f} "
              f"{info['students']:>8,}  {info['source']}")

    print(f"\n[다음 단계] generate_data.py 실행:")
    print(f"  python data/generate_data.py")


if __name__ == "__main__":
    main()
