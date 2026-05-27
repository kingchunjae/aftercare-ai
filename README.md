# 방과후·초등돌봄 수요-공급 불균형 AI 진단 시스템

제8회 교육 공공데이터 AI 활용대회 일반부 프로토타입

---

## 빠른 시작 (로컬 실행)

```bash
# 1. 저장소 클론
git clone https://github.com/[팀명]/aftercare-ai-diagnosis.git
cd aftercare-ai-diagnosis

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. 의존성 설치
pip install -r requirements.txt

# 4. API 키 설정
cp .env.example .env
# .env 파일을 열고 ANTHROPIC_API_KEY 값 입력

# 5. 샘플 데이터 생성
python data/generate_data.py

# 6. 모델 학습
python src/model.py

# 7. 앱 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## Streamlit Cloud 배포

1. GitHub에 이 저장소를 push
2. [streamlit.io/cloud](https://streamlit.io/cloud) 에서 새 앱 생성
3. **Settings → Secrets** 에 아래 내용 입력:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-여기에_실제_키_입력"
   ```
4. 배포 완료 후 URL을 PPT 표지에 기재

---

## 프로젝트 구조

```
aftercare_ai/
├── app.py                  # Streamlit 메인 앱 (4탭)
├── requirements.txt
├── .env.example
│
├── data/
│   ├── generate_data.py    # 샘플 데이터 생성
│   └── regions.csv         # 27개 시군구 데이터
│
├── src/
│   ├── preprocessing.py    # 지수 산출·유형 분류·예산 시뮬레이션
│   ├── model.py            # Random Forest 학습·예측·중요도
│   ├── ai_report.py        # Claude API 연동·프롬프트 4종
│   └── visualize.py        # folium 지도·plotly 차트
│
└── models/
    ├── rf_regressor.pkl    # 수요 예측 모델 (자동 생성)
    ├── rf_classifier.pkl   # 유형 분류 모델 (자동 생성)
    └── scaler.pkl          # MinMaxScaler (자동 생성)
```

---

## 앱 기능 설명

### 탭 1 — 지도 대시보드
- folium 인터랙티브 지도에 27개 시군구를 4유형 색상으로 표시
- 마커 클릭 → 팝업으로 기본 지표 확인
- 우측 패널: 유형 분포 파이차트·위험 점수 Top 5

### 탭 2 — 지역 상세 분석
- 지역 선택 → 수요/공급/불균형 지수 게이지
- 5년 후 수요 예측 그래프 (Random Forest 모델)
- 위험 요인 Top 3 · 변수 중요도 차트

### 탭 3 — 예산 배분 시뮬레이터
- 총 예산(억원) 슬라이더 입력
- A형 45% · C형 40% · D형 10% · B형 5% 우선 배분
- 배분 전후 불균형 지수 비교 차트

### 탭 4 — AI 정책 보고서
- Claude API (claude-sonnet-4-20250514) 활용
- 유형별 맞춤 프롬프트 4종 자동 선택
- 스트리밍 출력 · 텍스트 저장

---

## 활용 데이터 (실제 서비스 연동 시)

| 번호 | 데이터명 | 출처 | 라이선스 |
|------|----------|------|----------|
| ① | NEIS 방과후학교 운영현황 | open.neis.go.kr | 공공누리 1유형 |
| ② | 교육통계서비스 | kess.kedi.re.kr | 공공누리 1유형 |
| ③ | 학교알리미 방과후·돌봄 공시 | schoolinfo.go.kr | 공공누리 1유형 |
| ④ | 인구감소지역 지정현황 | data.go.kr (행안부) | 공공누리 1유형 |
| ⑤ | 맞벌이가구·출생통계 | data.go.kr (통계청) | 공공누리 1유형 |

현재 버전은 위 데이터 구조와 동일한 **시뮬레이션 데이터**로 동작합니다.
실제 API 연동은 `data/generate_data.py`를 각 API 호출로 교체하면 됩니다.

---

## 기술 스택

- **Frontend**: Streamlit 1.35
- **데이터**: pandas 2.0, numpy 1.25
- **ML**: scikit-learn 1.3 (Random Forest)
- **지도**: folium 0.17, streamlit-folium
- **차트**: plotly 5.22
- **생성형 AI**: Anthropic Claude API (claude-sonnet-4-20250514)

---

## 라이선스

MIT License — 자유로운 복제·수정·배포 허용
교육청 정책 실무 활용 시 팀에 사전 문의 바랍니다.
