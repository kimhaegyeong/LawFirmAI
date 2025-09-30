# ⚖️ LawFirmAI - 법률 AI 어시스턴트

법률 관련 질문에 답변해드리는 AI 어시스턴트입니다. 판례, 법령, Q&A 데이터베이스를 기반으로 정확한 법률 정보를 제공합니다.

## 🚀 주요 기능

- **하이브리드 검색**: 정확한 매칭 검색 + 의미적 검색 결합
- **판례 검색**: 법원 판례 검색 및 분석
- **헌재결정례 수집**: 날짜 기반 체계적 헌재결정례 수집 (신규)
- **법령 해설**: 법령 조문 해석 및 설명  
- **계약서 분석**: 계약서 검토 및 위험 요소 분석
- **Q&A**: 자주 묻는 법률 질문 답변
- **RAG 기반 답변**: 검색 증강 생성으로 정확한 답변 제공

## 🔧 최신 업데이트

### 2025-09-30: Raw 데이터 전처리 파이프라인 구축
- ✅ **전처리 스크립트 구현**: 수집된 raw 데이터를 벡터 DB에 적합한 형태로 변환
- ✅ **배치 전처리 지원**: 특정 데이터 유형만 선택적으로 전처리 가능
- ✅ **데이터 검증 시스템**: 전처리된 데이터의 품질 자동 검증
- ✅ **법률 용어 정규화**: 국가법령정보센터 OpenAPI 기반 용어 정규화 시스템

### 2025-09-26: 네트워크 안정성 향상
- ✅ **DNS 해결 실패 처리**: 네트워크 연결 문제 자동 감지 및 재시도
- ✅ **타임아웃 설정 개선**: 연결 타임아웃(30초)과 읽기 타임아웃(120초) 분리
- ✅ **재시도 로직 강화**: 지수 백오프 방식으로 재시도 간격 점진적 증가
- ✅ **재시도 횟수 증가**: 5회 → 10회로 증가

### 메모리 관리 강화
- ✅ **실시간 메모리 모니터링**: `psutil`을 사용한 메모리 사용량 추적
- ✅ **자동 메모리 정리**: 매 10페이지마다 가비지 컬렉션 실행
- ✅ **메모리 임계값 관리**: 800MB 이상 사용 시 자동 정리
- ✅ **PyTorch 크래시 방지**: 대용량 데이터 구조 제한 및 메모리 최적화

### 에러 핸들링 개선
- ✅ **상세한 오류 메시지**: 네트워크, 메모리 관련 오류에 대한 구체적인 해결 방법 제시
- ✅ **사용자 친화적 메시지**: 이모지와 함께 명확한 오류 설명 및 해결책 제공
- ✅ **오류 분류**: DNS, 연결, 타임아웃, 메모리 오류를 각각 다르게 처리

## 🛠️ 기술 스택

### AI/ML
- **KoBART**: 한국어 생성 모델 (법률 특화 파인튜닝)
- **Sentence-BERT**: 텍스트 임베딩 모델 (jhgan/ko-sroberta-multitask)
- **FAISS**: 벡터 검색 엔진

### Backend
- **FastAPI**: RESTful API 서버
- **SQLite**: 관계형 데이터베이스 (정확한 매칭 검색)
- **FAISS**: 벡터 데이터베이스 (의미적 검색)
- **Pydantic**: 데이터 검증
- **psutil**: 메모리 모니터링 및 시스템 리소스 관리

### Frontend
- **Gradio**: 웹 인터페이스
- **HuggingFace Spaces**: 배포 플랫폼

## 📁 프로젝트 구조

```
LawFirmAI/
├── gradio/                  # Gradio 애플리케이션
│   ├── app.py              # Gradio 메인 애플리케이션
│   ├── requirements.txt    # Gradio 의존성
│   ├── Dockerfile         # Gradio Docker 설정
│   └── docker-compose.yml # Gradio 로컬 개발 환경
├── api/                    # FastAPI 애플리케이션
│   ├── main.py            # FastAPI 메인 애플리케이션
│   ├── requirements.txt   # FastAPI 의존성
│   ├── Dockerfile        # FastAPI Docker 설정
│   └── docker-compose.yml # FastAPI 로컬 개발 환경
├── source/                 # Core Modules (공통 소스 코드)
│   ├── models/            # AI 모델 관련
│   ├── services/          # 비즈니스 로직
│   ├── data/              # 데이터 처리
│   ├── api/               # API 관련
│   └── utils/             # 유틸리티
├── data/                  # 데이터 파일
│   ├── raw/               # 원본 데이터
│   ├── processed/         # 전처리된 데이터
│   └── embeddings/        # 벡터 임베딩
├── tests/                 # 테스트 코드
├── docs/                  # 문서
├── scripts/               # 유틸리티 스크립트
│   ├── collect_data_only.py    # 데이터 수집 전용 (JSON 저장)
│   ├── build_vector_db.py      # 벡터DB 구축 전용
│   ├── run_data_pipeline.py    # 통합 데이터 파이프라인 실행
│   ├── collect_laws.py         # 법령 데이터 수집 (기존)
│   ├── collect_precedents.py   # 판례 데이터 수집 (기존)
│   ├── collect_legal_terms.py  # 법령용어 데이터 수집
│   ├── collect_administrative_rules.py # 행정규칙 데이터 수집
│   ├── collect_local_ordinances.py # 자치법규 데이터 수집
│   ├── collect_all_data.py     # 통합 데이터 수집 (기존)
│   └── validate_data_quality.py # 데이터 품질 검증
├── env.example            # 환경 변수 템플릿
├── .gitignore             # Git 무시 파일
└── README.md              # 프로젝트 문서
```

## 📊 데이터 수집

### 국가법령정보센터 LAW OPEN API 연동

LawFirmAI는 국가법령정보센터의 LAW OPEN API를 통해 법률 데이터를 수집합니다.

#### 지원 데이터 유형

- **법령**: 주요 법령 20개 (민법, 상법, 형법 등) - 모든 조문 및 개정이력 포함
- **판례**: 판례 5,000건 (최근 5년간)
- **헌재결정례**: 1,000건 (최근 5년간)
- **법령해석례**: 2,000건 (최근 3년간)
- **행정규칙**: 1,000건 (주요 부처별)
- **자치법규**: 500건 (주요 지자체별)
- **위원회결정문**: 500건 (주요 위원회별)
- **행정심판례**: 1,000건 (최근 3년간)
- **조약**: 100건 (주요 조약)

#### 데이터 수집 실행

```bash
# 새로운 분리된 데이터 파이프라인 (권장)
python scripts/run_data_pipeline.py --mode full --oc your_email_id

# 데이터 수집만 실행
python scripts/run_data_pipeline.py --mode collect --oc your_email_id

# 벡터DB 구축만 실행
python scripts/run_data_pipeline.py --mode build

# 개별 데이터 타입별 수집
python scripts/run_data_pipeline.py --mode laws --oc your_email_id --query "민법"
python scripts/run_data_pipeline.py --mode precedents --oc your_email_id --query "계약 해지"
python scripts/run_data_pipeline.py --mode constitutional --oc your_email_id --query "헌법"
python scripts/run_data_pipeline.py --mode interpretations --oc your_email_id --query "법령해석"
python scripts/run_data_pipeline.py --mode administrative --oc your_email_id --query "행정규칙"
python scripts/run_data_pipeline.py --mode local --oc your_email_id --query "자치법규"

# 여러 데이터 타입 동시 수집
python scripts/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents constitutional
python scripts/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents --query "민법"

# 개별 데이터 수집 스크립트 (직접 사용)
python scripts/collect_data_only.py --mode laws --oc your_email_id --query "민법"
python scripts/collect_data_only.py --mode multiple --oc your_email_id --types laws precedents constitutional

# 벡터DB 구축 (개별 타입별)
python scripts/build_vector_db.py --mode laws
python scripts/build_vector_db.py --mode multiple --types laws precedents constitutional
```

### 📦 데이터 전처리 (NEW)

수집된 raw 데이터를 벡터 DB에 적합한 형태로 전처리합니다.

#### 전처리 실행

```bash
# 전체 전처리 실행 (모든 데이터 유형)
python scripts/preprocess_raw_data.py

# 특정 데이터 유형만 전처리
python scripts/batch_preprocess.py --data-type laws
python scripts/batch_preprocess.py --data-type precedents
python scripts/batch_preprocess.py --data-type constitutional
python scripts/batch_preprocess.py --data-type interpretations
python scripts/batch_preprocess.py --data-type terms

# 드라이런 모드 (계획만 확인)
python scripts/batch_preprocess.py --data-type all --dry-run

# 전처리된 데이터 검증
python scripts/validate_processed_data.py

# 특정 데이터 유형만 검증
python scripts/validate_processed_data.py --data-type laws
```

#### 전처리 기능

- ✅ **텍스트 정리**: HTML 태그 제거, 공백 정규화, 특수문자 처리
- ✅ **법률 용어 정규화**: 국가법령정보센터 API 기반 용어 표준화
- ✅ **텍스트 청킹**: 벡터 검색에 최적화된 크기로 분할 (200-3000자)
- ✅ **법률 엔티티 추출**: 법률명, 조문, 사건번호, 법원명 등 자동 추출
- ✅ **품질 검증**: 완성도, 정확도, 일관성 자동 검증
- ✅ **중복 제거**: 해시 기반 중복 데이터 자동 제거

#### 상세 문서

- [데이터 전처리 계획서](docs/development/raw_data_preprocessing_plan.md)
- [법률 용어 정규화 전략](docs/development/legal_term_normalization_strategy.md)

```bash

# 기존 통합 스크립트 (레거시)
python scripts/collect_laws.py                    # 법령 수집
python scripts/collect_precedents.py              # 판례 수집
python scripts/collect_constitutional_decisions.py # 헌재결정례 수집
python scripts/collect_legal_interpretations.py   # 법령해석례 수집
python scripts/collect_all_data.py                # 통합 데이터 수집

# Q&A 데이터셋 생성
python scripts/generate_qa_dataset.py

# 데이터 품질 검증
python scripts/validate_data_quality.py
```

#### API 설정

1. [국가법령정보센터 LAW OPEN API](https://open.law.go.kr/LSO/openApi/guideList.do)에서 OC 파라미터 발급
2. 환경변수 설정:
   ```bash
   export LAW_OPEN_API_OC='your_email_id_here'
   ```

#### 사용 예시

**1. 전체 데이터 수집 및 벡터DB 구축**
```bash
# 모든 데이터 타입 수집 + 벡터DB 구축
python scripts/run_data_pipeline.py --mode full --oc your_email_id
```

**2. 특정 데이터 타입만 수집**
```bash
# 법령 데이터만 수집
python scripts/run_data_pipeline.py --mode laws --oc your_email_id --query "민법" --display 50

# 판례 데이터만 수집
python scripts/run_data_pipeline.py --mode precedents --oc your_email_id --query "손해배상" --display 100
```

**3. 여러 데이터 타입 동시 수집**
```bash
# 법령, 판례, 헌재결정례 동시 수집
python scripts/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents constitutional

# 특정 쿼리로 여러 타입 수집
python scripts/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents --query "계약"
```

**4. 데이터 수집과 벡터DB 구축 분리**
```bash
# 1단계: 데이터 수집만
python scripts/run_data_pipeline.py --mode collect --oc your_email_id

# 2단계: 벡터DB 구축만
python scripts/run_data_pipeline.py --mode build
```

**5. 개별 스크립트 사용**
```bash
# 데이터 수집만 (JSON 저장)
python scripts/collect_data_only.py --mode multiple --oc your_email_id --types laws precedents

# 벡터DB 구축만
python scripts/build_vector_db.py --mode multiple --types laws precedents
```

자세한 내용은 [데이터 수집 가이드](docs/data_collection_guide.md)를 참조하세요.

## 🔍 하이브리드 검색 시스템

LawFirmAI는 관계형 데이터베이스(SQLite)와 벡터 데이터베이스(FAISS)를 결합한 하이브리드 검색 시스템을 사용합니다.

### 검색 타입

1. **정확한 매칭 검색**: 법령명, 조문번호, 사건번호 등 정확한 검색
2. **의미적 검색**: 자연어 쿼리를 통한 맥락적 검색
3. **하이브리드 검색**: 두 검색 방식의 결과를 통합하여 최적의 결과 제공

### 장점

- **정확성**: 정확한 매칭으로 필요한 정보를 빠르게 찾을 수 있음
- **유연성**: 의미적 검색으로 다양한 표현의 질문에 답변 가능
- **포괄성**: 두 검색 방식의 장점을 결합하여 더 나은 검색 결과 제공

자세한 내용은 [하이브리드 검색 아키텍처](docs/architecture/hybrid_search_architecture.md)를 참조하세요.

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/LawFirmAI.git
cd LawFirmAI
```

### 2. 가상환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate
```

### 3. 환경 변수 설정

```bash
# 환경 변수 파일 복사
copy .env.example .env

# .env 파일 편집하여 설정값 수정
```

### 4. 데이터 수집 (NEW)

#### 판례 수집
```bash
# 2025년 판례 수집 (무제한) - 안정성 향상
python scripts/precedent/collect_by_date.py --strategy yearly --year 2025 --unlimited

# 2024년 판례 수집 (무제한) - 안정성 향상
python scripts/precedent/collect_by_date.py --strategy yearly --year 2024 --unlimited

# 연도별 수집 (최근 5년, 연간 2000건)
python scripts/precedent/collect_by_date.py --strategy yearly --target 10000
```

**최신 개선사항 (2025-09-26)**:
- ✅ **네트워크 안정성**: DNS 해결 실패, 타임아웃 오류 자동 처리
- ✅ **메모리 관리**: 실시간 메모리 모니터링 및 자동 정리
- ✅ **에러 핸들링**: 상세한 오류 메시지 및 해결 방법 제시

#### 헌재결정례 수집 (신규)
```bash
# 2025년 헌재결정례 수집 (종국일자 기준) - 안정성 향상
python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2025 --final-date

# 2024년 헌재결정례 수집 (선고일자 기준) - 안정성 향상
python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2024

# 특정 건수만 수집
python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2025 --target 100 --final-date

# 분기별 수집
python scripts/constitutional_decision/collect_by_date.py --strategy quarterly --year 2025 --quarter 1

# 월별 수집
python scripts/constitutional_decision/collect_by_date.py --strategy monthly --year 2025 --month 8
```

**최신 개선사항 (2025-09-26)**:
- ✅ **네트워크 안정성**: DNS 해결 실패, 타임아웃 오류 자동 처리
- ✅ **메모리 관리**: 실시간 메모리 모니터링 및 자동 정리
- ✅ **에러 핸들링**: 상세한 오류 메시지 및 해결 방법 제시

#### 기타 데이터 수집
```bash
# 전체 데이터 수집
python scripts/run_data_pipeline.py --mode full --oc your_email_id

# 법령 데이터만 수집
python scripts/run_data_pipeline.py --mode laws --oc your_email_id --query "민법" --display 50
```

### 5. 애플리케이션 실행

#### Gradio 인터페이스 실행

```bash
cd gradio
pip install -r requirements.txt
python app.py
```

#### FastAPI 서버 실행

```bash
cd api
pip install -r requirements.txt
python main.py
```

### 5. 접속

- **Gradio 인터페이스**: http://localhost:7860
- **FastAPI 서버**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

## 🐳 Docker 사용

### Gradio 인터페이스 실행

```bash
cd gradio
docker-compose up -d
```

### FastAPI 서버 실행

```bash
cd api
docker-compose up -d
```

### 전체 서비스 실행 (개발용)

```bash
# Gradio와 FastAPI를 동시에 실행하려면 각각의 폴더에서 실행
cd gradio && docker-compose up -d &
cd api && docker-compose up -d &
```

## 📊 벤치마킹 결과

### AI 모델 성능 비교

| 지표 | KoBART | KoGPT-2 | 승자 |
|------|--------|---------|------|
| 모델 크기 | 472.5 MB | 477.5 MB | KoBART |
| 메모리 사용량 | 400.8 MB | 748.3 MB | KoBART |
| 추론 속도 | 13.18초 | 8.34초 | **KoGPT-2** |
| 응답 품질 | 낮음 | 보통 | **KoGPT-2** |

### 벡터 스토어 성능 비교

| 지표 | FAISS | ChromaDB | 승자 |
|------|-------|----------|------|
| 안정성 | 정상 동작 | 정상 동작 | **동점** |
| 검색 속도 | 0.15초 | 0.17초 | **FAISS** |
| 메모리 사용량 | 낮음 | 높음 | **FAISS** |
| 확장성 | 높음 | 보통 | **FAISS** |

## 🔧 개발

### 개발 환경 설정

```bash
# 프로젝트 의존성 설치
pip install -r requirements.txt

# 메모리 모니터링 라이브러리 설치
pip install psutil>=5.9.0

# 개발 의존성 설치
pip install -e .[dev]

# 코드 포맷팅
black source/
isort source/

# 린팅
flake8 source/
mypy source/

# 테스트 실행
pytest tests/
```

### 코드 스타일

- **Python**: PEP 8 준수
- **타입 힌트**: 모든 함수에 타입 힌트 사용
- **문서화**: 모든 클래스와 함수에 docstring 작성
- **테스트**: 핵심 기능에 대한 단위 테스트 작성

## 📚 API 문서

### 주요 엔드포인트

- `POST /api/v1/chat` - 채팅 메시지 처리
- `POST /api/v1/search/hybrid` - 하이브리드 검색 (정확한 매칭 + 의미적 검색)
- `POST /api/v1/search/exact` - 정확한 매칭 검색
- `POST /api/v1/search/semantic` - 의미적 검색
- `POST /api/v1/external/law/search` - 법령 검색 (국가법령정보 API)
- `POST /api/v1/external/precedent/search` - 판례 검색 (국가법령정보 API)
- `GET /api/v1/health` - 헬스체크
- `GET /docs` - API 문서 (Swagger UI)

### API 문서 구조

- **[API 설계 명세서](docs/api/api_specification.md)** - LawFirmAI API 전체 명세
- **[국가법령정보 OPEN API 가이드](docs/api/law_open_api_complete_guide.md)** - 외부 API 연동 가이드
- **[API별 상세 가이드](docs/api/law_open_api/README.md)** - 각 API별 상세 문서

### 사용 예제

#### 하이브리드 검색 API
```python
import requests

# 하이브리드 검색 요청
response = requests.post(
    "http://localhost:8000/api/v1/search/hybrid",
    json={
        "query": "계약 해지 손해배상",
        "search_type": "hybrid",
        "filters": {
            "document_type": "precedent",
            "court_name": "대법원"
        },
        "limit": 10
    }
)

result = response.json()
print(f"총 {result['total_count']}건의 결과")
for doc in result['results']:
    print(f"제목: {doc['title']}")
    print(f"정확한 매칭: {doc['exact_match']}")
    print(f"유사도 점수: {doc['similarity_score']:.3f}")
```

#### 채팅 API
```python
import requests

# 채팅 요청
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "message": "계약서에서 주의해야 할 조항은 무엇인가요?",
        "context": "부동산 매매계약"
    }
)

result = response.json()
print(result["response"])
```

#### 외부 API 연동 (법령 검색)
```python
import requests

# 법령 검색 요청
response = requests.post(
    "http://localhost:8000/api/v1/external/law/search",
    json={
        "query": "자동차관리법",
        "filters": {
            "date_from": "20240101",
            "date_to": "20241231"
        },
        "limit": 10
    }
)

result = response.json()
for law in result["results"]:
    print(f"법령명: {law['법령명한글']}")
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.


## 🙏 감사의 말

- [HuggingFace](https://huggingface.co/) - AI 모델 제공
- [FastAPI](https://fastapi.tiangolo.com/) - 웹 프레임워크
- [Gradio](https://gradio.app/) - UI 프레임워크
- [ChromaDB](https://www.trychroma.com/) - 벡터 데이터베이스

---



*LawFirmAI는 법률 전문가의 도구로 사용되며, 법률 자문을 대체하지 않습니다. 중요한 법률 문제는 반드시 자격을 갖춘 법률 전문가와 상담하시기 바랍니다.*
