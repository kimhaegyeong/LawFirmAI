# ⚖️ LawFirmAI - 법률 AI 어시스턴트

법률 관련 질문에 답변해드리는 AI 어시스턴트입니다. 판례, 법령, Q&A 데이터베이스를 기반으로 정확한 법률 정보를 제공합니다.


## 📋 개발 규칙 및 가이드라인

### ⚠️ 중요: Streamlit 서버 관리 규칙

**절대 사용하지 말 것**:
```bash
# 모든 Python 프로세스 종료 (위험!)
taskkill /f /im python.exe
```

**올바른 서버 종료 방법**:
```bash
# PID 기반 종료 (권장)
python streamlit/stop_server.py

# 또는 배치 파일 사용
streamlit/stop_server.bat
```

### 📚 상세 개발 규칙

자세한 개발 규칙, 코딩 스타일, 운영 가이드라인은 다음 문서를 참조하세요:
- **[개발 규칙 및 가이드라인](docs/01_project_overview/development_rules.md)**: 프로세스 관리, 로깅, 보안, 테스트 규칙
- **[한국어 인코딩 개발 규칙](docs/01_project_overview/encoding_development_rules.md)**: Windows 환경의 CP949 인코딩 문제 해결을 위한 개발 규칙
- **[TASK별 상세 개발 계획](docs/development/TASK/TASK별%20상세%20개발%20계획_v1.0.md)**: 프로젝트 진행 현황 및 계획
- **[성능 최적화 완료 보고서](docs/07_performance_optimization/performance_optimization_report.md)**: 응답 시간 78% 단축 성과 및 기술적 세부사항
- **[성능 최적화 가이드](docs/07_performance_optimization/performance_optimization_guide.md)**: 최적화된 컴포넌트 사용법 및 성능 튜닝 방법

## 🔧 최신 업데이트

### 주요 업데이트 🎯
- ✅ **LangGraph 워크플로우 최적화**: State 최적화, Pruning 전략, Recursion limit 100 증가
- ✅ **하이브리드 검색 통합**: FAISS 벡터 검색 (155,819개 벡터) + 키워드 검색 결합
- ✅ **성능 최적화**: 응답 시간 78% 단축 (10.05초 → 2.21초)
- ✅ **문서 재작성**: 최신 구조 반영한 LangGraph 통합 가이드 작성

### 데이터 시스템 구축 📊
- ✅ **Assembly 데이터 수집**: 국회 법률정보시스템 기반 데이터 수집 (300개 법률)
- ✅ **벡터 임베딩**: 6,285개 텍스트 청크 벡터화, FAISS 기반 초고속 검색
- ✅ **증분 전처리**: 자동화된 데이터 파이프라인, 180,684개 조문 처리
- ✅ **Q&A 데이터셋**: 2,709개 법률 Q&A 쌍 생성 (품질 점수 93.5%)
- ✅ **메모리 최적화**: Float16 양자화, 지연 로딩, 자동 메모리 정리

### 초기 구축 🏗️
- ✅ **프로젝트 구조**: 스크립트 통합 및 구조 개편
- ✅ **벡터DB 구축**: 642개 법률 문서 임베딩, 평균 0.0003초 검색 시간
- ✅ **네트워크 안정성**: DNS 해결 실패 처리, 재시도 로직 강화
- ✅ **메모리 관리**: 실시간 모니터링 및 자동 정리 시스템

> 📌 자세한 업데이트 내역은 [아카이브 문서](docs/archive/)에서 확인하실 수 있습니다.

## 🛠️ 기술 스택

### AI/ML
- **KoBART**: 한국어 생성 모델 (법률 특화 파인튜닝)
- **Sentence-BERT**: 텍스트 임베딩 모델 (jhgan/ko-sroberta-multitask)
- **FAISS**: 벡터 검색 엔진
- **Ollama Qwen2.5:7b**: 로컬 LLM 모델 (Q&A 생성, 답변 생성)
- **질문 분류 모델**: 사용자 질문 유형 자동 분류 (신규)

### Backend
- **FastAPI**: RESTful API 서버
- **SQLite**: 관계형 데이터베이스 (정확한 매칭 검색)
- **FAISS**: 벡터 데이터베이스 (의미적 검색)
- **Pydantic**: 데이터 검증
- **psutil**: 메모리 모니터링 및 시스템 리소스 관리
- **지능형 검색 엔진**: 질문 유형별 동적 가중치 검색 (신규)
- **신뢰도 계산 시스템**: 답변 신뢰성 수치화 (신규)
- **최적화된 모델 관리자**: 싱글톤 패턴과 지연 로딩 (신규)
- **병렬 검색 엔진**: 동시 처리로 성능 향상 (신규)
- **통합 캐싱 시스템**: 다층 캐싱으로 응답 속도 최적화 (신규)

### Frontend
- **Streamlit**: 웹 인터페이스
- **HuggingFace Spaces**: 배포 플랫폼

## 📁 프로젝트 구조

```
LawFirmAI/
├── streamlit/              # Streamlit 애플리케이션
│   ├── app.py              # Streamlit 메인 애플리케이션
│   ├── requirements.txt    # Streamlit 의존성
│   ├── Dockerfile         # Streamlit Docker 설정
│   └── docker-compose.yml # Streamlit 로컬 개발 환경
├── api/                    # FastAPI 애플리케이션
│   ├── main.py            # FastAPI 메인 애플리케이션
│   ├── requirements.txt   # FastAPI 의존성
│   ├── Dockerfile        # FastAPI Docker 설정
│   └── docker-compose.yml # FastAPI 로컬 개발 환경
├── source/                 # Core Modules (공통 소스 코드)
│   ├── models/            # AI 모델 관련
│   ├── services/          # 비즈니스 로직
│   │   ├── hybrid_search_engine.py      # 하이브리드 검색 엔진 (확장)
│   │   ├── question_classifier.py       # 질문 분류기 (신규)
│   │   ├── precedent_search_engine.py   # 판례 검색 엔진 (신규)
│   │   ├── prompt_templates.py          # 프롬프트 템플릿 (신규)
│   │   ├── confidence_calculator.py     # 신뢰도 계산기 (신규)
│   │   ├── legal_term_expander.py       # 법률 용어 확장기 (신규)
│   │   ├── ollama_client.py             # Ollama 클라이언트 (신규)
│   │   ├── improved_answer_generator.py  # 개선된 답변 생성기 (신규)
│   │   ├── answer_formatter.py          # 답변 포맷터 (신규)
│   │   ├── context_builder.py           # 컨텍스트 빌더 (신규)
│   │   ├── optimized_model_manager.py   # 최적화된 모델 관리자 (신규)
│   │   ├── optimized_hybrid_search_engine.py # 병렬 검색 엔진 (신규)
│   │   ├── integrated_cache_system.py   # 통합 캐싱 시스템 (신규)
│   │   └── optimized_chat_service.py    # 최적화된 채팅 서비스 (신규)
│   ├── data/              # 데이터 처리
│   ├── api/               # API 관련
│   │   └── endpoints.py   # API 엔드포인트 (확장)
│   └── utils/             # 유틸리티
├── data/                  # 데이터 파일
│   ├── raw/               # 원본 데이터
│   ├── processed/         # 전처리된 데이터
│   ├── embeddings/        # 벡터 임베딩
│   │   ├── ml_enhanced_ko_sroberta/        # 법률 벡터 임베딩
│   │   └── ml_enhanced_ko_sroberta_precedents/ # 판례 벡터 임베딩 (신규)
│   ├── qa_dataset/        # Q&A 데이터셋
│   └── legal_term_dictionary.json # 법률 용어 사전 (신규)
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
│   ├── validate_data_quality.py # 데이터 품질 검증
│   ├── generate_qa_dataset.py  # Q&A 데이터셋 생성 (기본)
│   ├── enhanced_generate_qa_dataset.py # Q&A 데이터셋 생성 (향상)
│   ├── large_scale_generate_qa_dataset.py # Q&A 데이터셋 생성 (대규모)
│   ├── llm_qa_generator.py     # LLM 기반 Q&A 생성기 (신규)
│   └── generate_qa_with_llm.py # LLM Q&A 생성 실행 스크립트 (신규)
├── env.example            # 환경 변수 템플릿
├── .gitignore             # Git 무시 파일
└── README.md              # 프로젝트 문서
```

## 📊 데이터 수집

### 국가법령정보센터 LAW OPEN API 연동

LawFirmAI는 국가법령정보센터의 LAW OPEN API를 통해 법률 데이터를 수집합니다.

### 국회 법률정보시스템 웹 스크래핑 (NEW)

API 서비스 중단으로 인해 국회 법률정보시스템(https://likms.assembly.go.kr/law)을 대안으로 사용합니다.

```bash
# Assembly 시스템으로 법률 수집
python scripts/assembly/collect_laws.py --sample 100

# Assembly 시스템으로 판례 수집 (NEW)
python scripts/assembly/collect_precedents.py --sample 50

# 분야별 판례 수집 (NEW)
python scripts/assembly/collect_precedents_by_category.py --category civil --sample 20
python scripts/assembly/collect_precedents_by_category.py --category criminal --sample 20
python scripts/assembly/collect_precedents_by_category.py --category family --sample 20

# 모든 분야 한번에 수집
python scripts/assembly/collect_precedents_by_category.py --all-categories --sample 10

# 특정 페이지부터 수집
python scripts/assembly/collect_laws.py --sample 50 --start-page 5 --no-resume
python scripts/assembly/collect_precedents.py --sample 30 --start-page 3 --no-resume
```

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

### 📝 Q&A 데이터셋 생성 (NEW)

법령/판례 데이터를 기반으로 자동으로 Q&A 데이터셋을 생성합니다.

#### Q&A 생성 실행

```bash
# 기본 Q&A 데이터셋 생성
python scripts/generate_qa_dataset.py

# 향상된 Q&A 데이터셋 생성 (더 많은 패턴)
python scripts/enhanced_generate_qa_dataset.py

# 대규모 Q&A 데이터셋 생성 (최대 규모)
python scripts/large_scale_generate_qa_dataset.py

# LLM 기반 Q&A 데이터셋 생성 (자연스러운 질문-답변)
python scripts/generate_qa_with_llm.py

# LLM 기반 생성 옵션 지정
python scripts/generate_qa_with_llm.py \
  --model qwen2.5:7b \
  --data-type laws precedents \
  --output data/qa_dataset/llm_generated \
  --target 1000 \
  --max-items 20
```

#### 생성 결과

**템플릿 기반 생성 (기존)**
- **총 Q&A 쌍 수**: 2,709개 (목표 대비 90.3%)
- **평균 품질 점수**: 93.5% (목표 90% 초과)
- **고품질 비율**: 99.96% (2,708개/2,709개)
- **데이터 소스**: 법령 42개, 판례 621개

**LLM 기반 생성 (신규)**
- **총 Q&A 쌍 수**: 36개 (테스트 단계)
- **평균 품질 점수**: 68.3% (개선 중)
- **질문 유형**: 12가지 다양한 유형
- **자연스러움**: 템플릿 방식 대비 400% 향상
- **실용성**: 법률 실무 중심 질문 생성

#### 생성된 파일

**템플릿 기반 파일**
- `data/qa_dataset/large_scale_qa_dataset.json` - 전체 데이터셋
- `data/qa_dataset/large_scale_qa_dataset_high_quality.json` - 고품질 데이터셋
- `data/qa_dataset/large_scale_qa_dataset_statistics.json` - 통계 정보
- `docs/qa_dataset_quality_report.md` - 품질 보고서

**LLM 기반 파일**
- `data/qa_dataset/llm_generated/llm_qa_dataset.json` - LLM 생성 전체 데이터셋
- `data/qa_dataset/llm_generated/llm_qa_dataset_high_quality.json` - 고품질 데이터셋
- `data/qa_dataset/llm_generated/llm_qa_dataset_statistics.json` - 통계 정보
- `docs/llm_qa_dataset_quality_report.md` - LLM 품질 보고서

#### Q&A 유형

**템플릿 기반 유형**
- **법령 정의 Q&A**: 법률의 목적과 정의에 관한 질문
- **조문 내용 Q&A**: 특정 조문의 내용과 의미
- **조문 제목 Q&A**: 조문의 제목과 주제
- **키워드 기반 Q&A**: 법률 용어와 개념 설명
- **판례 쟁점 Q&A**: 사건의 핵심 쟁점과 문제
- **판결 내용 Q&A**: 법원의 판단과 결론

**LLM 기반 유형 (자연스러운 질문)**
- **개념 설명**: "~란 무엇인가요?"
- **실제 적용**: "~한 경우 어떻게 해야 하나요?"
- **요건/효과**: "~의 요건은 무엇인가요?"
- **비교/차이**: "~와 ~의 차이는 무엇인가요?"
- **절차**: "~하려면 어떤 절차를 거쳐야 하나요?"
- **예시**: "~의 구체적인 예시를 들어주세요"
- **주의사항**: "~할 때 주의할 점은 무엇인가요?"
- **적용 범위**: "~이 적용되는 대상은 무엇인가요?"
- **목적**: "~의 목적은 무엇인가요?"
- **법적 근거**: "~의 법적 근거는 무엇인가요?"
- **실무 적용**: "실무에서 ~는 어떻게 적용되나요?"
- **예외 사항**: "~의 예외 사항은 무엇인가요?"

```bash

# 기존 통합 스크립트 (레거시)
python scripts/collect_laws.py                    # 법령 수집
python scripts/collect_precedents.py              # 판례 수집
python scripts/collect_constitutional_decisions.py # 헌재결정례 수집
python scripts/collect_legal_interpretations.py   # 법령해석례 수집
python scripts/collect_all_data.py                # 통합 데이터 수집

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

## 📊 모니터링 시스템

### Grafana + Prometheus 기반 실시간 모니터링

LawFirmAI는 법률 수집 성능을 실시간으로 모니터링하는 시스템을 제공합니다.

#### 주요 기능
- **실시간 메트릭 수집**: 페이지 처리, 법률 수집, 에러율 등
- **지속적 메트릭 누적**: 여러 실행에 걸쳐 메트릭 값 누적
- **Grafana 대시보드**: 시각적 모니터링 및 알림
- **성능 분석**: 처리량, 메모리 사용량, CPU 사용률 추적

#### 빠른 시작

```bash
# 1. 모니터링 스택 시작
cd monitoring
docker-compose up -d

# 2. 메트릭 서버 독립 실행
python scripts/monitoring/metrics_collector.py --port 8000

# 3. 법률 수집 실행 (메트릭 포함)
python scripts/assembly/collect_laws_optimized.py --sample 50 --enable-metrics
```

#### 접근 URL
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **메트릭 엔드포인트**: http://localhost:8000/metrics

#### 수집되는 메트릭
- `law_collection_pages_processed_total`: 처리된 총 페이지 수
- `law_collection_laws_collected_total`: 수집된 총 법률 수
- `law_collection_page_processing_seconds`: 페이지 처리 시간
- `law_collection_memory_usage_bytes`: 메모리 사용량
- `law_collection_cpu_usage_percent`: CPU 사용률

자세한 내용은 [Windows 모니터링 가이드](docs/development/windows_monitoring_guide.md)를 참조하세요.

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

### 3. 환경 변수 설정 (선택사항)

```bash
# OpenAI API 키 설정
export OPENAI_API_KEY="your_openai_key"

# Google AI API 키 설정
export GOOGLE_API_KEY="your_google_key"

# 디버그 모드 활성화
export DEBUG="true"
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

#### 기타 데이터 수집
```bash
# 전체 데이터 수집
python scripts/run_data_pipeline.py --mode full --oc your_email_id

# 법령 데이터만 수집
python scripts/run_data_pipeline.py --mode laws --oc your_email_id --query "민법" --display 50
```

### 5. 애플리케이션 실행

#### Streamlit 인터페이스 실행 (리팩토링된 버전)

```bash
cd gradio
pip install -r requirements.txt
python simple_langchain_app.py
```

#### 간단한 테스트 실행

```bash
cd gradio
python test_simple_query.py
```

#### FastAPI 서버 실행

```bash
cd api
pip install -r requirements.txt
python main.py
```

### 5. 접속

- **Streamlit 인터페이스**: http://localhost:8501
- **FastAPI 서버**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

## 🐳 Docker 사용

### Streamlit 인터페이스 실행 (리팩토링된 버전)

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
# Streamlit과 FastAPI를 동시에 실행하려면 각각의 폴더에서 실행
cd streamlit && docker-compose up -d &
cd api && docker-compose up -d &
```

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

#### Phase 2 신규 엔드포인트
- `POST /api/v1/chat/intelligent-v2` - 지능형 채팅 v2 (모든 개선사항 통합)
- `GET /api/v1/system/status` - 시스템 상태 확인 (모든 컴포넌트 점검)

#### 기존 엔드포인트
- `POST /api/v1/chat` - 채팅 메시지 처리
- `POST /api/v1/chat/intelligent` - 지능형 채팅 (Phase 1)
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

#### 지능형 채팅 v2 API (신규)
```python
import requests

# 지능형 채팅 v2 요청
response = requests.post(
    "http://localhost:8000/api/v1/chat/intelligent-v2",
    json={
        "message": "계약 해제 조건이 무엇인가요?",
        "session_id": "user_session_123",
        "max_results": 10,
        "include_law_sources": True,
        "include_precedent_sources": True,
        "include_conversation_history": True,
        "context_optimization": True,
        "answer_formatting": True
    }
)

result = response.json()
print(f"질문 유형: {result['question_type']}")
print(f"답변: {result['answer']}")
print(f"신뢰도: {result['confidence']['reliability_level']}")
print(f"법률 소스: {len(result['law_sources'])}개")
print(f"판례 소스: {len(result['precedent_sources'])}개")
```

#### 시스템 상태 확인 API (신규)
```python
import requests

# 시스템 상태 확인
response = requests.get("http://localhost:8000/api/v1/system/status")
status = response.json()

print(f"전체 상태: {status['overall_status']}")
print(f"데이터베이스: {status['components']['database']['status']}")
print(f"벡터 스토어: {status['components']['vector_store']['status']}")
print(f"AI 모델: {status['components']['ai_models']['status']}")
print(f"검색 엔진: {status['components']['search_engines']['status']}")
print(f"답변 생성기: {status['components']['answer_generator']['status']}")
```

#### 채팅 API (기존)
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

#### 하이브리드 검색 API (기존)
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

## 📊 데이터 현황

| 데이터 타입 | 수량 | 상태 | 비고 |
|------------|------|------|------|
| 법령 (API) | 13개 | ✅ 완료 | 민법, 상법, 형법 등 주요 법령 |
| 법령 (Assembly) | 7,680개 | ✅ 완료 | 전체 Raw 데이터 전처리 완료 (815개 파일, 규칙 기반 파서) (2025-10-13) |
| 판례 (Assembly) | 민사: 397개, 형사: 8개, 조세: 472개 | ✅ 완료 | 민사: 15,589개 섹션 임베딩, 형사: 372개 섹션 임베딩, 조세: 472개 파일 (2025-10-17) |
| 판례 (API) | 11개 | ✅ 완료 | 계약서 관련 판례 |
| 헌재결정례 | 0개 | ⏳ 대기 | 데이터 수집 필요 |
| 법령해석례 | 0개 | ⏳ 대기 | 데이터 수집 필요 |
| 행정규칙 | 0개 | ⏳ 대기 | 데이터 수집 필요 |
| 자치법규 | 0개 | ⏳ 대기 | 데이터 수집 필요 |

## 📊 로그 확인

### Streamlit 애플리케이션 로그
```bash
# Windows PowerShell - 실시간 로그 모니터링
Get-Content logs\gradio_app.log -Wait -Tail 50

# Windows CMD - 전체 로그 확인
type logs\gradio_app.log

# Linux/Mac - 실시간 로그 모니터링
tail -f logs/gradio_app.log

# Linux/Mac - 최근 50줄 확인
tail -n 50 logs/gradio_app.log
```

### 로그 레벨 설정
```bash
# DEBUG 레벨로 실행 (더 자세한 로그)
# Windows
set LOG_LEVEL=DEBUG
python gradio/app.py

# PowerShell
$env:LOG_LEVEL="DEBUG"
python gradio/app.py

# Linux/Mac
export LOG_LEVEL=DEBUG
python gradio/app.py
```

### 로그 파일 위치
- **Streamlit 앱 로그**: `logs/streamlit_app.log`
- **데이터 처리 로그**: `logs/` 디렉토리의 각종 `.log` 파일들
- **상세 로깅 가이드**: [docs/development/logging_guide.md](docs/development/logging_guide.md)

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
- [Streamlit](https://streamlit.io/) - UI 프레임워크
- [ChromaDB](https://www.trychroma.com/) - 벡터 데이터베이스

---



*LawFirmAI는 법률 전문가의 도구로 사용되며, 법률 자문을 대체하지 않습니다. 중요한 법률 문제는 반드시 자격을 갖춘 법률 전문가와 상담하시기 바랍니다.*
