# 법령정보지식베이스 법령용어 수집 시스템

법령정보지식베이스 API를 활용한 법령용어 데이터 수집, 처리, 벡터 임베딩 생성 시스템입니다.

## 📋 개요

이 시스템은 법령정보지식베이스의 법령용어 조회 API를 통해 법령용어를 수집하고, `base_legal_terms` 폴더 구조에 체계적으로 저장하며, 벡터 임베딩을 생성하여 검색 시스템에 활용할 수 있도록 합니다.

## 🗂️ 폴더 구조

```
data/base_legal_terms/
├── raw/                             # 원본 수집 데이터
│   ├── term_lists/                  # 용어 목록 데이터
│   ├── term_details/                # 용어 상세 데이터
│   ├── term_relations/              # 용어 관계 데이터
│   └── api_responses/               # API 응답 원본
├── processed/                       # 전처리된 데이터
│   ├── cleaned_terms/               # 정제된 용어 데이터
│   ├── normalized_terms/            # 정규화된 용어 데이터
│   ├── validated_terms/             # 검증된 용어 데이터
│   └── integrated_terms/             # 통합된 용어 데이터
├── embeddings/                      # 벡터 임베딩
│   ├── base_legal_terms_index.faiss
│   ├── base_legal_terms_metadata.json
│   └── cache/
├── database/                        # 데이터베이스 파일
│   └── base_legal_terms.db
├── config/                          # 설정 파일
│   ├── base_legal_term_collection_config.py
│   └── collection_config.yaml
├── logs/                           # 로그 파일
├── progress/                       # 진행 상황 파일
└── reports/                        # 수집 보고서
```

## 🚀 사용법

### 1. 환경 설정

#### API 키 설정
```bash
# 환경변수 설정
export BASE_LEGAL_API_OC_ID=your_email_id
export BASE_LOG_LEVEL=INFO
```

#### Windows 환경변수 설정
```cmd
set BASE_LEGAL_API_OC_ID=your_email_id
set BASE_LOG_LEVEL=INFO
```

### 2. 개별 단계 실행

#### 목록 수집만 실행
```bash
python scripts/data_collection/base_legal_terms/base_legal_term_collector.py \
  --collect-lists \
  --start-page 1 \
  --end-page 10 \
  --batch-size 20 \
  --verbose
```

#### 상세 정보 수집만 실행
```bash
python scripts/data_collection/base_legal_terms/base_legal_term_collector.py \
  --collect-details \
  --detail-batch-size 50 \
  --verbose
```

#### 번갈아가면서 수집
```bash
python scripts/data_collection/base_legal_terms/base_legal_term_collector.py \
  --collect-alternating \
  --start-page 1 \
  --end-page 5 \
  --batch-size 20 \
  --detail-batch-size 50 \
  --verbose
```

#### 데이터 처리 실행
```bash
python scripts/data_processing/base_legal_terms/process_terms.py
```

#### 벡터 임베딩 생성 실행
```bash
python scripts/data_processing/base_legal_terms/generate_embeddings.py
```

### 3. 통합 파이프라인 실행

#### 전체 파이프라인 실행
```bash
python scripts/data_collection/base_legal_terms/run_pipeline.py \
  --collect-alternating \
  --start-page 1 \
  --end-page 5 \
  --batch-size 20 \
  --detail-batch-size 50 \
  --verbose
```

#### 특정 단계 건너뛰기
```bash
# 수집 단계 건너뛰고 처리부터 실행
python scripts/data_collection/base_legal_terms/run_pipeline.py \
  --skip-collection \
  --verbose

# 임베딩 단계만 실행
python scripts/data_collection/base_legal_terms/run_pipeline.py \
  --skip-collection \
  --skip-processing \
  --verbose
```

### 4. 배치 스크립트 실행

#### Windows
```cmd
# 기본 수집 실행
scripts\data_collection\base_legal_terms\run_collection.bat

# 통합 파이프라인 실행
scripts\data_collection\base_legal_terms\run_full_pipeline.bat
```

#### Linux/Mac
```bash
# 실행 권한 부여
chmod +x scripts/data_collection/base_legal_terms/run_collection.sh

# 기본 수집 실행
scripts/data_collection/base_legal_terms/run_collection.sh
```

## ⚙️ 설정 옵션

### 주요 설정 파라미터

- `--start-page`: 시작 페이지 (기본값: 1)
- `--end-page`: 종료 페이지 (기본값: 무제한)
- `--batch-size`: 목록 수집 배치 크기 (기본값: 20)
- `--detail-batch-size`: 상세 수집 배치 크기 (기본값: 50)
- `--query`: 검색 쿼리
- `--homonym-yn`: 동음이의어 포함 여부 (Y/N, 기본값: Y)
- `--display-count`: 페이지당 결과 수 (기본값: 100, 최대: 100)
- `--rate-limit-delay`: 요청 간 대기 시간(초) (기본값: 1.0)
- `--detail-delay`: 상세 조회 간 대기 시간(초) (기본값: 1.0)
- `--verbose`: 상세 로그 출력

### 환경 변수

- `BASE_LEGAL_API_OC_ID`: API 사용자 ID
- `BASE_LOG_LEVEL`: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
- `BASE_SENTENCE_BERT_MODEL`: 임베딩 모델명

## 📊 API 정보

### 법령정보지식베이스 법령용어 조회 API

- **URL**: `https://www.law.go.kr/DRF/lawSearch.do?target=lstrmAI`
- **응답 형태**: JSON
- **페이지당 최대**: 100개
- **새로운 필드**: 동음이의어 정보, 용어간관계 링크, 조문간관계 링크

### 요청 파라미터

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| OC | string | 필수 | 사용자 이메일 ID |
| target | string | 필수 | 서비스 대상 (lstrmAI) |
| type | char | 필수 | 출력 형태 (JSON) |
| query | string | 선택 | 법령용어명 검색 쿼리 |
| display | int | 선택 | 검색 결과 개수 (기본: 20, 최대: 100) |
| page | int | 선택 | 검색 결과 페이지 (기본: 1) |
| homonymYn | char | 선택 | 동음이의어 존재여부 (Y/N) |

### 응답 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| 법령용어ID | string | 법령용어 순번 |
| 법령용어명 | string | 법령용어명 |
| 동음이의어존재여부 | string | 동음이의어 존재여부 |
| 비고 | string | 동음이의어 내용 |
| 용어간관계링크 | string | 법령용어-일상용어 연계 정보 상세링크 |
| 조문간관계링크 | string | 법령용어-조문 연계 정보 상세링크 |

## 🔧 데이터 처리 과정

### 1. 수집 단계
- 법령용어 목록 수집
- 상세 정보 수집
- 관계 정보 수집

### 2. 처리 단계
- 데이터 정제 및 정규화
- 키워드 추출
- 카테고리 분류
- 품질 점수 계산
- 중복 제거

### 3. 임베딩 단계
- 텍스트 임베딩 생성
- FAISS 인덱스 구축
- 메타데이터 저장

## 📈 예상 결과

### 수집 규모
- **용어 수**: 10,000-15,000개 (기존 시스템 대비 대폭 확장)
- **상세 정보**: 각 용어별 정의, 동음이의어, 관계 정보
- **관계 데이터**: 용어간관계, 조문간관계 링크

### 품질 향상
- 동음이의어 구분으로 정확도 향상
- 관계 정보로 컨텍스트 이해도 증대
- 대규모 데이터로 검색 성능 개선

## 🛠️ 문제 해결

### 일반적인 문제

1. **API 키 오류**
   - 환경변수 `BASE_LEGAL_API_OC_ID` 설정 확인
   - 올바른 이메일 ID 사용 확인

2. **메모리 부족**
   - 배치 크기 조정 (`--batch-size`, `--detail-batch-size`)
   - GPU 사용 가능시 자동으로 GPU 사용

3. **네트워크 오류**
   - 재시도 로직 자동 실행 (3분, 5분, 10분 대기)
   - 요청 간 대기 시간 조정 (`--rate-limit-delay`)

### 로그 확인

```bash
# 로그 파일 위치
data/base_legal_terms/logs/collection.log
data/base_legal_terms/logs/error.log
```

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.

## 📞 지원

문제가 발생하면 이슈를 생성하거나 로그 파일을 확인해주세요.
