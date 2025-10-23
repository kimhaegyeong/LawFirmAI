# 현행법령 데이터 수집 시스템

현행법령 목록 조회 API와 본문 조회 API를 사용하여 법령 데이터를 수집하고, 데이터베이스와 벡터 저장소에 저장하는 분리된 시스템입니다.

## 📋 시스템 구성

### 1. 핵심 컴포넌트

- **현행법령 수집기** (`current_law_collector.py`): API를 통한 데이터 수집
- **데이터베이스 관리자** (`database.py`): SQLite 데이터베이스 관리
- **벡터 저장소** (`vector_store.py`): FAISS 기반 벡터 검색

### 2. 실행 스크립트

- **데이터 수집** (`collect_current_laws.py`): 현행법령 데이터 수집
- **데이터베이스 업데이트** (`update_database.py`): 데이터베이스 저장
- **벡터 저장소 업데이트** (`update_vectors.py`): 벡터 임베딩 생성
- **통합 실행** (`run_pipeline.py`): 전체 파이프라인 실행

## 🚀 사용법

### 환경 설정

```bash
# 환경변수 설정
export LAW_OPEN_API_OC='your_email@example.com'

# 또는 .env 파일에 추가
echo "LAW_OPEN_API_OC=your_email@example.com" >> .env
```

### 1. 개별 단계 실행

#### 데이터 수집만 실행
```bash
# 기본 수집 (모든 현행법령)
python scripts/data_collection/law_open_api/current_laws/collect_current_laws.py

# 특정 키워드로 검색
python scripts/data_collection/law_open_api/current_laws/collect_current_laws.py --query "자동차"

# 샘플 수집 (10개만)
python scripts/data_collection/law_open_api/current_laws/collect_current_laws.py --sample 10

# 상세 정보 제외하고 수집
python scripts/data_collection/law_open_api/current_laws/collect_current_laws.py --no-details
```

#### 데이터베이스 업데이트만 실행
```bash
# 기본 업데이트
python scripts/data_collection/law_open_api/current_laws/update_database.py

# 기존 데이터 삭제 후 업데이트
python scripts/data_collection/law_open_api/current_laws/update_database.py --clear-existing

# 특정 배치 디렉토리 지정
python scripts/data_collection/law_open_api/current_laws/update_database.py --batch-dir "data/raw/law_open_api/current_laws/batches"
```

#### 벡터 저장소 업데이트만 실행
```bash
# 기본 업데이트
python scripts/data_collection/law_open_api/current_laws/update_vectors.py

# 기존 벡터 삭제 후 업데이트
python scripts/data_collection/law_open_api/current_laws/update_vectors.py --clear-existing

# 다른 모델 사용
python scripts/data_collection/law_open_api/current_laws/update_vectors.py --model-name "jhgan/ko-sroberta-multitask"
```

### 2. 통합 파이프라인 실행

#### 모든 단계 실행
```bash
# 전체 파이프라인 (수집 → 데이터베이스 → 벡터)
python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --all

# 특정 키워드로 전체 파이프라인 실행
python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --all --query "자동차"
```

#### 선택적 단계 실행
```bash
# 수집 + 데이터베이스만
python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --collect --database

# 데이터베이스 + 벡터만
python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --database --vectors
```

### 3. 테스트 및 검증

#### 연결 테스트
```bash
# 모든 연결 테스트
python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --test

# 개별 테스트
python scripts/data_collection/law_open_api/current_laws/collect_current_laws.py --test
python scripts/data_collection/law_open_api/current_laws/update_database.py --test
python scripts/data_collection/law_open_api/current_laws/update_vectors.py --test
```

#### Dry run (실행 계획 확인)
```bash
# 통합 파이프라인 계획 확인
python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --all --dry-run

# 개별 단계 계획 확인
python scripts/data_collection/law_open_api/current_laws/update_database.py --dry-run
python scripts/data_collection/law_open_api/current_laws/update_vectors.py --dry-run
```

## 📊 데이터 구조

### 배치 파일 구조
```json
{
  "batch_number": 1,
  "batch_size": 10,
  "start_page": 1,
  "end_page": 1,
  "timestamp": "2025-01-22T12:00:00",
  "laws": [
    {
      "법령ID": "1747",
      "법령명한글": "자동차관리법",
      "공포일자": 20151007,
      "시행일자": 20151007,
      "소관부처명": "국토교통부",
      "detailed_info": {
        "법령ID": "1747",
        "법령명_한글": "자동차관리법",
        "조문내용": "전체 조문 내용...",
        "별표내용": "별표 내용...",
        "부칙내용": "부칙 내용..."
      },
      "document_type": "current_law",
      "collected_at": "2025-01-22T12:00:00"
    }
  ]
}
```

### 데이터베이스 테이블
- **current_laws**: 현행법령 기본 정보
- **current_laws_fts**: 전문 검색용 FTS 테이블

### 벡터 저장소
- **문서 타입**: `current_law`
- **메타데이터**: 법령ID, 법령명, 소관부처, 시행일자 등
- **검색 기능**: 유사도 검색, 소관부처별 검색, 키워드 검색

## 🔧 주요 옵션

### 수집 옵션
- `--query`: 검색 질의
- `--max-pages`: 최대 페이지 수
- `--batch-size`: 배치 크기 (기본값: 10)
- `--sort-order`: 정렬 순서 (기본값: ldes)
- `--no-details`: 상세 정보 제외
- `--resume-checkpoint`: 체크포인트에서 재시작
- `--sample`: 샘플 수집

### 데이터베이스 옵션
- `--batch-dir`: 배치 파일 디렉토리
- `--pattern`: 파일 패턴
- `--db-batch-size`: 데이터베이스 배치 크기 (기본값: 100)
- `--clear-existing`: 기존 데이터 삭제
- `--summary-file`: 요약 파일 경로

### 벡터 저장소 옵션
- `--vector-batch-size`: 벡터화 배치 크기 (기본값: 50)
- `--model-name`: 임베딩 모델명
- `--clear-existing`: 기존 벡터 삭제

## 📁 파일 구조

```
scripts/data_collection/law_open_api/current_laws/
├── current_law_collector.py    # 수집기 클래스
├── collect_current_laws.py     # 데이터 수집 스크립트
├── update_database.py          # 데이터베이스 업데이트 스크립트
├── update_vectors.py           # 벡터 저장소 업데이트 스크립트
└── run_pipeline.py             # 통합 실행 스크립트

data/raw/law_open_api/current_laws/batches/
├── current_law_batch_20250122_120000_001.json
├── current_law_batch_20250122_120000_002.json
└── current_law_batch_summary_20250122_120000.json

results/
├── current_laws_collection_20250122_120000.json
├── current_laws_database_update_20250122_120000.json
├── current_laws_vector_update_20250122_120000.json
└── current_laws_integration_20250122_120000.json

logs/
├── current_laws_collection_20250122_120000.log
├── current_laws_database_update_20250122_120000.log
├── current_laws_vector_update_20250122_120000.log
└── current_laws_integration_20250122_120000.log
```

## 🎯 사용 시나리오

### 1. 처음 실행 (전체 수집)
```bash
# 모든 현행법령 수집 및 저장
python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --all
```

### 2. 특정 키워드 수집
```bash
# 자동차 관련 법령만 수집
python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --all --query "자동차"
```

### 3. 증분 업데이트
```bash
# 새로운 데이터만 수집
python scripts/data_collection/law_open_api/current_laws/collect_current_laws.py --resume-checkpoint

# 수집된 데이터를 데이터베이스에 추가
python scripts/data_collection/law_open_api/current_laws/update_database.py

# 벡터 저장소 업데이트
python scripts/data_collection/law_open_api/current_laws/update_vectors.py
```

### 4. 데이터 재구성
```bash
# 기존 데이터 삭제 후 전체 재구성
python scripts/data_collection/law_open_api/current_laws/run_pipeline.py --all --clear-existing
```

## ⚠️ 주의사항

1. **API 제한**: 국가법령정보센터 API는 일일 요청 제한이 있습니다.
2. **메모리 사용량**: 대량 데이터 처리 시 메모리 사용량을 모니터링하세요.
3. **디스크 공간**: 배치 파일과 벡터 인덱스는 상당한 디스크 공간을 사용합니다.
4. **네트워크**: 안정적인 인터넷 연결이 필요합니다.

## 🔍 문제 해결

### 일반적인 오류

1. **API 연결 실패**
   - OC 파라미터 확인
   - 네트워크 연결 확인
   - API 서버 상태 확인

2. **메모리 부족**
   - 배치 크기 줄이기
   - 시스템 메모리 확인

3. **디스크 공간 부족**
   - 불필요한 배치 파일 삭제
   - 디스크 공간 확보

### 로그 확인
```bash
# 최근 로그 파일 확인
ls -la logs/current_laws_*.log

# 로그 내용 확인
tail -f logs/current_laws_collection_*.log
```

## 📈 성능 최적화

1. **배치 크기 조정**: 시스템 성능에 따라 배치 크기 조정
2. **병렬 처리**: 여러 터미널에서 다른 키워드로 동시 수집
3. **체크포인트 활용**: 중단 시 재시작 기능 사용
4. **모델 선택**: 더 빠른 임베딩 모델 사용 고려

이 시스템을 통해 현행법령 데이터를 체계적으로 수집하고 관리할 수 있습니다.
