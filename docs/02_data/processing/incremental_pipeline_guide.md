# LawFirmAI 증분 전처리 파이프라인 가이드

## 개요

LawFirmAI 프로젝트의 **증분 전처리 파이프라인**은 새로운 법률 데이터가 추가될 때마다 자동으로 감지하고 처리하는 완전 자동화된 시스템입니다. 이 시스템은 기존 데이터를 보존하면서 새로운 데이터만 효율적으로 처리하여 리소스를 절약하고 처리 속도를 최적화합니다.

## 🚀 주요 특징

### 🔍 자동 데이터 감지
- **파일 패턴 인식**: 파일명과 메타데이터를 기반으로 데이터 유형 자동 분류
- **중복 방지**: 이미 처리된 파일은 자동으로 스킵
- **해시 기반 추적**: 파일 내용 변경 감지를 위한 SHA256 해시 사용
- **배치 처리**: 여러 파일을 한 번에 감지하고 처리

### ⚡ 증분 처리
- **새 데이터만 처리**: 기존 데이터는 건드리지 않고 새로운 데이터만 처리
- **상태 추적**: 데이터베이스에서 각 파일의 처리 상태를 추적
- **체크포인트 시스템**: 중단 시 이어서 처리 가능
- **메모리 최적화**: 대용량 파일도 효율적으로 처리

### 🔄 완전 자동화
- **원스톱 처리**: 데이터 감지 → 전처리 → 벡터 임베딩 → DB 저장
- **오류 복구**: 실패한 파일은 별도 추적하여 재처리 가능
- **로깅 시스템**: 모든 처리 과정을 상세히 기록
- **통계 제공**: 처리 결과에 대한 상세한 통계 정보

## 📁 시스템 구조

```
scripts/
├── data_processing/
│   ├── auto_data_detector.py                    # 자동 데이터 감지
│   ├── incremental_preprocessor.py               # 증분 전처리 (법률)
│   ├── incremental_precedent_preprocessor.py     # 증분 전처리 (판례)
│   ├── precedent_preprocessor.py                 # 판례 전용 전처리기
│   ├── auto_pipeline_orchestrator.py            # 통합 오케스트레이터
│   ├── preprocessing/                            # 기본 전처리 스크립트
│   │   └── preprocess_laws.py                   # ML-enhanced 전처리
│   ├── quality/                                  # 품질 관리 모듈
│   │   ├── data_quality_validator.py
│   │   ├── automated_data_cleaner.py
│   │   └── real_time_quality_monitor.py
│   └── utilities/                                # 유틸리티 스크립트
│       ├── import_laws_to_db.py                  # DB 임포트 (법률)
│       └── import_precedents_to_db.py            # DB 임포트 (판례)
└── ml_training/
    └── vector_embedding/
        ├── incremental_vector_builder.py          # 증분 벡터 임베딩 (법률)
        └── incremental_precedent_vector_builder.py  # 증분 벡터 임베딩 (판례)
```

## 🛠️ 설치 및 설정

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 설정

```bash
# 환경변수 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LOG_LEVEL=INFO
```

### 3. 데이터베이스 초기화

```bash
# 데이터베이스 테이블 생성 (processed_files 테이블 포함)
python -c "from source.data.database import DatabaseManager; DatabaseManager()"
```

## 📖 사용법

### 1. 전체 파이프라인 실행

```bash
# law_only 데이터에 대한 전체 파이프라인 실행
python scripts/data_processing/auto_pipeline_orchestrator.py --data-type law_only

# 판례 데이터에 대한 전체 파이프라인 실행 (민사)
python scripts/data_processing/auto_pipeline_orchestrator.py --data-type precedent_civil

# 판례 데이터에 대한 전체 파이프라인 실행 (형사)
python scripts/data_processing/auto_pipeline_orchestrator.py --data-type precedent_criminal

# 판례 데이터에 대한 전체 파이프라인 실행 (가사)
python scripts/data_processing/auto_pipeline_orchestrator.py --data-type precedent_family

# 모든 데이터 유형 처리
python scripts/data_processing/auto_pipeline_orchestrator.py --data-type all
```

### 2. 개별 단계 실행

#### 데이터 감지
```bash
python scripts/data_processing/auto_data_detector.py --base-path data/raw/assembly/law_only --data-type law_only --verbose
```

#### 증분 전처리
```bash
# 법률 데이터 전처리
python scripts/data_processing/incremental_preprocessor.py --data-type law_only --verbose

# 판례 데이터 전처리 (민사)
python scripts/data_processing/incremental_precedent_preprocessor.py --category civil --verbose

# 판례 데이터 전처리 (형사)
python scripts/data_processing/incremental_precedent_preprocessor.py --category criminal --verbose

# 판례 데이터 전처리 (가사)
python scripts/data_processing/incremental_precedent_preprocessor.py --category family --verbose
```

#### 증분 벡터 임베딩
```bash
# 법률 데이터 벡터 임베딩
python scripts/ml_training/vector_embedding/incremental_vector_builder.py

# 판례 데이터 벡터 임베딩 (민사)
python scripts/ml_training/vector_embedding/incremental_precedent_vector_builder.py --category civil

# 판례 데이터 벡터 임베딩 (형사)
python scripts/ml_training/vector_embedding/incremental_precedent_vector_builder.py --category criminal

# 판례 데이터 벡터 임베딩 (가사)
python scripts/ml_training/vector_embedding/incremental_precedent_vector_builder.py --category family
```

#### DB 임포트 (증분 모드)
```bash
# 법률 데이터 DB 임포트
python scripts/data_processing/utilities/import_laws_to_db.py --input data/processed/assembly/law_only/20251016 --incremental

# 판례 데이터 DB 임포트 (민사)
python scripts/data_processing/utilities/import_precedents_to_db.py --input data/processed/assembly/precedent/civil/20251016 --category civil --incremental

# 판례 데이터 DB 임포트 (형사)
python scripts/data_processing/utilities/import_precedents_to_db.py --input data/processed/assembly/precedent/criminal/20251016 --category criminal --incremental

# 판례 데이터 DB 임포트 (가사)
python scripts/data_processing/utilities/import_precedents_to_db.py --input data/processed/assembly/precedent/family/20251016 --category family --incremental
```

### 3. 설정 파일 사용

```yaml
# config/pipeline_config.yaml
data_sources:
  law_only:
    enabled: true
    priority: 1
    raw_path: "data/raw/assembly/law_only"
    processed_path: "data/processed/assembly/law_only"
    
  precedent_civil:
    enabled: true
    priority: 2
    raw_path: "data/raw/assembly/precedent"
    processed_path: "data/processed/assembly/precedent/civil"
    
  precedent_criminal:
    enabled: true
    priority: 3
    raw_path: "data/raw/assembly/precedent"
    processed_path: "data/processed/assembly/precedent/criminal"
    
  precedent_family:
    enabled: true
    priority: 4
    raw_path: "data/raw/assembly/precedent"
    processed_path: "data/processed/assembly/precedent/family"

paths:
  raw_data_base: "data/raw/assembly"
  processed_data_base: "data/processed/assembly"
  embedding_output: "data/embeddings/ml_enhanced_ko_sroberta"
  precedent_embedding_output: "data/embeddings/ml_enhanced_ko_sroberta_precedents"
  database: "data/lawfirm.db"

embedding:
  model_name: "jhgan/ko-sroberta-multitask"
  dimension: 768
  index_type: "flat"
  batch_size: 100

preprocessing:
  enable_term_normalization: true
  max_memory_usage: 0.8
  batch_size: 50
```

## 🔧 주요 컴포넌트

### AutoDataDetector
- **기능**: 새로운 데이터 파일 자동 감지 및 분류
- **특징**: 파일 패턴 매칭, 메타데이터 분석, 중복 제거
- **출력**: 처리할 파일 목록과 데이터 유형 정보

### IncrementalPreprocessor
- **기능**: 새로운 파일만 선별하여 전처리
- **특징**: 체크포인트 지원, 배치 처리, 오류 복구
- **출력**: ML 강화된 전처리된 데이터

### IncrementalPrecedentPreprocessor
- **기능**: 판례 데이터에 대한 새로운 파일만 선별하여 전처리
- **특징**: 카테고리별 처리 (민사/형사/가사), 판례 전용 파싱, 체크포인트 지원
- **출력**: ML 강화된 전처리된 판례 데이터

### IncrementalPrecedentVectorBuilder
- **기능**: 전처리된 판례 데이터로부터 벡터 임베딩 생성
- **특징**: 별도 FAISS 인덱스, 판례 섹션별 임베딩, 카테고리별 관리
- **출력**: 업데이트된 판례 벡터 인덱스

### AutoPipelineOrchestrator
- **기능**: 전체 파이프라인 통합 관리
- **특징**: 단계별 실행, 오류 처리, 통계 수집
- **출력**: 전체 처리 결과 리포트

## 📊 처리 결과 예시

### 판례 처리 결과 예시
```bash
==================================================
PRECEDENT PIPELINE EXECUTION SUMMARY
==================================================
Overall Status: completed
Duration: 1066.3 seconds (17.8 minutes)
Category: civil

Step 1 - Data Detection:
  Total new files: 397
  Files by type: {'precedent_civil': 397}

Step 2 - Precedent Preprocessing:
  Successfully processed: 397 files
  Failed to process: 0 files
  Processing time: 16.2 seconds

Step 3 - Vector Embedding:
  Successfully embedded: 397 files
  Total chunks added: 15,589
  Embedding time: 1044.9 seconds

Step 4 - Database Import:
  Imported cases: 0
  Updated cases: 0
  Skipped cases: 0
  Import time: 0.0 seconds

==================================================
DATABASE STATISTICS
==================================================
Total precedent cases: 0
Total precedent sections: 0
Total precedent parties: 0
```
### 법률 처리 결과 예시
```bash
==================================================
AUTOMATED PIPELINE EXECUTION SUMMARY
==================================================
Overall Status: completed
Duration: 45.2 seconds

Step 1 - Data Detection:
  Total new files: 373
  Files by type: {'law_only': 373}

Step 2 - Incremental Preprocessing:
  Successfully processed: 373 files
  Failed to process: 0 files
  Processing time: 14.85 seconds

Step 3 - Vector Embedding:
  Successfully embedded: 373 files
  Total chunks added: 1,962
  Embedding time: 8.3 seconds

Step 4 - Database Import:
  Imported laws: 1,895
  Updated laws: 67
  Skipped laws: 0
  Import time: 30.35 seconds

==================================================
DATABASE STATISTICS
==================================================
Total laws in database: 4,321
Total articles: 180,684
FTS laws: 4,321
FTS articles: 180,684
```

## 🚨 오류 처리 및 복구

### 일반적인 오류와 해결방법

#### 1. 처리 상태 초기화
```bash
# 특정 날짜의 파일 처리 상태 초기화
python -c "
import sys
sys.path.append('.')
from source.data.database import DatabaseManager
db = DatabaseManager()
rows = db.execute_update('DELETE FROM processed_files WHERE file_path LIKE \"%20251016%\"')
print(f'Cleared {rows} processing records')
"
```

#### 2. 체크포인트에서 재개
```bash
# 체크포인트에서 전처리 재개
python scripts/data_processing/incremental_preprocessor.py --data-type law_only --resume
```

#### 3. 특정 파일 재처리
```bash
# 특정 파일만 재처리
python scripts/data_processing/incremental_preprocessor.py --input-files data/raw/assembly/law_only/20251016/problem_file.json
```

### 로그 확인
```bash
# 처리 로그 확인
tail -f logs/pipeline.log

# 특정 오류 검색
grep "ERROR" logs/pipeline.log
```

## 🔍 모니터링 및 디버깅

### 처리 상태 확인
```python
from source.data.database import DatabaseManager

db = DatabaseManager()

# 처리된 파일 통계
stats = db.get_processing_statistics()
print(f"Total processed files: {stats['total_files']}")
print(f"Completed: {stats['completed']}")
print(f"Failed: {stats['failed']}")

# 특정 데이터 유형별 통계
law_only_stats = db.get_processed_files_by_type('law_only')
print(f"Law-only files: {len(law_only_stats)}")
```

### 벡터 인덱스 상태 확인
```python
from source.data.vector_store import LegalVectorStore

vector_store = LegalVectorStore()
vector_store.load_index("data/embeddings/ml_enhanced_ko_sroberta")
print(f"Total vectors in index: {vector_store.index.ntotal}")
```

## 🎯 성능 최적화

### 배치 크기 조정
- **전처리**: `batch_size=50` (메모리 사용량에 따라 조정)
- **벡터 임베딩**: `batch_size=100` (GPU 메모리에 따라 조정)
- **DB 임포트**: 기본값 사용 (SQLite 최적화)

### 메모리 관리
- **스트리밍 처리**: 대용량 파일을 청크 단위로 처리
- **가비지 컬렉션**: 각 단계 후 메모리 정리
- **인덱스 압축**: FAISS 인덱스 주기적 압축

### 병렬 처리
- **멀티프로세싱**: CPU 집약적 작업에 적용
- **비동기 I/O**: 파일 읽기/쓰기 최적화
- **배치 처리**: 데이터베이스 작업 최적화

## 🔮 향후 개선 계획

### 단기 계획
- [ ] 실시간 모니터링 대시보드 구축
- [ ] 자동 재시도 메커니즘 강화
- [ ] 처리 성능 메트릭 수집

### 중기 계획
- [ ] 분산 처리 지원 (여러 서버)
- [ ] 클라우드 스토리지 연동
- [ ] 실시간 알림 시스템

### 장기 계획
- [ ] AI 기반 데이터 품질 검증
- [ ] 자동 스키마 진화 지원
- [ ] 멀티 테넌트 아키텍처

## 📚 관련 문서

- [데이터 전처리 가이드](preprocessing_guide.md): 기본 전처리 파이프라인
- [벡터 임베딩 가이드](../embedding/README.md): 벡터 임베딩 생성
- [데이터베이스 스키마](../../10_technical_reference/database_schema.md): 데이터베이스 구조
- [API 문서](../../07_api/README.md): API 사용법
