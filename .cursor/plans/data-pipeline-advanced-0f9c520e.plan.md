<!-- 0f9c520e-cde8-4420-876f-029ffb4531a0 ccc8c471-6b4e-4018-b2c2-2edb91a4f7fd -->
# 데이터 파이프라인 고도화 계획

## 목표

새로운 law_only 데이터를 자동으로 감지하여 전처리, 벡터 임베딩, DB 저장까지 완전 자동화된 증분 처리 파이프라인 구축

## 핵심 요구사항

- ko-sroberta-multitask 모델 사용 (768차원)
- law_only 데이터 우선 처리 후 점진적 확장
- 전체 자동화 (감지 → 전처리 → 임베딩 → DB 저장)
- 증분 처리 (기존 데이터 유지, 신규 데이터만 추가)

## 1. 자동 데이터 감지 시스템

### 1.1 데이터 소스 감지기 생성

새로운 스크립트: `scripts/data_processing/auto_data_detector.py`

주요 기능:

- `data/raw/assembly/law_only/` 디렉토리 스캔
- 날짜별 폴더 감지 (예: `20251016/`)
- 파일 패턴 매칭 (`law_only_page_*.json`)
- JSON 메타데이터 분석하여 데이터 유형 자동 분류

핵심 클래스:

```python
class AutoDataDetector:
    def detect_new_data_sources(base_path: str) -> Dict[str, List[Path]]
    def classify_data_type(file_path: Path) -> str
    def get_data_statistics(files: List[Path]) -> Dict
```

### 1.2 처리 이력 추적 시스템

새로운 데이터베이스 테이블: `processed_files` (data/lawfirm.db에 추가)

스키마:

```sql
CREATE TABLE IF NOT EXISTS processed_files (
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    data_type TEXT NOT NULL,
    processed_at TIMESTAMP,
    processing_status TEXT,
    record_count INTEGER
)
```

구현 파일: `source/data/database.py`에 메서드 추가

- `mark_file_as_processed(file_path, file_hash, data_type)`
- `is_file_processed(file_path) -> bool`
- `get_unprocessed_files(data_type) -> List[Path]`

## 2. 증분 전처리 시스템

### 2.1 증분 전처리 프로세서

새로운 스크립트: `scripts/data_processing/incremental_preprocessor.py`

주요 기능:

- 미처리 파일만 선별하여 전처리
- 기존 `LegalDataProcessor` 재사용
- 배치 단위 처리 및 메모리 관리
- 체크포인트 시스템 통합

핵심 클래스:

```python
class IncrementalPreprocessor:
    def __init__(self, checkpoint_manager: CheckpointManager)
    def process_new_files_only(data_path: str) -> ProcessingResult
    def resume_from_checkpoint() -> bool
    def create_checkpoint(current_state: Dict) -> bool
```

처리 흐름:

1. 미처리 파일 목록 조회
2. 파일 해시 계산 및 중복 확인
3. 배치 단위로 전처리 (기존 `LegalDataProcessor` 사용)
4. 처리 결과를 `data/processed/assembly/law_only/YYYYMMDD/` 저장
5. DB에 처리 이력 기록

## 3. 자동 벡터 임베딩 시스템

### 3.1 증분 벡터 임베딩 생성기

새로운 스크립트: `scripts/ml_training/vector_embedding/incremental_vector_builder.py`

기존 `MLEnhancedVectorBuilder` 확장:

- 기존 FAISS 인덱스 로드
- 신규 문서만 임베딩 생성
- 기존 인덱스에 추가 (faiss.IndexFlatIP.add)
- 메타데이터 병합

핵심 클래스:

```python
class IncrementalVectorBuilder:
    def __init__(self, model_name="jhgan/ko-sroberta-multitask")
    def load_existing_index(index_path: str) -> bool
    def add_new_documents(processed_files: List[Path]) -> bool
    def save_updated_index(output_path: str) -> bool
```

처리 흐름:

1. 기존 FAISS 인덱스 로드 (`data/embeddings/ml_enhanced_ko_sroberta/`)
2. 신규 처리된 파일만 식별
3. 문서 청크 생성 (법률 조문별)
4. ko-sroberta-multitask로 임베딩 생성 (배치 크기: 20)
5. 기존 인덱스에 추가 (증분 업데이트)
6. 업데이트된 인덱스 저장

## 4. 데이터베이스 통합 시스템

### 4.1 증분 DB 임포터

기존 스크립트 확장: `scripts/data_processing/utilities/import_laws_to_db.py`

추가 기능:

- `--incremental` 모드 추가
- 기존 법률 중복 확인 (law_id 기반)
- 업데이트 vs 신규 삽입 자동 판단
- FTS(Full-Text Search) 인덱스 자동 업데이트

핵심 메서드:

```python
def import_incremental(file_path: Path, db_path: str) -> ImportResult:
    # law_id 존재 여부 확인
    # 존재하면 UPDATE, 없으면 INSERT
    # FTS 인덱스 자동 동기화
```

## 5. 통합 자동화 파이프라인

### 5.1 파이프라인 오케스트레이터

새로운 스크립트: `scripts/data_processing/auto_pipeline_orchestrator.py`

전체 워크플로우 자동화:

```python
class AutoPipelineOrchestrator:
    def run_auto_pipeline(data_source="law_only") -> PipelineResult:
        # Step 1: 신규 데이터 감지
        detector = AutoDataDetector()
        new_files = detector.detect_new_data_sources(
            "data/raw/assembly/law_only/"
        )
        
        # Step 2: 증분 전처리
        preprocessor = IncrementalPreprocessor()
        processed_files = preprocessor.process_new_files_only(new_files)
        
        # Step 3: 증분 벡터 임베딩
        vector_builder = IncrementalVectorBuilder()
        vector_builder.load_existing_index()
        vector_builder.add_new_documents(processed_files)
        vector_builder.save_updated_index()
        
        # Step 4: DB 증분 임포트
        db_importer = IncrementalDBImporter()
        db_importer.import_incremental(processed_files)
        
        # Step 5: 처리 통계 생성
        generate_pipeline_report()
```

실행 명령:

```bash
python scripts/data_processing/auto_pipeline_orchestrator.py \
    --data-source law_only \
    --auto-detect
```

### 5.2 모니터링 및 로깅

모든 단계별 진행 상황 추적:

- 실시간 진행률 표시
- 단계별 소요 시간 측정
- 에러 발생 시 체크포인트 저장
- 처리 통계 JSON 리포트 생성

로그 파일: `logs/auto_pipeline_{date}.log`

리포트 파일: `reports/pipeline_report_{date}.json`

## 6. 체크포인트 및 복구 시스템

### 6.1 통합 체크포인트 관리

기존 `CheckpointManager` 확장하여 파이프라인 전체 상태 저장:

```python
checkpoint_data = {
    "pipeline_stage": "preprocessing",  # 또는 "vectorizing", "importing"
    "current_file_index": 42,
    "total_files": 200,
    "processed_files": [...],
    "failed_files": [...],
    "vector_index_updated": False,
    "db_import_completed": False
}
```

복구 메커니즘:

- 각 단계 완료 시 체크포인트 저장
- 중단 시 마지막 체크포인트부터 재개
- 부분 완료된 작업 롤백 지원

## 7. 설정 및 실행

### 7.1 설정 파일

새로운 설정: `config/pipeline_config.yaml`

```yaml
data_sources:
  law_only:
    enabled: true
    priority: 1
    raw_path: "data/raw/assembly/law_only"
    processed_path: "data/processed/assembly/law_only"

preprocessing:
  batch_size: 100
  enable_term_normalization: true
  enable_ml_enhancement: true

vectorization:
  model_name: "jhgan/ko-sroberta-multitask"
  dimension: 768
  batch_size: 20
  chunk_size: 200
  index_type: "flat"

incremental:
  enabled: true
  check_file_hash: true
  skip_duplicates: true
```

### 7.2 실행 명령어

전체 자동 파이프라인:

```bash
python scripts/data_processing/auto_pipeline_orchestrator.py --auto-detect
```

특정 날짜 데이터만 처리:

```bash
python scripts/data_processing/auto_pipeline_orchestrator.py \
    --data-path data/raw/assembly/law_only/20251016
```

체크포인트에서 재개:

```bash
python scripts/data_processing/auto_pipeline_orchestrator.py --resume
```

## 8. 검증 및 테스트

### 8.1 통합 테스트 스크립트

새로운 테스트: `scripts/tests/test_auto_pipeline.py`

테스트 항목:

- 신규 데이터 자동 감지 검증
- 증분 전처리 정확성 검증
- 벡터 임베딩 무결성 검증
- DB 중복 방지 검증
- 체크포인트 복구 검증

### 8.2 성능 벤치마크

예상 처리 성능 (20251016 데이터 기준):

- 파일 감지: < 1초
- 전처리: ~200 파일/시간
- 벡터 임베딩: ~100 법률/시간 (ko-sroberta)
- DB 임포트: ~500 법률/시간

## 핵심 파일 구조

```
scripts/
├── data_processing/
│   ├── auto_data_detector.py          (새로 생성)
│   ├── incremental_preprocessor.py    (새로 생성)
│   ├── auto_pipeline_orchestrator.py  (새로 생성)
│   └── utilities/
│       └── import_laws_to_db.py       (기능 확장)
├── ml_training/
│   └── vector_embedding/
│       └── incremental_vector_builder.py  (새로 생성)
└── tests/
    └── test_auto_pipeline.py          (새로 생성)

source/
└── data/
    └── database.py                     (메서드 추가)

config/
└── pipeline_config.yaml                (새로 생성)
```

### To-dos

- [ ] 데이터베이스에 processed_files 테이블 추가 및 처리 이력 추적 메서드 구현
- [ ] 자동 데이터 감지 시스템 구현 (auto_data_detector.py)
- [ ] 증분 전처리 프로세서 구현 (incremental_preprocessor.py)
- [ ] 증분 벡터 임베딩 생성기 구현 (incremental_vector_builder.py)
- [ ] DB 임포터에 증분 모드 추가 (import_laws_to_db.py 확장)
- [ ] 통합 자동화 파이프라인 오케스트레이터 구현 (auto_pipeline_orchestrator.py)
- [ ] 파이프라인 설정 파일 생성 (pipeline_config.yaml)
- [ ] 통합 테스트 스크립트 작성 및 검증 (test_auto_pipeline.py)