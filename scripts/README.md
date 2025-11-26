# Scripts Directory Structure

LawFirmAI 프로젝트의 스크립트들이 목적과 용도에 따라 체계적으로 분류되어 관리됩니다.

## 📊 현재 상태 (2025-01-XX)

- **루트 레벨 파일**: 0개 ✅ (정리 완료)
- **카테고리별 폴더**: 19개
- **전체 스크립트 파일**: 약 275개

### ✅ 정리 완료

루트 레벨 파일들이 모두 적절한 하위 폴더로 이동되었습니다:

- **테스트 파일 (15개)** → `testing/` (하위 폴더별 분류)
- **검증 파일 (3개)** → `verification/`
- **체크 파일 (6개)** → `checks/`
- **도구 파일 (3개)** → `tools/`
- **기타 파일 (8개)** → `analysis/`, `migrations/`, `monitoring/`, `setup/`, `scripts/`

자세한 정리 내용은 `docs/scripts_organization_plan.md`를 참조하세요.

## 📁 폴더 구조

### 📊 **data_collection/** - 데이터 수집
다양한 법률 데이터 소스에서 데이터를 수집하는 스크립트들

#### Assembly 수집
- `assembly/collect_laws.py` - Assembly 법률 수집
- `assembly/collect_laws_only.py` - 법률만 수집
- `assembly/collect_laws_optimized.py` - 최적화된 법률 수집
- `assembly/collect_precedents.py` - 판례 수집
- `assembly/collect_precedents_by_category.py` - 카테고리별 판례 수집

#### 기타 데이터 수집
- `precedent/` - 판례 수집 스크립트
- `constitutional/` - 헌재결정례 수집 스크립트
- `legal_interpretation/` - 법령해석례 수집 스크립트
- `administrative_appeal/` - 행정심판례 수집 스크립트
- `legal_term/` - 법률용어 수집 스크립트
- `qa_generation/` - QA 데이터셋 생성 스크립트

#### 공통 유틸리티
- `common/assembly_collector.py` - Assembly 수집기
- `common/assembly_logger.py` - Assembly 로거
- `common/checkpoint_manager.py` - 체크포인트 관리자
- `common/common_utils.py` - 공통 유틸리티

### 🔧 **data_processing/** - 데이터 처리
법률 데이터의 전처리, 정제, 최적화를 담당하는 스크립트들

#### 전처리 파이프라인
- `preprocessing/preprocess_raw_data.py` - 원본 데이터 전처리
- `preprocessing/quality_improved_preprocess.py` - 품질 개선된 전처리
- `preprocessing/optimize_law_data.py` - 법률 데이터 최적화
- `preprocessing/batch_preprocess.py` - 배치 전처리

#### 파서 시스템
- `parsers/` - 법률 문서 파서 모듈들
  - `article_parser.py` - 조문 파서
  - `legal_structure_parser.py` - 법률 구조 파서
  - `html_parser.py` - HTML 파서
  - `text_normalizer.py` - 텍스트 정규화

#### 데이터 검증
- `validation/validate_data_quality.py` - 데이터 품질 검증
- `validation/check_parsing_quality.py` - 파싱 품질 확인
- `validation/verify_clean_data.py` - 정제된 데이터 검증

#### 처리 유틸리티
- `utilities/batch_update_law_content.py` - 배치 법률 내용 업데이트
- `utilities/update_law_content.py` - 법률 내용 업데이트
- `utilities/refine_law_data_from_html.py` - HTML에서 법률 데이터 정제
- `utilities/run_data_pipeline.py` - 데이터 파이프라인 실행
- `utilities/setup_env.py` - 환경 설정
- `utilities/add_missing_data_types.py` - 누락된 데이터 타입 추가

### 🧠 **ml_training/** - ML 및 벡터 임베딩
AI 모델의 훈련, 평가, 벡터 임베딩 생성을 담당하는 스크립트들

#### 모델 훈련
- `model_training/evaluate_legal_model.py` - 법률 모델 평가
- `model_training/finetune_legal_model.py` - 법률 모델 파인튜닝
- `model_training/prepare_expanded_training_dataset.py` - 확장된 훈련 데이터셋 준비
- `model_training/generate_expanded_training_dataset.py` - 확장된 훈련 데이터셋 생성
- `model_training/generate_comprehensive_training_dataset.py` - 포괄적 훈련 데이터셋 생성
- `model_training/prepare_training_dataset.py` - 훈련 데이터셋 준비
- `model_training/setup_lora_environment.py` - LoRA 환경 설정
- `model_training/analyze_kogpt2_structure.py` - KoGPT-2 구조 분석

#### 벡터 임베딩
- `vector_embedding/build_ml_enhanced_vector_db.py` - ML 강화 벡터 DB 구축
- `vector_embedding/build_ml_enhanced_vector_db_optimized.py` - 최적화된 ML 강화 벡터 DB 구축
- `vector_embedding/build_ml_enhanced_vector_db_cpu_optimized.py` - CPU 최적화된 ML 강화 벡터 DB 구축
- `vector_embedding/build_resumable_vector_db.py` - 재시작 가능한 벡터 DB 구축
- `vector_embedding/rebuild_improved_vector_db.py` - 개선된 벡터 DB 재구축
- `vector_embedding/test_faiss_direct.py` - FAISS 직접 테스트
- `vector_embedding/test_vector_embedding_basic.py` - 기본 벡터 임베딩 테스트

#### 훈련 데이터 준비
- `training_data/prepare_training_data.py` - 훈련 데이터 준비
- `training_data/optimized_prepare_training_data.py` - 최적화된 훈련 데이터 준비

### 🗄️ **database/** - 데이터베이스
데이터베이스 스키마, 백업, 분석을 담당하는 스크립트들

- `migrate_database_schema.py` - 데이터베이스 스키마 마이그레이션
- `backup_database.py` - 데이터베이스 백업
- `analyze_database_content.py` - 데이터베이스 내용 분석

### 📥 **ingest/** - 데이터 수집 및 저장
데이터를 데이터베이스에 수집하고 저장하는 스크립트들

- `ingest_cases.py` - 판례 데이터 수집 및 저장 (참조조문 자동 추출)
- `ingest_decisions.py` - 결정례 데이터 수집 및 저장 (참조조문 자동 추출)
- `ingest_interpretations.py` - 해석례 데이터 수집 및 저장 (참조조문 자동 추출)
- `ingest_aihub_from_r2.py` - AIHub 데이터 R2에서 다운로드 및 PostgreSQL 적재

### ⚙️ **setup/** - 환경 설정 스크립트
프로젝트 환경 설정 및 초기화 스크립트들

- `setup_aihub_env.bat` - AIHub 데이터 적재 환경 설정 (Windows)
- `setup_aihub_env.sh` - AIHub 데이터 적재 환경 설정 (Linux/Mac)

### 🔧 **utils/** - 유틸리티
공통 유틸리티 및 헬퍼 함수들

- `reference_statute_extractor.py` - 참조조문 추출기 (판례/결정례/해석례에서 법령 정보 추출)

### 🔄 **migrations/** - 데이터베이스 스키마 초기화 및 검증
데이터베이스 스키마 초기화, 검증, 유지보수 스크립트들

#### 구조
- `schema/` - 초기 스키마 SQL 파일들
- `scripts/init/` - 스키마 초기화 스크립트
- `scripts/validate/` - 스키마 검증 스크립트
- `scripts/maintenance/` - 유지보수 스크립트
- `utils/` - 공통 유틸리티 모듈

#### 주요 스크립트
- `scripts/init/run_postgresql_migration.py` - PostgreSQL 메인 스키마 초기화
- `scripts/init/init_open_law_schema.py` - Open Law 스키마 초기화
- `scripts/validate/validate_postgresql_schema.py` - 스키마 검증
- `scripts/validate/check_extensions.py` - 확장 확인

### 📊 **analysis/** - 데이터 분석
데이터 품질 분석, 모델 성능 분석을 담당하는 스크립트들

- `analyze_model_optimization.py` - 모델 최적화 분석
- `analyze_precedent_data.py` - 판례 데이터 분석
- `check_refined_data.py` - 정제된 데이터 확인
- `check_updated_file.py` - 업데이트된 파일 확인
- `improve_precedent_accuracy.py` - 판례 정확도 개선
- `validate_data_quality.py` - 데이터 품질 검증
- `validate_processed_data.py` - 전처리된 데이터 검증

### ⚡ **benchmarking/** - 성능 벤치마킹
모델과 벡터 저장소의 성능을 측정하는 스크립트들

- `benchmark_models.py` - 모델 성능 벤치마킹
- `benchmark_vector_stores.py` - 벡터 저장소 성능 벤치마킹

### 📈 **monitoring/** - 모니터링
시스템 모니터링, 로그 분석을 담당하는 스크립트들

- `analyze_logs.py` - 로그 분석
- `metrics_collector.py` - 메트릭 수집
- `quality_monitor.py` - 품질 모니터링

### 🧪 **testing/** - 테스트
각종 기능과 모듈의 테스트를 담당하는 스크립트들

#### 통합 테스트 (`integration/`)
- `test_v2_integration.py` - v2 통합 테스트
- `test_faiss_version_with_real_data.py` - FAISS 버전 실제 데이터 테스트
- `test_ingest_with_new_chunking.py` - 새 청킹 방식 수집 테스트

#### 품질 검증 테스트 (`quality/`)
- `test_reference_quality_improvements.py` - 참조 품질 개선 테스트
- `test_reference_quality_with_workflow.py` - 워크플로우 참조 품질 테스트
- `test_content_quality_validation.py` - 콘텐츠 품질 검증 테스트
- `test_performance_monitoring.py` - 성능 모니터링 테스트

#### 검색 테스트 (`search/`)
- `test_search_engine_hybrid_integration.py` - 하이브리드 검색 엔진 통합 테스트
- `test_search_quality_with_hybrid_chunking.py` - 하이브리드 청킹 검색 품질 테스트
- `test_partial_match_improvements.py` - 부분 일치 개선 테스트

#### 청킹 테스트 (`chunking/`)
- `test_chunking_strategies.py` - 청킹 전략 테스트

#### 추출 테스트 (`extraction/`)
- `test_complex_keyword_extraction.py` - 복잡한 키워드 추출 테스트
- `test_statute_content_extraction.py` - 법령 내용 추출 테스트
- `test_reference_statutes_in_sources.py` - 소스 내 참조 법령 테스트
- `test_stream_handler_integration.py` - 스트림 핸들러 통합 테스트

### ✅ **verification/** - 검증
데이터 검증 및 결과 확인 스크립트들

- `verify_reference_statutes.py` - 참조 법령 검증
- `verify_extraction_quality.py` - 추출 품질 검증
- `verify_dynamic_chunking_results.py` - 다이나믹 청킹 결과 검증

### 🔍 **checks/** - 체크
상태 확인 및 시스템 체크 스크립트들

- `check_auto_complete_status.py` - 자동 완료 상태 확인
- `check_pytorch_threads.py` - PyTorch 스레드 확인
- `check_re_embedding_status.ps1` - 재임베딩 상태 확인 (PowerShell)
- `check_search_logs.py` - 검색 로그 확인
- `check_statute_article_status.py` - 법령 조문 상태 확인
- `check_system_specs.py` - 시스템 사양 확인

### 🛠️ **tools/** - 도구
유틸리티 도구 스크립트들

- `create_test_version.py` - 테스트 버전 생성
- `assign_version_to_existing_embeddings.py` - 기존 임베딩에 버전 할당
- `wait_and_build_faiss_index.py` - 대기 후 FAISS 인덱스 빌드
- `analyze_scripts.py` - 스크립트 분석 도구

### 📜 **scripts/** - 래퍼 스크립트
자동화 래퍼 스크립트들

- `start_auto_complete.ps1` - 자동 완료 스크립트 시작 (PowerShell)

## 🚀 사용법

### 데이터 수집
```bash
# Assembly 법률 수집
python scripts/data_collection/assembly/collect_laws.py --sample 100

# 판례 수집
python scripts/data_collection/precedent/collect_precedents.py

# 헌재결정례 수집
python scripts/data_collection/constitutional/collect_constitutional_decisions.py
```

### 데이터 전처리
```bash
# 원본 데이터 전처리
python scripts/data_processing/preprocessing/preprocess_raw_data.py

# 품질 개선된 전처리
python scripts/data_processing/preprocessing/quality_improved_preprocess.py

# 배치 전처리
python scripts/data_processing/preprocessing/batch_preprocess.py
```

### ML 훈련 및 벡터 임베딩
```bash
# 벡터 DB 구축
python scripts/ml_training/vector_embedding/build_ml_enhanced_vector_db.py

# 모델 평가
python scripts/ml_training/model_training/evaluate_legal_model.py

# 훈련 데이터 준비
python scripts/ml_training/training_data/prepare_training_data.py
```

### 데이터베이스 관리
```bash
# 데이터베이스 백업
python scripts/database/backup_database.py

# 스키마 마이그레이션
python scripts/database/migrate_database_schema.py
```

### 데이터 수집 및 저장
```bash
# 판례 수집 (참조조문 자동 추출)
python scripts/ingest/ingest_cases.py --file data/... --domain "민사법"

# 결정례 수집 (참조조문 자동 추출)
python scripts/ingest/ingest_decisions.py --file data/... --domain "민사법"

# 해석례 수집 (참조조문 자동 추출)
python scripts/ingest/ingest_interpretations.py --file data/... --domain "민사법"

# AIHub 데이터 적재 (R2에서 다운로드)
python scripts/ingest/ingest_aihub_from_r2.py --dataset civil --object-key aihub/civil/data.zip
```

### 환경 설정
```bash
# AIHub 데이터 적재 환경 설정
# Windows
scripts\setup\setup_aihub_env.bat

# Linux/Mac
chmod +x scripts/setup/setup_aihub_env.sh
./scripts/setup/setup_aihub_env.sh
```

### 참조조문 마이그레이션
```bash
# 기존 데이터 참조조문 재추출
python scripts/migrations/migrate_reference_statutes.py --db data/lawfirm_v2.db --force

# 특정 타입만 재추출
python scripts/migrations/migrate_reference_statutes.py --db data/lawfirm_v2.db --type cases --force

# 추출 품질 검증
python scripts/verify_reference_statutes.py --db data/lawfirm_v2.db
```

## 📝 주의사항

1. **환경 설정**: 스크립트 실행 전 필요한 환경변수와 패키지가 설치되어 있는지 확인하세요.
2. **데이터 백업**: 중요한 데이터 처리 전에는 반드시 백업을 수행하세요.
3. **로그 확인**: 각 스크립트 실행 시 로그를 확인하여 오류가 없는지 점검하세요.
4. **메모리 관리**: 대용량 데이터 처리 시 메모리 사용량을 모니터링하세요.

## 🔄 마이그레이션 정보

이 디렉토리 구조는 2025-10-16에 대규모 재구성을 통해 개선되었습니다. 자세한 변경사항은 `docs/archive/structure_migration_2025-10-16.md`를 참조하세요.

### 📊 **analysis/** - 분석
데이터 분석, 품질 검증, 모델 최적화 분석을 담당하는 스크립트들

- `analyze_model_optimization.py` - 모델 최적화 분석
- `analyze_precedent_data.py` - 판례 데이터 분석
- `validate_data_quality.py` - 데이터 품질 검증
- `validate_processed_data.py` - 처리된 데이터 검증
- `check_updated_file.py` - 업데이트된 파일 확인
- `check_refined_data.py` - 정제된 데이터 확인
- `improve_precedent_accuracy.py` - 판례 정확도 개선
- `improve_precedent_accuracy_fixed.py` - 수정된 판례 정확도 개선
- `improve_precedent_accuracy_utf8.py` - UTF-8 판례 정확도 개선

### 📥 **collection/** - 데이터 수집
다양한 법률 데이터 수집 및 QA 데이터셋 생성을 담당하는 스크립트들

- `collect_administrative_appeals_new.py` - 새로운 행정심판 수집
- `collect_administrative_rules.py` - 행정규칙 수집
- `collect_committee_decisions.py` - 위원회 결정 수집
- `collect_laws.py` - 법률 수집
- `collect_local_ordinances.py` - 지방자치단체 조례 수집
- `collect_treaties.py` - 조약 수집
- `generate_qa_dataset.py` - QA 데이터셋 생성
- `generate_qa_with_llm.py` - LLM을 사용한 QA 생성
- `large_scale_generate_qa_dataset.py` - 대규모 QA 데이터셋 생성
- `llm_qa_generator.py` - LLM QA 생성기

### ⚡ **benchmarking/** - 벤치마킹
모델 및 벡터 스토어 성능 벤치마킹을 담당하는 스크립트들

- `benchmark_models.py` - 모델 벤치마킹
- `benchmark_vector_stores.py` - 벡터 스토어 벤치마킹

### 🧪 **tests/** - 테스트
다양한 기능의 테스트 스크립트들

- `test_ko_sroberta_korean.py` - ko-sroberta Korean 모델 테스트
- `test_law_record.py` - 법률 레코드 테스트
- `test_real_data.py` - 실제 데이터 테스트
- `test_simple_embedding.py` - 간단한 임베딩 테스트
- `test_vector_builder.py` - 벡터 빌더 테스트
- `test_vector_store.py` - 벡터 스토어 테스트
- `test_final_vector_embedding_performance.py` - 최종 벡터 임베딩 성능 테스트

### 📁 **기존 폴더들**
- `assembly/` - 국회 법률 데이터 처리
- `monitoring/` - 시스템 모니터링
- `precedent/` - 판례 데이터 처리
- `legal_interpretation/` - 법률 해석
- `legal_term/` - 법률 용어
- `constitutional_decision/` - 헌법재판소 결정
- `administrative_appeal/` - 행정심판

## 🚀 사용법

### 데이터 처리 파이프라인
```bash
# 전체 데이터 처리 파이프라인 실행
python scripts/data_processing/run_data_pipeline.py

# 특정 데이터 전처리
python scripts/data_processing/preprocess_raw_data.py
```

### 벡터 임베딩 생성
```bash
# ML 강화 벡터 임베딩 생성
python scripts/vector_embedding/build_ml_enhanced_vector_db_cpu_optimized.py

# 벡터 임베딩 테스트
python scripts/tests/test_vector_embedding_basic.py
```

### 모델 훈련
```bash
# 훈련 데이터셋 준비
python scripts/model_training/prepare_training_dataset.py

# 모델 파인튜닝
python scripts/model_training/finetune_legal_model.py
```

### 데이터베이스 관리
```bash
# 스키마 마이그레이션
python scripts/database/migrate_database_schema.py

# 데이터베이스 백업
python scripts/database/backup_database.py
```

## 📋 개발 가이드

### 새로운 스크립트 추가 시
1. **목적에 맞는 폴더 선택**: 스크립트의 주요 기능에 따라 적절한 폴더에 배치
2. **명명 규칙 준수**: 기능을 명확히 나타내는 파일명 사용
3. **문서화**: 스크립트 상단에 목적과 사용법 주석 추가
4. **README 업데이트**: 해당 폴더의 README에 새 스크립트 정보 추가

### 폴더별 책임
- **data_processing**: 데이터 전처리, 정제, 변환
- **model_training**: AI 모델 훈련, 평가, 데이터셋 준비
- **vector_embedding**: 벡터 임베딩 생성, 관리, 최적화
- **database**: 데이터베이스 스키마, 백업, 마이그레이션
- **analysis**: 데이터 분석, 품질 검증, 성능 분석
- **collection**: 외부 데이터 수집, QA 데이터셋 생성
- **benchmarking**: 성능 벤치마킹, 비교 분석
- **tests**: 기능 테스트, 검증 스크립트

---

**마지막 업데이트**: 2025-10-15  
**관리자**: LawFirmAI 개발팀
