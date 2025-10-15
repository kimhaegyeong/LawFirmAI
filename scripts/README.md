# Scripts Directory Structure

LawFirmAI 프로젝트의 스크립트들이 목적과 용도에 따라 체계적으로 분류되어 관리됩니다.

## 📁 폴더 구조

### 🔧 **data_processing/** - 데이터 처리
법률 데이터의 전처리, 정제, 최적화를 담당하는 스크립트들

- `preprocess_raw_data.py` - 원본 데이터 전처리
- `quality_improved_preprocess.py` - 품질 개선된 전처리
- `optimize_law_data.py` - 법률 데이터 최적화
- `batch_update_law_content.py` - 배치 법률 내용 업데이트
- `update_law_content.py` - 법률 내용 업데이트
- `refine_law_data_from_html.py` - HTML에서 법률 데이터 정제
- `batch_preprocess.py` - 배치 전처리
- `run_data_pipeline.py` - 데이터 파이프라인 실행
- `setup_env.py` - 환경 설정
- `add_missing_data_types.py` - 누락된 데이터 타입 추가

### 🧠 **model_training/** - 모델 훈련
AI 모델의 훈련, 평가, 데이터셋 준비를 담당하는 스크립트들

- `evaluate_legal_model.py` - 법률 모델 평가
- `finetune_legal_model.py` - 법률 모델 파인튜닝
- `prepare_expanded_training_dataset.py` - 확장된 훈련 데이터셋 준비
- `generate_expanded_training_dataset.py` - 확장된 훈련 데이터셋 생성
- `generate_comprehensive_training_dataset.py` - 포괄적 훈련 데이터셋 생성
- `prepare_training_dataset.py` - 훈련 데이터셋 준비
- `setup_lora_environment.py` - LoRA 환경 설정
- `analyze_kogpt2_structure.py` - KoGPT-2 구조 분석

### 🔍 **vector_embedding/** - 벡터 임베딩
벡터 임베딩 생성, 관리, 테스트를 담당하는 스크립트들

- `build_ml_enhanced_vector_db.py` - ML 강화 벡터 DB 구축
- `build_ml_enhanced_vector_db_optimized.py` - 최적화된 ML 강화 벡터 DB 구축
- `build_ml_enhanced_vector_db_cpu_optimized.py` - CPU 최적화된 ML 강화 벡터 DB 구축
- `build_resumable_vector_db.py` - 재시작 가능한 벡터 DB 구축
- `rebuild_improved_vector_db.py` - 개선된 벡터 DB 재구축
- `test_faiss_direct.py` - FAISS 직접 테스트
- `test_vector_embedding_basic.py` - 기본 벡터 임베딩 테스트

### 🗄️ **database/** - 데이터베이스
데이터베이스 스키마, 백업, 분석을 담당하는 스크립트들

- `migrate_database_schema.py` - 데이터베이스 스키마 마이그레이션
- `backup_database.py` - 데이터베이스 백업
- `analyze_database_content.py` - 데이터베이스 내용 분석

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

- `test_bge_m3_korean.py` - BGE-M3 Korean 모델 테스트
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
