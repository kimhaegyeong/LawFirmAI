# Tests Scripts Directory

이 디렉토리는 LawFirmAI 프로젝트의 테스트 및 유틸리티 스크립트를 포함합니다.

## 📁 주요 스크립트 카테고리

### 🔧 유틸리티 스크립트

#### 버전 관리
- `activate_version_5.py` - 특정 embedding 버전 활성화
- `check_active_version.py` - 현재 활성 embedding 버전 확인
- `build_faiss_for_version.py` - 특정 버전용 FAISS 인덱스 빌드

#### 인덱스 관리
- `build_indexivfpq.py` - IndexIVFPQ 인덱스 빌드
- `check_index_type.py` - FAISS 인덱스 타입 확인
- `check_index_loading.py` - 인덱스 로드 상태 확인
- `create_external_index_metadata.py` - 외부 인덱스 메타데이터 생성

#### 데이터 검증 및 수정
- `fix_data_consistency.py` - 데이터 일관성 수정 (메타데이터 복원)
- `validate_metadata_completeness.py` - 메타데이터 완전성 검증
- `check_chunk_metadata.py` - 청크 메타데이터 확인
- `check_data_structure.py` - 데이터 구조 확인

#### 분석 및 모니터링
- `check_embeddings_distribution.py` - 임베딩 분포 확인
- `check_vector_store_distribution.py` - 벡터 스토어 분포 확인
- `check_source_types.py` - 소스 타입 확인
- `check_answer_quality.py` - 답변 품질 확인
- `analyze_langgraph_queries.py` - LangGraph 쿼리 분석
- `analyze_answer_issues.py` - 답변 이슈 분석

### 🧪 테스트 스크립트

#### 워크플로우 테스트
- `run_query_test.py` - 메인 쿼리 테스트 스크립트
- `test_full_workflow.py` - 전체 워크플로우 테스트
- `test_full_workflow_prompt.py` - 프롬프트 포함 전체 워크플로우 테스트
- `test_workflow_simple.py` - 간단한 워크플로우 테스트
- `test_workflow_with_improvements.py` - 개선사항 포함 워크플로우 테스트
- `test_langgraph_with_indexivfpq.py` - IndexIVFPQ 인덱스 사용 테스트

#### 검색 테스트
- `test_search_validation.py` - 검색 검증 테스트
- `test_semantic_search_engine_delivery.py` - 의미적 검색 엔진 전달 테스트
- `test_statute_search.py` - 법령 검색 테스트

#### 메타데이터 테스트
- `test_metadata_restoration.py` - 메타데이터 복원 테스트
- `test_metadata_improvement.py` - 메타데이터 개선 테스트
- `test_statute_metadata_restoration.py` - 법령 메타데이터 복원 테스트

#### 기능 테스트
- `test_sources_extraction.py` - 출처 추출 테스트
- `test_sources_workflow.py` - 출처 워크플로우 테스트
- `test_type_based_document_sections.py` - 타입 기반 문서 섹션 테스트
- `test_document_inclusion_improvements.py` - 문서 포함 개선 테스트
- `test_conversation_context_features.py` - 대화 컨텍스트 기능 테스트
- `test_generate_answer_stream_integration.py` - 답변 스트림 생성 통합 테스트

#### 프롬프트 테스트
- `test_prompt_analysis.py` - 프롬프트 분석 테스트
- `test_prompt_improvements.py` - 프롬프트 개선 테스트

## 📝 문서

- `faiss_test_summary.md` - FAISS 테스트 요약
- `final_test_analysis.md` - 최종 테스트 분석
- `answer_quality_issues_analysis.md` - 답변 품질 이슈 분석

## 🗑️ 정리된 파일들

다음 파일들은 일시적/디버깅용으로 삭제되었습니다:
- `investigate_*.py` - 디버깅/조사용 스크립트
- `test_version_5_*.py` - 특정 버전 테스트 (일시적)
- `test_new_faiss_index.py` - 특정 인덱스 테스트 (일시적)
- `check_version_details.py` - 중복 체크 스크립트
- `check_external_index_metadata.py` - 중복 체크 스크립트
- `test_indexivfpq_search.py`, `test_indexivfpq_support.py` - `test_langgraph_with_indexivfpq.py`로 통합
- `debug_*.py` - 디버깅용 스크립트

## 📌 사용 가이드

### 메인 테스트 실행
```bash
python lawfirm_langgraph/tests/scripts/run_query_test.py -f query.txt
```

### 버전 관리
```bash
# 활성 버전 확인
python lawfirm_langgraph/tests/scripts/check_active_version.py

# 버전 활성화
python lawfirm_langgraph/tests/scripts/activate_version_5.py
```

### 인덱스 관리
```bash
# 인덱스 타입 확인
python lawfirm_langgraph/tests/scripts/check_index_type.py

# IndexIVFPQ 인덱스 빌드
python lawfirm_langgraph/tests/scripts/build_indexivfpq.py
```

### 데이터 검증
```bash
# 메타데이터 완전성 검증
python lawfirm_langgraph/tests/scripts/validate_metadata_completeness.py

# 데이터 일관성 수정
python lawfirm_langgraph/tests/scripts/fix_data_consistency.py
```

