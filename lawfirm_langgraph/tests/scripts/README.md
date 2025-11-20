# Tests Scripts Directory

이 디렉토리는 LawFirmAI 프로젝트의 테스트 및 유틸리티 스크립트를 포함합니다.

## 📁 폴더 구조

```
lawfirm_langgraph/tests/scripts/
├── README.md                          # 이 파일
├── run_query_test.py                  # 메인 쿼리 테스트 스크립트 (권장)
│
├── tests/                             # 테스트 스크립트
│   ├── workflow/                      # 워크플로우 테스트
│   │   ├── test_langgraph_with_indexivfpq.py
│   │   └── test_conversation_context_features.py
│   │
│   ├── search/                        # 검색 테스트
│   │   ├── test_search_validation.py
│   │   ├── test_statute_search.py
│   │   ├── test_semantic_search_engine_delivery.py
│   │   └── test_production_integration.py
│   │
│   ├── metadata/                      # 메타데이터 테스트
│   │   ├── test_metadata_restoration.py
│   │   └── test_statute_metadata_restoration.py
│   │
│   ├── features/                      # 기능 테스트
│   │   ├── test_sources_workflow.py
│   │   ├── test_type_based_document_sections.py
│   │   ├── test_document_inclusion_improvements.py
│   │   └── test_generate_answer_stream_integration.py
│   │
│   ├── performance/                   # 성능 테스트
│   │   ├── test_performance_improvements.py
│   │   ├── test_classification_performance.py
│   │   ├── test_weight_combinations.py
│   │   ├── test_hybrid_query_processor.py
│   │   └── test_keyword_extraction_hf.py
│   │
│   ├── prompts/                       # 프롬프트 테스트
│   │   ├── test_prompt_analysis.py
│   │   └── test_multi_query_prompt.py
│   │
│   └── mlflow/                        # MLflow 테스트
│       ├── test_mlflow_only_index.py
│       └── verify_mlflow_integration.py
│
├── evaluation/                        # 평가 및 비교
│   ├── test_search_quality_evaluation.py
│   ├── compare_search_quality.py
│   ├── quick_evaluation_test.py
│   └── create_comparison_from_existing.py
│
├── utils/                             # 유틸리티 스크립트
│   ├── data/                          # 데이터 처리
│   │   ├── fix_data_consistency.py
│   │   └── validate_metadata_completeness.py
│   │
│   ├── analysis/                      # 분석 도구
│   │   ├── analyze_langgraph_queries.py
│   │   ├── analyze_answer_issues.py
│   │   └── check_answer_quality.py
│   │
│   └── verification/                  # 검증 도구
│       ├── check_evaluation_results.py
│       ├── verify_classification_optimization.py
│       └── check_evaluation_status.ps1
│
└── docs/                              # 문서 파일
    ├── faiss_test_summary.md
    ├── final_test_analysis.md
    └── answer_quality_issues_analysis.md
```

## 📌 주요 스크립트

### 🚀 메인 테스트

- **`run_query_test.py`** - 메인 쿼리 테스트 스크립트 (권장)
  ```bash
  python lawfirm_langgraph/tests/scripts/run_query_test.py "질의 내용"
  python lawfirm_langgraph/tests/scripts/run_query_test.py -f query.txt
  ```

### 🧪 테스트 스크립트

#### 워크플로우 테스트 (`tests/workflow/`)
- `test_langgraph_with_indexivfpq.py` - IndexIVFPQ 인덱스 사용 테스트
- `test_conversation_context_features.py` - 대화 컨텍스트 기능 테스트

#### 검색 테스트 (`tests/search/`)
- `test_search_validation.py` - 검색 검증 테스트
- `test_statute_search.py` - 법령 검색 테스트
- `test_semantic_search_engine_delivery.py` - 의미적 검색 엔진 전달 테스트
- `test_production_integration.py` - 프로덕션 인덱스 및 최적 파라미터 통합 테스트

#### 메타데이터 테스트 (`tests/metadata/`)
- `test_metadata_restoration.py` - 메타데이터 복원 테스트
- `test_statute_metadata_restoration.py` - 법령 메타데이터 복원 테스트

#### 기능 테스트 (`tests/features/`)
- `test_sources_workflow.py` - 출처 워크플로우 테스트
- `test_type_based_document_sections.py` - 타입 기반 문서 섹션 테스트
- `test_document_inclusion_improvements.py` - 문서 포함 개선 테스트
- `test_generate_answer_stream_integration.py` - 답변 스트림 생성 통합 테스트

#### 성능 테스트 (`tests/performance/`)
- `test_performance_improvements.py` - 성능 개선 사항 테스트
- `test_classification_performance.py` - 분류 성능 테스트
- `test_weight_combinations.py` - 가중치 조합 테스트
- `test_hybrid_query_processor.py` - 하이브리드 쿼리 프로세서 테스트
- `test_keyword_extraction_hf.py` - HuggingFace 키워드 추출 테스트

#### 프롬프트 테스트 (`tests/prompts/`)
- `test_prompt_analysis.py` - 프롬프트 분석 테스트
- `test_multi_query_prompt.py` - 다중 쿼리 프롬프트 테스트

#### MLflow 테스트 (`tests/mlflow/`)
- `test_mlflow_only_index.py` - MLflow 인덱스 테스트
- `verify_mlflow_integration.py` - MLflow 통합 검증

### 📊 평가 및 비교 (`evaluation/`)

- `test_search_quality_evaluation.py` - 검색 품질 평가 스크립트
- `compare_search_quality.py` - 검색 품질 Before/After 비교
- `quick_evaluation_test.py` - 빠른 평가 테스트
- `create_comparison_from_existing.py` - 기존 결과로부터 비교 생성

### 🔧 유틸리티 스크립트 (`utils/`)

#### 데이터 처리 (`utils/data/`)
- `fix_data_consistency.py` - 데이터 일관성 수정 (메타데이터 복원)
- `validate_metadata_completeness.py` - 메타데이터 완전성 검증

#### 분석 도구 (`utils/analysis/`)
- `analyze_langgraph_queries.py` - LangGraph 쿼리 분석
- `analyze_answer_issues.py` - 답변 이슈 분석
- `check_answer_quality.py` - 답변 품질 확인

#### 검증 도구 (`utils/verification/`)
- `check_evaluation_results.py` - 평가 결과 확인
- `verify_classification_optimization.py` - 분류 최적화 검증
- `check_evaluation_status.ps1` - 평가 상태 확인 (PowerShell)

## 📝 문서 (`docs/`)

- `faiss_test_summary.md` - FAISS 테스트 요약
- `final_test_analysis.md` - 최종 테스트 분석
- `answer_quality_issues_analysis.md` - 답변 품질 이슈 분석

## 🚀 사용 가이드

### 메인 테스트 실행

```bash
# 기본 사용법
python lawfirm_langgraph/tests/scripts/run_query_test.py "질의 내용"

# 파일에서 질의 읽기
python lawfirm_langgraph/tests/scripts/run_query_test.py -f query.txt

# 환경 변수 사용
$env:TEST_QUERY='질의내용'; python run_query_test.py
```

### 워크플로우 테스트 실행

```bash
# IndexIVFPQ 인덱스 테스트
python lawfirm_langgraph/tests/scripts/tests/workflow/test_langgraph_with_indexivfpq.py

# 대화 컨텍스트 테스트
python lawfirm_langgraph/tests/scripts/tests/workflow/test_conversation_context_features.py
```

### 검색 테스트 실행

```bash
# 검색 검증 테스트
python lawfirm_langgraph/tests/scripts/tests/search/test_search_validation.py

# 법령 검색 테스트
python lawfirm_langgraph/tests/scripts/tests/search/test_statute_search.py

# 프로덕션 통합 테스트
python lawfirm_langgraph/tests/scripts/tests/search/test_production_integration.py
```

### 평가 실행

```bash
# 검색 품질 평가
python lawfirm_langgraph/tests/scripts/evaluation/test_search_quality_evaluation.py

# 검색 품질 비교
python lawfirm_langgraph/tests/scripts/evaluation/compare_search_quality.py

# 빠른 평가 테스트
python lawfirm_langgraph/tests/scripts/evaluation/quick_evaluation_test.py
```

### 데이터 처리

```bash
# 메타데이터 완전성 검증
python lawfirm_langgraph/tests/scripts/utils/data/validate_metadata_completeness.py

# 데이터 일관성 수정
python lawfirm_langgraph/tests/scripts/utils/data/fix_data_consistency.py
```

### 분석 도구

```bash
# LangGraph 쿼리 분석
python lawfirm_langgraph/tests/scripts/utils/analysis/analyze_langgraph_queries.py

# 답변 이슈 분석
python lawfirm_langgraph/tests/scripts/utils/analysis/analyze_answer_issues.py

# 답변 품질 확인
python lawfirm_langgraph/tests/scripts/utils/analysis/check_answer_quality.py
```

## 📋 파일 찾기 가이드

### 테스트 파일 찾기
- **워크플로우 관련**: `tests/workflow/`
- **검색 관련**: `tests/search/`
- **메타데이터 관련**: `tests/metadata/`
- **기능 관련**: `tests/features/`
- **성능 관련**: `tests/performance/`
- **프롬프트 관련**: `tests/prompts/`
- **MLflow 관련**: `tests/mlflow/`

### 유틸리티 파일 찾기
- **데이터 처리**: `utils/data/`
- **분석 도구**: `utils/analysis/`
- **검증 도구**: `utils/verification/`

### 평가 파일 찾기
- **모든 평가 관련**: `evaluation/`

## 🔄 변경 사항

### 2024년 폴더 구조 개선
- 파일들을 카테고리별 폴더로 재구성
- 테스트 파일, 평가 파일, 유틸리티 파일 분리
- 각 파일의 import 경로 자동 수정 완료

## 💡 팁

1. **메인 테스트는 `run_query_test.py` 사용 권장**
   - 가장 완전한 테스트 기능 제공
   - 상세한 로깅 및 평가 포함

2. **특정 기능 테스트는 해당 폴더에서 찾기**
   - 예: 검색 테스트 → `tests/search/`
   - 예: 성능 테스트 → `tests/performance/`

3. **평가 및 비교는 `evaluation/` 폴더 참조**
   - 검색 품질 평가 및 비교 도구 모음

4. **유틸리티는 `utils/` 폴더 참조**
   - 데이터 처리, 분석, 검증 도구 모음
