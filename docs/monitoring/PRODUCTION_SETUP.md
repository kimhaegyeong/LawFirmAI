# 프로덕션 설정 가이드

LangGraph에서 MLflow를 통해 관리되는 프로덕션 FAISS 인덱스와 최적 파라미터를 사용하는 방법을 설명합니다.

## 개요

이 가이드는 다음을 설정하는 방법을 설명합니다:
1. **MLflow 인덱스**: MLflow를 통해 관리되는 최적화된 FAISS 인덱스 사용
2. **최적 검색 파라미터**: MLflow 최적화를 통해 찾은 최적 검색 파라미터 사용

## 사전 요구사항

- MLflow가 설치되어 있어야 합니다: `pip install mlflow>=2.8.0`
- 프로덕션 인덱스가 MLflow에 저장되어 있어야 합니다
- 최적 파라미터 파일이 있어야 합니다: `data/ml_config/optimized_search_params.json`

## 설정 방법

### 1. 환경 변수 설정

#### 프로젝트 루트 `.env` 파일

```env
# MLflow 인덱스 사용
USE_MLFLOW_INDEX=true

# MLflow Tracking URI (절대 경로 사용, Windows)
MLFLOW_TRACKING_URI=file:///D:/project/LawFirmAI/LawFirmAI/mlflow/mlruns

# MLflow Run ID (비워두면 프로덕션 run 자동 조회)
MLFLOW_RUN_ID=

# MLflow 실험 이름 (기본값: faiss_index_versions)
MLFLOW_EXPERIMENT_NAME=faiss_index_versions

# 최적 파라미터 경로 (선택사항, 기본값 사용 시 생략 가능)
OPTIMIZED_SEARCH_PARAMS_PATH=D:/project/LawFirmAI/LawFirmAI/data/ml_config/optimized_search_params.json
```

#### `lawfirm_langgraph/.env` 파일

동일한 환경 변수를 `lawfirm_langgraph/.env` 파일에도 설정합니다.

### 2. 프로덕션 Run 자동 조회

`MLFLOW_RUN_ID`를 비워두면 시스템이 자동으로 프로덕션 태그가 있는 run을 찾아 사용합니다:

```env
MLFLOW_RUN_ID=
```

시스템은 `tags.status='production_ready'` 태그가 있는 run을 최신 순으로 조회하여 사용합니다.

### 3. 특정 Run 사용

특정 MLflow run의 인덱스를 사용하려면 run ID를 지정하세요:

```env
MLFLOW_RUN_ID=5fe69543e53d4c9dad6421b3cefff7d4
```

## 검증

### 통합 테스트 실행

```bash
python lawfirm_langgraph/tests/scripts/test_production_integration.py
```

이 테스트는 다음을 확인합니다:
- ✅ 환경 변수 설정 확인
- ✅ MLflow 인덱스 로드 확인
- ✅ 최적 파라미터 파일 존재 확인
- ✅ SemanticSearchEngineV2 초기화 확인
- ✅ QueryEnhancer 최적 파라미터 로드 확인
- ✅ LangGraph 워크플로우 통합 확인

### 수동 확인

#### 1. MLflow Run 확인

```python
from scripts.rag.mlflow_manager import MLflowFAISSManager
import mlflow

mlflow.set_tracking_uri('file:///D:/project/LawFirmAI/LawFirmAI/mlflow/mlruns')
manager = MLflowFAISSManager()

# 프로덕션 run 조회
run_id = manager.get_production_run()
print(f"Production run ID: {run_id}")

# 모든 run 목록 조회
runs = manager.list_runs()
for run in runs[:5]:
    print(f"Run ID: {run['run_id']}, Version: {run.get('version', 'N/A')}, Status: {run.get('status', 'N/A')}")
```

#### 2. 최적 파라미터 확인

```bash
# 최적 파라미터 파일 확인
cat data/ml_config/optimized_search_params.json
```

#### 3. 로그 확인

LangGraph 실행 시 다음 로그 메시지를 확인하세요:

```
Loaded MLflow FAISS index: IndexIVFPQ (26,630 vectors) from run <run_id>
MLflow version: production-20251117-094811
최적 파라미터 로드 완료: data/ml_config/optimized_search_params.json
SemanticSearchEngineV2 initialized successfully with ...
```

## 최적 파라미터

현재 적용된 최적 파라미터:

```json
{
  "top_k": 15,
  "similarity_threshold": 0.9,
  "use_reranking": true,
  "rerank_top_n": 30,
  "query_enhancement": true,
  "hybrid_search_ratio": 1.0,
  "use_keyword_search": true
}
```

이 파라미터는 `QueryEnhancer.determine_search_parameters()` 메서드에서 자동으로 사용됩니다.

## 동작 방식

### 1. MLflow 인덱스 로드

`SemanticSearchEngineV2`는 초기화 시 다음 순서로 인덱스를 로드합니다:

1. `USE_MLFLOW_INDEX=true`이고 `MLFLOW_RUN_ID`가 설정되어 있으면 해당 run의 인덱스 사용
2. `USE_MLFLOW_INDEX=true`이고 `MLFLOW_RUN_ID`가 비어있으면 프로덕션 run 자동 조회
3. MLflow 인덱스 로드 실패 시 폴백 모드로 DB 기반 인덱스 사용

### 2. 최적 파라미터 로드

`QueryEnhancer`는 초기화 시 다음 순서로 최적 파라미터를 로드합니다:

1. `OPTIMIZED_SEARCH_PARAMS_PATH` 환경 변수가 설정되어 있으면 해당 경로의 파일 사용
2. 기본값: `data/ml_config/optimized_search_params.json`
3. 파일이 없으면 기본 파라미터 사용 (기존 동작 유지)

### 3. 검색 파라미터 결정

`QueryEnhancer.determine_search_parameters()` 메서드는:

1. 최적 파라미터가 로드되어 있으면 이를 기본값으로 사용
2. 질문 유형, 복잡도, 키워드 수에 따른 동적 조정은 그대로 유지
3. 최적 파라미터의 `use_reranking`, `query_enhancement` 등의 기능 활성화

## 문제 해결

### MLflow 인덱스가 로드되지 않는 경우

1. **MLflow 설정 확인**
   ```bash
   # MLflow tracking URI 확인
   echo $MLFLOW_TRACKING_URI
   echo $USE_MLFLOW_INDEX
   ```

2. **Run ID 확인**
   ```python
   from scripts.rag.mlflow_manager import MLflowFAISSManager
   manager = MLflowFAISSManager()
   run_id = manager.get_production_run()
   print(f"Production run ID: {run_id}")
   ```

3. **로그 확인**
   - `SemanticSearchEngineV2` 초기화 로그 확인
   - "Failed to load MLflow FAISS index" 경고 확인
   - 인덱스 로드 실패 시 폴백 모드로 동작

### 최적 파라미터가 적용되지 않는 경우

1. **파일 경로 확인**
   ```bash
   # 최적 파라미터 파일 존재 확인
   ls -la data/ml_config/optimized_search_params.json
   ```

2. **환경 변수 확인**
   ```bash
   echo $OPTIMIZED_SEARCH_PARAMS_PATH
   ```

3. **로그 확인**
   - `QueryEnhancer` 초기화 시 "최적 파라미터 로드 완료" 메시지 확인
   - 로드 실패 시 DEBUG 레벨 로그 확인

### Windows 경로 문제

Windows에서는 경로 구분자와 절대 경로 사용에 주의하세요:

```env
# ❌ 잘못된 예
MLFLOW_TRACKING_URI=file://D:\project\LawFirmAI\LawFirmAI\mlflow\mlruns

# ✅ 올바른 예 (절대 경로, 권장)
MLFLOW_TRACKING_URI=file:///D:/project/LawFirmAI/LawFirmAI/mlflow/mlruns
```

**주의**: `file:///` 뒤에 슬래시(`/`) 3개를 사용하고, 경로 구분자는 슬래시(`/`)를 사용하세요.

## MLflow UI 사용

### UI 시작

```bash
mlflow ui --backend-store-uri file:///D:/project/LawFirmAI/LawFirmAI/mlflow/mlruns
```

브라우저에서 `http://localhost:5000`으로 접속하여 실험 결과를 확인할 수 있습니다.

### 주요 기능

- **실험 비교**: 여러 run의 성능 비교
- **아티팩트 확인**: 저장된 FAISS 인덱스 파일 확인
- **메트릭 추적**: 인덱스 통계 및 성능 메트릭 확인
- **파라미터 추적**: 인덱스 빌드 시 사용된 파라미터 확인

## 성능 모니터링

프로덕션 인덱스와 최적 파라미터 적용 후 다음 메트릭을 모니터링하세요:

- **검색 성능**: 검색 응답 시간
- **검색 정확도**: NDCG@10, Recall@10, Precision@10
- **검색 결과 개수**: Semantic search 결과 개수 (목표: 50개 이상)
- **chunk_id not found**: 0개 유지
- **사용자 만족도**: 사용자 피드백

## 업데이트

### 새로운 최적 파라미터 적용

1. `data/ml_config/optimized_search_params.json` 파일 업데이트
2. LangGraph 재시작 (환경 변수 변경 시)
3. 통합 테스트 실행하여 검증

### 새로운 프로덕션 인덱스 사용

1. MLflow에 새 인덱스 저장:
   ```bash
   python scripts/rag/build_production_index.py
   ```

2. `MLFLOW_RUN_ID` 환경 변수 업데이트 (또는 자동 조회 사용)
3. LangGraph 재시작
4. 통합 테스트 실행하여 검증

## 마이그레이션 가이드

기존 `USE_EXTERNAL_VECTOR_STORE` 방식에서 MLflow 방식으로 마이그레이션하는 방법은 [MLflow 마이그레이션 요약](../../docs/03_rag_system/mlflow_migration_summary.md)을 참고하세요.

## 관련 문서

- [MLflow 인덱스 사용 가이드](../../docs/03_rag_system/mlflow_index_usage_guide.md)
- [MLflow 최적화 요약](../../docs/03_rag_system/mlflow_optimization_summary.md)
- [최적 파라미터 통합 가이드](../../docs/03_rag_system/optimized_params_integration_guide.md)
- [RAG 시스템 README](../../scripts/rag/README.md)

---

**작성일**: 2025-11-17  
**버전**: 2.0.0 (MLflow 통합)
