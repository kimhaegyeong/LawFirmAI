# MLflow 인덱스 마이그레이션 요약

## 변경 사항

### 환경 변수 변경

**제거된 변수**:
- `USE_EXTERNAL_VECTOR_STORE`
- `EXTERNAL_VECTOR_STORE_BASE_PATH`
- `VECTOR_STORE_VERSION`

**추가된 변수**:
- `USE_MLFLOW_INDEX`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_RUN_ID`
- `MLFLOW_EXPERIMENT_NAME`

### 코드 변경

**주요 파일**:
- `lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py`
- `lawfirm_langgraph/core/search/engines/hybrid_search_engine_v2.py`
- `lawfirm_langgraph/core/search/engines/semantic_search_engine_v2.py`

**변경 내용**:
- `use_external_index` → `use_mlflow_index`
- `external_index_path` → MLflow run ID 기반 로드
- `vector_store_version` → MLflow run ID

## 마이그레이션 체크리스트

- [ ] `.env` 파일 업데이트
- [ ] `USE_MLFLOW_INDEX=true` 설정
- [ ] `MLFLOW_TRACKING_URI` 설정 (절대 경로 권장)
- [ ] `MLFLOW_RUN_ID` 설정 (또는 자동 조회 사용)
- [ ] 기존 인덱스를 MLflow에 저장 (필요시)
- [ ] 테스트 실행하여 정상 작동 확인

## 롤백 방법

기존 방식으로 롤백하려면:

```env
USE_MLFLOW_INDEX=false
```

또는 환경 변수를 제거하면 기본 DB 기반 인덱스를 사용합니다.

