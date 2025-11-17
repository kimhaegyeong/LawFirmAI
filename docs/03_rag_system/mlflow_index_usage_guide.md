# MLflow 인덱스 사용 가이드

## 개요

LawFirmAI 프로젝트는 FAISS 인덱스 버전 관리를 MLflow를 통해 수행합니다. 이를 통해 인덱스 버전 추적, 실험 관리, 그리고 프로덕션 배포가 용이해집니다.

## 환경 설정

### 1. 환경 변수 설정

`.env` 파일 또는 환경 변수에 다음 설정을 추가하세요:

```env
# MLflow 인덱스 사용
USE_MLFLOW_INDEX=true

# MLflow Tracking URI (선택사항, 기본값: ./mlflow/mlruns)
MLFLOW_TRACKING_URI=file:///D:/project/LawFirmAI/LawFirmAI/mlflow/mlruns

# MLflow Run ID (선택사항, None이면 프로덕션 run 자동 조회)
MLFLOW_RUN_ID=

# MLflow 실험 이름 (기본값: faiss_index_versions)
MLFLOW_EXPERIMENT_NAME=faiss_index_versions
```

### 2. Windows 경로 설정

Windows 환경에서는 절대 경로를 사용하는 것이 권장됩니다:

```env
MLFLOW_TRACKING_URI=file:///D:/project/LawFirmAI/LawFirmAI/mlflow/mlruns
```

**주의**: `file:///` 뒤에 슬래시(`/`) 3개를 사용하고, 경로 구분자는 슬래시(`/`)를 사용하세요.

## 사용 방법

### 프로덕션 Run 자동 조회

`MLFLOW_RUN_ID`를 비워두면 시스템이 자동으로 프로덕션 태그가 있는 run을 찾아 사용합니다:

```env
MLFLOW_RUN_ID=
```

시스템은 `tags.status='production_ready'` 태그가 있는 run을 최신 순으로 조회하여 사용합니다.

### 특정 Run 사용

특정 MLflow run의 인덱스를 사용하려면 run ID를 지정하세요:

```env
MLFLOW_RUN_ID=5fe69543e53d4c9dad6421b3cefff7d4
```

### Run ID 확인 방법

MLflow UI를 사용하거나 Python 코드로 확인할 수 있습니다:

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
for run in runs:
    print(f"Run ID: {run['run_id']}, Version: {run['version']}, Status: {run['status']}")
```

## 인덱스 빌드 및 저장

### 인덱스 빌드

새로운 FAISS 인덱스를 빌드하고 MLflow에 저장:

```bash
python scripts/rag/build_index.py \
    --version-name production-20251117-094811 \
    --embedding-version-id 5 \
    --chunking-strategy dynamic \
    --db-path data/lawfirm_v2.db \
    --use-mlflow \
    --local-backup
```

### 프로덕션 인덱스 빌드

최적화된 파라미터를 사용하여 프로덕션 인덱스 빌드:

```bash
python scripts/rag/build_production_index.py
```

이 스크립트는:
1. 최적화된 파라미터를 로드
2. FAISS 인덱스를 빌드
3. MLflow에 저장
4. `status='production_ready'` 태그 추가

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

## 문제 해결

### MLflow 모듈 Import 오류

```
MLflowFAISSManager not available: No module named 'mlflow_manager'
```

**해결 방법**:
1. `scripts/rag` 디렉토리가 Python 경로에 포함되어 있는지 확인
2. 프로젝트 루트에서 실행하는지 확인

### 인덱스 로드 실패

```
Failed to load index from MLflow run: <run_id>
```

**해결 방법**:
1. Run ID가 올바른지 확인
2. MLflow tracking URI가 올바른지 확인
3. 해당 run에 FAISS 인덱스 아티팩트가 있는지 확인

### 프로덕션 Run을 찾을 수 없음

```
No production run found in MLflow
```

**해결 방법**:
1. `status='production_ready'` 태그가 있는 run이 있는지 확인
2. `MLFLOW_RUN_ID`를 직접 지정하여 사용

## 성능 비교

MLflow 인덱스를 사용하면 다음과 같은 개선을 기대할 수 있습니다:

- **검색 결과 개선**: Semantic search 결과가 3배 증가 (22개 → 67개)
- **안정성 향상**: chunk_id not found 문제 완전 해결 (1,068개 → 0개)
- **타입별 검색 다양성**: 다양한 문서 타입에서 검색 결과 확보

## 모니터링

### 인덱스 통계 확인

MLflow UI에서 각 run의 메트릭을 확인할 수 있습니다:

- `num_vectors`: 인덱스에 포함된 벡터 수
- `dimension`: 벡터 차원
- `id_mapping_size`: ID 매핑 크기
- `metadata_size`: 메타데이터 크기

### 로그 확인

애플리케이션 로그에서 다음 메시지를 확인할 수 있습니다:

```
Loaded MLflow FAISS index: IndexIVFPQ (26,630 vectors) from run <run_id>
MLflow version: production-20251117-094811
```

## 마이그레이션 가이드

### 기존 인덱스에서 MLflow로 마이그레이션

1. 기존 인덱스 확인:
   ```bash
   ls data/vector_store/
   ```

2. MLflow에 인덱스 저장:
   ```bash
   python scripts/rag/build_index.py --version-name <version> ...
   ```

3. 환경 변수 업데이트:
   ```env
   USE_MLFLOW_INDEX=true
   MLFLOW_RUN_ID=<run_id>
   ```

4. 테스트:
   ```bash
   python lawfirm_langgraph/tests/scripts/run_query_test.py "테스트 질의"
   ```

## 참고 자료

- [MLflow 공식 문서](https://mlflow.org/docs/latest/index.html)
- [FAISS 인덱스 빌드 가이드](../scripts/rag/README.md)
- [RAG 시스템 최적화 가이드](./mlflow_optimization_summary.md)

