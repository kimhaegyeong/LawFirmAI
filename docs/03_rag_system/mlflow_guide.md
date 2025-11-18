# MLflow 통합 가이드

## 개요

LawFirmAI 프로젝트는 MLflow를 활용하여 FAISS 인덱스 버전 관리, 검색 품질 최적화, 그리고 최적 파라미터 관리를 수행합니다.

## 목차

1. [환경 설정](#환경-설정)
2. [FAISS 인덱스 관리](#faiss-인덱스-관리)
3. [검색 품질 최적화](#검색-품질-최적화)
4. [최적 파라미터 통합](#최적-파라미터-통합)
5. [MLflow UI 사용](#mlflow-ui-사용)
6. [문제 해결](#문제-해결)

## 환경 설정

### 1. MLflow 설치

```bash
pip install mlflow>=2.8.0
```

### 2. 환경 변수 설정

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

# 최적 파라미터 경로 (선택사항)
OPTIMIZED_SEARCH_PARAMS_PATH=data/ml_config/optimized_search_params.json
```

### 3. Windows 경로 설정

Windows 환경에서는 절대 경로를 사용하는 것이 권장됩니다:

```env
MLFLOW_TRACKING_URI=file:///D:/project/LawFirmAI/LawFirmAI/mlflow/mlruns
```

**주의**: `file:///` 뒤에 슬래시(`/`) 3개를 사용하고, 경로 구분자는 슬래시(`/`)를 사용하세요.

## FAISS 인덱스 관리

### 인덱스 빌드 및 저장

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

### 인덱스 사용

#### 프로덕션 Run 자동 조회

`MLFLOW_RUN_ID`를 비워두면 시스템이 자동으로 프로덕션 태그가 있는 run을 찾아 사용합니다:

```env
MLFLOW_RUN_ID=
```

시스템은 `tags.status='production_ready'` 태그가 있는 run을 최신 순으로 조회하여 사용합니다.

#### 특정 Run 사용

특정 MLflow run의 인덱스를 사용하려면 run ID를 지정하세요:

```env
MLFLOW_RUN_ID=5fe69543e53d4c9dad6421b3cefff7d4
```

#### Run ID 확인 방법

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

### 마이그레이션 가이드

기존 인덱스에서 MLflow로 마이그레이션:

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

## 검색 품질 최적화

### Ground Truth 데이터 준비

```bash
# Pseudo Query 생성
python scripts/ml_training/evaluation/generate_pseudo_queries.py \
    --vector-store-path data/embeddings/ml_enhanced_ko_sroberta \
    --output-path data/evaluation/ground_truth/pseudo_queries.json

# 데이터셋 분할
python scripts/ml_training/evaluation/generate_rag_evaluation_dataset.py \
    --ground-truth-path data/evaluation/ground_truth/pseudo_queries.json \
    --output-dir data/evaluation/datasets \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### 최적화 워크플로우

#### Step 1: 빠른 탐색 (Quick Exploration)

작은 샘플로 빠르게 테스트:

```bash
python scripts/rag/optimize_search_quality.py \
    --ground-truth-path data/evaluation/datasets/val.json \
    --vector-store-path data/embeddings/ml_enhanced_ko_sroberta \
    --experiment-name rag_optimization_quick \
    --max-combinations 20 \
    --sample-size 50 \
    --primary-metric ndcg@10
```

#### Step 2: 상세 최적화 (Detailed Optimization)

전체 데이터셋으로 상세 테스트:

```bash
python scripts/rag/optimize_search_quality.py \
    --ground-truth-path data/evaluation/datasets/val.json \
    --vector-store-path data/embeddings/ml_enhanced_ko_sroberta \
    --experiment-name rag_optimization_detailed \
    --max-combinations 100 \
    --primary-metric ndcg@10 \
    --output-path data/evaluation/optimization_results.json
```

#### Step 3: 최적 파라미터 검증

최적 파라미터로 테스트 데이터셋 평가:

```bash
python scripts/rag/evaluate.py \
    --ground-truth-path data/evaluation/datasets/test.json \
    --vector-store-path data/embeddings/ml_enhanced_ko_sroberta \
    --run-name optimized_validation \
    --experiment-name rag_evaluation \
    --top-k-list 5,10,20
```

### 결과 분석

#### 주요 메트릭

- **Recall@K**: 관련 문서를 얼마나 찾았는지
- **Precision@K**: 찾은 문서 중 관련 문서 비율
- **NDCG@K**: 순위를 고려한 정규화 점수
- **MRR**: 첫 번째 관련 문서의 역순위

#### 파라미터 영향도 분석

MLflow UI에서:
1. 실험 선택
2. "Compare" 탭 클릭
3. 파라미터별 성능 비교
4. 최적 조합 확인

### 주기적 개선 프로세스

```bash
# 1. 새로운 데이터로 재최적화
python scripts/rag/optimize_search_quality.py \
    --ground-truth-path data/evaluation/datasets/latest_val.json \
    --experiment-name rag_optimization_$(date +%Y%m%d) \
    --max-combinations 30

# 2. 결과 비교
python scripts/rag/compare_experiments.py \
    --experiment-name rag_optimization_* \
    --output-path data/evaluation/weekly_comparison.txt

# 3. 개선 확인 후 적용
```

## 최적 파라미터 통합

### 최적 파라미터

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

### 자동 로드

`QueryEnhancer` 클래스는 초기화 시 자동으로 최적 파라미터를 로드합니다:

1. 환경 변수 `OPTIMIZED_SEARCH_PARAMS_PATH` 확인
2. 기본 경로: `data/ml_config/optimized_search_params.json`
3. 파일이 존재하면 자동 로드
4. 파일이 없으면 기본값 사용 (기존 동작 유지)

### 사용 방법

```python
from lawfirm_langgraph.core.search.optimizers.query_enhancer import QueryEnhancer

# 최적 파라미터가 자동으로 로드됨
query_enhancer = QueryEnhancer(llm, llm_fast, term_integrator, config)

# 검색 파라미터 결정 시 최적 파라미터가 기본값으로 사용됨
search_params = query_enhancer.determine_search_parameters(
    query_type="precedent_search",
    query_complexity=50,
    keyword_count=5,
    is_retry=False
)
```

### 파라미터 적용 방식

1. **기본값으로 사용**: 최적 파라미터의 `top_k`와 `similarity_threshold`가 기본값으로 사용됩니다.
2. **동적 조정 유지**: 질문 유형, 복잡도, 키워드 수에 따른 동적 조정은 그대로 유지됩니다.
3. **추가 기능 활성화**: `use_reranking`, `query_enhancement`, `use_keyword_search` 등의 기능이 활성화됩니다.

### 성능 개선

최적 파라미터 적용 시 예상되는 성능 개선:

- **NDCG@10**: 0.005621 (최적화 전 대비 개선)
- **Recall@10**: 향상
- **Precision@10**: 향상

### 주기적 재최적화

새로운 데이터가 추가되거나 모델이 업데이트되면 주기적으로 재최적화를 수행하세요:

```bash
python scripts/rag/automated_optimization_pipeline.py \
    --vector-store-path data/vector_store/v2.0.0-dynamic-dynamic-ivfpq \
    --ground-truth-path data/evaluation/rag_ground_truth_combined_test.json
```

## MLflow UI 사용

### UI 시작

```bash
mlflow ui --backend-store-uri file:///D:/project/LawFirmAI/LawFirmAI/mlflow/mlruns --port 5000
```

브라우저에서 `http://localhost:5000`으로 접속하여 실험 결과를 확인할 수 있습니다.

### 주요 기능

- **실험 비교**: 여러 run의 성능 비교
- **아티팩트 확인**: 저장된 FAISS 인덱스 파일 확인
- **메트릭 추적**: 인덱스 통계 및 성능 메트릭 확인
- **파라미터 추적**: 인덱스 빌드 시 사용된 파라미터 확인

### 인덱스 통계 확인

MLflow UI에서 각 run의 메트릭을 확인할 수 있습니다:

- `num_vectors`: 인덱스에 포함된 벡터 수
- `dimension`: 벡터 차원
- `id_mapping_size`: ID 매핑 크기
- `metadata_size`: 메타데이터 크기

### 실험 결과 비교

```bash
python scripts/rag/compare_experiments.py \
    --experiment-name rag_optimization_detailed \
    --output-path data/evaluation/comparison_report.txt
```

### 실험 결과 찾기

```python
import mlflow

# 모든 실험 조회
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.experiment_id}")

# 특정 실험의 runs 조회
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.primary_score DESC"]
)
```

## 문제 해결

### MLflow 연결 오류

```bash
# Tracking URI 확인
echo $MLFLOW_TRACKING_URI

# 수동 설정
export MLFLOW_TRACKING_URI=file://./mlflow/mlruns
```

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

### 최적 파라미터가 로드되지 않는 경우

1. 파일 경로 확인:
   ```bash
   ls -la data/ml_config/optimized_search_params.json
   ```

2. 파일 형식 확인:
   ```bash
   cat data/ml_config/optimized_search_params.json | python -m json.tool
   ```

3. 로그 확인:
   - INFO 레벨: "최적 파라미터 로드 완료" 메시지 확인
   - DEBUG 레벨: 로드 실패 시 상세 오류 메시지 확인

### 메모리 부족

```bash
# 샘플 크기 줄이기
python scripts/rag/optimize_search_quality.py \
    --sample-size 100 \
    --max-combinations 20
```

### 기본값으로 되돌리기

최적 파라미터 파일을 삭제하거나 이름을 변경하면 기본값이 사용됩니다:

```bash
mv data/ml_config/optimized_search_params.json data/ml_config/optimized_search_params.json.bak
```

## 성능 비교

MLflow 인덱스를 사용하면 다음과 같은 개선을 기대할 수 있습니다:

- **검색 결과 개선**: Semantic search 결과가 3배 증가 (22개 → 67개)
- **안정성 향상**: chunk_id not found 문제 완전 해결 (1,068개 → 0개)
- **타입별 검색 다양성**: 다양한 문서 타입에서 검색 결과 확보

## 모니터링

### 로그 확인

애플리케이션 로그에서 다음 메시지를 확인할 수 있습니다:

```
Loaded MLflow FAISS index: IndexIVFPQ (26,630 vectors) from run <run_id>
MLflow version: production-20251117-094811
```

## 관련 파일

- 최적 파라미터 설정: `data/ml_config/optimized_search_params.json`
- MLflow 관리자: `scripts/rag/mlflow_manager.py`
- 인덱스 빌드: `scripts/rag/build_index.py`
- 프로덕션 인덱스 빌드: `scripts/rag/build_production_index.py`
- 검색 품질 최적화: `scripts/rag/optimize_search_quality.py`
- RAG 평가: `scripts/rag/evaluate.py`
- 파라미터 로더: `scripts/rag/load_optimized_params.py`
- QueryEnhancer: `lawfirm_langgraph/core/search/optimizers/query_enhancer.py`
- 자동화 파이프라인: `scripts/rag/automated_optimization_pipeline.py`

## 참고 자료

- [MLflow 공식 문서](https://www.mlflow.org/docs/latest/index.html)
- [FAISS 인덱스 빌드 가이드](../scripts/rag/README.md)
- [RAG 시스템 아키텍처](./rag_architecture.md)

