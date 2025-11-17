# MLflow RAG 검색 품질 개선 가이드

## 개요

MLflow를 활용하여 RAG 문서 검색 품질을 체계적으로 개선하는 방법을 안내합니다.

## 목차

1. [시작하기](#시작하기)
2. [검색 품질 최적화 워크플로우](#검색-품질-최적화-워크플로우)
3. [실험 관리](#실험-관리)
4. [결과 분석](#결과-분석)
5. [최적 파라미터 적용](#최적-파라미터-적용)

## 시작하기

### 1. 환경 설정

```bash
# MLflow 설치 확인
pip install mlflow>=2.8.0

# 환경 변수 설정
export MLFLOW_TRACKING_URI=file://./mlflow/mlruns
```

### 2. Ground Truth 데이터 준비

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

## 검색 품질 최적화 워크플로우

### Step 1: 빠른 탐색 (Quick Exploration)

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

### Step 2: 상세 최적화 (Detailed Optimization)

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

### Step 3: 최적 파라미터 검증

최적 파라미터로 테스트 데이터셋 평가:

```bash
python scripts/rag/evaluate.py \
    --ground-truth-path data/evaluation/datasets/test.json \
    --vector-store-path data/embeddings/ml_enhanced_ko_sroberta \
    --run-name optimized_validation \
    --experiment-name rag_evaluation \
    --top-k-list 5,10,20
```

## 실험 관리

### MLflow UI 실행

```bash
mlflow ui --backend-store-uri file://./mlflow/mlruns --port 5000
```

브라우저에서 `http://localhost:5000` 접속

### 실험 결과 비교

```bash
python scripts/rag/compare_experiments.py \
    --experiment-name rag_optimization_detailed \
    --output-path data/evaluation/comparison_report.txt
```

## 결과 분석

### 주요 메트릭

- **Recall@K**: 관련 문서를 얼마나 찾았는지
- **Precision@K**: 찾은 문서 중 관련 문서 비율
- **NDCG@K**: 순위를 고려한 정규화 점수
- **MRR**: 첫 번째 관련 문서의 역순위

### 파라미터 영향도 분석

MLflow UI에서:
1. 실험 선택
2. "Compare" 탭 클릭
3. 파라미터별 성능 비교
4. 최적 조합 확인

## 최적 파라미터 적용

### 1. 최적 파라미터 확인

```bash
python scripts/rag/compare_experiments.py \
    --experiment-name rag_optimization_detailed
```

### 2. 코드에 적용

최적 파라미터를 `SemanticSearchEngineV2` 또는 검색 설정에 적용

### 3. 프로덕션 검증

```bash
python scripts/rag/evaluate.py \
    --ground-truth-path data/evaluation/datasets/test.json \
    --vector-store-path data/embeddings/ml_enhanced_ko_sroberta \
    --run-name production_validation
```

## 주기적 개선 프로세스

### 주간 개선 사이클

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

## 문제 해결

### MLflow 연결 오류

```bash
# Tracking URI 확인
echo $MLFLOW_TRACKING_URI

# 수동 설정
export MLFLOW_TRACKING_URI=file://./mlflow/mlruns
```

### 메모리 부족

```bash
# 샘플 크기 줄이기
python scripts/rag/optimize_search_quality.py \
    --sample-size 100 \
    --max-combinations 20
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

## 참고 자료

- [MLflow 공식 문서](https://www.mlflow.org/docs/latest/index.html)
- [FAISS 버전 관리 가이드](../02_data/embedding/faiss_version_management_guide.md)
- [RAG 평가 가이드](./rag_evaluation_guide.md)

