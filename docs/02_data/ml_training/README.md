# ML 훈련 및 평가 시스템

LawFirmAI의 ML 훈련 및 평가 시스템에 대한 문서입니다.

## 📋 개요

LawFirmAI는 RAG 검색 시스템의 성능을 평가하고 개선하기 위한 ML 훈련 및 평가 도구를 제공합니다.

### 주요 기능

- **Ground Truth 생성**: 의사 쿼리 및 클러스터링 기반 Ground Truth 생성
- **RAG 검색 평가**: Recall@K, Precision@K, MRR 등 검색 성능 평가
- **검색 파라미터 튜닝**: 최적의 검색 파라미터 탐색
- **평가 결과 분석**: Test/Val/Train 데이터셋 비교 분석

## 📁 스크립트 구조

```
scripts/ml_training/
├── evaluation/              # 평가 스크립트
│   ├── generate_pseudo_queries.py          # 의사 쿼리 생성
│   ├── generate_clustering_ground_truth.py  # 클러스터링 기반 Ground Truth 생성
│   ├── generate_rag_evaluation_dataset.py  # 평가 데이터셋 생성
│   ├── evaluate_rag_search.py              # RAG 검색 평가
│   ├── analyze_rag_evaluation_results.py    # 평가 결과 분석
│   ├── tune_search_parameters.py            # 검색 파라미터 튜닝
│   └── check_progress.py                    # 진행 상황 확인
├── model_training/          # 모델 훈련 스크립트
└── vector_embedding/        # 벡터 임베딩 스크립트
```

## 🚀 빠른 시작

### 1. Ground Truth 생성

#### 의사 쿼리 생성

```bash
python scripts/ml_training/evaluation/generate_pseudo_queries.py \
    --vector-store-path data/embeddings/ml_enhanced_ko_sroberta \
    --output-path data/evaluation/ground_truth/pseudo_queries.json \
    --model-name jhgan/ko-sroberta-multitask \
    --llm-provider gemini \
    --batch-size 10 \
    --checkpoint-dir data/evaluation/checkpoints
```

#### 클러스터링 기반 Ground Truth 생성

```bash
python scripts/ml_training/evaluation/generate_clustering_ground_truth.py \
    --vector-store-path data/embeddings/ml_enhanced_ko_sroberta \
    --output-path data/evaluation/ground_truth/clustering_ground_truth.json \
    --model-name jhgan/ko-sroberta-multitask \
    --clustering-method hdbscan \
    --min-cluster-size 5 \
    --checkpoint-dir data/evaluation/checkpoints
```

### 2. 평가 데이터셋 생성

```bash
python scripts/ml_training/evaluation/generate_rag_evaluation_dataset.py \
    --ground-truth-path data/evaluation/ground_truth/pseudo_queries.json \
    --output-dir data/evaluation/datasets \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### 3. RAG 검색 평가

```bash
python scripts/ml_training/evaluation/evaluate_rag_search.py \
    --ground-truth-path data/evaluation/datasets/test.json \
    --vector-store-path data/embeddings/ml_enhanced_ko_sroberta \
    --output-path data/evaluation/evaluation_reports/rag_evaluation_report_test.json \
    --top-k-list 5,10,20 \
    --checkpoint-dir data/evaluation/checkpoints \
    --checkpoint-interval 100
```

### 4. 평가 결과 분석

```bash
python scripts/ml_training/evaluation/analyze_rag_evaluation_results.py \
    --reports-dir data/evaluation/evaluation_reports \
    --output-path data/evaluation/analysis/comparison_report.json
```

### 5. 검색 파라미터 튜닝

```bash
python scripts/ml_training/evaluation/tune_search_parameters.py \
    --ground-truth-path data/evaluation/datasets/val.json \
    --vector-store-path data/embeddings/ml_enhanced_ko_sroberta \
    --output-path data/evaluation/tuning/parameter_tuning_results.json \
    --top-k-range 5,50,5
```

## 📊 평가 메트릭

### 검색 성능 메트릭

- **Recall@K**: 상위 K개 결과 중 관련 문서 비율
- **Precision@K**: 상위 K개 결과 중 관련 문서 비율
- **NDCG@K**: 정규화된 할인 누적 이득 (Normalized Discounted Cumulative Gain)
- **MRR**: 평균 상호 순위 (Mean Reciprocal Rank)

### 평가 결과 예시

```json
{
  "aggregated_metrics": {
    "recall@5_mean": 0.7234,
    "recall@5_std": 0.1234,
    "precision@5_mean": 0.6543,
    "precision@5_std": 0.0987,
    "ndcg@5_mean": 0.7890,
    "ndcg@5_std": 0.1123,
    "mrr_mean": 0.8123,
    "mrr_std": 0.0987,
    "total_queries": 1000
  }
}
```

## 🔧 주요 기능 상세

### 1. 의사 쿼리 생성 (Pseudo Query Generation)

문서 기반으로 LLM을 사용하여 질문을 생성하고, 원본 문서를 Ground Truth로 사용합니다.

**특징**:
- 배치 처리로 효율적인 생성
- 체크포인트 지원으로 중단 후 재개 가능
- Gemini API 비용 최적화
- 메모리 효율적인 처리

**사용 예시**:
```python
from scripts.ml_training.evaluation.generate_pseudo_queries import PseudoQueryGenerator

generator = PseudoQueryGenerator(
    vector_store_path="data/embeddings/ml_enhanced_ko_sroberta",
    model_name="jhgan/ko-sroberta-multitask",
    llm_provider="gemini"
)

ground_truth = generator.generate(
    output_path="data/evaluation/ground_truth/pseudo_queries.json",
    batch_size=10,
    checkpoint_dir="data/evaluation/checkpoints"
)
```

### 2. 클러스터링 기반 Ground Truth 생성

벡터 스토어의 모든 문서를 클러스터링하여, 같은 클러스터 내 문서들을 서로 관련 문서로 간주합니다.

**특징**:
- HDBSCAN 및 K-Means 클러스터링 지원
- 최적 클러스터 수 자동 탐색
- 대규모 데이터셋 처리 지원
- 체크포인트 지원

**사용 예시**:
```python
from scripts.ml_training.evaluation.generate_clustering_ground_truth import ClusteringGroundTruthGenerator

generator = ClusteringGroundTruthGenerator(
    vector_store_path="data/embeddings/ml_enhanced_ko_sroberta",
    model_name="jhgan/ko-sroberta-multitask",
    clustering_method="hdbscan"
)

ground_truth = generator.generate(
    output_path="data/evaluation/ground_truth/clustering_ground_truth.json",
    min_cluster_size=5,
    checkpoint_dir="data/evaluation/checkpoints"
)
```

### 3. RAG 검색 평가

생성된 Ground Truth를 사용하여 RAG 검색 시스템의 성능을 평가합니다.

**특징**:
- Recall@K, Precision@K, NDCG@K, MRR 메트릭 계산
- 체크포인트 지원으로 대규모 평가 가능
- 상세한 쿼리별 메트릭 제공

**사용 예시**:
```python
from scripts.ml_training.evaluation.evaluate_rag_search import RAGSearchEvaluator

evaluator = RAGSearchEvaluator(
    vector_store_path="data/embeddings/ml_enhanced_ko_sroberta",
    model_name="jhgan/ko-sroberta-multitask",
    checkpoint_dir="data/evaluation/checkpoints"
)

results = evaluator.run(
    ground_truth_path="data/evaluation/datasets/test.json",
    top_k_list=[5, 10, 20],
    resume_from_checkpoint=True,
    checkpoint_interval=100
)
```

## 📈 성능 최적화

### 메모리 최적화

- 배치 처리로 메모리 사용량 제어
- 체크포인트를 통한 중간 결과 저장
- 불필요한 데이터 즉시 삭제

### 비용 최적화

- Gemini API 호출 최소화
- 배치 처리로 API 호출 횟수 감소
- 캐싱을 통한 중복 호출 방지

### 처리 속도 최적화

- 병렬 처리 지원
- 체크포인트를 통한 중단 후 재개
- 효율적인 데이터 구조 사용

## 🔍 진행 상황 확인

```bash
python scripts/ml_training/evaluation/check_progress.py \
    --checkpoint-dir data/evaluation/checkpoints
```

## 📚 관련 문서

- [의사 쿼리 최적화 요약](../05_quality/pseudo_query_optimization_summary.md)
- [Ground Truth 생성 성능 개선](../05_quality/ground_truth_generation_performance_improvements.md)
- [Gemini API 비용 최적화](../05_quality/gemini_api_cost_optimization.md)
- [메모리 최적화 요약](../05_quality/memory_optimization_summary.md)
- [근사 Ground Truth 생성 계획](../05_quality/approximate_ground_truth_generation_plan.md)

## 🛠️ 문제 해결

### 체크포인트에서 재개

대부분의 스크립트는 체크포인트를 지원합니다. 중단된 작업은 체크포인트에서 자동으로 재개됩니다.

### 메모리 부족 오류

배치 크기를 줄이거나 체크포인트 간격을 줄여서 메모리 사용량을 제어하세요.

### API 비용 문제

배치 크기를 조정하거나 LLM 제공자를 변경하여 비용을 최적화하세요.

