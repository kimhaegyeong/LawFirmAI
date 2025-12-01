# RAG 빌드 및 평가 모듈

FAISS 인덱스 빌드, MLflow 버전 관리, RAG 평가 기능을 제공합니다.

## 모듈 구조

```
scripts/rag/
├── __init__.py
├── mlflow_manager.py          # MLflow FAISS 관리자
├── build_index.py             # 인덱스 빌드 및 MLflow 저장
├── build_production_index.py  # 프로덕션 인덱스 빌드
├── evaluate.py                # MLflow 통합 RAG 평가
├── optimize_search_quality.py # 검색 품질 최적화
├── compare_experiments.py    # 실험 결과 비교
├── verify_mlflow_setup.py    # MLflow 설정 검증
├── apply_optimized_params.py # 최적 파라미터 적용
├── load_optimized_params.py  # 최적 파라미터 로더
├── visualize_results.py      # 결과 시각화
└── automated_optimization_pipeline.py # 자동화 파이프라인
```

## 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 활성화 (scripts/venv)
cd scripts
.\venv\Scripts\activate  # Windows PowerShell
# 또는
source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install mlflow>=2.8.0 faiss-cpu sentence-transformers tqdm
```

### 2. 설정 검증

```bash
python scripts/rag/verify_mlflow_setup.py
```

### 3. 인덱스 빌드 및 MLflow 저장

```bash
# 활성 버전 확인 후
python scripts/rag/build_index.py \
    --version-name v2.0.0-dynamic \
    --embedding-version-id 5 \
    --chunking-strategy dynamic \
    --db-path data/lawfirm_v2.db
```

### 4. MLflow UI 실행

```bash
mlflow ui --backend-store-uri file://./mlflow/mlruns
```

## 주요 기능

### MLflowFAISSManager

FAISS 인덱스를 MLflow로 관리:

```python
from scripts.rag.mlflow_manager import MLflowFAISSManager

manager = MLflowFAISSManager()

# Run 생성
run_id = manager.create_run(
    version_name="v2.0.0-standard",
    embedding_version_id=1,
    chunking_strategy="standard",
    chunking_config={"chunk_size": 1000},
    embedding_config={"model": "jhgan/ko-sroberta-multitask"},
    document_count=1000,
    total_chunks=5000
)

# 인덱스 저장
manager.save_index(run_id, index, id_mapping, metadata)

# 인덱스 로드
index_data = manager.load_index(run_id)
```

### 검색 품질 최적화

```bash
python scripts/rag/optimize_search_quality.py \
    --ground-truth-path data/evaluation/datasets/test.json \
    --vector-store-path data/embeddings/ml_enhanced_ko_sroberta \
    --max-combinations 50 \
    --primary-metric ndcg@10
```

### 실험 결과 비교

```bash
python scripts/rag/compare_experiments.py \
    --experiment-name rag_search_optimization \
    --output-path data/evaluation/comparison_report.txt
```

## 문서

- [MLflow 통합 가이드](../../docs/03_rag_system/mlflow_integration_guide.md)

