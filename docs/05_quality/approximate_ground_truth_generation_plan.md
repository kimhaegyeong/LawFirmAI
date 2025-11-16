# 근사 Ground Truth 생성 계획

## 1. 개요

RAG 검색 평가를 위한 근사 Ground Truth 생성 시스템 구축 계획입니다. 정확한 Ground Truth는 없지만, RAG 검색 평가용 **pseudo Ground Truth**를 생성하여 Recall 평가의 근사치를 얻을 수 있습니다.

## 2. 목표

- RAG 검색 시스템의 Recall@K, Precision@K, MRR 등 평가 메트릭 계산을 위한 Ground Truth 데이터셋 생성
- 두 가지 방법을 병행하여 더 신뢰성 있는 평가 데이터셋 구축
  - 문서 간 클러스터링 기반 Ground Truth
  - 문서 기반 Pseudo-Query 생성 (LLM 사용)

## 3. 현재 사용 중인 벡터 임베딩 버전

- **모델**: `jhgan/ko-sroberta-multitask`
- **차원**: 768
- **법령 벡터 스토어 경로**: `data/embeddings/ml_enhanced_ko_sroberta`
- **판례 벡터 스토어 경로**: `data/embeddings/ml_enhanced_ko_sroberta_precedents`
- **버전 관리**: `VectorStoreVersionManager` 사용

## 4. 구현 방법

### 4.1 문서 간 클러스터링 기반 Ground Truth

**원리**: 비슷한 문서끼리 묶고, 하나의 클러스터 안에서는 전부 "서로 관련 있다"고 간주하는 방식

**구현 위치**: `scripts/ml_training/evaluation/generate_clustering_ground_truth.py`

**기능**:
- 벡터 스토어의 모든 문서 임베딩 수집
- K-means 또는 HDBSCAN으로 클러스터링
- 같은 클러스터 내 문서를 서로 관련 문서로 간주
- Ground Truth 데이터셋 생성

**출력 형식**:
```json
{
  "query_doc_id": "doc_123",
  "query_text": "문서 내용...",
  "relevant_doc_ids": ["doc_456", "doc_789"],
  "cluster_id": 5,
  "cluster_size": 12,
  "metadata": {
    "law_id": "law_001",
    "article_number": "1",
    "type": "statute_article"
  }
}
```

**클러스터링 알고리즘**:
- 기본: K-means (자동 클러스터 수 결정 - Elbow method)
- 옵션: HDBSCAN (밀도 기반 클러스터링)
- 유사도 임계값: 클러스터 내 문서 간 최소 유사도 설정
- 필터링: 클러스터 크기 최소값 설정 (너무 작은 클러스터 제외)

### 4.2 문서 기반 Pseudo-Query 생성

**원리**: LLM을 사용하여 문서 내용에서 질문을 생성하고, 원본 문서를 Ground Truth로 사용

**구현 위치**: `scripts/ml_training/evaluation/generate_pseudo_queries.py`

**기능**:
- 벡터 스토어의 각 문서에 대해 LLM으로 질문 생성
- 문서 내용 → 질문 3-5개 생성
- 생성된 질문과 원본 문서를 Ground Truth로 매핑

**출력 형식**:
```json
{
  "query": "반품은 어디에서 할 수 있나요?",
  "ground_truth_doc_id": "doc_123",
  "ground_truth_text": "반품은 주문 조회 페이지에서 가능합니다.",
  "generated_queries": [
    "반품은 어디에서 할 수 있나요?",
    "반품 요청 위치는?",
    "주문 조회에서 어떤 기능을 할 수 있나요?"
  ],
  "metadata": {
    "law_id": "law_001",
    "article_number": "1",
    "type": "statute_article"
  }
}
```

**LLM 설정**:
- 모델: 프로젝트에서 사용 중인 LLM 모델 활용
- 프롬프트 템플릿: 법률 문서 특화
- 생성 개수: 문서당 3-5개 질문
- 품질 필터링: 생성된 질문의 유효성 검증

### 4.3 통합 평가 데이터셋 생성

**구현 위치**: `scripts/ml_training/evaluation/generate_rag_evaluation_dataset.py`

**기능**:
- 두 방법의 결과 통합
- 중복 제거 및 품질 필터링
- train/val/test 분할
- 평가 메트릭 계산용 포맷 변환

**출력**:
- `data/evaluation/rag_ground_truth_clustering.json`
- `data/evaluation/rag_ground_truth_pseudo_queries.json`
- `data/evaluation/rag_ground_truth_combined.json`
- `data/evaluation/rag_ground_truth_train.json`
- `data/evaluation/rag_ground_truth_val.json`
- `data/evaluation/rag_ground_truth_test.json`

### 4.4 RAG 검색 평가

**구현 위치**: `scripts/ml_training/evaluation/evaluate_rag_search.py`

**기능**:
- 생성된 Ground Truth로 RAG 검색 평가
- Recall@K, Precision@K, MRR 계산
- 결과 리포트 생성

**평가 메트릭**:
- **Recall@K**: 상위 K개 결과 중 Ground Truth 문서가 포함된 비율
- **Precision@K**: 상위 K개 결과 중 Ground Truth 문서의 비율
- **MRR (Mean Reciprocal Rank)**: 첫 번째 관련 문서의 역순위 평균
- **NDCG@K**: 정규화된 누적 할인 이득

## 5. 파일 구조

```
scripts/ml_training/evaluation/
├── __init__.py
├── generate_clustering_ground_truth.py      # 클러스터링 기반 GT 생성
├── generate_pseudo_queries.py                # LLM 기반 pseudo-query 생성
├── generate_rag_evaluation_dataset.py        # 통합 데이터셋 생성
├── evaluate_rag_search.py                    # RAG 검색 평가
└── utils/
    ├── __init__.py
    ├── clustering_utils.py                   # 클러스터링 유틸리티
    └── query_generation_utils.py             # 질문 생성 유틸리티

data/evaluation/
├── rag_ground_truth_clustering.json
├── rag_ground_truth_pseudo_queries.json
├── rag_ground_truth_combined.json
├── rag_ground_truth_train.json
├── rag_ground_truth_val.json
├── rag_ground_truth_test.json
└── evaluation_reports/
    └── rag_evaluation_report_YYYYMMDD.json
```

## 6. 사용 방법

### 6.1 클러스터링 기반 Ground Truth 생성

```bash
python scripts/ml_training/evaluation/generate_clustering_ground_truth.py \
    --vector_store_path data/embeddings/ml_enhanced_ko_sroberta \
    --output_path data/evaluation/rag_ground_truth_clustering.json \
    --algorithm kmeans \
    --n_clusters auto \
    --min_cluster_size 3 \
    --similarity_threshold 0.7
```

### 6.2 Pseudo-Query 기반 Ground Truth 생성

```bash
python scripts/ml_training/evaluation/generate_pseudo_queries.py \
    --vector_store_path data/embeddings/ml_enhanced_ko_sroberta \
    --output_path data/evaluation/rag_ground_truth_pseudo_queries.json \
    --queries_per_doc 3 \
    --llm_model gpt-4 \
    --batch_size 10
```

### 6.3 통합 데이터셋 생성

```bash
python scripts/ml_training/evaluation/generate_rag_evaluation_dataset.py \
    --clustering_path data/evaluation/rag_ground_truth_clustering.json \
    --pseudo_queries_path data/evaluation/rag_ground_truth_pseudo_queries.json \
    --output_path data/evaluation/rag_ground_truth_combined.json \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

### 6.4 RAG 검색 평가

```bash
python scripts/ml_training/evaluation/evaluate_rag_search.py \
    --ground_truth_path data/evaluation/rag_ground_truth_test.json \
    --vector_store_path data/embeddings/ml_enhanced_ko_sroberta \
    --output_path data/evaluation/evaluation_reports/rag_evaluation_report.json \
    --top_k_list 5,10,20
```

## 7. 의존성

필요 패키지:
- `scikit-learn` (K-means 클러스터링)
- `hdbscan` (HDBSCAN 클러스터링, 선택사항)
- `openai` 또는 프로젝트 LLM 인터페이스 (pseudo-query 생성)
- 기존 프로젝트 의존성 (FAISS, Sentence-BERT 등)

## 8. 제한사항

- 현재 사용 중인 벡터 임베딩 버전으로만 제한
  - 모델: `jhgan/ko-sroberta-multitask`
  - 법령: `data/embeddings/ml_enhanced_ko_sroberta`
  - 판례: `data/embeddings/ml_enhanced_ko_sroberta_precedents`

## 9. 구현 상태

1. ✅ 계획 문서 작성
2. ✅ evaluation 디렉토리 구조 생성
3. ✅ 클러스터링 기반 Ground Truth 생성 스크립트 구현
4. ✅ Pseudo-Query 생성 스크립트 구현
5. ✅ 통합 데이터셋 생성 스크립트 구현
6. ✅ 평가 스크립트 구현
7. ⏳ 테스트 및 검증

## 10. 구현 완료 파일 목록

### 스크립트 파일
- `scripts/ml_training/evaluation/__init__.py`
- `scripts/ml_training/evaluation/generate_clustering_ground_truth.py`
- `scripts/ml_training/evaluation/generate_pseudo_queries.py`
- `scripts/ml_training/evaluation/generate_rag_evaluation_dataset.py`
- `scripts/ml_training/evaluation/evaluate_rag_search.py`
- `scripts/ml_training/evaluation/utils/__init__.py`

### 문서 파일
- `docs/05_quality/approximate_ground_truth_generation_plan.md`

## 11. 참고 자료

- OpenAI / Google / LangChain의 실제 RAG 평가 데이터셋 생성 방법
- FAISS hierarchical clustering
- K-means, HDBSCAN 클러스터링 알고리즘

