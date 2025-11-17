# 최적 파라미터 통합 가이드

MLflow를 통해 최적화된 검색 파라미터를 RAG 시스템에 통합하는 방법을 설명합니다.

## 개요

최적화된 검색 파라미터는 `data/ml_config/optimized_search_params.json` 파일에 저장되어 있으며, RAG 시스템의 `QueryEnhancer` 클래스에서 자동으로 로드됩니다.

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

## 자동 로드

`QueryEnhancer` 클래스는 초기화 시 자동으로 최적 파라미터를 로드합니다:

1. 환경 변수 `OPTIMIZED_SEARCH_PARAMS_PATH` 확인
2. 기본 경로: `data/ml_config/optimized_search_params.json`
3. 파일이 존재하면 자동 로드
4. 파일이 없으면 기본값 사용 (기존 동작 유지)

## 사용 방법

### 1. 기본 사용 (자동 로드)

최적 파라미터 파일이 기본 경로에 있으면 자동으로 적용됩니다:

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

### 2. 환경 변수로 경로 지정

다른 경로의 최적 파라미터 파일을 사용하려면:

```bash
export OPTIMIZED_SEARCH_PARAMS_PATH=/path/to/your/optimized_params.json
```

또는 `.env` 파일에 추가:

```env
OPTIMIZED_SEARCH_PARAMS_PATH=data/ml_config/optimized_search_params.json
```

### 3. 최적 파라미터 확인

최적 파라미터가 로드되었는지 확인:

```python
if query_enhancer.optimized_params:
    print("최적 파라미터 로드됨:", query_enhancer.optimized_params)
else:
    print("최적 파라미터 미로드 (기본값 사용)")
```

## 파라미터 적용 방식

최적 파라미터는 다음과 같이 적용됩니다:

1. **기본값으로 사용**: 최적 파라미터의 `top_k`와 `similarity_threshold`가 기본값으로 사용됩니다.
2. **동적 조정 유지**: 질문 유형, 복잡도, 키워드 수에 따른 동적 조정은 그대로 유지됩니다.
3. **추가 기능 활성화**: `use_reranking`, `query_enhancement`, `use_keyword_search` 등의 기능이 활성화됩니다.

## 성능 개선

최적 파라미터 적용 시 예상되는 성능 개선:

- **NDCG@10**: 0.005621 (최적화 전 대비 개선)
- **Recall@10**: 향상
- **Precision@10**: 향상

## 주기적 재최적화

새로운 데이터가 추가되거나 모델이 업데이트되면 주기적으로 재최적화를 수행하세요:

```bash
python scripts/rag/automated_optimization_pipeline.py \
    --vector-store-path data/vector_store/v2.0.0-dynamic-dynamic-ivfpq \
    --ground-truth-path data/evaluation/rag_ground_truth_combined_test.json
```

## 문제 해결

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

### 기본값으로 되돌리기

최적 파라미터 파일을 삭제하거나 이름을 변경하면 기본값이 사용됩니다:

```bash
mv data/ml_config/optimized_search_params.json data/ml_config/optimized_search_params.json.bak
```

## 관련 파일

- 최적 파라미터 설정: `data/ml_config/optimized_search_params.json`
- 파라미터 로더: `scripts/rag/load_optimized_params.py`
- QueryEnhancer: `lawfirm_langgraph/core/search/optimizers/query_enhancer.py`
- 최적화 스크립트: `scripts/rag/optimize_search_quality.py`
- 자동화 파이프라인: `scripts/rag/automated_optimization_pipeline.py`

