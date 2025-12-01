# RAG 검색 성능 개선 계획

## 개요

LawFirmAI의 RAG 검색 성능을 개선하기 위한 종합 계획입니다. 검색 정확도, 관련성, 다양성을 향상시키기 위한 단계별 개선 방안을 제시합니다.

## 현재 상태 분석

### 주요 문제점
1. 검색 결과의 관련성이 낮음
2. 문서 다양성 부족
3. 질문 유형별 특화 부족
4. 검색 후보 수가 부족하여 필터링 후 결과 부족

### 현재 검색 파이프라인
1. 쿼리 임베딩 생성
2. FAISS 벡터 검색 (k * 2 후보)
3. 필터링 및 재정렬
4. 문서 선택 (최대 7개)

## 개선 계획

### Phase 1: 즉시 적용 (High Impact, Low Effort)

#### 1.1 검색 후보 수 확대
- **현재**: `search_k = k * 2`
- **개선**: `search_k = k * 5`
- **효과**: 필터링 후에도 충분한 결과 확보
- **위치**: `semantic_search_engine_v2.py`의 `_search_with_threshold` 메서드

#### 1.2 쿼리 확장 강화
- **현재**: 상위 3개 키워드만 사용
- **개선**: 상위 5개 키워드 사용, 구조화된 정보(법령 조문 번호 등) 추가
- **효과**: 검색 범위 확대 및 정확도 향상
- **위치**: `semantic_search_engine_v2.py`의 `search_with_query_expansion` 메서드

#### 1.3 동적 임계값 조정
- **현재**: 고정된 임계값 사용
- **개선**: 검색 결과 점수 분포에 따라 동적 조정
- **효과**: 검색 결과가 적을 때 자동으로 임계값 완화
- **위치**: `workflow_document_processor.py`의 `build_prompt_optimized_context` 메서드

### Phase 2: 단기 개선 (High Impact, Medium Effort)

#### 2.1 Cross-Encoder Reranker 도입
- **목적**: Query-Document 쌍을 직접 평가하여 더 정확한 재정렬
- **모델**: `Dongjin-kr/ko-reranker` (한국어 특화)
- **효과**: 재정렬 정확도 20-30% 향상 예상
- **위치**: `core/search/processors/result_ranker.py`

#### 2.2 MMR (Maximal Marginal Relevance) 다양성 알고리즘
- **목적**: 관련성과 다양성의 균형을 맞춘 문서 선택
- **효과**: 중복 문서 제거, 다양한 관점의 문서 제공
- **위치**: `workflow_document_processor.py`

#### 2.3 질문 유형별 가중치 조정
- **목적**: 질문 유형에 따라 검색 전략 최적화
- **구현**:
  - 법령 조회: 키워드 검색 가중치 증가
  - 판례 검색: 의미적 검색 가중치 증가
  - 복잡한 질문: 균형 가중치
- **위치**: `hybrid_search_engine_v2.py`

### Phase 3: 중장기 개선 (Medium Impact, High Effort)

#### 3.1 의미 단위 청킹 전략
- 법령: 조항 단위 청킹
- 판례: 판결 요지, 판단 단위 청킹
- 효과: 더 정확한 문서 매칭

#### 3.2 검색 성능 메트릭 수집 및 자동 튜닝
- 검색 품질 메트릭 수집
- 자동 파라미터 튜닝
- A/B 테스트 지원

## 구현 세부사항

### 즉시 적용 방법

#### 1. 검색 후보 수 확대
```python
# semantic_search_engine_v2.py
search_k = k * 5  # 기존: k * 2
```

#### 2. 쿼리 확장 강화
```python
# semantic_search_engine_v2.py
top_keywords = expanded_keywords[:5]  # 기존: [:3]
```

#### 3. 동적 임계값 조정
```python
# workflow_document_processor.py
# 검색 결과 점수 분포 분석 후 동적 임계값 계산
scores = [doc.get("relevance_score", 0.0) for doc in retrieved_docs]
if scores:
    avg_score = sum(scores) / len(scores)
    dynamic_threshold = max(0.25, avg_score - 0.1)
```

### 단기 개선 방법

#### 1. Cross-Encoder Reranker
```python
from sentence_transformers import CrossEncoder

class ResultRanker:
    def __init__(self):
        self.cross_encoder = CrossEncoder('Dongjin-kr/ko-reranker')
    
    def cross_encoder_rerank(self, documents, query, top_k=10):
        # Cross-Encoder를 사용한 재정렬
        pass
```

#### 2. MMR 다양성 알고리즘
```python
def select_diverse_documents(self, documents, query, max_docs=7, diversity_weight=0.3):
    # MMR 알고리즘 구현
    pass
```

#### 3. 질문 유형별 가중치
```python
weights = {
    "law_inquiry": {"semantic": 0.4, "keyword": 0.6},
    "precedent_search": {"semantic": 0.6, "keyword": 0.4},
    "complex_question": {"semantic": 0.5, "keyword": 0.5}
}
```

## 성능 지표

### 측정 지표
1. **평균 관련성 점수**: 검색 결과의 평균 relevance_score
2. **다양성 점수**: 선택된 문서의 고유 키워드 비율
3. **키워드 커버리지**: 질문 키워드가 문서에 포함된 비율
4. **검색 시간**: 검색 수행 시간

### 목표 개선 수치
- 평균 관련성 점수: 0.45 → 0.60 (33% 향상)
- 다양성 점수: 0.30 → 0.50 (67% 향상)
- 키워드 커버리지: 0.60 → 0.80 (33% 향상)
- 검색 시간: 현재 수준 유지 또는 10% 이내 증가

## 테스트 계획

### 단위 테스트
1. 검색 후보 수 확대 테스트
2. 동적 임계값 조정 테스트
3. Cross-Encoder Reranker 테스트
4. MMR 다양성 알고리즘 테스트

### 통합 테스트
1. 전체 검색 파이프라인 테스트
2. 질문 유형별 검색 테스트
3. 성능 벤치마크 테스트

### 테스트 쿼리 예시
- 법령 조회: "계약 해지 사유는 무엇인가요?"
- 판례 검색: "임대차 계약 해지 관련 판례"
- 복잡한 질문: "계약 해지 시 손해배상 범위와 판례"

## 롤백 계획

각 개선사항은 독립적으로 활성화/비활성화 가능하도록 구현:
- 환경 변수로 제어
- 설정 파일로 관리
- 단계별 배포 가능

## 일정

- **Week 1**: 즉시 적용 방법 구현 및 테스트
- **Week 2**: 단기 개선 방법 구현 및 테스트
- **Week 3**: 통합 테스트 및 성능 측정
- **Week 4**: 프로덕션 배포 및 모니터링

## 참고 자료

- [FAISS 최적화 가이드](https://github.com/facebookresearch/faiss/wiki)
- [Cross-Encoder Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [MMR 알고리즘](https://en.wikipedia.org/wiki/Maximal_marginal_relevance)

