# FAISS 인덱스 테스트 결과 요약

## 테스트 일시
2025-11-09 22:38

## 발견된 문제점

### 1. FAISS 인덱스 불일치 문제
- **문제**: FAISS 인덱스에 16,411개의 벡터만 포함되어 있음
- **실제 데이터**: embeddings 테이블에는 26,630개의 벡터 존재
- **원인**: 오래된 인덱스 파일이 사용되고 있었음
- **해결**: 인덱스 파일 삭제 후 재빌드 예정

### 2. 검색 결과 타입 분포
**embeddings 테이블 분포**:
- case_paragraph: 16,203개 (60.8%)
- decision_paragraph: 7,246개 (27.2%)
- statute_article: 2,093개 (7.9%)
- interpretation_paragraph: 1,088개 (4.1%)

**검색 결과 분포** (인덱스 삭제 후):
- statute_article: 6개
- interpretation_paragraph: 1개
- unknown: 3개

## 개선 사항

### 1. 로깅 강화
- FAISS 인덱스 로드/사용 여부 로깅 추가
- 검색 결과 타입 분포 로깅 추가
- 다양성 보장 로직 작동 여부 확인

### 2. 다양성 보장 로직
- 검색 결과 재분배 로직이 작동 중
- 하지만 초기 검색 결과에 판례/결정례가 없어 재분배 불가

## 테스트 결과

### Before (FAISS 인덱스 사용 시)
- retrieved_docs: statute_article 7개, unknown 3개
- sources: law 2개, document 1개
- 판례/해석례/결정례: 없음

### After (인덱스 삭제 후 선형 검색)
- retrieved_docs: statute_article 6개, interpretation_paragraph 1개, unknown 3개
- sources: law 2개, interpretation 1개, document 1개
- 해석례: ✅ 포함됨
- 판례/결정례: ❌ 여전히 없음

## 완료된 개선 사항

### 1. FAISS 인덱스 재빌드 ✅
- 모든 타입의 벡터가 포함되도록 재빌드 완료
- 인덱스 크기: 26,630개 벡터 (모든 타입 포함)

### 2. 검색 쿼리 개선 ✅
- 검색 결과 수 증가: `k * 2` → `k * 3` (search_handler)
- similarity_threshold 조정: `0.5` → `0.4` (더 많은 결과 확보)
- 검색 후보 수 증가: `k * 2` → `k * 3` (semantic_search_engine_v2)

### 3. 유사도 계산 개선 ✅
- 타입별 가중치 적용:
  - `statute_article`: 1.0 (기본)
  - `case_paragraph`: 1.15 (15% 보정)
  - `decision_paragraph`: 1.15 (15% 보정)
  - `interpretation_paragraph`: 1.10 (10% 보정)
- 가중치가 적용된 점수로 relevance_score 및 hybrid_score 계산

## 최종 테스트 결과

### 검색 결과 (개선 후)
- retrieved_docs: statute_article 6개, interpretation_paragraph 1개, unknown 3개
- sources: law 2개, interpretation 1개, document 1개
- 해석례: ✅ 포함됨
- 판례/결정례: ❌ 여전히 검색되지 않음

### 다양성 보장 로직 작동 확인
- 로그: "Rebalanced results: law=1, precedent=1, decision=1, interpretation=2"
- 재분배 로직은 작동하지만, 초기 검색 결과에 판례/결정례가 없어 재분배 불가

## 분석

1. **검색 쿼리 특성**: "계약서 작성 시 주의사항"이라는 쿼리가 법령에 더 관련이 높아 판례/결정례의 유사도가 낮을 수 있음
2. **데이터 분포**: embeddings 테이블에는 판례(60.8%), 결정례(27.2%)가 대부분이지만, 검색 결과에는 법령이 우세
3. **개선 효과**: 해석례는 포함되었지만, 판례/결정례는 여전히 검색되지 않음

## 추가 개선 제안

1. **쿼리 확장**: 판례/결정례 관련 키워드를 자동으로 추가
2. **별도 검색**: 초기 검색 결과에 판례/결정례가 없을 경우 별도 검색 수행
3. **타입별 최소 결과 수 보장**: 각 타입별로 최소 1개씩은 포함되도록 강제

## 로그 확인 사항

- `🔍 [FAISS] Using FAISS index for search`: FAISS 인덱스 사용 중
- `🔍 [LINEAR] Loaded X vectors`: 선형 검색 사용 중
- `🔍 [DIVERSITY] Type distribution before rebalancing`: 재분배 전 타입 분포
- `🔍 [DIVERSITY] Rebalanced results`: 재분배 후 결과

