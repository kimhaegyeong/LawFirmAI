# sources_detail 추출 문제 원인 분석 결과

## 테스트 결과 요약

### ✅ 통과한 테스트
1. **sources_detail 추출**: `retrieved_docs`에서 정상적으로 추출됨
2. **sources_by_type 생성**: `sources_detail`이 있으면 정상적으로 생성됨
3. **sources_by_type 생성 (참조 법령 포함)**: 참조 법령도 정상적으로 추가됨
4. **stream_handler._generate_sources_by_type**: 정상 작동
5. **_create_sources_event**: 빈 `sources_detail`에서도 기본 구조 생성

### 🔍 발견된 문제

#### 1. 실제 문제 상황
사용자가 제공한 JSON:
```json
{
  "type": "sources",
  "metadata": {
    "sources_by_type": {
      "statutes_articles": [],
      "precedent_contents": [],
      "precedent_chunks": []
    },
    "sources_detail": []
  }
}
```

**문제**: `sources_detail`이 빈 배열 → `sources_by_type`도 빈 배열

#### 2. 가능한 원인

##### 원인 1: `retrieved_docs`가 비어있음
- `stream_handler.py`의 `stream_final_answer`에서 `retrieved_docs` 추출 실패
- LangGraph 워크플로우에서 검색 결과가 없음
- 검색 쿼리가 제대로 실행되지 않음

##### 원인 2: `sources_detail` 추출 로직 실패
- `sources_extractor._extract_sources_detail()` 호출 실패
- `retrieved_docs`는 있지만 `sources_detail` 변환 실패
- `UnifiedSourceFormatter` 초기화 실패

##### 원인 3: State에서 `sources_detail` 추출 실패
- `stream_final_answer`에서 state를 가져오는 과정에서 실패
- `retrieved_docs`가 state에 저장되지 않음
- 타임아웃으로 인한 state 조회 실패

## 해결 방법

### 1. 로그 확인 필요 사항

#### 백엔드 로그에서 확인:
```python
# api/services/streaming/stream_handler.py
# Line 730-780: sources_detail 추출 로그
[stream_final_answer] Attempting to extract sources
[stream_final_answer] ✅ Extracted {len(sources_detail)} sources_detail from retrieved_docs
[stream_final_answer] Failed to extract sources_detail
```

#### 확인할 로그:
1. `retrieved_docs` 개수: `retrieved_docs_count={len(retrieved_docs)}`
2. `sources_detail` 추출 성공 여부: `✅ Extracted {len(sources_detail)} sources_detail`
3. 추출 실패 원인: `Failed to extract sources_detail: {e}`

### 2. 디버깅 포인트

#### 포인트 1: `retrieved_docs` 확인
```python
# api/services/streaming/stream_handler.py Line 700-730
retrieved_docs = state_values.get("retrieved_docs", [])
if not retrieved_docs:
    logger.warning("retrieved_docs is empty!")
```

#### 포인트 2: `sources_extractor` 초기화 확인
```python
# api/services/streaming/stream_handler.py Line 737
if retrieved_docs and self.sources_extractor:
    # sources_detail 추출 시도
```

#### 포인트 3: `_extract_sources_detail` 호출 확인
```python
# api/services/sources_extractor.py Line 1240-1264
def _extract_sources_detail(self, state_values: Dict[str, Any]) -> List[Dict[str, Any]]:
    # retrieved_docs에서 sources_detail 생성
    if not sources_detail and "retrieved_docs" in state_values:
        sources_detail = self._generate_sources_detail_from_retrieved_docs(
            state_values.get("retrieved_docs", [])
        )
```

### 3. 예상되는 문제 시나리오

#### 시나리오 1: 검색 결과 없음
- 질문에 대한 검색 결과가 없음
- `retrieved_docs`가 빈 배열
- → `sources_detail`도 빈 배열

#### 시나리오 2: State 조회 실패
- `stream_final_answer`에서 state를 가져오는 과정에서 타임아웃
- `retrieved_docs`가 state에 저장되지 않음
- → `sources_detail` 추출 불가

#### 시나리오 3: `sources_extractor` 초기화 실패
- `chat_service.sources_extractor`가 None
- `sources_detail` 추출 시도하지 않음
- → 빈 `sources_detail` 반환

## 권장 조치 사항

### 1. 로그 강화
`stream_handler.py`의 `stream_final_answer`에 더 상세한 로그 추가:
- `retrieved_docs` 개수 및 내용
- `sources_detail` 추출 시도 여부
- 추출 실패 시 상세 에러 메시지

### 2. 폴백 메커니즘 강화
`retrieved_docs`가 없을 때:
- 메시지 metadata에서 `sources_detail` 가져오기
- 세션에서 이전 검색 결과 가져오기
- 최소한 빈 구조라도 반환

### 3. 검증 로직 추가
`sources_detail`이 비어있을 때:
- 검색이 실행되었는지 확인
- 검색 결과가 있는지 확인
- 추출 로직이 실행되었는지 확인

## 테스트 코드 위치
`scripts/tests/test_sources_extraction.py`

## 다음 단계
1. 실제 스트리밍 로그 확인
2. `retrieved_docs` 추출 과정 확인
3. `sources_detail` 추출 실패 원인 확인

