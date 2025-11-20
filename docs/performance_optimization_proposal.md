# LangGraph 성능 최적화 제안서

## 📊 현재 성능 병목 지점 분석

### 1. 동기/비동기 혼용 문제
- **위치**: `legal_workflow_enhanced.py`, `search_execution_processor.py`
- **문제**: `ThreadPoolExecutor`를 사용하여 비동기 작업을 동기적으로 실행
- **영향**: 비동기 이점을 활용하지 못하고 오버헤드 발생

### 2. 불필요한 대기 시간
- **위치**: 여러 파일에서 `time.sleep()` 호출
- **문제**: 
  - `semantic_search_engine_v2.py`: `time.sleep(0.5)` 재시도 대기
  - `legal_workflow_enhanced.py`: `time.sleep(1)` 긴급도 평가 대기
  - `prompt_chain_executor.py`: `time.sleep(0.5)` 재시도 대기
- **영향**: 누적 대기 시간이 전체 실행 시간에 영향

### 3. 순차 실행으로 인한 병목
- **위치**: 검색 실행, 키워드 확장, 분류 작업
- **문제**: 병렬로 실행 가능한 작업들이 순차적으로 실행됨
- **영향**: 전체 실행 시간 = 각 작업 시간의 합

### 4. 모델 로딩 최적화 부족
- **위치**: `semantic_search_engine_v2.py`
- **문제**: SentenceTransformer 모델이 매번 로드되거나 느리게 로드될 수 있음
- **영향**: 첫 실행 시 긴 대기 시간

### 5. State 접근 최적화 부족
- **위치**: `search_execution_processor.py`의 `get_search_params()`
- **문제**: State에서 값을 가져올 때 여러 번 접근 (6단계 확인)
- **영향**: 불필요한 딕셔너리 탐색 오버헤드

### 6. 캐싱 활용 부족
- **위치**: 검색 결과, 키워드 확장, 쿼리 최적화
- **문제**: 일부 작업에 캐싱이 없거나 제한적
- **영향**: 동일한 작업 반복 실행

## 🚀 개선 방안

### 우선순위 1: 비동기 처리 개선 (High Impact)

#### 1.1 ThreadPoolExecutor → asyncio.gather 전환
**현재 코드**:
```python
# legal_workflow_enhanced.py:3711
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        'urgency': executor.submit(self._assess_urgency_internal, query),
        'multi_turn': executor.submit(self._resolve_multi_turn_internal, query, session_id),
    }
```

**개선안**:
```python
# 비동기 함수로 변경
async def classification_parallel(self, state: LegalWorkflowState) -> LegalWorkflowState:
    query = self._get_state_value(state, "query", "")
    session_id = self._get_state_value(state, "session_id", "")
    
    # 병렬 비동기 실행
    urgency_task = asyncio.create_task(
        self._assess_urgency_async(query)
    )
    multi_turn_task = asyncio.create_task(
        self._resolve_multi_turn_async(query, session_id)
    )
    
    # 동시 실행 및 결과 수집
    urgency_result, multi_turn_result = await asyncio.gather(
        urgency_task, multi_turn_task, return_exceptions=True
    )
```

**예상 개선**: 30-50% 시간 단축 (2개 작업 병렬 실행)

#### 1.2 검색 작업 병렬화
**현재 코드**:
```python
# search_execution_processor.py:357
with ThreadPoolExecutor(max_workers=2) as executor:
    # 순차적 실행
```

**개선안**:
```python
# 모든 검색 작업을 비동기로 병렬 실행
async def execute_searches_parallel_async(self, state: LegalWorkflowState):
    search_tasks = []
    
    # 모든 검색 작업을 태스크로 생성
    for search_type in search_types:
        task = asyncio.create_task(
            self._execute_single_search_async(search_type, state)
        )
        search_tasks.append(task)
    
    # 모든 검색을 동시에 실행
    results = await asyncio.gather(*search_tasks, return_exceptions=True)
    return results
```

**예상 개선**: 40-60% 시간 단축 (검색 작업 수에 비례)

### 우선순위 2: 불필요한 대기 시간 제거 (Medium Impact)

#### 2.1 time.sleep() 제거 또는 최소화
**현재 코드**:
```python
# semantic_search_engine_v2.py:1112
time.sleep(0.5)  # 재시도 대기
```

**개선안**:
```python
# exponential backoff 사용
import asyncio

async def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = min(0.1 * (2 ** attempt), 1.0)  # 최대 1초
                await asyncio.sleep(wait_time)
            else:
                raise
```

**예상 개선**: 10-20% 시간 단축 (대기 시간 누적 제거)

### 우선순위 3: 모델 로딩 최적화 (High Impact)

#### 3.1 싱글톤 패턴으로 모델 재사용
**현재 코드**:
```python
# semantic_search_engine_v2.py:106
self.model = SentenceTransformer(model_name, device="cpu")
```

**개선안**:
```python
# 모델 싱글톤 매니저
class ModelManager:
    _instances = {}
    _lock = asyncio.Lock()
    
    @classmethod
    async def get_model(cls, model_name: str):
        if model_name not in cls._instances:
            async with cls._lock:
                if model_name not in cls._instances:
                    cls._instances[model_name] = SentenceTransformer(
                        model_name, device="cpu"
                    )
        return cls._instances[model_name]

# 사용
self.model = await ModelManager.get_model(model_name)
```

**예상 개선**: 첫 실행 후 80-90% 시간 단축 (모델 로딩 제거)

### 우선순위 4: State 접근 최적화 (Medium Impact)

#### 4.1 State 접근 캐싱
**현재 코드**:
```python
# search_execution_processor.py:58-200
# 6단계로 State에서 optimized_queries 찾기
```

**개선안**:
```python
# State 접근 결과 캐싱
class StateAccessCache:
    def __init__(self):
        self._cache = {}
        self._cache_key = None
    
    def get_optimized_queries(self, state: LegalWorkflowState):
        # State 해시로 캐시 키 생성
        state_hash = hash(str(sorted(state.items())))
        
        if self._cache_key != state_hash:
            # 한 번만 접근하여 모든 값 가져오기
            optimized_queries = self._get_optimized_queries_once(state)
            self._cache = {
                'optimized_queries': optimized_queries,
                # ... 기타 값들
            }
            self._cache_key = state_hash
        
        return self._cache['optimized_queries']
```

**예상 개선**: 5-10% 시간 단축 (State 접근 오버헤드 감소)

### 우선순위 5: 캐싱 강화 (Medium Impact)

#### 5.1 검색 결과 캐싱 확대
**개선안**:
```python
# 검색 결과 캐싱 강화
class EnhancedSearchCache:
    def __init__(self):
        self.query_cache = {}  # 쿼리 -> 검색 결과
        self.embedding_cache = {}  # 텍스트 -> 임베딩
    
    async def get_or_search(self, query: str, search_func):
        # 쿼리 해시로 캐시 확인
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash in self.query_cache:
            return self.query_cache[query_hash]
        
        # 캐시 미스 시 검색 실행
        result = await search_func(query)
        self.query_cache[query_hash] = result
        return result
```

**예상 개선**: 반복 쿼리에서 70-90% 시간 단축

### 우선순위 6: 배치 처리 최적화 (Low-Medium Impact)

#### 6.1 임베딩 배치 처리
**개선안**:
```python
# 단일 텍스트가 아닌 배치로 임베딩 생성
async def embed_batch(self, texts: List[str], batch_size: int = 32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = await self.model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

**예상 개선**: 임베딩 생성 시 20-30% 시간 단축

## 📈 예상 성능 개선 효과

### 개별 개선 효과
| 개선 항목 | 예상 시간 단축 | 우선순위 |
|---------|--------------|---------|
| 비동기 처리 개선 | 30-60% | High |
| 불필요한 대기 제거 | 10-20% | Medium |
| 모델 로딩 최적화 | 80-90% (첫 실행 후) | High |
| State 접근 최적화 | 5-10% | Medium |
| 캐싱 강화 | 70-90% (반복 쿼리) | Medium |
| 배치 처리 최적화 | 20-30% | Low-Medium |

### 종합 예상 효과
- **첫 실행**: 40-50% 시간 단축
- **반복 실행 (캐시 히트)**: 70-85% 시간 단축
- **병렬 처리 가능한 작업**: 50-70% 시간 단축

## 🔧 구현 단계

### Phase 1: 빠른 개선 (1-2일)
1. ✅ time.sleep() 제거 또는 최소화
2. ✅ 모델 싱글톤 패턴 구현
3. ✅ State 접근 캐싱

### Phase 2: 중기 개선 (3-5일)
1. ✅ ThreadPoolExecutor → asyncio.gather 전환
2. ✅ 검색 작업 병렬화
3. ✅ 캐싱 강화

### Phase 3: 장기 개선 (1주일)
1. ✅ 배치 처리 최적화
2. ✅ 성능 모니터링 추가
3. ✅ 프로파일링 및 추가 최적화

## 📝 구현 시 주의사항

1. **비동기 전환 시**: 기존 동기 함수를 점진적으로 전환
2. **캐싱 시**: 메모리 사용량 모니터링 필요
3. **병렬 처리 시**: 동시성 제한 (max_workers) 설정
4. **모델 싱글톤 시**: 메모리 누수 방지 (정기적 정리)

## 🎯 성능 측정 방법

```python
# 성능 측정 데코레이터
import time
from functools import wraps

def measure_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        
        logger.info(f"{func.__name__} 실행 시간: {duration:.2f}초")
        return result
    return wrapper
```

## 📚 참고 자료

- LangGraph 공식 문서: 비동기 처리 가이드
- Python asyncio 모범 사례
- Sentence Transformers 배치 처리 가이드

