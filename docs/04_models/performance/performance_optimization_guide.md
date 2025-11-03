# 🚀 LawFirmAI 성능 최적화 가이드

## 📋 개요

이 가이드는 LawFirmAI 시스템의 성능을 최적화하는 방법과 최적화된 컴포넌트 사용법을 설명합니다.

## 🎯 최적화 목표

- **응답 시간 단축**: 10초 → 2초 이하
- **메모리 효율성**: 모델 재사용으로 메모리 절약
- **동시 처리**: 여러 요청 병렬 처리
- **캐싱**: 동일 질문에 대한 빠른 응답

## 🔧 최적화된 컴포넌트

### 1. OptimizedModelManager

#### 개요
AI 모델의 로딩과 관리를 최적화하는 싱글톤 패턴 기반 모델 관리자입니다.

#### 주요 기능
- **싱글톤 패턴**: 모델 인스턴스를 한 번만 생성하고 재사용
- **지연 로딩**: 필요 시에만 모델 로딩
- **Float16 양자화**: 메모리 사용량 50% 감소
- **미리 로딩**: 백그라운드에서 모델 미리 로딩

#### 사용법
```python
from source.services.optimized_model_manager import model_manager

# Sentence-BERT 모델 가져오기
model = model_manager.get_sentence_transformer(
    model_name="jhgan/ko-sroberta-multitask",
    enable_quantization=True,
    device="cpu"
)

# 성능 통계 확인
stats = model_manager.get_stats()
print(f"모델 로딩 횟수: {stats['model_loads']}")
print(f"캐시 히트율: {stats['cache_hit_rate']:.1%}")
```

#### 설정 옵션
```python
# 모델 미리 로딩
model_configs = {
    'jhgan/ko-sroberta-multitask': {
        'device': 'cpu',
        'enable_quantization': True
    }
}
model_manager.preload_models(model_configs)
```

### 2. OptimizedHybridSearchEngine

#### 개요
병렬 처리와 캐싱을 지원하는 최적화된 하이브리드 검색 엔진입니다.

#### 주요 기능
- **병렬 검색**: 정확 검색과 의미 검색을 동시 실행
- **결과 캐싱**: 검색 결과를 캐시하여 재사용
- **타임아웃 적용**: 검색 작업에 타임아웃 설정
- **점수 임계값**: 최소 점수 이하 결과 필터링

#### 사용법
```python
from source.services.optimized_hybrid_search_engine import OptimizedHybridSearchEngine, OptimizedSearchConfig

# 검색 설정
config = OptimizedSearchConfig(
    max_results_per_type=5,
    parallel_search=True,
    cache_enabled=True,
    timeout_seconds=3.0,
    min_score_threshold=0.3
)

# 검색 엔진 초기화
search_engine = OptimizedHybridSearchEngine(config)

# 검색 실행
results = await search_engine.search_with_question_type(
    query="계약서 검토 요청",
    question_type=QuestionType.CONTRACT_REVIEW,
    max_results=20
)

# 성능 통계 확인
stats = search_engine.get_stats()
print(f"총 검색 횟수: {stats['total_searches']}")
print(f"캐시 히트율: {stats['cache_hit_rate']:.1%}")
```

#### 설정 옵션
```python
# 검색 설정 조정
config = OptimizedSearchConfig(
    max_results_per_type=10,      # 각 검색 타입별 최대 결과 수
    parallel_search=True,         # 병렬 검색 사용
    cache_enabled=True,          # 캐싱 사용
    timeout_seconds=5.0,         # 검색 타임아웃
    min_score_threshold=0.5      # 최소 점수 임계값
)
```

### 3. IntegratedCacheSystem

#### 개요
질문 분류, 검색 결과, 답변 생성을 위한 다층 캐싱 시스템입니다.

#### 주요 기능
- **다층 캐싱**: 각 단계별 캐싱
- **LRU 캐시**: 최근 사용된 항목 우선 보관
- **영구 캐시**: 디스크 기반 장기 캐싱
- **TTL 관리**: 캐시 유효 시간 자동 관리

#### 사용법
```python
from source.services.integrated_cache_system import cache_system

# 질문 분류 결과 캐싱
classification = cache_system.get_question_classification("계약서 검토 요청")
if not classification:
    classification = classify_question("계약서 검토 요청")
    cache_system.put_question_classification("계약서 검토 요청", classification)

# 검색 결과 캐싱
search_results = cache_system.get_search_results("계약서 검토", "contract", 20)
if not search_results:
    search_results = perform_search("계약서 검토", 20)
    cache_system.put_search_results("계약서 검토", "contract", 20, search_results)

# 답변 캐싱
answer = cache_system.get_answer("계약서 검토", "contract", "context_hash")
if not answer:
    answer = generate_answer("계약서 검토", context)
    cache_system.put_answer("계약서 검토", "contract", "context_hash", answer)

# 성능 통계 확인
stats = cache_system.get_stats()
print(f"질문 분류 캐시 히트율: {stats['classification_cache']['hit_rate']:.1%}")
print(f"검색 캐시 히트율: {stats['search_cache']['hit_rate']:.1%}")
```

#### 캐시 설정
```python
# 캐시 크기 조정
cache_system.question_classification_cache.max_size = 1000
cache_system.search_results_cache.max_size = 2000
cache_system.answer_generation_cache.max_size = 500

# 캐시 정리
cache_system.clear_all()
```

### 4. OptimizedChatService

#### 개요
모든 최적화 기술을 통합한 최적화된 채팅 서비스입니다.

#### 주요 기능
- **통합 최적화**: 모든 최적화 기술 통합
- **성능 모니터링**: 실시간 성능 통계 수집
- **동적 설정**: 런타임에 성능 설정 조정
- **에러 처리**: 강화된 에러 처리 및 복구

#### 사용법
```python
from source.services.optimized_chat_service import OptimizedChatService
from source.utils.config import Config

# 설정 초기화
config = Config()

# 최적화된 서비스 초기화
service = OptimizedChatService(config)

# 질문 처리
result = await service.process_message("계약서 검토 요청")
print(f"응답: {result['response']}")
print(f"처리 시간: {result['processing_time']:.2f}초")
print(f"캐시 사용: {result['cached']}")
```

#### 성능 설정 조정
```python
# 성능 설정 조정
performance_config = {
    'enable_caching': True,
    'enable_parallel_search': True,
    'enable_model_preloading': True,
    'max_concurrent_requests': 10,
    'response_timeout': 15.0
}

service.optimize_performance(performance_config)
```

#### 상태 확인
```python
# 서비스 상태 확인
status = service.get_service_status()
print(f"전체 상태: {status['status']}")
print(f"컴포넌트 상태: {status['components']}")

# 성능 통계 확인
stats = service.get_performance_stats()
print(f"총 요청 수: {stats['total_requests']}")
print(f"평균 응답 시간: {stats['avg_response_time']:.2f}초")
print(f"캐시 히트율: {stats['cache_hit_rate']:.1%}")
print(f"에러율: {stats['error_rate']:.1%}")
```

## 📊 성능 모니터링

### 실시간 모니터링

#### 성능 지표 수집
```python
# 성능 통계 수집
stats = service.get_performance_stats()

# 주요 지표
print(f"평균 응답 시간: {stats['avg_response_time']:.2f}초")
print(f"총 요청 수: {stats['total_requests']}")
print(f"캐시 히트율: {stats['cache_hit_rate']:.1%}")
print(f"에러율: {stats['error_rate']:.1%}")

# 캐시 통계
cache_stats = stats['cache_stats']
print(f"질문 분류 캐시: {cache_stats['classification_cache']['hit_rate']:.1%}")
print(f"검색 캐시: {cache_stats['search_cache']['hit_rate']:.1%}")
print(f"답변 캐시: {cache_stats['answer_cache']['hit_rate']:.1%}")

# 모델 통계
model_stats = stats['model_stats']
print(f"모델 로딩 횟수: {model_stats['model_loads']}")
print(f"평균 로딩 시간: {model_stats['avg_load_time']:.2f}초")

# 검색 통계
search_stats = stats['search_stats']
print(f"총 검색 횟수: {search_stats['total_searches']}")
print(f"병렬 검색 비율: {search_stats['parallel_search_rate']:.1%}")
```

#### 모니터링 대시보드
```python
# 주기적 모니터링
import time
import asyncio

async def monitor_performance():
    while True:
        stats = service.get_performance_stats()
        
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"응답시간: {stats['avg_response_time']:.2f}초, "
              f"캐시히트율: {stats['cache_hit_rate']:.1%}, "
              f"에러율: {stats['error_rate']:.1%}")
        
        await asyncio.sleep(60)  # 1분마다 모니터링

# 모니터링 시작
asyncio.create_task(monitor_performance())
```

### 성능 분석

#### 응답 시간 분석
```python
# 응답 시간 분포 분석
response_times = []
for i in range(100):
    start_time = time.time()
    result = await service.process_message(f"테스트 질문 {i}")
    response_time = time.time() - start_time
    response_times.append(response_time)

# 통계 계산
import statistics
print(f"평균 응답 시간: {statistics.mean(response_times):.2f}초")
print(f"중간값: {statistics.median(response_times):.2f}초")
print(f"표준편차: {statistics.stdev(response_times):.2f}초")
print(f"최대값: {max(response_times):.2f}초")
print(f"최소값: {min(response_times):.2f}초")
```

#### 캐시 효과 분석
```python
# 캐시 효과 측정
test_query = "계약서 검토 요청"

# 첫 번째 질문 (캐시 없음)
start_time = time.time()
result1 = await service.process_message(test_query)
first_time = time.time() - start_time

# 두 번째 질문 (캐시 있음)
start_time = time.time()
result2 = await service.process_message(test_query)
second_time = time.time() - start_time

print(f"첫 번째 질문: {first_time:.2f}초")
print(f"두 번째 질문: {second_time:.2f}초")
print(f"캐시 효과: {first_time / second_time:.1f}배 빠름")
```

## 🔧 성능 튜닝

### 메모리 최적화

#### 모델 메모리 관리
```python
# 모델 메모리 사용량 모니터링
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"메모리 사용량: {memory_info.rss / 1024 / 1024:.1f}MB")

# 모델 로딩 전후 메모리 비교
monitor_memory()  # 로딩 전
model = model_manager.get_sentence_transformer("jhgan/ko-sroberta-multitask")
monitor_memory()  # 로딩 후
```

#### 캐시 메모리 관리
```python
# 캐시 크기 조정
cache_system.question_classification_cache.max_size = 500
cache_system.search_results_cache.max_size = 1000
cache_system.answer_generation_cache.max_size = 200

# 메모리 사용량에 따른 캐시 정리
def cleanup_cache_if_needed():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    if memory_info.rss > 1024 * 1024 * 1024:  # 1GB 이상
        cache_system.clear_all()
        print("메모리 사용량이 높아 캐시를 정리했습니다.")
```

### 검색 성능 튜닝

#### 검색 파라미터 조정
```python
# 검색 성능 튜닝
config = OptimizedSearchConfig(
    max_results_per_type=3,       # 결과 수 감소로 속도 향상
    parallel_search=True,        # 병렬 검색 활성화
    cache_enabled=True,         # 캐싱 활성화
    timeout_seconds=2.0,        # 타임아웃 단축
    min_score_threshold=0.4     # 임계값 상향 조정
)
```

#### 검색 결과 품질 향상
```python
# 검색 결과 품질 향상
def improve_search_quality(query, results):
    # 점수 기준 정렬
    results.sort(key=lambda x: x.score, reverse=True)
    
    # 중복 제거
    seen = set()
    unique_results = []
    for result in results:
        if result.content not in seen:
            seen.add(result.content)
            unique_results.append(result)
    
    return unique_results[:10]  # 상위 10개만 반환
```

### 캐시 최적화

#### 캐시 전략 조정
```python
# 캐시 TTL 조정
cache_system.put_question_classification(
    query, classification, ttl=3600  # 1시간
)
cache_system.put_search_results(
    query, question_type, max_results, results, ttl=1800  # 30분
)
cache_system.put_answer(
    query, question_type, context_hash, answer, ttl=7200  # 2시간
)
```

#### 캐시 히트율 향상
```python
# 질문 정규화로 캐시 히트율 향상
def normalize_query(query):
    # 소문자 변환
    normalized = query.lower().strip()
    
    # 불필요한 문자 제거
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # 공백 정규화
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized

# 정규화된 질문으로 캐시 확인
normalized_query = normalize_query(query)
cached_result = cache_system.get_question_classification(normalized_query)
```

## 🧪 성능 테스트

### 기본 성능 테스트

#### 테스트 스크립트
```python
import asyncio
import time
from source.services.optimized_chat_service import OptimizedChatService
from source.utils.config import Config

async def performance_test():
    config = Config()
    service = OptimizedChatService(config)
    
    test_queries = [
        "안녕하세요",
        "계약서 검토 요청",
        "민법 제750조의 내용이 무엇인가요?",
        "손해배상 관련 판례를 찾아주세요",
        "이혼 절차는 어떻게 진행하나요?"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries):
        print(f"[{i+1}/{len(test_queries)}] Processing: {query[:50]}...")
        start_time = time.time()
        result = await service.process_message(query)
        processing_time = time.time() - start_time
        
        results.append({
            "query": query,
            "processing_time": processing_time,
            "success": bool(result.get("response")),
            "cached": result.get("cached", False)
        })
        
        print(f"완료: {processing_time:.2f}초 (캐시: {result.get('cached', False)})")
    
    # 결과 분석
    avg_time = sum(r["processing_time"] for r in results) / len(results)
    success_rate = sum(1 for r in results if r["success"]) / len(results)
    cache_hit_rate = sum(1 for r in results if r["cached"]) / len(results)
    
    print(f"\n=== 성능 테스트 결과 ===")
    print(f"평균 응답 시간: {avg_time:.2f}초")
    print(f"성공률: {success_rate:.1%}")
    print(f"캐시 히트율: {cache_hit_rate:.1%}")

# 테스트 실행
asyncio.run(performance_test())
```

### 부하 테스트

#### 동시 요청 테스트
```python
async def load_test():
    config = Config()
    service = OptimizedChatService(config)
    
    test_query = "계약서 검토 요청"
    concurrent_requests = 10
    
    # 동시 요청 생성
    tasks = []
    for i in range(concurrent_requests):
        task = service.process_message(f"{test_query} {i}")
        tasks.append(task)
    
    # 동시 실행
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    # 결과 분석
    successful_results = [r for r in results if isinstance(r, dict)]
    failed_count = len(results) - len(successful_results)
    
    print(f"=== 부하 테스트 결과 ===")
    print(f"동시 요청 수: {concurrent_requests}")
    print(f"총 처리 시간: {total_time:.2f}초")
    print(f"성공한 요청: {len(successful_results)}")
    print(f"실패한 요청: {failed_count}")
    print(f"처리량: {len(successful_results) / total_time:.1f} 요청/초")

# 부하 테스트 실행
asyncio.run(load_test())
```

### 캐시 효과 테스트

#### 캐시 성능 측정
```python
async def cache_performance_test():
    config = Config()
    service = OptimizedChatService(config)
    
    test_query = "계약서 검토 요청"
    iterations = 5
    
    first_run_time = None
    cached_times = []
    
    for i in range(iterations):
        start_time = time.time()
        result = await service.process_message(test_query)
        processing_time = time.time() - start_time
        
        if i == 0:
            first_run_time = processing_time
        else:
            cached_times.append(processing_time)
        
        print(f"Iteration {i+1}: {processing_time:.2f}초 (cached: {result.get('cached', False)})")
    
    # 캐시 효과 계산
    avg_cached_time = sum(cached_times) / len(cached_times)
    speedup_factor = first_run_time / avg_cached_time
    
    print(f"\n=== 캐시 성능 테스트 결과 ===")
    print(f"첫 번째 실행: {first_run_time:.2f}초")
    print(f"평균 캐시 시간: {avg_cached_time:.2f}초")
    print(f"속도 향상: {speedup_factor:.1f}배")

# 캐시 테스트 실행
asyncio.run(cache_performance_test())
```

## 🚨 문제 해결

### 일반적인 문제

#### 메모리 부족
```python
# 메모리 부족 시 해결 방법
def handle_memory_shortage():
    # 캐시 정리
    cache_system.clear_all()
    
    # 모델 캐시 정리
    model_manager.clear_cache()
    
    # 가비지 컬렉션 강제 실행
    import gc
    gc.collect()
    
    print("메모리 정리 완료")
```

#### 응답 시간 지연
```python
# 응답 시간 지연 시 해결 방법
def handle_slow_response():
    # 캐시 설정 확인
    if not cache_system.question_classification_cache:
        print("캐시가 비활성화되어 있습니다.")
    
    # 병렬 검색 확인
    if not search_engine.config.parallel_search:
        print("병렬 검색이 비활성화되어 있습니다.")
    
    # 모델 로딩 상태 확인
    stats = model_manager.get_stats()
    if stats['model_loads'] > 1:
        print("모델이 여러 번 로딩되고 있습니다.")
```

#### 캐시 히트율 낮음
```python
# 캐시 히트율 향상 방법
def improve_cache_hit_rate():
    # 질문 정규화 강화
    def normalize_query_advanced(query):
        # 더 정교한 정규화
        normalized = query.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # 법률 용어 표준화
        legal_terms = {
            '계약서': '계약',
            '계약': '계약서',
            '계약 해지': '계약해지',
            '계약해지': '계약 해지'
        }
        
        for term, standard in legal_terms.items():
            normalized = normalized.replace(term, standard)
        
        return normalized
    
    # 캐시 TTL 조정
    cache_system.put_question_classification(
        query, classification, ttl=7200  # 2시간으로 연장
    )
```

### 성능 모니터링

#### 실시간 모니터링
```python
# 실시간 성능 모니터링
async def real_time_monitoring():
    while True:
        stats = service.get_performance_stats()
        
        # 임계값 확인
        if stats['avg_response_time'] > 5.0:
            print("⚠️ 응답 시간이 5초를 초과했습니다.")
        
        if stats['cache_hit_rate'] < 0.3:
            print("⚠️ 캐시 히트율이 30% 미만입니다.")
        
        if stats['error_rate'] > 0.1:
            print("⚠️ 에러율이 10%를 초과했습니다.")
        
        await asyncio.sleep(60)  # 1분마다 확인
```

## 📚 추가 자료

### 관련 문서
- [성능 최적화 완료 보고서](performance_optimization_report.md)
- [API 문서](../07_api/)
- [개발 가이드](../10_technical_reference/development_rules.md)

### 외부 자료
- [Python asyncio 공식 문서](https://docs.python.org/3/library/asyncio.html)
- [FAISS 공식 문서](https://faiss.ai/)
- [Sentence-BERT GitHub](https://github.com/UKPLab/sentence-transformers)

---

**작성일**: 2025년 1월 10일  
**작성자**: LawFirmAI 개발팀  
**문서 버전**: 1.0
