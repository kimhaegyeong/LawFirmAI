# Langfuse 설정 가이드

## 개요

Langfuse는 LLM 애플리케이션의 관찰성(Observability)을 위한 오픈소스 플랫폼입니다. LawFirmAI 프로젝트에서 LangChain 기반 RAG 시스템의 성능을 모니터링하고 디버깅하기 위해 Langfuse를 통합했습니다.

## Langfuse란?

Langfuse는 다음과 같은 기능을 제공합니다:
- **LLM 호출 추적**: 모든 LLM 호출의 실시간 모니터링
- **성능 메트릭**: 응답 시간, 토큰 사용량, 비용 분석
- **오류 추적**: 실패한 요청의 상세 분석
- **A/B 테스트**: 다양한 프롬프트 및 모델 비교
- **사용자 피드백**: 답변 품질 평가 및 개선점 도출

## 설치 및 설정

### 1. Langfuse 패키지 설치

```bash
# Langfuse 및 관련 패키지 설치
pip install langfuse>=2.0.0
pip install langchain>=0.1.0
pip install langchain-openai>=0.0.5
pip install langchain-community>=0.0.10

# 참고: sqlite3는 Python 내장 모듈이므로 별도 설치 불필요
```

### 2. 환경 변수 설정

`.env` 파일에 다음 환경 변수를 추가합니다:

```bash
# Langfuse 기본 설정
LANGFUSE_ENABLED=true
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_DEBUG=false
LANGFUSE_FLUSH_INTERVAL=5

# 선택적 설정
LANGFUSE_RELEASE=production
LANGFUSE_ENVIRONMENT=production
```

### 3. Langfuse 계정 생성

1. [Langfuse 웹사이트](https://langfuse.com/)에 접속
2. 계정 생성 및 로그인
3. 프로젝트 생성
4. API 키 생성 (Secret Key, Public Key)

### 4. 설정 검증

```python
from source.utils.langchain_config import LangChainConfig
from source.services.langfuse_client import LangfuseClient

# 설정 로드
config = LangChainConfig.from_env()

# Langfuse 클라이언트 초기화
client = LangfuseClient(config)

# 연결 확인
if client.is_enabled():
    print("✅ Langfuse 연결 성공")
    trace_id = client.get_current_trace_id()
    print(f"현재 추적 ID: {trace_id}")
else:
    print("❌ Langfuse 연결 실패")
```

## 사용 방법

### 1. 기본 사용법

```python
from source.services.langchain_rag_service import LangChainRAGService
from source.utils.langchain_config import LangChainConfig

# 설정 로드
config = LangChainConfig.from_env()

# RAG 서비스 초기화 (Langfuse 자동 통합)
rag_service = LangChainRAGService(config)

# 쿼리 처리 (자동으로 Langfuse 추적)
result = rag_service.process_query(
    query="계약서 검토 요청",
    session_id="user-session-1",
    template_type="legal_qa"
)

print(f"답변: {result.answer}")
print(f"신뢰도: {result.confidence}")
print(f"추적 ID: {result.trace_id}")
```

### 2. 수동 추적

```python
from source.services.langfuse_client import LangfuseClient, RAGMetrics
from datetime import datetime

# Langfuse 클라이언트 초기화
client = LangfuseClient(config)

# RAG 메트릭 생성
metrics = RAGMetrics(
    query="테스트 질문",
    response_time=1.5,
    retrieved_docs_count=3,
    context_length=1000,
    response_length=200,
    similarity_scores=[0.8, 0.7, 0.9],
    confidence_score=0.8,
    timestamp=datetime.now()
)

# RAG 쿼리 추적
trace_id = client.track_rag_query(
    query="테스트 질문",
    response="테스트 응답",
    metrics=metrics,
    sources=[{"title": "test_doc.txt", "similarity": 0.8}]
)

print(f"추적 ID: {trace_id}")
```

### 3. LLM 호출 추적

```python
# LLM 호출 추적
trace_id = client.track_llm_call(
    model="gpt-3.5-turbo",
    prompt="법률 질문 프롬프트",
    response="LLM 응답",
    tokens_used=150,
    response_time=2.5
)
```

### 4. 오류 추적

```python
try:
    # RAG 처리
    result = rag_service.process_query("테스트 질문")
except Exception as e:
    # 오류 추적
    client.track_error(
        error_type=type(e).__name__,
        error_message=str(e),
        context={"query": "테스트 질문", "session_id": "test-session"}
    )
    raise
```

## Langfuse 대시보드 사용법

### 1. 대시보드 접속

1. [Langfuse 대시보드](https://cloud.langfuse.com/)에 로그인
2. 프로젝트 선택
3. 대시보드 메뉴 탐색

### 2. 주요 기능

#### Traces (추적)
- **실시간 모니터링**: 모든 LLM 호출의 실시간 추적
- **상세 분석**: 각 호출의 입력, 출력, 메트릭 확인
- **오류 분석**: 실패한 요청의 상세 분석

#### Metrics (메트릭)
- **성능 지표**: 응답 시간, 토큰 사용량, 비용 분석
- **품질 지표**: 신뢰도, 정확도, 사용자 만족도
- **사용량 통계**: 일일/월간 사용량 추이

#### Prompts (프롬프트)
- **템플릿 관리**: 다양한 프롬프트 템플릿 관리
- **A/B 테스트**: 프롬프트 성능 비교
- **버전 관리**: 프롬프트 변경 이력 추적

#### Evaluations (평가)
- **자동 평가**: 시스템 기반 답변 품질 평가
- **사용자 평가**: 사용자 피드백 수집
- **성능 비교**: 모델 및 프롬프트 성능 비교

### 3. 알림 설정

```python
# Langfuse에서 알림 설정
# 1. 대시보드에서 Settings > Alerts 이동
# 2. 알림 조건 설정:
#    - 응답 시간 > 10초
#    - 오류율 > 5%
#    - 토큰 사용량 > 일일 한도
# 3. 알림 방법 선택 (이메일, Slack 등)
```

## 고급 설정

### 1. 커스텀 메트릭

```python
from langfuse import Langfuse

# 커스텀 메트릭 추가
langfuse = Langfuse(
    secret_key="your-secret-key",
    public_key="your-public-key"
)

# 법률 특화 메트릭
langfuse.score(
    name="legal_accuracy",
    value=0.95,
    trace_id="trace-123",
    comment="법률 정확도 점수"
)

langfuse.score(
    name="citation_quality",
    value=0.88,
    trace_id="trace-123",
    comment="인용 품질 점수"
)
```

### 2. 배치 처리 추적

```python
# 여러 쿼리 배치 처리
queries = [
    {"query": "계약서 검토", "context": "..."},
    {"query": "판례 검색", "context": "..."},
    {"query": "법령 해석", "context": "..."}
]

results = []
for i, query_data in enumerate(queries):
    # 각 쿼리별 추적
    trace_id = client.get_current_trace_id()
    
    result = rag_service.process_query(
        query=query_data["query"],
        context=query_data["context"]
    )
    
    results.append(result)
    
    # 배치 진행률 추적
    langfuse.score(
        name="batch_progress",
        value=(i + 1) / len(queries),
        trace_id=trace_id,
        comment=f"배치 처리 진행률: {i+1}/{len(queries)}"
    )
```

### 3. 사용자 세션 추적

```python
# 사용자별 세션 추적
user_id = "user-123"
session_id = f"{user_id}-session-{datetime.now().strftime('%Y%m%d%H%M%S')}"

# 세션 시작 추적
client.track_session_start(user_id, session_id)

# 쿼리 처리
result = rag_service.process_query(
    query="사용자 질문",
    session_id=session_id
)

# 세션 종료 추적
client.track_session_end(session_id, result.confidence)
```

## 문제 해결

### 1. 연결 문제

**문제**: Langfuse 연결 실패
```python
# 연결 상태 확인
client = LangfuseClient(config)
if not client.is_enabled():
    print("Langfuse 연결 실패")
    # 환경 변수 확인
    print(f"LANGFUSE_ENABLED: {config.langfuse_enabled}")
    print(f"LANGFUSE_SECRET_KEY: {'설정됨' if config.langfuse_secret_key else '미설정'}")
    print(f"LANGFUSE_PUBLIC_KEY: {'설정됨' if config.langfuse_public_key else '미설정'}")
```

**해결책**:
1. 환경 변수 확인
2. API 키 유효성 검증
3. 네트워크 연결 확인
4. Langfuse 서비스 상태 확인

### 2. 성능 문제

**문제**: 추적으로 인한 성능 저하
```python
# 비동기 추적 사용
config.langfuse_flush_interval = 10  # 10초마다 배치 전송
config.langfuse_debug = False  # 디버그 모드 비활성화

# 샘플링 설정 (선택적 추적)
config.langfuse_sampling_rate = 0.1  # 10%만 추적
```

### 3. 메모리 사용량 문제

**문제**: 메모리 사용량 증가
```python
# 정기적인 세션 정리
rag_service.cleanup_old_sessions(max_age_hours=24)

# 캐시 크기 제한
config.cache_ttl = 3600  # 1시간
config.max_sessions = 100  # 최대 100개 세션
```

## 모니터링 대시보드 예시

### 1. 주요 지표

```python
# 서비스 통계 조회
stats = rag_service.get_service_statistics()

print("=== RAG 서비스 통계 ===")
print(f"총 쿼리 수: {stats['rag_stats']['total_queries']}")
print(f"평균 응답 시간: {stats['rag_stats']['avg_response_time']:.2f}초")
print(f"평균 신뢰도: {stats['rag_stats']['avg_confidence']:.2f}")
print(f"Langfuse 활성화: {stats['langfuse_enabled']}")
```

### 2. 성능 모니터링

```python
# 성능 메트릭 수집
from source.services.langfuse_client import MetricsCollector

collector = MetricsCollector(client)
summary = collector.get_performance_summary()

print("=== 성능 요약 ===")
print(f"총 쿼리 수: {summary['total_queries']}")
print(f"평균 응답 시간: {summary['avg_response_time']:.2f}초")
print(f"평균 신뢰도: {summary['avg_confidence']:.2f}")
print(f"총 검색 문서 수: {summary['total_documents_retrieved']}")
```

## 보안 고려사항

### 1. API 키 보안

```bash
# 환경 변수로 API 키 관리
export LANGFUSE_SECRET_KEY="your-secret-key"
export LANGFUSE_PUBLIC_KEY="your-public-key"

# .env 파일 사용 (프로덕션에서는 제외)
echo "LANGFUSE_SECRET_KEY=your-secret-key" >> .env
echo "LANGFUSE_PUBLIC_KEY=your-public-key" >> .env
```

### 2. 데이터 민감성

```python
# 민감한 데이터 마스킹
def mask_sensitive_data(text):
    # 개인정보 마스킹
    import re
    text = re.sub(r'\d{3}-\d{4}-\d{4}', '***-****-****', text)  # 전화번호
    text = re.sub(r'\d{6}-\d{7}', '******-*******', text)  # 주민등록번호
    return text

# 추적 전 데이터 마스킹
masked_query = mask_sensitive_data(query)
client.track_rag_query(masked_query, response, metrics, sources)
```

## 결론

Langfuse를 통한 관찰성 구현으로 LawFirmAI의 RAG 시스템 성능을 효과적으로 모니터링하고 개선할 수 있습니다. 실시간 추적, 성능 메트릭, 오류 분석을 통해 시스템의 안정성과 사용자 만족도를 높일 수 있습니다.

자세한 내용은 [Langfuse 공식 문서](https://langfuse.com/docs)를 참조하시기 바랍니다.
