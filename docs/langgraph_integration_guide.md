# LangGraph 통합 가이드

## 개요

LawFirmAI에 LangGraph 워크플로우 관리 기능이 추가되었습니다. 이 기능을 통해 복잡한 법률 질문 처리 워크플로우를 상태 기반으로 관리하고, 체크포인트를 통한 세션 지속성을 제공합니다.

## 주요 기능

- **상태 기반 워크플로우 관리**: 질문 분류 → 문서 검색 → 컨텍스트 분석 → 답변 생성 → 응답 포맷팅
- **자동 체크포인트 저장**: SQLite 기반으로 워크플로우 상태 자동 저장
- **세션 기반 대화 이력 관리**: 사용자 세션별 대화 컨텍스트 유지
- **Ollama 로컬 LLM 통합**: 로컬 환경에서 실행 가능한 LLM 사용
- **기존 시스템과의 하위 호환성**: 기존 기능에 영향 없이 점진적 통합

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install langgraph>=0.2.0
pip install langgraph-checkpoint>=0.1.0
pip install langgraph-checkpoint-sqlite>=0.0.1
```

### 2. 환경 변수 설정

`.env` 파일에 다음 설정을 추가하세요:

```bash
# LangGraph 설정
LANGGRAPH_ENABLED=true
LANGGRAPH_CHECKPOINT_DB=./data/checkpoints/langgraph.db

# Ollama 설정
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# LLM 제공자 (local, openai, google)
LLM_PROVIDER=local

# LangGraph 활성화
USE_LANGGRAPH=true
```

### 3. Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치 (Windows)
# https://ollama.ai/download

# 모델 다운로드
ollama pull qwen2.5:7b

# Ollama 서버 시작
ollama serve
```

## 사용 방법

### 1. 기본 사용법

```python
from source.services.chat_service import ChatService
from source.utils.config import Config

# ChatService 초기화 (자동으로 LangGraph 사용)
config = Config()
chat_service = ChatService(config)

# 질문 처리
result = await chat_service.process_message("계약서 작성 시 주의사항은?")
print(result["response"])
```

### 2. 세션 기반 대화

```python
# 세션 ID와 함께 질문 처리
session_id = "user-session-123"
result = await chat_service.process_message(
    "이혼 절차는 어떻게 되나요?", 
    session_id=session_id
)

# 같은 세션으로 후속 질문
result2 = await chat_service.process_message(
    "위자료는 어떻게 되나요?", 
    session_id=session_id
)
```

### 3. 워크플로우 서비스 직접 사용

```python
from source.services.langgraph.workflow_service import LangGraphWorkflowService
from source.utils.langgraph_config import LangGraphConfig

# 설정 및 서비스 초기화
config = LangGraphConfig.from_env()
service = LangGraphWorkflowService(config)

# 질문 처리
result = await service.process_query("손해배상 관련 판례를 찾아주세요")

# 세션 재개
result2 = await service.resume_session(result["session_id"], "관련 법령도 알려주세요")
```

## 워크플로우 구조

### 노드 구성

1. **classify_query**: 질문 유형 분류
2. **search_documents**: 관련 문서 검색
3. **analyze_context**: 컨텍스트 분석 및 법률 참조 추출
4. **generate_answer**: Ollama를 사용한 답변 생성
5. **format_response**: 응답 포맷팅 및 소스 정리

### 상태 관리

```python
class LegalWorkflowState(TypedDict):
    query: str                    # 사용자 질문
    session_id: str              # 세션 ID
    query_type: str              # 질문 유형
    confidence: float            # 분류 신뢰도
    retrieved_docs: List[Dict]   # 검색된 문서
    legal_references: List[str]   # 법률 참조
    answer: str                  # 최종 답변
    sources: List[str]          # 소스 목록
    processing_steps: List[str]   # 처리 단계
    errors: List[str]           # 오류 목록
```

## 체크포인트 관리

### 자동 저장

워크플로우 실행 중 각 노드의 상태가 자동으로 SQLite 데이터베이스에 저장됩니다.

### 수동 관리

```python
# 체크포인트 관리자 직접 사용
from source.services.langgraph.checkpoint_manager import CheckpointManager

manager = CheckpointManager("./data/checkpoints/langgraph.db")

# 세션 체크포인트 목록 조회
checkpoints = manager.list_checkpoints("session-id")

# 오래된 체크포인트 정리 (24시간 이상)
cleaned = manager.cleanup_old_checkpoints(24)
```

## 설정 옵션

### LangGraphConfig 클래스

```python
@dataclass
class LangGraphConfig:
    # 체크포인트 설정
    checkpoint_storage: CheckpointStorageType = CheckpointStorageType.SQLITE
    checkpoint_db_path: str = "./data/checkpoints/langgraph.db"
    checkpoint_ttl: int = 3600
    
    # 워크플로우 설정
    max_iterations: int = 10
    recursion_limit: int = 25
    enable_streaming: bool = True
    
    # LLM 설정
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_timeout: int = 120
```

## 모니터링 및 디버깅

### 서비스 상태 확인

```python
# ChatService 상태 확인
status = chat_service.get_service_status()
print(f"LangGraph 활성화: {status['langgraph_enabled']}")
print(f"워크플로우 서비스 사용 가능: {status['langgraph_service_available']}")

# 워크플로우 서비스 상태 확인
workflow_status = service.get_service_status()
print(f"서비스 상태: {workflow_status['status']}")
print(f"데이터베이스 정보: {workflow_status['database_info']}")
```

### 테스트 실행

```python
# 서비스 테스트
test_result = await chat_service.test_service("테스트 질문")
print(f"테스트 통과: {test_result['test_passed']}")

# 워크플로우 테스트
workflow_test = await service.test_workflow("계약서 검토 요청")
print(f"워크플로우 테스트 통과: {workflow_test['test_passed']}")
```

## 문제 해결

### 일반적인 문제

1. **LangGraph 모듈을 찾을 수 없음**
   ```bash
   pip install langgraph langgraph-checkpoint langgraph-checkpoint-sqlite
   ```

2. **Ollama 연결 실패**
   ```bash
   # Ollama 서버 상태 확인
   curl http://localhost:11434/api/tags
   
   # 모델 확인
   ollama list
   ```

3. **체크포인트 데이터베이스 오류**
   ```bash
   # 데이터베이스 디렉토리 생성
   mkdir -p ./data/checkpoints
   
   # 권한 확인
   ls -la ./data/checkpoints/
   ```

### 로그 확인

```python
import logging

# 로그 레벨 설정
logging.basicConfig(level=logging.DEBUG)

# LangGraph 관련 로그만 확인
logger = logging.getLogger("source.services.langgraph")
logger.setLevel(logging.DEBUG)
```

## 성능 최적화

### 메모리 관리

```python
# 메모리 사용량 모니터링
import psutil

def monitor_memory():
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        print(f"높은 메모리 사용량: {memory_percent}%")
```

### 체크포인트 정리

```python
# 정기적인 체크포인트 정리
import schedule
import time

def cleanup_checkpoints():
    manager = CheckpointManager("./data/checkpoints/langgraph.db")
    cleaned = manager.cleanup_old_checkpoints(24)  # 24시간 이상된 것 삭제
    print(f"정리된 체크포인트: {cleaned}개")

# 매일 자정에 실행
schedule.every().day.at("00:00").do(cleanup_checkpoints)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 향후 확장 계획

### Phase 2: 멀티 에이전트 시스템

- 법률 연구 에이전트
- 판례 분석 에이전트
- 계약서 검토 에이전트
- 답변 종합 에이전트

### Phase 3: 고급 기능

- OpenAI/Gemini 통합
- PostgreSQL 체크포인트 저장소
- 실시간 스트리밍 응답
- 동적 워크플로우 생성

### Phase 4: 배포 최적화

- Docker 컨테이너 지원
- 클라우드 배포 최적화
- 자동 스케일링
- 모니터링 대시보드

## API 참조

### LangGraphWorkflowService

#### 메서드

- `process_query(query, session_id, enable_checkpoint)`: 질문 처리
- `resume_session(session_id, query)`: 세션 재개
- `get_session_info(session_id)`: 세션 정보 조회
- `cleanup_old_sessions(ttl_hours)`: 오래된 세션 정리
- `get_service_status()`: 서비스 상태 조회
- `test_workflow(test_query)`: 워크플로우 테스트

### CheckpointManager

#### 메서드

- `save_checkpoint(thread_id, state)`: 체크포인트 저장
- `load_checkpoint(thread_id)`: 체크포인트 로드
- `list_checkpoints(thread_id)`: 체크포인트 목록 조회
- `delete_checkpoint(thread_id, checkpoint_id)`: 체크포인트 삭제
- `cleanup_old_checkpoints(ttl_hours)`: 오래된 체크포인트 정리
- `get_database_info()`: 데이터베이스 정보 조회

## 예제 코드

### 완전한 예제

```python
import asyncio
from source.services.chat_service import ChatService
from source.utils.config import Config

async def main():
    # 설정 및 서비스 초기화
    config = Config()
    chat_service = ChatService(config)
    
    # 질문 처리
    questions = [
        "계약서 작성 시 주의사항은?",
        "이혼 절차는 어떻게 되나요?",
        "손해배상 관련 판례를 찾아주세요"
    ]
    
    session_id = "demo-session"
    
    for i, question in enumerate(questions, 1):
        print(f"\n=== 질문 {i}: {question} ===")
        
        result = await chat_service.process_message(question, session_id=session_id)
        
        print(f"답변: {result['response']}")
        print(f"신뢰도: {result['confidence']}")
        print(f"처리 시간: {result['processing_time']:.2f}초")
        print(f"소스: {result['sources']}")
        
        if 'legal_references' in result:
            print(f"법률 참조: {result['legal_references']}")
        
        if 'processing_steps' in result:
            print(f"처리 단계: {result['processing_steps']}")
    
    # 서비스 상태 확인
    status = chat_service.get_service_status()
    print(f"\n=== 서비스 상태 ===")
    print(f"LangGraph 활성화: {status['langgraph_enabled']}")
    print(f"워크플로우 서비스 사용 가능: {status['langgraph_service_available']}")

if __name__ == "__main__":
    asyncio.run(main())
```

이 가이드를 통해 LangGraph 통합 기능을 효과적으로 활용할 수 있습니다. 추가 질문이나 문제가 있으시면 프로젝트 이슈를 통해 문의해주세요.
