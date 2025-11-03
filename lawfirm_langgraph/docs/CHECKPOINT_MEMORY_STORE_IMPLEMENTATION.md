# LangGraph Memory Store 구현 완료

## 개요

LangGraph에서 Memory Store (MemorySaver) 사용을 위한 체크포인터 시스템을 구현했습니다.

## 주요 변경 사항

### 1. 설정 파일 개선 (`lawfirm_langgraph/config/langgraph_config.py`)

- **체크포인터 활성화 옵션 추가**: `enable_checkpoint` 필드 추가
- **MemorySaver 기본값 설정**: `checkpoint_storage`의 기본값을 `MEMORY`로 변경
- **저장소 타입 확장**: `MEMORY`, `DISABLED` 옵션 추가

```python
# 체크포인트 설정
enable_checkpoint: bool = True  # 체크포인터 활성화 여부
checkpoint_storage: CheckpointStorageType = CheckpointStorageType.MEMORY  # 기본값: MemorySaver
checkpoint_db_path: str = "./data/checkpoints/langgraph.db"
checkpoint_ttl: int = 3600  # 1시간
```

### 2. CheckpointManager 개선 (`source/agents/checkpoint_manager.py`)

- **유연한 초기화**: 저장소 타입에 따라 MemorySaver 또는 SqliteSaver 선택
- **MemorySaver 기본값**: SqliteSaver 초기화 실패 시 자동으로 MemorySaver로 폴백
- **새로운 메서드 추가**:
  - `get_checkpointer()`: LangGraph compile에서 사용할 체크포인터 반환
  - `is_enabled()`: 체크포인터 활성화 여부 확인

```python
# 사용 예시
checkpoint_manager = CheckpointManager(
    storage_type="memory",  # 또는 "sqlite", "disabled"
    db_path="./data/checkpoints/langgraph.db"  # sqlite 사용 시만 필요
)
```

### 3. WorkflowService 수정 (`lawfirm_langgraph/langgraph_core/services/workflow_service.py`)

- **체크포인터 자동 초기화**: 설정에 따라 CheckpointManager 자동 생성
- **컴파일 시 체크포인터 전달**: `graph.compile(checkpointer=...)`에 실제 체크포인터 전달
- **세션별 thread_id 설정**: `process_query`에서 체크포인터 활성화 시 `thread_id` 설정

## 사용 방법

### 환경 변수 설정

`.env` 파일에 다음 설정 추가:

```env
# 체크포인터 활성화 (기본값: true)
ENABLE_CHECKPOINT=true

# 저장소 타입: memory, sqlite, disabled
CHECKPOINT_STORAGE=memory

# SQLite 사용 시 데이터베이스 경로
LANGGRAPH_CHECKPOINT_DB=./data/checkpoints/langgraph.db

# 체크포인트 TTL (초 단위, 기본값: 3600)
CHECKPOINT_TTL=3600
```

### 저장소 타입별 특징

#### 1. MemorySaver (기본값, 개발용)
- **장점**: 
  - 빠른 초기화 및 실행
  - 별도의 데이터베이스 설정 불필요
  - 개발/테스트 환경에 적합
  
- **단점**:
  - 애플리케이션 재시작 시 상태 손실
  - 메모리 사용량 증가 가능

- **사용 예시**:
```python
# 환경 변수 또는 코드에서
CHECKPOINT_STORAGE=memory
```

#### 2. SqliteSaver (프로덕션용)
- **장점**:
  - 영구 저장 (애플리케이션 재시작 후에도 상태 유지)
  - 대용량 세션 관리 가능
  - 프로덕션 환경에 적합

- **단점**:
  - 데이터베이스 파일 관리 필요
  - 초기 설정 복잡도 증가

- **사용 예시**:
```python
# 환경 변수 또는 코드에서
CHECKPOINT_STORAGE=sqlite
LANGGRAPH_CHECKPOINT_DB=./data/checkpoints/langgraph.db
```

#### 3. Disabled (체크포인터 비활성화)
- **사용 예시**:
```python
# 환경 변수 또는 코드에서
CHECKPOINT_STORAGE=disabled
# 또는
ENABLE_CHECKPOINT=false
```

## 코드 변경 요약

### Before (체크포인터 미사용)

```python
# workflow_service.py
self.checkpoint_manager = None  # Checkpoint manager is disabled
self.app = self.legal_workflow.graph.compile(
    checkpointer=None,
    ...
)

# process_query
config = {}  # 체크포인터 비활성화
```

### After (체크포인터 사용)

```python
# workflow_service.py
if self.config.enable_checkpoint:
    self.checkpoint_manager = CheckpointManager(
        storage_type=self.config.checkpoint_storage.value,
        db_path=self.config.checkpoint_db_path if storage_type == "sqlite" else None
    )

checkpointer = self.checkpoint_manager.get_checkpointer() if self.checkpoint_manager.is_enabled() else None
self.app = self.legal_workflow.graph.compile(
    checkpointer=checkpointer,
    ...
)

# process_query
if enable_checkpoint and self.checkpoint_manager.is_enabled():
    config = {"configurable": {"thread_id": session_id}}
```

## 테스트

### 기본 테스트

```python
from lawfirm_langgraph.langgraph_core.services.workflow_service import LangGraphWorkflowService
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig

# MemorySaver 사용
config = LangGraphConfig.from_env()
config.checkpoint_storage = CheckpointStorageType.MEMORY
config.enable_checkpoint = True

service = LangGraphWorkflowService(config)

# 쿼리 처리 (체크포인터 활성화)
result = await service.process_query(
    query="법률 질문",
    session_id="test-session-1",
    enable_checkpoint=True
)
```

### 세션 재개 테스트

같은 `session_id`로 두 번째 쿼리를 보내면 이전 상태를 기반으로 실행됩니다:

```python
# 첫 번째 쿼리
result1 = await service.process_query(
    query="첫 번째 질문",
    session_id="my-session",
    enable_checkpoint=True
)

# 두 번째 쿼리 (같은 세션)
result2 = await service.process_query(
    query="두 번째 질문",
    session_id="my-session",  # 같은 세션 ID
    enable_checkpoint=True
)
```

## 주의사항

1. **MemorySaver 사용 시**: 애플리케이션 재시작 시 모든 체크포인트가 손실됩니다.
2. **SqliteSaver 사용 시**: 데이터베이스 파일 경로가 올바른지 확인하세요.
3. **세션 관리**: 각 세션은 고유한 `thread_id`를 가져야 합니다.
4. **성능**: 체크포인터 활성화 시 약간의 성능 오버헤드가 발생할 수 있습니다.

## 문제 해결

### 체크포인터가 초기화되지 않는 경우

1. `CheckpointManager` 클래스가 import 가능한지 확인
2. `langgraph.checkpoint.memory` 모듈이 설치되어 있는지 확인:
   ```bash
   pip install langgraph
   ```
3. 로그에서 초기화 오류 메시지 확인

### SqliteSaver 초기화 실패 시

- 자동으로 MemorySaver로 폴백됩니다
- 로그에서 경고 메시지 확인:
  ```
  WARNING: Failed to initialize SqliteSaver: ..., falling back to MemorySaver
  ```

## 향후 개선 사항

- [ ] PostgresSaver 지원
- [ ] RedisSaver 지원
- [ ] 체크포인트 TTL 자동 정리 기능
- [ ] 체크포인트 메타데이터 조회 API
- [ ] 체크포인트 수동 삭제 기능

## 관련 파일

- `lawfirm_langgraph/config/langgraph_config.py`: 설정 관리
- `source/agents/checkpoint_manager.py`: 체크포인터 관리자
- `lawfirm_langgraph/langgraph_core/services/workflow_service.py`: 워크플로우 서비스

## 참고 문서

- [LangGraph Checkpointing Guide](https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointing)
- [MemorySaver Documentation](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.memory.MemorySaver)
- [SqliteSaver Documentation](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.sqlite.SqliteSaver)

