# 스트리밍 기능 가이드

## 개요

LawFirmAI는 LangGraph 기반의 실시간 스트리밍 기능을 제공합니다. LLM 응답을 토큰 단위로 실시간 전달하여 사용자 경험을 향상시킵니다.

## 아키텍처

### 스트리밍 흐름

```
Client Request (POST /chat/stream)
    ↓
ChatService.stream_final_answer()
    ↓
StreamingCallbackHandler 생성 (asyncio.Queue)
    ↓
LangGraphWorkflowService.app.astream_events()
    ↓
generate_answer_stream 노드 실행
    ↓
LLM.stream() → on_llm_stream 이벤트
    ↓
StreamingCallbackHandler.on_llm_stream()
    ↓
asyncio.Queue에 청크 저장
    ↓
ChatService에서 큐 모니터링 및 SSE 전송
    ↓
Client (실시간 스트리밍 수신)
```

## 구현 세부사항

### 1. StreamingCallbackHandler

**위치**: `lawfirm_langgraph/core/workflow/callbacks/streaming_callback_handler.py`

**기능**:
- LangChain의 `BaseCallbackHandler`를 상속
- `on_llm_stream` 이벤트를 캡처하여 `asyncio.Queue`에 저장
- 스트리밍 통계 수집

**주요 메서드**:
- `on_llm_start()`: LLM 시작 시 호출
- `on_llm_stream()`: LLM 스트리밍 청크 수신 시 호출
- `on_llm_end()`: LLM 종료 시 호출
- `get_stats()`: 스트리밍 통계 반환

### 2. 노드 분리

#### generate_answer_stream
- **용도**: API용 스트리밍 전용 노드
- **특징**: 실시간 토큰 스트리밍만 수행
- **환경 변수**: `USE_STREAMING_MODE=true`일 때 사용

#### generate_answer_final
- **용도**: 테스트용 최종 검증 노드
- **특징**: 답변 품질 검증, 법률 참조 추출, 포맷팅 포함
- **환경 변수**: `USE_STREAMING_MODE=false`일 때 사용

### 3. 환경 변수 설정

```bash
# API용: 스트리밍 노드 사용 (기본값)
USE_STREAMING_MODE=true

# 테스트용: 최종 검증 노드 사용
USE_STREAMING_MODE=false
```

## API 사용법

### 엔드포인트

**POST** `/api/v1/chat/stream`

### Request

```json
{
  "message": "계약 해제 조건은?",
  "session_id": "session_123"
}
```

### Response (Server-Sent Events)

```
data: {"type":"progress","content":"답변 생성 중...","timestamp":"2025-11-12T09:30:00"}

data: {"type":"stream","content":"계약","timestamp":"2025-11-12T09:30:01"}

data: {"type":"stream","content":" 해제","timestamp":"2025-11-12T09:30:01"}

data: {"type":"stream","content":" 조건은","timestamp":"2025-11-12T09:30:01"}

...

data: {"type":"final","content":"전체 답변...","sources":[...],"legal_references":[...]}
```

### 이벤트 타입

- `progress`: 진행 상황 알림
- `stream`: 실시간 답변 청크 (토큰 단위)
- `final`: 최종 답변 및 메타데이터
- `error`: 오류 발생 시

## 클라이언트 구현 예시

### JavaScript (Fetch API)

```javascript
async function streamChat(message, sessionId) {
  const response = await fetch('/api/v1/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: message,
      session_id: sessionId
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        
        if (data.type === 'stream') {
          // 실시간 청크 표시
          appendToChat(data.content);
        } else if (data.type === 'final') {
          // 최종 답변 처리
          handleFinalAnswer(data);
        }
      }
    }
  }
}
```

### Python (requests)

```python
import requests
import json

def stream_chat(message, session_id):
    url = 'http://localhost:8000/api/v1/chat/stream'
    data = {
        'message': message,
        'session_id': session_id
    }
    
    response = requests.post(url, json=data, stream=True)
    
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                event_data = json.loads(line_str[6:])
                
                if event_data['type'] == 'stream':
                    print(event_data['content'], end='', flush=True)
                elif event_data['type'] == 'final':
                    print('\n\n최종 답변:', event_data['content'])
```

## 트러블슈팅

### 스트리밍이 작동하지 않는 경우

1. **환경 변수 확인**
   ```bash
   echo $USE_STREAMING_MODE  # 또는
   echo %USE_STREAMING_MODE%
   ```

2. **콜백 핸들러 확인**
   - 로그에서 "StreamingCallbackHandler created" 메시지 확인
   - `on_llm_stream` 이벤트 발생 여부 확인

3. **노드 선택 확인**
   - 로그에서 "generate_answer_stream" 노드 실행 여부 확인

### 성능 최적화

- 큐 크기 모니터링: `callback_queue.qsize()`
- 청크 처리 지연 최소화
- 네트워크 버퍼 크기 조정

## 관련 문서

- [LangGraph 통합 가이드](docs/03_rag_system/langgraph_integration_guide.md)
- [API 엔드포인트 문서](docs/07_api/api_endpoints.md)
- [Core Modules 가이드](docs/10_technical_reference/core_modules_guide.md)

