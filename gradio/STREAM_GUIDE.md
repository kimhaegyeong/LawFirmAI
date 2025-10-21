# LawFirmAI 스트림 기능 사용 가이드

## 개요

LawFirmAI에 스트림 형태의 답변 기능이 추가되었습니다. 이 기능을 통해 사용자는 실시간으로 답변이 생성되는 과정을 볼 수 있으며, 더 나은 사용자 경험을 제공합니다.

## 주요 기능

### 1. 실시간 답변 생성
- 질문 분석 과정을 실시간으로 표시
- 관련 법령 및 판례 검색 상태 표시
- 답변 생성 과정을 단어별로 스트림

### 2. 스트림 모드 토글
- 사용자가 스트림 모드와 일반 모드를 선택할 수 있음
- 기본값: 스트림 모드 활성화

### 3. 진행률 표시
- 답변 생성 진행률을 퍼센트로 표시
- 처리 단계별 상태 메시지 제공

## 사용 방법

### Gradio 인터페이스에서 사용

1. **스트림 모드 활성화**
   - 체크박스 "🔄 스트림 모드 (실시간 답변)"을 체크
   - 기본적으로 활성화되어 있음

2. **질문 입력**
   - 텍스트 박스에 법률 관련 질문 입력
   - Enter 키 또는 "전송" 버튼 클릭

3. **실시간 답변 확인**
   - 질문 분석 → 검색 → 답변 생성 과정을 실시간으로 확인
   - 답변이 단어별로 점진적으로 표시됨

### 프로그래밍 방식으로 사용

```python
import asyncio
from source.services.chat_service import ChatService
from source.utils.config import Config

async def use_stream():
    # ChatService 초기화
    config = Config()
    chat_service = ChatService(config)
    
    # 스트림 처리
    async for chunk in chat_service.process_message_stream(
        "계약 해제 조건이 무엇인가요?",
        session_id="my_session",
        user_id="my_user"
    ):
        chunk_type = chunk.get("type")
        content = chunk.get("content")
        
        if chunk_type == "status":
            print(f"상태: {content}")
        elif chunk_type == "content":
            print(f"답변: {content}")
        elif chunk_type == "metadata":
            print(f"메타데이터: {content}")

# 실행
asyncio.run(use_stream())
```

## 스트림 청크 타입

### 1. status
- 처리 상태 메시지
- 예: "질문을 분석하고 있습니다...", "관련 법령을 검색하고 있습니다..."

### 2. content
- 실제 답변 내용
- 단어별로 청크가 전송됨

### 3. metadata
- 처리 완료 후 메타데이터
- 처리 시간, 신뢰도, 질문 유형 등 포함

### 4. error
- 오류 발생 시 오류 메시지

## 설정 옵션

### 스트림 속도 조절
```python
# ChatService에서 스트림 속도 조절
chunk_size = 5  # 한 번에 전송할 단어 수
delay = 0.1     # 청크 간 지연 시간 (초)
```

### 스트림 모드 비활성화
```python
# Gradio 인터페이스에서 체크박스 해제
stream_mode = gr.Checkbox(label="🔄 스트림 모드", value=False)
```

## 성능 고려사항

### 1. 메모리 사용량
- 스트림 모드는 일반 모드보다 약간 더 많은 메모리를 사용
- 대화 기록이 실시간으로 업데이트됨

### 2. 네트워크 사용량
- 스트림 모드는 더 많은 네트워크 요청을 생성
- 각 청크마다 별도의 응답이 전송됨

### 3. 처리 시간
- 전체 처리 시간은 일반 모드와 동일
- 사용자 경험은 더 나아짐 (실시간 피드백)

## 문제 해결

### 1. 스트림이 작동하지 않는 경우
- 브라우저가 WebSocket을 지원하는지 확인
- 네트워크 연결 상태 확인
- JavaScript가 활성화되어 있는지 확인

### 2. 답변이 중간에 끊기는 경우
- 네트워크 연결 상태 확인
- 브라우저 새로고침 후 재시도
- 일반 모드로 전환하여 사용

### 3. 성능이 느린 경우
- 스트림 모드를 비활성화
- 브라우저 캐시 삭제
- 다른 탭이나 애플리케이션 종료

## 향후 개선 계획

### 1. WebSocket 지원
- 현재는 시뮬레이션된 스트림
- 실제 WebSocket을 통한 실시간 스트림 구현 예정

### 2. 사용자 설정
- 스트림 속도 사용자 정의
- 청크 크기 조절 옵션
- 애니메이션 효과 선택

### 3. 모바일 최적화
- 모바일 환경에서의 스트림 성능 최적화
- 터치 인터페이스 개선

## 예제 코드

### 기본 스트림 사용
```python
async def basic_stream_example():
    chat_service = ChatService(Config())
    
    message = "이혼 절차는 어떻게 진행하나요?"
    
    async for chunk in chat_service.process_message_stream(message):
        if chunk["type"] == "content":
            print(chunk["content"], end="", flush=True)
```

### 고급 스트림 처리
```python
async def advanced_stream_example():
    chat_service = ChatService(Config())
    
    message = "손해배상 청구 방법을 알려주세요"
    full_response = ""
    
    async for chunk in chat_service.process_message_stream(message):
        chunk_type = chunk["type"]
        content = chunk["content"]
        
        if chunk_type == "status":
            print(f"\n[상태] {content}")
        elif chunk_type == "content":
            full_response += content
            print(content, end="", flush=True)
        elif chunk_type == "metadata":
            metadata = content
            print(f"\n[완료] 처리시간: {metadata['processing_time']:.2f}초")
            print(f"[완료] 신뢰도: {metadata['confidence']:.1%}")
```

## 결론

스트림 기능은 LawFirmAI의 사용자 경험을 크게 향상시키는 중요한 기능입니다. 실시간으로 답변이 생성되는 과정을 보여줌으로써 사용자에게 더 나은 피드백을 제공하고, 시스템의 투명성을 높입니다.

이 기능을 통해 사용자는 AI가 어떻게 답변을 생성하는지 이해할 수 있으며, 더 신뢰할 수 있는 법률 AI 어시스턴트를 경험할 수 있습니다.
