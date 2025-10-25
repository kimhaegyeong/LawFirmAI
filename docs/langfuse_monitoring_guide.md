# Langfuse 모니터링 설정 가이드

이 가이드는 LawFirmAI 프로젝트에서 Langfuse를 사용하여 LangChain과 LangGraph를 모니터링하는 방법을 설명합니다.

## 📋 목차

1. [설치 및 설정](#설치-및-설정)
2. [환경 변수 구성](#환경-변수-구성)
3. [LangChain 모니터링](#langchain-모니터링)
4. [LangGraph 모니터링](#langgraph-모니터링)
5. [커스텀 모니터링](#커스텀-모니터링)
6. [사용 예시](#사용-예시)
7. [문제 해결](#문제-해결)

## 🚀 설치 및 설정

### 1. Langfuse 설치

```bash
pip install langfuse
```

### 2. LangChain 설치 (선택사항)

```bash
pip install langchain
pip install openai  # OpenAI 모델 사용 시
```

### 3. LangGraph 설치 (선택사항)

```bash
pip install langgraph
```

## 🔧 환경 변수 구성

### 1. Langfuse 계정 설정

1. [Langfuse 웹사이트](https://langfuse.com)에서 계정 생성
2. 프로젝트 생성 후 API 키 확인

### 2. 환경 변수 설정

`.env` 파일에 다음 변수들을 추가하세요:

```env
# Langfuse API 설정
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_HOST=https://cloud.langfuse.com

# 모니터링 활성화 여부
LANGFUSE_ENABLED=true

# 추가 설정 (선택사항)
LANGFUSE_RELEASE=production
LANGFUSE_ENVIRONMENT=development
```

## 🔍 LangChain 모니터링

### 기본 사용법

```python
from source.utils.langchain_monitor import monitor_llm, monitor_chain
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# LLM 모니터링
llm = OpenAI(temperature=0.7)
monitored_llm = monitor_llm(llm, name="legal_assistant")

# 체인 모니터링
chain = LLMChain(llm=monitored_llm, prompt=prompt_template)
monitored_chain = monitor_chain(chain, name="legal_qa_chain")

# 실행 (모니터링 포함)
result = monitored_chain.run(
    question="계약서 작성 방법을 알려주세요",
    user_id="user123",
    session_id="session456"
)
```

### 채팅 모델 모니터링

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# 채팅 모델 모니터링
chat_model = ChatOpenAI(temperature=0.7)
monitored_chat = monitor_llm(chat_model, name="legal_chat")

# 메시지 실행
messages = [HumanMessage(content="계약서 검토 요청")]
response = monitored_chat.invoke(
    messages=messages,
    user_id="user123",
    session_id="session456"
)
```

## 🌐 LangGraph 모니터링

### 기본 사용법

```python
from source.utils.langchain_monitor import monitor_langgraph
from langgraph.graph import StateGraph, END

# 그래프 정의
workflow = StateGraph(LegalState)
workflow.add_node("analyze", analyze_question)
workflow.add_node("generate", generate_answer)
workflow.add_edge("analyze", "generate")
workflow.add_edge("generate", END)
workflow.set_entry_point("analyze")

# 그래프 모니터링
monitored_graph = monitor_langgraph(workflow, name="legal_workflow")
compiled_graph = monitored_graph.compile()

# 실행 (모니터링 포함)
result = monitored_graph.invoke(
    {"question": "계약서 작성 방법을 알려주세요"},
    user_id="user123",
    session_id="session456"
)
```

## 🛠️ 커스텀 모니터링

### 직접 트레이스 생성

```python
from source.utils.langfuse_monitor import get_langfuse_monitor

monitor = get_langfuse_monitor()

# 트레이스 생성
trace = monitor.create_trace(
    name="custom_legal_analysis",
    user_id="user123",
    session_id="session456"
)

# 이벤트 로깅
monitor.log_event(
    trace_id=trace.id,
    name="question_received",
    input_data={"question": "계약서 검토 요청"},
    metadata={"source": "api"}
)

# 생성 로깅
monitor.log_generation(
    trace_id=trace.id,
    name="legal_analysis",
    input_data={"question": "계약서 검토 요청"},
    output_data={"analysis": "계약서 검토가 필요합니다."},
    metadata={"model": "custom", "confidence": 0.9}
)

# 데이터 플러시
monitor.flush()
```

### 데코레이터 사용

```python
from source.utils.langfuse_monitor import observe_function

@observe_function(name="process_legal_question")
def process_legal_question(question: str) -> str:
    """법률 질문 처리"""
    # 실제 처리 로직
    return f"질문 '{question}'에 대한 답변입니다."
```

## 📊 사용 예시

### 1. 데모 실행

```bash
python demos/langfuse_monitoring_demo.py
```

### 2. 통합 예시

```bash
python examples/langfuse_integration_example.py
```

### 3. 기존 서비스에 통합

```python
from source.utils.langfuse_monitor import get_langfuse_monitor
from source.utils.langchain_monitor import get_monitored_callback_manager

class YourService:
    def __init__(self):
        self.monitor = get_langfuse_monitor()
        self.callback_manager = get_monitored_callback_manager()
    
    def process_request(self, request: str, user_id: str):
        # 모니터링이 활성화된 경우에만 트레이스 생성
        if self.monitor.is_enabled():
            trace = self.monitor.create_trace(
                name="request_processing",
                user_id=user_id
            )
            # 처리 로직
            # 결과 로깅
            self.monitor.flush()
        else:
            # 기본 처리
            pass
```

## 🔧 문제 해결

### 1. 모니터링이 작동하지 않는 경우

- 환경 변수 `LANGFUSE_PUBLIC_KEY`와 `LANGFUSE_SECRET_KEY`가 올바르게 설정되었는지 확인
- 네트워크 연결 상태 확인
- Langfuse 대시보드에서 API 키 상태 확인

### 2. 데이터가 표시되지 않는 경우

- `monitor.flush()` 호출 확인
- 트레이스 ID가 올바른지 확인
- Langfuse 대시보드에서 프로젝트 설정 확인

### 3. 성능 문제

- 모니터링이 비활성화된 경우 기본 처리로 폴백
- 배치 처리 시 주기적으로 `flush()` 호출
- 불필요한 메타데이터 제거

## 📈 모니터링 데이터 확인

1. [Langfuse 대시보드](https://cloud.langfuse.com)에 로그인
2. 프로젝트 선택
3. "Traces" 탭에서 실행된 트레이스 확인
4. "Analytics" 탭에서 성능 메트릭 확인

## 🔒 보안 고려사항

- API 키를 환경 변수로 관리
- 민감한 데이터는 메타데이터에 포함하지 않음
- 프로덕션 환경에서는 적절한 로그 레벨 설정

## 📚 추가 자료

- [Langfuse 공식 문서](https://langfuse.com/docs)
- [LangChain 콜백 문서](https://python.langchain.com/docs/modules/callbacks/)
- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)

## 🤝 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해주세요.
