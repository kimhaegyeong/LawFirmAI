# LangChain & LangGraph 개발 규칙

## 개요

LawFirmAI 프로젝트에서 LLM 사용 시 LangChain과 LangGraph를 활용한 개발 규칙을 정의합니다. 이 규칙은 일관성 있는 코드 작성, 유지보수성 향상, 그리고 확장 가능한 아키텍처 구축을 목표로 합니다.

## 핵심 원칙

### 1. 아키텍처 원칙
- **상태 기반 워크플로우**: LangGraph를 통한 복잡한 법률 질문 처리 워크플로우 관리
- **체크포인트 기반 지속성**: SQLite 기반 세션 상태 저장 및 복원
- **모듈화된 컴포넌트**: 재사용 가능한 서비스 컴포넌트 설계
- **하위 호환성**: 기존 시스템과의 점진적 통합

### 2. 개발 원칙
- **타입 안전성**: TypedDict를 활용한 상태 정의
- **에러 처리**: 포괄적인 예외 처리 및 로깅
- **성능 최적화**: 메모리 관리 및 캐싱 전략
- **테스트 가능성**: 단위 테스트 및 통합 테스트 지원

## 프로젝트 구조 규칙

### 디렉토리 구조
```
source/
├── services/
│   ├── langgraph/              # LangGraph 워크플로우 서비스
│   │   ├── __init__.py
│   │   ├── workflow_service.py        # 메인 워크플로우 서비스
│   │   ├── workflow_service_enhanced.py # 향상된 워크플로우 서비스
│   │   ├── legal_workflow.py          # 기본 법률 워크플로우
│   │   ├── legal_workflow_enhanced.py # 향상된 법률 워크플로우
│   │   ├── legal_workflow_simple.py   # 단순화된 워크플로우
│   │   ├── state_definitions.py       # 상태 정의
│   │   ├── checkpoint_manager.py      # 체크포인트 관리
│   │   ├── keyword_mapper.py         # 키워드 매핑 시스템
│   │   └── agents/                   # 에이전트 시스템 (향후 확장)
│   ├── langchain/              # LangChain 서비스
│   │   ├── __init__.py
│   │   ├── rag_service.py            # RAG 서비스
│   │   ├── llm_client.py             # LLM 클라이언트
│   │   ├── prompt_templates.py       # 프롬프트 템플릿
│   │   └── chains/                   # 체인 구현
│   └── chat_service.py         # 통합 채팅 서비스
├── utils/
│   ├── langgraph_config.py    # LangGraph 설정 관리
│   ├── llm_config.py          # LLM 설정 관리
│   └── prompt_manager.py      # 프롬프트 관리
└── models/
    ├── langchain_models.py    # LangChain 모델 래퍼
    └── langgraph_models.py    # LangGraph 모델 래퍼
```

## LangGraph 개발 규칙

### 1. 상태 정의 규칙

#### 상태 클래스 정의
```python
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add

class LegalWorkflowState(TypedDict):
    """법률 워크플로우 상태 정의"""
    
    # 필수 입력 데이터
    query: str
    session_id: str
    
    # 질문 분류 결과
    query_type: str
    confidence: float
    
    # 문서 검색 결과
    retrieved_docs: Annotated[List[Dict[str, Any]], add]
    search_metadata: Dict[str, Any]
    
    # 컨텍스트 분석 결과
    analysis: Optional[str]
    legal_references: List[str]
    
    # 최종 답변
    answer: str
    sources: List[str]
    
    # 처리 과정 메타데이터
    processing_steps: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    metadata: Dict[str, Any]
    
    # 성능 메트릭
    processing_time: float
    tokens_used: int
```

#### 상태 초기화 헬퍼 함수
```python
def create_initial_legal_state(query: str, session_id: str) -> LegalWorkflowState:
    """법률 워크플로우 초기 상태 생성"""
    return LegalWorkflowState(
        query=query,
        session_id=session_id,
        query_type="",
        confidence=0.0,
        retrieved_docs=[],
        search_metadata={},
        analysis=None,
        legal_references=[],
        answer="",
        sources=[],
        processing_steps=[],
        errors=[],
        metadata={},
        processing_time=0.0,
        tokens_used=0
    )
```

### 2. 워크플로우 구현 규칙

#### 워크플로우 클래스 구조
```python
from langgraph.graph import StateGraph, END
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class LegalQuestionWorkflow:
    """법률 질문 처리 워크플로우"""
    
    def __init__(self, config: LangGraphConfig):
        """워크플로우 초기화"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 워크플로우 그래프 생성
        self.graph = StateGraph(LegalWorkflowState)
        self._build_workflow()
    
    def _build_workflow(self):
        """워크플로우 그래프 구성"""
        # 노드 추가
        self.graph.add_node("classify_query", self._classify_query)
        self.graph.add_node("search_documents", self._search_documents)
        self.graph.add_node("analyze_context", self._analyze_context)
        self.graph.add_node("generate_answer", self._generate_answer)
        self.graph.add_node("format_response", self._format_response)
        
        # 엣지 정의
        self.graph.set_entry_point("classify_query")
        self.graph.add_edge("classify_query", "search_documents")
        self.graph.add_edge("search_documents", "analyze_context")
        self.graph.add_edge("analyze_context", "generate_answer")
        self.graph.add_edge("generate_answer", "format_response")
        self.graph.add_edge("format_response", END)
    
    async def _classify_query(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """질문 분류 노드"""
        try:
            # 질문 분류 로직 구현
            query_type, confidence = await self._perform_classification(state["query"])
            
            return {
                "query_type": query_type,
                "confidence": confidence,
                "processing_steps": ["질문 분류 완료"]
            }
        except Exception as e:
            self.logger.error(f"질문 분류 중 오류: {e}")
            return {
                "errors": [f"질문 분류 오류: {str(e)}"],
                "processing_steps": ["질문 분류 실패"]
            }
```

#### 노드 함수 구현 규칙
```python
async def _node_function(self, state: LegalWorkflowState) -> Dict[str, Any]:
    """
    노드 함수 구현 규칙
    
    Args:
        state: 현재 워크플로우 상태
        
    Returns:
        Dict[str, Any]: 상태 업데이트 딕셔너리
    """
    try:
        # 1. 로깅 시작
        self.logger.info(f"노드 실행 시작: {state['query'][:50]}...")
        
        # 2. 비즈니스 로직 구현
        result = await self._perform_business_logic(state)
        
        # 3. 성공 응답 반환
        return {
            "result_field": result,
            "processing_steps": [f"노드 완료: {result}"],
            "metadata": {"node_completed": True}
        }
        
    except ValueError as e:
        # 4. 비즈니스 로직 오류 처리
        self.logger.warning(f"비즈니스 로직 오류: {e}")
        return {
            "errors": [f"처리 오류: {str(e)}"],
            "processing_steps": ["노드 실패"]
        }
        
    except Exception as e:
        # 5. 예상치 못한 오류 처리
        self.logger.error(f"예상치 못한 오류: {e}")
        return {
            "errors": [f"시스템 오류: {str(e)}"],
            "processing_steps": ["노드 실패"]
        }
```

### 3. 체크포인트 관리 규칙

#### 체크포인트 매니저 구현
```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.base import Checkpointer
import sqlite3
import os
from typing import Optional

class CheckpointManager:
    """체크포인트 관리 클래스"""
    
    def __init__(self, db_path: str):
        """체크포인트 매니저 초기화"""
        self.db_path = db_path
        self._ensure_db_directory()
        self._initialize_database()
    
    def _ensure_db_directory(self):
        """데이터베이스 디렉토리 생성"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.close()
            self.logger.info(f"체크포인트 데이터베이스 초기화: {self.db_path}")
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {e}")
            raise
    
    def get_memory(self) -> Optional[Checkpointer]:
        """체크포인트 메모리 반환"""
        try:
            return SqliteSaver.from_conn_string(self.db_path)
        except Exception as e:
            self.logger.error(f"체크포인트 메모리 생성 실패: {e}")
            return None
    
    def cleanup_old_checkpoints(self, ttl_hours: int = 24) -> int:
        """오래된 체크포인트 정리"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 오래된 체크포인트 삭제
            cursor.execute("""
                DELETE FROM checkpoints 
                WHERE created_at < datetime('now', '-{} hours')
            """.format(ttl_hours))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"정리된 체크포인트: {deleted_count}개")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"체크포인트 정리 실패: {e}")
            return 0
```

### 4. 워크플로우 서비스 규칙

#### 서비스 클래스 구조
```python
import logging
import uuid
import time
from typing import Dict, Any, Optional
from datetime import datetime

class LangGraphWorkflowService:
    """LangGraph 워크플로우 서비스"""
    
    def __init__(self, config: Optional[LangGraphConfig] = None):
        """워크플로우 서비스 초기화"""
        self.config = config or LangGraphConfig.from_env()
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_db_path)
        self.legal_workflow = EnhancedLegalQuestionWorkflow(self.config)
        
        # 워크플로우 컴파일
        checkpointer = self.checkpoint_manager.get_memory()
        if checkpointer is not None:
            self.app = self.legal_workflow.graph.compile(checkpointer=checkpointer)
        else:
            self.app = self.legal_workflow.graph.compile()
            self.logger.warning("워크플로우가 체크포인트 없이 컴파일되었습니다")
        
        if self.app is None:
            self.logger.error("Failed to compile workflow")
            raise RuntimeError("워크플로우 컴파일에 실패했습니다")
        
        self.logger.info("LangGraphWorkflowService initialized successfully")
    
    async def process_query(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        enable_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """질문 처리"""
        try:
            # 세션 ID 생성 또는 사용
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # 초기 상태 생성
            initial_state = create_initial_legal_state(query, session_id)
            
            # 워크플로우 실행
            start_time = time.time()
            
            if enable_checkpoint:
                result = await self.app.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": session_id}}
                )
            else:
                result = await self.app.ainvoke(initial_state)
            
            processing_time = time.time() - start_time
            
            # 결과 포맷팅
            return {
                "response": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "sources": result.get("sources", []),
                "legal_references": result.get("legal_references", []),
                "processing_steps": result.get("processing_steps", []),
                "session_id": session_id,
                "processing_time": processing_time,
                "tokens_used": result.get("tokens_used", 0),
                "errors": result.get("errors", [])
            }
            
        except Exception as e:
            self.logger.error(f"질문 처리 중 오류: {e}")
            return {
                "response": "죄송합니다. 처리 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "legal_references": [],
                "processing_steps": ["오류 발생"],
                "session_id": session_id or str(uuid.uuid4()),
                "processing_time": 0.0,
                "tokens_used": 0,
                "errors": [str(e)]
            }
```

## LangChain 개발 규칙

### 1. RAG 서비스 구현 규칙

#### RAG 서비스 클래스 구조
```python
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LangChainRAGService:
    """LangChain 기반 RAG 서비스"""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        prompt_template: ChatPromptTemplate
    ):
        """RAG 서비스 초기화"""
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
        self.logger = logging.getLogger(__name__)
        
        # 체인 구성
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """RAG 체인 구성"""
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    async def query(self, question: str) -> Dict[str, Any]:
        """질문 처리"""
        try:
            # 문서 검색
            docs = await self.retriever.ainvoke(question)
            
            # 답변 생성
            answer = await self.chain.ainvoke(question)
            
            return {
                "answer": answer,
                "sources": [doc.metadata.get("source", "") for doc in docs],
                "retrieved_docs": len(docs),
                "confidence": self._calculate_confidence(docs, answer)
            }
            
        except Exception as e:
            self.logger.error(f"RAG 쿼리 처리 중 오류: {e}")
            return {
                "answer": "죄송합니다. 답변을 생성할 수 없습니다.",
                "sources": [],
                "retrieved_docs": 0,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _calculate_confidence(self, docs: List[Document], answer: str) -> float:
        """신뢰도 계산"""
        if not docs:
            return 0.0
        
        # 간단한 신뢰도 계산 로직
        doc_count = len(docs)
        answer_length = len(answer)
        
        # 문서 수와 답변 길이를 기반으로 한 신뢰도
        confidence = min(0.9, (doc_count * 0.1) + (min(answer_length, 500) / 1000))
        return round(confidence, 2)
```

### 2. 프롬프트 템플릿 규칙

#### 프롬프트 템플릿 정의
```python
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 시스템 프롬프트 템플릿
SYSTEM_PROMPT = """당신은 전문적인 법률 AI 어시스턴트입니다.

역할:
- 사용자의 법률 질문에 정확하고 도움이 되는 답변을 제공합니다
- 관련 법령, 판례, 법률 원칙을 참조하여 답변합니다
- 복잡한 법률 개념을 이해하기 쉽게 설명합니다

답변 규칙:
1. 정확성: 법률적으로 정확한 정보만 제공합니다
2. 명확성: 명확하고 이해하기 쉬운 언어를 사용합니다
3. 근거: 답변에 대한 법적 근거를 제시합니다
4. 한계: 법률 조언이 아님을 명시합니다

답변 형식:
- 핵심 답변을 먼저 제시합니다
- 관련 법령이나 판례를 참조합니다
- 필요한 경우 단계별 설명을 제공합니다
- 주의사항이나 예외사항을 명시합니다"""

# 인간 메시지 프롬프트 템플릿
HUMAN_PROMPT = """질문: {question}

컨텍스트:
{context}

위 컨텍스트를 참조하여 질문에 답변해주세요."""

# 프롬프트 템플릿 생성
LEGAL_QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)
])
```

### 3. LLM 클라이언트 규칙

#### LLM 클라이언트 구현
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    """LLM 클라이언트 관리 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """LLM 클라이언트 초기화"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm = self._create_llm()
    
    def _create_llm(self):
        """LLM 인스턴스 생성"""
        provider = self.config.get("provider", "google")
        
        try:
            if provider == "google":
                return ChatGoogleGenerativeAI(
                    model=self.config.get("model", "gemini-2.5-flash-lite"),
                    google_api_key=self.config.get("api_key"),
                    temperature=self.config.get("temperature", 0.7),
                    max_output_tokens=self.config.get("max_tokens", 2048)
                )
            elif provider == "ollama":
                return ChatOllama(
                    model=self.config.get("model", "qwen2.5:7b"),
                    base_url=self.config.get("base_url", "http://localhost:11434"),
                    temperature=self.config.get("temperature", 0.7)
                )
            elif provider == "openai":
                return ChatOpenAI(
                    model=self.config.get("model", "gpt-3.5-turbo"),
                    api_key=self.config.get("api_key"),
                    temperature=self.config.get("temperature", 0.7),
                    max_tokens=self.config.get("max_tokens", 2048)
                )
            else:
                raise ValueError(f"지원하지 않는 LLM 제공자: {provider}")
                
        except Exception as e:
            self.logger.error(f"LLM 생성 실패: {e}")
            raise
    
    async def generate_response(self, prompt: str) -> str:
        """응답 생성"""
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"응답 생성 실패: {e}")
            raise
```

## 설정 관리 규칙

### 1. 환경 변수 설정

#### .env 파일 설정
```bash
# LangGraph 설정
LANGGRAPH_ENABLED=true
LANGGRAPH_CHECKPOINT_DB=./data/checkpoints/langgraph.db
CHECKPOINT_TTL=3600
MAX_ITERATIONS=10
RECURSION_LIMIT=25
ENABLE_STREAMING=true

# LLM 설정
LLM_PROVIDER=google
GOOGLE_MODEL=gemini-2.5-flash-lite
GOOGLE_API_KEY=your_google_api_key_here

# Ollama 설정 (백업용)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TIMEOUT=15

# 키워드 매핑 시스템 설정
KEYWORD_MAPPING_ENABLED=true
KEYWORD_EFFECTIVENESS_FILE=data/keyword_effectiveness.json
KEYWORD_LEARNING_ENABLED=true
SEMANTIC_SIMILARITY_THRESHOLD=0.6
CONTEXT_AWARE_MAPPING=true
```

### 2. 설정 클래스 구현

#### 설정 클래스 구조
```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class LangGraphConfig:
    """LangGraph 설정 클래스"""
    
    # 체크포인트 설정
    checkpoint_storage: CheckpointStorageType = CheckpointStorageType.SQLITE
    checkpoint_db_path: str = "./data/checkpoints/langgraph.db"
    checkpoint_ttl: int = 3600
    
    # 워크플로우 설정
    max_iterations: int = 10
    recursion_limit: int = 25
    enable_streaming: bool = True
    
    # LLM 설정
    llm_provider: str = "google"
    google_model: str = "gemini-2.5-flash-lite"
    google_api_key: str = ""
    
    # Ollama 설정 (백업용)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_timeout: int = 15
    
    # LangGraph 활성화 설정
    langgraph_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'LangGraphConfig':
        """환경 변수에서 설정 로드"""
        config = cls()
        
        # 환경 변수에서 값 로드
        config.langgraph_enabled = os.getenv("LANGGRAPH_ENABLED", "true").lower() == "true"
        config.checkpoint_db_path = os.getenv("LANGGRAPH_CHECKPOINT_DB", config.checkpoint_db_path)
        config.checkpoint_ttl = int(os.getenv("CHECKPOINT_TTL", config.checkpoint_ttl))
        config.max_iterations = int(os.getenv("MAX_ITERATIONS", config.max_iterations))
        config.recursion_limit = int(os.getenv("RECURSION_LIMIT", config.recursion_limit))
        config.enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
        
        config.llm_provider = os.getenv("LLM_PROVIDER", config.llm_provider)
        config.google_model = os.getenv("GOOGLE_MODEL", config.google_model)
        config.google_api_key = os.getenv("GOOGLE_API_KEY", config.google_api_key)
        
        config.ollama_base_url = os.getenv("OLLAMA_BASE_URL", config.ollama_base_url)
        config.ollama_model = os.getenv("OLLAMA_MODEL", config.ollama_model)
        config.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", config.ollama_timeout))
        
        logger.info(f"LangGraph configuration loaded: enabled={config.langgraph_enabled}")
        return config
    
    def validate(self) -> List[str]:
        """설정 유효성 검사"""
        errors = []
        
        if self.langgraph_enabled:
            if not self.checkpoint_db_path:
                errors.append("LANGGRAPH_CHECKPOINT_DB is required when LangGraph is enabled")
            
            if self.checkpoint_ttl <= 0:
                errors.append("CHECKPOINT_TTL must be positive")
            
            if self.max_iterations <= 0:
                errors.append("MAX_ITERATIONS must be positive")
            
            if self.recursion_limit <= 0:
                errors.append("RECURSION_LIMIT must be positive")
        
        return errors
```

## 에러 처리 규칙

### 1. 예외 처리 계층

#### 예외 클래스 정의
```python
class LangGraphError(Exception):
    """LangGraph 기본 예외 클래스"""
    pass

class WorkflowError(LangGraphError):
    """워크플로우 실행 오류"""
    pass

class CheckpointError(LangGraphError):
    """체크포인트 관리 오류"""
    pass

class LLMError(LangGraphError):
    """LLM 관련 오류"""
    pass

class ConfigurationError(LangGraphError):
    """설정 관련 오류"""
    pass
```

#### 에러 처리 패턴
```python
async def safe_node_execution(self, state: LegalWorkflowState) -> Dict[str, Any]:
    """안전한 노드 실행"""
    try:
        # 노드 로직 실행
        result = await self._execute_node_logic(state)
        return result
        
    except ValueError as e:
        # 비즈니스 로직 오류
        self.logger.warning(f"비즈니스 로직 오류: {e}")
        return {
            "errors": [f"처리 오류: {str(e)}"],
            "processing_steps": ["노드 실패"]
        }
        
    except WorkflowError as e:
        # 워크플로우 오류
        self.logger.error(f"워크플로우 오류: {e}")
        return {
            "errors": [f"워크플로우 오류: {str(e)}"],
            "processing_steps": ["워크플로우 실패"]
        }
        
    except Exception as e:
        # 예상치 못한 오류
        self.logger.error(f"예상치 못한 오류: {e}")
        return {
            "errors": [f"시스템 오류: {str(e)}"],
            "processing_steps": ["시스템 실패"]
        }
```

## 로깅 규칙

### 1. 로깅 설정

#### 로깅 구성
```python
import logging
import sys
from datetime import datetime

def setup_langgraph_logging(log_level: str = "INFO"):
    """LangGraph 로깅 설정"""
    
    # 로거 생성
    logger = logging.getLogger("source.services.langgraph")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 핸들러 생성
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper()))
    
    # 포맷터 생성
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(handler)
    
    return logger
```

#### 로깅 사용 패턴
```python
class WorkflowNode:
    """워크플로우 노드 예제"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """노드 실행"""
        # 시작 로깅
        self.logger.info(f"노드 실행 시작: {state['query'][:50]}...")
        
        try:
            # 비즈니스 로직
            result = await self._process_state(state)
            
            # 성공 로깅
            self.logger.info(f"노드 실행 완료: {result.get('status', 'success')}")
            return result
            
        except Exception as e:
            # 오류 로깅
            self.logger.error(f"노드 실행 실패: {e}")
            raise
```

## 테스트 규칙

### 1. 단위 테스트

#### 워크플로우 테스트
```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
from source.services.langgraph.workflow_service import LangGraphWorkflowService
from source.utils.langgraph_config import LangGraphConfig

class TestLangGraphWorkflowService:
    """LangGraph 워크플로우 서비스 테스트"""
    
    @pytest.fixture
    def mock_config(self):
        """모킹된 설정"""
        config = LangGraphConfig()
        config.langgraph_enabled = True
        config.checkpoint_db_path = ":memory:"
        return config
    
    @pytest.fixture
    async def workflow_service(self, mock_config):
        """워크플로우 서비스 인스턴스"""
        with patch('source.services.langgraph.workflow_service.CheckpointManager'):
            service = LangGraphWorkflowService(mock_config)
            return service
    
    @pytest.mark.asyncio
    async def test_process_query_success(self, workflow_service):
        """질문 처리 성공 테스트"""
        query = "계약서 검토 요청"
        
        result = await workflow_service.process_query(query)
        
        assert "response" in result
        assert "session_id" in result
        assert "processing_time" in result
        assert result["processing_time"] >= 0
    
    @pytest.mark.asyncio
    async def test_process_query_with_session(self, workflow_service):
        """세션 ID와 함께 질문 처리 테스트"""
        query = "이혼 절차 문의"
        session_id = "test-session-123"
        
        result = await workflow_service.process_query(query, session_id=session_id)
        
        assert result["session_id"] == session_id
        assert "response" in result
    
    @pytest.mark.asyncio
    async def test_process_query_error_handling(self, workflow_service):
        """오류 처리 테스트"""
        with patch.object(workflow_service.app, 'ainvoke', side_effect=Exception("Test error")):
            result = await workflow_service.process_query("test query")
            
            assert "error" in result["response"].lower()
            assert len(result["errors"]) > 0
```

### 2. 통합 테스트

#### 전체 워크플로우 테스트
```python
import pytest
import asyncio
from source.services.chat_service import ChatService
from source.utils.config import Config

class TestLangGraphIntegration:
    """LangGraph 통합 테스트"""
    
    @pytest.fixture
    async def chat_service(self):
        """채팅 서비스 인스턴스"""
        config = Config()
        service = ChatService(config)
        return service
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, chat_service):
        """전체 워크플로우 테스트"""
        questions = [
            "계약서 작성 시 주의사항은?",
            "이혼 절차는 어떻게 되나요?",
            "손해배상 관련 판례를 찾아주세요"
        ]
        
        session_id = "integration-test-session"
        
        for question in questions:
            result = await chat_service.process_message(
                question, 
                session_id=session_id
            )
            
            # 기본 응답 구조 확인
            assert "response" in result
            assert "confidence" in result
            assert "sources" in result
            assert "processing_time" in result
            
            # 응답 품질 확인
            assert len(result["response"]) > 0
            assert 0 <= result["confidence"] <= 1
            assert result["processing_time"] > 0
    
    @pytest.mark.asyncio
    async def test_session_persistence(self, chat_service):
        """세션 지속성 테스트"""
        session_id = "persistence-test-session"
        
        # 첫 번째 질문
        result1 = await chat_service.process_message(
            "계약서 검토 요청", 
            session_id=session_id
        )
        
        # 두 번째 질문 (같은 세션)
        result2 = await chat_service.process_message(
            "위 계약서의 주의사항은?", 
            session_id=session_id
        )
        
        # 세션 ID 일치 확인
        assert result1["session_id"] == session_id
        assert result2["session_id"] == session_id
        
        # 두 응답 모두 유효한지 확인
        assert len(result1["response"]) > 0
        assert len(result2["response"]) > 0
```

## 성능 최적화 규칙

### 1. 메모리 관리

#### 메모리 사용량 모니터링
```python
import psutil
import gc
from typing import Dict, Any

class MemoryMonitor:
    """메모리 사용량 모니터"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 조회"""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total / (1024**3),  # GB
            "available": memory.available / (1024**3),  # GB
            "used": memory.used / (1024**3),  # GB
            "percent": memory.percent
        }
    
    def check_memory_threshold(self, threshold: float = 80.0) -> bool:
        """메모리 사용량 임계값 확인"""
        memory_usage = self.get_memory_usage()
        return memory_usage["percent"] > threshold
    
    def cleanup_memory(self):
        """메모리 정리"""
        gc.collect()
        self.logger.info("메모리 정리 완료")
```

### 2. 캐싱 전략

#### 응답 캐싱
```python
from functools import lru_cache
from typing import Dict, Any, Optional
import hashlib
import json

class ResponseCache:
    """응답 캐시 관리"""
    
    def __init__(self, max_size: int = 128):
        self.cache = {}
        self.max_size = max_size
        self.logger = logging.getLogger(__name__)
    
    def _generate_cache_key(self, query: str, context: Optional[str] = None) -> str:
        """캐시 키 생성"""
        content = f"{query}:{context or ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_response(self, query: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """캐시된 응답 조회"""
        cache_key = self._generate_cache_key(query, context)
        return self.cache.get(cache_key)
    
    def cache_response(self, query: str, response: Dict[str, Any], context: Optional[str] = None):
        """응답 캐시 저장"""
        cache_key = self._generate_cache_key(query, context)
        
        # 캐시 크기 제한
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = response
        self.logger.debug(f"응답 캐시 저장: {cache_key[:8]}...")
    
    def clear_cache(self):
        """캐시 정리"""
        self.cache.clear()
        self.logger.info("응답 캐시 정리 완료")
```

## 배포 규칙

### 1. Docker 설정

#### LangGraph Dockerfile
```dockerfile
# LangGraph 서비스용 Dockerfile
FROM python:3.9-slim as builder

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.9-slim

# 보안을 위한 non-root 사용자 생성
RUN useradd --create-home --shell /bin/bash app
USER app

WORKDIR /app

# 빌드된 패키지 복사
COPY --from=builder /root/.local /home/app/.local

# 애플리케이션 코드 복사
COPY --chown=app:app source/ ./source/
COPY --chown=app:app data/ ./data/

# 환경 변수 설정
ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV LANGGRAPH_ENABLED=true

# 체크포인트 디렉토리 생성
RUN mkdir -p /app/data/checkpoints

EXPOSE 8000

CMD ["python", "-m", "source.services.langgraph.workflow_service"]
```

### 2. 환경별 설정

#### 개발 환경 설정
```python
# config/development.py
class DevelopmentConfig(LangGraphConfig):
    """개발 환경 설정"""
    
    def __init__(self):
        super().__init__()
        
        # 개발 환경 특화 설정
        self.debug = True
        self.log_level = "DEBUG"
        self.checkpoint_ttl = 7200  # 2시간
        self.max_iterations = 5
        self.enable_streaming = True
```

#### 프로덕션 환경 설정
```python
# config/production.py
class ProductionConfig(LangGraphConfig):
    """프로덕션 환경 설정"""
    
    def __init__(self):
        super().__init__()
        
        # 프로덕션 환경 특화 설정
        self.debug = False
        self.log_level = "INFO"
        self.checkpoint_ttl = 3600  # 1시간
        self.max_iterations = 10
        self.enable_streaming = True
```

## 모니터링 규칙

### 1. 메트릭 수집

#### 성능 메트릭 수집
```python
import time
from functools import wraps
from typing import Dict, Any, Callable

def measure_performance(func: Callable) -> Callable:
    """성능 측정 데코레이터"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 성공 메트릭 기록
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # 실패 메트릭 기록
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper
```

### 2. 헬스 체크

#### 서비스 헬스 체크
```python
class HealthChecker:
    """서비스 헬스 체크"""
    
    def __init__(self, workflow_service: LangGraphWorkflowService):
        self.workflow_service = workflow_service
        self.logger = logging.getLogger(__name__)
    
    async def check_health(self) -> Dict[str, Any]:
        """헬스 체크 실행"""
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }
        
        # 워크플로우 서비스 체크
        try:
            test_result = await self.workflow_service.test_workflow("헬스 체크 테스트")
            health_status["checks"]["workflow_service"] = {
                "status": "healthy" if test_result.get("test_passed") else "unhealthy",
                "details": test_result
            }
        except Exception as e:
            health_status["checks"]["workflow_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "unhealthy"
        
        # 체크포인트 데이터베이스 체크
        try:
            db_info = self.workflow_service.checkpoint_manager.get_database_info()
            health_status["checks"]["checkpoint_db"] = {
                "status": "healthy",
                "details": db_info
            }
        except Exception as e:
            health_status["checks"]["checkpoint_db"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "unhealthy"
        
        return health_status
```

## 코드 리뷰 체크리스트

### LangGraph 관련 체크리스트
- [ ] 상태 정의가 TypedDict로 올바르게 정의되었는가?
- [ ] 노드 함수가 적절한 에러 처리를 포함하는가?
- [ ] 체크포인트 관리가 올바르게 구현되었는가?
- [ ] 워크플로우 그래프가 올바르게 구성되었는가?
- [ ] 세션 관리가 적절히 구현되었는가?

### LangChain 관련 체크리스트
- [ ] RAG 체인이 올바르게 구성되었는가?
- [ ] 프롬프트 템플릿이 적절히 정의되었는가?
- [ ] LLM 클라이언트가 에러 처리를 포함하는가?
- [ ] 체인 실행이 비동기로 구현되었는가?
- [ ] 출력 파싱이 적절히 처리되었는가?

### 공통 체크리스트
- [ ] 로깅이 적절히 구현되었는가?
- [ ] 설정 관리가 환경별로 구분되었는가?
- [ ] 테스트 코드가 작성되었는가?
- [ ] 성능 최적화가 고려되었는가?
- [ ] 보안 검증이 적용되었는가?

## 예제 코드

### 완전한 워크플로우 예제
```python
import asyncio
from source.services.chat_service import ChatService
from source.utils.config import Config

async def main():
    """LangGraph 워크플로우 사용 예제"""
    
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

이 규칙들을 통해 LawFirmAI 프로젝트에서 LangChain과 LangGraph를 활용한 일관성 있고 확장 가능한 LLM 시스템을 구축할 수 있습니다. 추가 질문이나 규칙 보완이 필요하시면 언제든 문의해주세요.
