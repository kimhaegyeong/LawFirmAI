# -*- coding: utf-8 -*-
"""
LangChain과 LangGraph 모니터링 래퍼
Langfuse를 사용한 모니터링 기능 제공
"""

import logging
from typing import Dict, Any, Optional, List, Union
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains.base import Chain
from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel

# LangGraph 관련 import (선택적)
try:
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    ToolNode = None

from ..utils.langfuse_monitor import get_langfuse_monitor

logger = logging.getLogger(__name__)

class LangfuseCallbackHandler(BaseCallbackHandler):
    """LangChain용 Langfuse 콜백 핸들러"""
    
    def __init__(self):
        super().__init__()
        self.monitor = get_langfuse_monitor()
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """LLM 시작 시 호출"""
        if self.monitor.is_enabled():
            logger.info(f"LLM 시작: {serialized.get('name', 'unknown')}")
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """LLM 종료 시 호출"""
        if self.monitor.is_enabled():
            logger.info("LLM 종료")
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """LLM 오류 시 호출"""
        if self.monitor.is_enabled():
            logger.error(f"LLM 오류: {error}")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """체인 시작 시 호출"""
        if self.monitor.is_enabled():
            logger.info(f"체인 시작: {serialized.get('name', 'unknown')}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """체인 종료 시 호출"""
        if self.monitor.is_enabled():
            logger.info("체인 종료")
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """체인 오류 시 호출"""
        if self.monitor.is_enabled():
            logger.error(f"체인 오류: {error}")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """도구 시작 시 호출"""
        if self.monitor.is_enabled():
            logger.info(f"도구 시작: {serialized.get('name', 'unknown')}")
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """도구 종료 시 호출"""
        if self.monitor.is_enabled():
            logger.info("도구 종료")
    
    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """도구 오류 시 호출"""
        if self.monitor.is_enabled():
            logger.error(f"도구 오류: {error}")

class MonitoredChain:
    """모니터링이 적용된 LangChain 체인 래퍼"""
    
    def __init__(self, chain: Chain, name: Optional[str] = None):
        self.chain = chain
        self.name = name or chain.__class__.__name__
        self.monitor = get_langfuse_monitor()
        self.callback_handler = LangfuseCallbackHandler()
        
        # 콜백 매니저 설정
        if hasattr(chain, 'callback_manager'):
            if isinstance(chain.callback_manager, CallbackManager):
                chain.callback_manager.add_handler(self.callback_handler)
            else:
                chain.callback_manager = CallbackManager([self.callback_handler])
    
    def __call__(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """체인 실행 (모니터링 포함)"""
        if not self.monitor.is_enabled():
            return self.chain(inputs, **kwargs)
        
        # 트레이스 생성
        trace = self.monitor.create_trace(
            name=f"chain_{self.name}",
            user_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id')
        )
        
        if trace:
            try:
                # 체인 실행
                result = self.chain(inputs, **kwargs)
                
                # 결과 로깅
                self.monitor.log_generation(
                    trace_id=trace.id,
                    name=self.name,
                    input_data=inputs,
                    output_data=result,
                    metadata={
                        "chain_type": self.chain.__class__.__name__,
                        "timestamp": str(kwargs.get('timestamp', ''))
                    }
                )
                
                return result
                
            except Exception as e:
                # 오류 로깅
                self.monitor.log_event(
                    trace_id=trace.id,
                    name=f"{self.name}_error",
                    input_data=inputs,
                    output_data={"error": str(e)},
                    metadata={"error_type": type(e).__name__}
                )
                raise
            finally:
                self.monitor.flush()
        else:
            return self.chain(inputs, **kwargs)
    
    def run(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """체인 실행 (단일 입력)"""
        if isinstance(inputs, str):
            inputs = {"input": inputs}
        return self(inputs, **kwargs)

class MonitoredLLM:
    """모니터링이 적용된 LangChain LLM 래퍼"""
    
    def __init__(self, llm: Union[LLM, BaseChatModel], name: Optional[str] = None):
        self.llm = llm
        self.name = name or llm.__class__.__name__
        self.monitor = get_langfuse_monitor()
        self.callback_handler = LangfuseCallbackHandler()
        
        # 콜백 매니저 설정
        if hasattr(llm, 'callback_manager'):
            if isinstance(llm.callback_manager, CallbackManager):
                llm.callback_manager.add_handler(self.callback_handler)
            else:
                llm.callback_manager = CallbackManager([self.callback_handler])
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """LLM 실행 (모니터링 포함)"""
        if not self.monitor.is_enabled():
            return self.llm(prompt, **kwargs)
        
        # 트레이스 생성
        trace = self.monitor.create_trace(
            name=f"llm_{self.name}",
            user_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id')
        )
        
        if trace:
            try:
                # LLM 실행
                result = self.llm(prompt, **kwargs)
                
                # 결과 로깅
                self.monitor.log_generation(
                    trace_id=trace.id,
                    name=self.name,
                    input_data={"prompt": prompt},
                    output_data={"response": result},
                    metadata={
                        "llm_type": self.llm.__class__.__name__,
                        "temperature": kwargs.get('temperature'),
                        "max_tokens": kwargs.get('max_tokens')
                    }
                )
                
                return result
                
            except Exception as e:
                # 오류 로깅
                self.monitor.log_event(
                    trace_id=trace.id,
                    name=f"{self.name}_error",
                    input_data={"prompt": prompt},
                    output_data={"error": str(e)},
                    metadata={"error_type": type(e).__name__}
                )
                raise
            finally:
                self.monitor.flush()
        else:
            return self.llm(prompt, **kwargs)
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        """채팅 모델 실행 (모니터링 포함)"""
        if not self.monitor.is_enabled():
            return self.llm.invoke(messages, **kwargs)
        
        # 트레이스 생성
        trace = self.monitor.create_trace(
            name=f"chat_{self.name}",
            user_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id')
        )
        
        if trace:
            try:
                # 채팅 모델 실행
                result = self.llm.invoke(messages, **kwargs)
                
                # 메시지 변환
                input_messages = [{"type": msg.__class__.__name__, "content": msg.content} for msg in messages]
                output_message = {"type": result.__class__.__name__, "content": result.content}
                
                # 결과 로깅
                self.monitor.log_generation(
                    trace_id=trace.id,
                    name=self.name,
                    input_data={"messages": input_messages},
                    output_data={"response": output_message},
                    metadata={
                        "model_type": self.llm.__class__.__name__,
                        "message_count": len(messages)
                    }
                )
                
                return result
                
            except Exception as e:
                # 오류 로깅
                self.monitor.log_event(
                    trace_id=trace.id,
                    name=f"{self.name}_error",
                    input_data={"messages": input_messages},
                    output_data={"error": str(e)},
                    metadata={"error_type": type(e).__name__}
                )
                raise
            finally:
                self.monitor.flush()
        else:
            return self.llm.invoke(messages, **kwargs)

class MonitoredLangGraph:
    """모니터링이 적용된 LangGraph 래퍼"""
    
    def __init__(self, graph: 'StateGraph', name: Optional[str] = None):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph가 설치되지 않았습니다.")
        
        self.graph = graph
        self.name = name or "langgraph"
        self.monitor = get_langfuse_monitor()
        self.compiled_graph = None
    
    def compile(self, **kwargs):
        """그래프 컴파일 (모니터링 포함)"""
        if not self.monitor.is_enabled():
            self.compiled_graph = self.graph.compile(**kwargs)
            return self.compiled_graph
        
        # 트레이스 생성
        trace = self.monitor.create_trace(
            name=f"compile_{self.name}",
            session_id=kwargs.get('session_id')
        )
        
        if trace:
            try:
                # 그래프 컴파일
                self.compiled_graph = self.graph.compile(**kwargs)
                
                # 컴파일 로깅
                self.monitor.log_event(
                    trace_id=trace.id,
                    name="graph_compile",
                    input_data={"compile_kwargs": kwargs},
                    output_data={"compiled": True},
                    metadata={"graph_name": self.name}
                )
                
                return self.compiled_graph
                
            except Exception as e:
                # 오류 로깅
                self.monitor.log_event(
                    trace_id=trace.id,
                    name="graph_compile_error",
                    input_data={"compile_kwargs": kwargs},
                    output_data={"error": str(e)},
                    metadata={"error_type": type(e).__name__}
                )
                raise
            finally:
                self.monitor.flush()
        else:
            self.compiled_graph = self.graph.compile(**kwargs)
            return self.compiled_graph
    
    def invoke(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """그래프 실행 (모니터링 포함)"""
        if not self.compiled_graph:
            self.compile()
        
        if not self.monitor.is_enabled():
            return self.compiled_graph.invoke(input_data, **kwargs)
        
        # 트레이스 생성
        trace = self.monitor.create_trace(
            name=f"invoke_{self.name}",
            user_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id')
        )
        
        if trace:
            try:
                # 그래프 실행
                result = self.compiled_graph.invoke(input_data, **kwargs)
                
                # 결과 로깅
                self.monitor.log_generation(
                    trace_id=trace.id,
                    name=self.name,
                    input_data=input_data,
                    output_data=result,
                    metadata={
                        "graph_type": "StateGraph",
                        "config": kwargs.get('config', {})
                    }
                )
                
                return result
                
            except Exception as e:
                # 오류 로깅
                self.monitor.log_event(
                    trace_id=trace.id,
                    name=f"{self.name}_error",
                    input_data=input_data,
                    output_data={"error": str(e)},
                    metadata={"error_type": type(e).__name__}
                )
                raise
            finally:
                self.monitor.flush()
        else:
            return self.compiled_graph.invoke(input_data, **kwargs)

# 편의 함수들
def monitor_chain(chain: Chain, name: Optional[str] = None) -> MonitoredChain:
    """체인 모니터링 래퍼 생성"""
    return MonitoredChain(chain, name)

def monitor_llm(llm: Union[LLM, BaseChatModel], name: Optional[str] = None) -> MonitoredLLM:
    """LLM 모니터링 래퍼 생성"""
    return MonitoredLLM(llm, name)

def monitor_langgraph(graph: 'StateGraph', name: Optional[str] = None) -> MonitoredLangGraph:
    """LangGraph 모니터링 래퍼 생성"""
    return MonitoredLangGraph(graph, name)

def get_monitored_callback_manager() -> CallbackManager:
    """모니터링이 적용된 콜백 매니저 반환"""
    return CallbackManager([LangfuseCallbackHandler()])
