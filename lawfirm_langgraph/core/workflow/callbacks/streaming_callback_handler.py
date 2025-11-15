# -*- coding: utf-8 -*-
"""
Streaming Callback Handler
LangGraph 스트리밍 콜백 핸들러 구현
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    LANCHAIN_CALLBACKS_AVAILABLE = True
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
        from langchain.schema import LLMResult
        LANCHAIN_CALLBACKS_AVAILABLE = True
    except ImportError:
        LANCHAIN_CALLBACKS_AVAILABLE = False
        BaseCallbackHandler = object
        LLMResult = None

logger = logging.getLogger(__name__)


class StreamingCallbackHandler(BaseCallbackHandler):
    """스트리밍 콜백 핸들러 - on_llm_stream 이벤트를 큐에 저장"""
    
    def __init__(self, queue: Optional[asyncio.Queue] = None):
        """
        초기화
        
        Args:
            queue: 청크를 저장할 asyncio.Queue. None이면 자동 생성
        """
        if not LANCHAIN_CALLBACKS_AVAILABLE:
            logger.warning("LangChain callbacks not available. Streaming may not work properly.")
        
        super().__init__()
        self.queue = queue if queue is not None else asyncio.Queue()
        self.streaming_active = False
        self.chunk_count = 0
        self.total_chunks = 0
        self.node_name = None
        
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """LLM 시작 시 호출"""
        self.streaming_active = True
        self.chunk_count = 0
        self.node_name = kwargs.get("name", "unknown")
        logger.debug(f"📡 [CALLBACK] on_llm_start: node={self.node_name}, prompts={len(prompts)}")
    
    def on_llm_stream(self, chunk: Any, **kwargs: Any) -> None:
        """LLM 스트리밍 청크 수신 시 호출"""
        if not self.streaming_active:
            return
        
        self.chunk_count += 1
        self.total_chunks += 1
        
        # 청크 내용 추출
        chunk_content = self._extract_chunk_content(chunk)
        
        if chunk_content:
            try:
                # 큐에 청크 추가 (비동기)
                if self.queue:
                    # 큐가 가득 찬 경우를 대비하여 non-blocking 시도
                    try:
                        self.queue.put_nowait({
                            "type": "chunk",
                            "content": chunk_content,
                            "chunk_index": self.chunk_count,
                            "node_name": self.node_name,
                            "timestamp": asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else None
                        })
                    except asyncio.QueueFull:
                        logger.warning(f"⚠️ [CALLBACK] Queue full, dropping chunk #{self.chunk_count}")
                    except RuntimeError:
                        # 이벤트 루프가 없는 경우 동기적으로 처리
                        logger.debug(f"📡 [CALLBACK] on_llm_stream (sync): chunk #{self.chunk_count}, content={chunk_content[:50]}...")
                
                # 디버그 로깅 (처음 10개 청크만)
                if self.chunk_count <= 10:
                    logger.info(
                        f"📡 [CALLBACK] on_llm_stream: chunk #{self.chunk_count}, "
                        f"content={chunk_content[:50]}..., node={self.node_name}, "
                        f"queue_size={self.queue.qsize() if self.queue else 0}"
                    )
            except Exception as e:
                logger.error(f"❌ [CALLBACK] Error processing chunk: {e}")
    
    def on_chat_model_stream(self, chunk: Any, **kwargs: Any) -> None:
        """Chat Model 스트리밍 청크 수신 시 호출 (ChatGoogleGenerativeAI 등)"""
        # on_llm_stream과 동일한 로직 사용
        self.on_llm_stream(chunk, **kwargs)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 종료 시 호출"""
        self.streaming_active = False
        logger.debug(
            f"✅ [CALLBACK] on_llm_end: node={self.node_name}, "
            f"total_chunks={self.chunk_count}, "
            f"generations={len(response.generations) if response and hasattr(response, 'generations') else 0}"
        )
        
        # 종료 신호를 큐에 추가
        if self.queue:
            try:
                self.queue.put_nowait({
                    "type": "end",
                    "node_name": self.node_name,
                    "total_chunks": self.chunk_count,
                    "timestamp": asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else None
                })
            except (asyncio.QueueFull, RuntimeError):
                pass
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """LLM 오류 시 호출"""
        self.streaming_active = False
        logger.error(f"❌ [CALLBACK] on_llm_error: node={self.node_name}, error={error}")
        
        # 오류 신호를 큐에 추가
        if self.queue:
            try:
                self.queue.put_nowait({
                    "type": "error",
                    "node_name": self.node_name,
                    "error": str(error),
                    "timestamp": asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else None
                })
            except (asyncio.QueueFull, RuntimeError):
                pass
    
    def _extract_chunk_content(self, chunk: Any) -> str:
        """청크에서 텍스트 내용 추출"""
        if chunk is None:
            return ""
        
        # AIMessageChunk 또는 유사한 객체 처리
        if hasattr(chunk, "content"):
            content = chunk.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list) and len(content) > 0:
                # 리스트인 경우 첫 번째 요소 추출
                first_item = content[0]
                if isinstance(first_item, str):
                    return first_item
                elif hasattr(first_item, "text"):
                    return first_item.text
                else:
                    return str(first_item)
            else:
                return str(content) if content else ""
        
        # 문자열인 경우
        if isinstance(chunk, str):
            return chunk
        
        # text 속성이 있는 경우
        if hasattr(chunk, "text"):
            return chunk.text
        
        # delta 형식 (LangGraph v2)
        if isinstance(chunk, dict):
            delta = chunk.get("delta", {})
            if isinstance(delta, dict):
                return delta.get("content", delta.get("text", ""))
            elif isinstance(delta, str):
                return delta
            return chunk.get("content", chunk.get("text", ""))
        
        # 기타 경우 문자열로 변환
        return str(chunk)
    
    def reset(self) -> None:
        """핸들러 상태 초기화"""
        self.streaming_active = False
        self.chunk_count = 0
        self.node_name = None
        # 큐는 비우지 않음 (이미 처리된 청크 보존)
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            "streaming_active": self.streaming_active,
            "chunk_count": self.chunk_count,
            "total_chunks": self.total_chunks,
            "node_name": self.node_name,
            "queue_size": self.queue.qsize() if self.queue else 0
        }

