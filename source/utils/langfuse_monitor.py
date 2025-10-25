# -*- coding: utf-8 -*-
"""
Langfuse 모니터링 설정
LangChain과 LangGraph를 모니터링하기 위한 설정
"""

import os
import logging
from typing import Dict, Any, Optional
from langfuse import Langfuse, observe

logger = logging.getLogger(__name__)

class LangfuseMonitor:
    """Langfuse 모니터링 클래스"""
    
    def __init__(self):
        """Langfuse 모니터 초기화"""
        self.langfuse = None
        self._initialize_langfuse()
    
    def _initialize_langfuse(self):
        """Langfuse 초기화"""
        try:
            # 환경 변수에서 설정 가져오기
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            
            if not public_key or not secret_key:
                logger.warning("Langfuse API 키가 설정되지 않았습니다. 모니터링이 비활성화됩니다.")
                return
            
            # Langfuse 클라이언트 초기화
            self.langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            
            logger.info("Langfuse 모니터링이 성공적으로 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"Langfuse 초기화 실패: {e}")
            self.langfuse = None
    
    def is_enabled(self) -> bool:
        """모니터링이 활성화되어 있는지 확인"""
        return self.langfuse is not None
    
    def create_trace(self, name: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Optional[Any]:
        """새로운 트레이스 생성"""
        if not self.is_enabled():
            return None
        
        try:
            # Langfuse의 올바른 API 사용
            trace_id = self.langfuse.create_trace_id()
            logger.info(f"트레이스 생성 성공: {trace_id}")
            return type('Trace', (), {'id': trace_id})()
        except Exception as e:
            logger.error(f"트레이스 생성 실패: {e}")
            return None
    
    def create_span(self, trace_id: str, name: str, input_data: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """새로운 스팬 생성"""
        if not self.is_enabled():
            return None
        
        try:
            span = self.langfuse.start_as_current_span(
                name=name,
                input=input_data
            )
            return span
        except Exception as e:
            logger.error(f"스팬 생성 실패: {e}")
            return None
    
    def log_generation(self, 
                      trace_id: str, 
                      name: str, 
                      input_data: Optional[Dict[str, Any]] = None,
                      output_data: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """생성 로그 기록"""
        if not self.is_enabled():
            return False
        
        try:
            # 메타데이터에 trace_id 추가
            if metadata is None:
                metadata = {}
            metadata['trace_id'] = trace_id
            
            self.langfuse.create_event(
                name=name,
                input=input_data,
                output=output_data,
                metadata=metadata
            )
            return True
        except Exception as e:
            logger.error(f"생성 로그 기록 실패: {e}")
            return False
    
    def log_event(self, 
                 trace_id: str, 
                 name: str, 
                 input_data: Optional[Dict[str, Any]] = None,
                 output_data: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """이벤트 로그 기록"""
        if not self.is_enabled():
            return False
        
        try:
            # 메타데이터에 trace_id 추가
            if metadata is None:
                metadata = {}
            metadata['trace_id'] = trace_id
            
            self.langfuse.create_event(
                name=name,
                input=input_data,
                output=output_data,
                metadata=metadata
            )
            return True
        except Exception as e:
            logger.error(f"이벤트 로그 기록 실패: {e}")
            return False
    
    def flush(self):
        """데이터 플러시"""
        if self.is_enabled():
            try:
                self.langfuse.flush()
            except Exception as e:
                logger.error(f"데이터 플러시 실패: {e}")

# 전역 인스턴스
langfuse_monitor = LangfuseMonitor()

def get_langfuse_monitor() -> LangfuseMonitor:
    """Langfuse 모니터 인스턴스 반환"""
    return langfuse_monitor

# 데코레이터 함수들
def observe_function(name: Optional[str] = None, 
                    capture_input: bool = True, 
                    capture_output: bool = True,
                    capture_exception: bool = True):
    """함수 관찰 데코레이터"""
    return observe(
        name=name,
        capture_input=capture_input,
        capture_output=capture_output,
        capture_exception=capture_exception
    )

def create_langfuse_context(trace_id: str, 
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None):
    """Langfuse 컨텍스트 생성 (간단한 버전)"""
    return {
        "trace_id": trace_id,
        "user_id": user_id,
        "session_id": session_id
    }
