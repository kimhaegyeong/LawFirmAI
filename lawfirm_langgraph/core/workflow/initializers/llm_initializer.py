# -*- coding: utf-8 -*-
"""
LLM Initializer
LLM 초기화 로직을 처리하는 초기화기
"""

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import os
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from lawfirm_langgraph.core.workflow.utils.workflow_constants import WorkflowConstants
except ImportError:
    from core.workflow.utils.workflow_constants import WorkflowConstants

logger = get_logger(__name__)


class LLMInitializer:
    """LLM 초기화기"""

    def __init__(self, config, main_llm=None):
        self.config = config
        self.main_llm = main_llm

    def initialize_llm(self) -> Any:
        """LLM 초기화 (Google Gemini)"""
        if self.config.llm_provider == "google":
            try:
                return self.initialize_gemini()
            except Exception as e:
                logger.warning(f"Failed to initialize Google Gemini LLM: {e}. Using Mock LLM.")

        return self.create_mock_llm()

    def initialize_gemini(self, timeout: Optional[int] = None) -> ChatGoogleGenerativeAI:
        """Google Gemini LLM 초기화 (최종 답변 생성용 - RAG QA)
        
        Args:
            timeout: 타임아웃 시간 (초). None이면 WorkflowConstants.TIMEOUT_RAG_QA 사용
        """
        # 최종 답변용 모델 설정 (환경 변수 우선, 없으면 gemini-2.5-flash-lite 기본값)
        answer_model = os.getenv("ANSWER_LLM_MODEL", "gemini-2.5-flash-lite")
        
        if timeout is None:
            timeout = WorkflowConstants.TIMEOUT_RAG_QA
        
        gemini_llm = ChatGoogleGenerativeAI(
            model=answer_model,
            temperature=WorkflowConstants.TEMPERATURE,
            max_output_tokens=WorkflowConstants.MAX_OUTPUT_TOKENS,
            timeout=timeout,
            api_key=self.config.google_api_key
        )
        logger.info(f"Initialized Google Gemini LLM for answer generation: {answer_model} (timeout: {timeout}s)")
        return gemini_llm
    
    def initialize_gemini_long_text(self) -> ChatGoogleGenerativeAI:
        """Google Gemini LLM 초기화 (긴 글/코드 생성용 - 30~60초 timeout)"""
        answer_model = os.getenv("LONG_TEXT_ANSWER_LLM_MODEL", "gemini-2.5-flash")
        
        gemini_llm = ChatGoogleGenerativeAI(
            model=answer_model,
            temperature=WorkflowConstants.TEMPERATURE,
            max_output_tokens=WorkflowConstants.MAX_OUTPUT_TOKENS,
            timeout=WorkflowConstants.TIMEOUT_LONG_TEXT,
            api_key=self.config.google_api_key
        )
        logger.info(f"Initialized Google Gemini LLM for long text/code generation: {answer_model} (timeout: {WorkflowConstants.TIMEOUT_LONG_TEXT}s)")
        return gemini_llm

    def create_mock_llm(self) -> Any:
        """Mock LLM 생성"""
        class MockLLM:
            def invoke(self, prompt):
                return "Mock LLM response for: " + prompt
            async def ainvoke(self, prompt):
                return "Mock LLM async response for: " + prompt

        logger.warning("No valid LLM provider configured or failed to initialize. Using Mock LLM.")
        return MockLLM()

    def initialize_llm_fast(self) -> Any:
        """빠른 LLM 초기화 (간단한 질문용 - Gemini 2.5 Flash Lite)"""
        if self.config.llm_provider == "google":
            try:
                # gemini-2.5-flash-lite를 기본값으로 사용
                flash_model = "gemini-2.5-flash-lite"
                # 환경 변수나 config에서 모델명이 지정되어 있으면 사용
                if self.config.google_model and "flash" in self.config.google_model.lower():
                    flash_model = self.config.google_model

                gemini_llm_fast = ChatGoogleGenerativeAI(
                    model=flash_model,
                    temperature=0.3,
                    max_output_tokens=500,
                    timeout=10,
                    api_key=self.config.google_api_key
                )
                logger.info(f"Initialized fast LLM: {flash_model}")
                return gemini_llm_fast
            except Exception as e:
                logger.warning(f"Failed to initialize fast LLM: {e}. Using main LLM.")
                return self.main_llm

        return self.main_llm

    def initialize_validator_llm(self) -> Any:
        """품질 검증용 LLM 초기화"""
        validator_provider = os.getenv("VALIDATOR_LLM_PROVIDER", self.config.llm_provider)
        validator_model = os.getenv("VALIDATOR_LLM_MODEL", None)

        if validator_provider == "google":
            try:
                if not validator_model:
                    validator_model = "gemini-2.5-flash-lite"
                    if self.config.google_model and "flash" in self.config.google_model.lower():
                        validator_model = self.config.google_model

                validator_llm = ChatGoogleGenerativeAI(
                    model=validator_model,
                    temperature=0.3,
                    max_output_tokens=1000,
                    timeout=15,
                    api_key=self.config.google_api_key
                )
                logger.info(f"Initialized validator LLM: {validator_model}")
                return validator_llm
            except Exception as e:
                logger.warning(f"Failed to initialize validator LLM: {e}. Using main LLM.")
                return self.main_llm

        return self.main_llm

