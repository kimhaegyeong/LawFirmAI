# -*- coding: utf-8 -*-
"""
LLM Initializer
LLM 초기화 로직을 처리하는 초기화기
"""

import logging
import os
from typing import Any, Optional

from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI

from core.workflow.utils.workflow_constants import WorkflowConstants

logger = logging.getLogger(__name__)


class LLMInitializer:
    """LLM 초기화기"""

    def __init__(self, config, main_llm=None):
        self.config = config
        self.main_llm = main_llm

    def initialize_llm(self) -> Any:
        """LLM 초기화 (Google Gemini 우선, Ollama 백업)"""
        if self.config.llm_provider == "google":
            try:
                return self.initialize_gemini()
            except Exception as e:
                logger.warning(f"Failed to initialize Google Gemini LLM: {e}. Falling back to Ollama.")

        if self.config.llm_provider == "ollama":
            try:
                return self.initialize_ollama()
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama LLM: {e}. Using Mock LLM.")

        return self.create_mock_llm()

    def initialize_gemini(self) -> ChatGoogleGenerativeAI:
        """Google Gemini LLM 초기화"""
        gemini_llm = ChatGoogleGenerativeAI(
            model=self.config.google_model,
            temperature=WorkflowConstants.TEMPERATURE,
            max_output_tokens=WorkflowConstants.MAX_OUTPUT_TOKENS,
            timeout=WorkflowConstants.TIMEOUT,
            api_key=self.config.google_api_key
        )
        logger.info(f"Initialized Google Gemini LLM: {self.config.google_model}")
        return gemini_llm

    def initialize_ollama(self) -> Ollama:
        """Ollama LLM 초기화"""
        ollama_llm = Ollama(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
            temperature=WorkflowConstants.TEMPERATURE,
            num_predict=WorkflowConstants.MAX_OUTPUT_TOKENS,
            timeout=20
        )
        logger.info(f"Initialized Ollama LLM: {self.config.ollama_model}")
        return ollama_llm

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
        """빠른 LLM 초기화 (간단한 질문용 - Gemini Flash 또는 작은 모델)"""
        if self.config.llm_provider == "google":
            try:
                flash_model = "gemini-1.5-flash"
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
                    validator_model = "gemini-1.5-flash"
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

