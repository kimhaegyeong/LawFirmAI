# -*- coding: utf-8 -*-
"""
Response Extractor
응답 관련 추출 유틸리티
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Any

logger = get_logger(__name__)


class ResponseExtractor:
    """응답 관련 추출 유틸리티"""

    @staticmethod
    def extract_response_content(response) -> str:
        """응답에서 내용 추출"""
        try:
            if hasattr(response, 'content'):
                content = response.content
                # content가 문자열인지 확인
                if isinstance(content, dict):
                    content = content.get("content", content.get("answer", str(content)))
                return str(content) if not isinstance(content, str) else content

            # response 자체를 처리
            if isinstance(response, dict):
                return response.get("content", response.get("answer", str(response)))

            return str(response)

        except Exception as e:
            logger.warning(f"Failed to extract response content: {e}")
            return str(response) if response else ""

