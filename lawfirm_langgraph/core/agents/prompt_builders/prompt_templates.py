# -*- coding: utf-8 -*-
"""
질문 유형별 프롬프트 템플릿
각 질문 유형에 최적화된 프롬프트 생성 시스템
UnifiedPromptManager 기반 단순화된 래퍼
"""

import logging
from typing import Any, Dict, Optional

from .question_classifier import QuestionType
from .unified_prompt_manager import LegalDomain, ModelType, UnifiedPromptManager

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """프롬프트 템플릿 관리자 (UnifiedPromptManager 통합)"""

    def __init__(self):
        """프롬프트 템플릿 초기화"""
        self.logger = logging.getLogger(__name__)
        # 통합 프롬프트 관리자와 연동
        self.unified_manager = UnifiedPromptManager()

    def get_template(self, question_type: QuestionType) -> Dict[str, Any]:
        """
        질문 유형에 맞는 프롬프트 템플릿 반환

        Args:
            question_type: 질문 유형

        Returns:
            Dict[str, Any]: 프롬프트 템플릿 정보
        """
        try:
            # UnifiedPromptManager에서 질문 유형별 템플릿 정보 가져오기
            question_type_config = self.unified_manager.question_type_templates.get(question_type)

            if question_type_config:
                return {
                    "template": question_type_config.get("template", ""),
                    "context_keys": question_type_config.get("context_keys", []),
                    "max_context_length": question_type_config.get("max_context_length", 2000),
                    "priority": question_type_config.get("priority", "medium")
                }
            else:
                # 기본 템플릿 반환
                default_config = self.unified_manager.question_type_templates.get(
                    QuestionType.GENERAL_QUESTION,
                    {"template": "", "context_keys": [], "max_context_length": 2000, "priority": "low"}
                )
                return {
                    "template": default_config.get("template", ""),
                    "context_keys": default_config.get("context_keys", []),
                    "max_context_length": default_config.get("max_context_length", 2000),
                    "priority": default_config.get("priority", "low")
                }

        except Exception as e:
            self.logger.error(f"Error getting template for {question_type}: {e}")
            return {
                "template": "",
                "context_keys": [],
                "max_context_length": 2000,
                "priority": "medium"
            }

    def format_prompt(self,
                     question_type: QuestionType,
                     context_data: Dict[str, Any],
                     user_query: Optional[str] = None,
                     domain: Optional[LegalDomain] = None,
                     model_type: ModelType = ModelType.GEMINI) -> str:
        """
        프롬프트 템플릿을 실제 데이터로 포맷팅 (통합 시스템 사용)

        Args:
            question_type: 질문 유형
            context_data: 컨텍스트 데이터
            user_query: 사용자 질문 (선택사항)
            domain: 법률 도메인 (선택사항)
            model_type: 모델 타입 (기본값: GEMINI)

        Returns:
            str: 포맷팅된 프롬프트
        """
        try:
            # UnifiedPromptManager 사용
            if user_query:
                return self.unified_manager.get_optimized_prompt(
                    query=user_query,
                    question_type=question_type,
                    domain=domain,
                    context=context_data,
                    model_type=model_type
                )
            else:
                # 기본 프롬프트 반환
                return self._get_basic_prompt(question_type, context_data)

        except Exception as e:
            self.logger.error(f"Error formatting prompt: {e}")
            return f"질문에 대한 답변을 제공하겠습니다.\n\n사용자 질문: {user_query or '질문이 없습니다.'}"

    def _get_basic_prompt(self, question_type: QuestionType, context_data: Dict[str, Any]) -> str:
        """기본 프롬프트 생성"""
        template_info = self.get_template(question_type)

        if not template_info or not template_info.get("template"):
            return "질문에 대한 답변을 제공하겠습니다."

        return template_info["template"].format(**context_data)

    def get_template_info(self, question_type: QuestionType) -> Dict[str, Any]:
        """
        템플릿 정보 조회

        Args:
            question_type: 질문 유형

        Returns:
            Dict[str, Any]: 템플릿 정보
        """
        try:
            template_info = self.get_template(question_type)

            return {
                "question_type": question_type.value,
                "context_keys": template_info.get("context_keys", []),
                "max_context_length": template_info.get("max_context_length", 2000),
                "priority": template_info.get("priority", "medium"),
                "template_preview": template_info.get("template", "")[:200] + "..." if len(template_info.get("template", "")) > 200 else template_info.get("template", "")
            }

        except Exception as e:
            self.logger.error(f"Error getting template info: {e}")
            return {
                "question_type": question_type.value,
                "context_keys": [],
                "max_context_length": 2000,
                "priority": "medium",
                "template_preview": ""
            }

    def validate_context_data(self,
                             question_type: QuestionType,
                             context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        컨텍스트 데이터 검증

        Args:
            question_type: 질문 유형
            context_data: 컨텍스트 데이터

        Returns:
            Dict[str, Any]: 검증 결과
        """
        try:
            template_info = self.get_template(question_type)
            context_keys = template_info.get("context_keys", [])
            max_length = template_info.get("max_context_length", 2000)

            warnings = []

            # 필수 키 확인
            for key in context_keys:
                if key not in context_data:
                    warnings.append(f"필수 컨텍스트 키 '{key}'가 없습니다.")
                elif isinstance(context_data[key], str):
                    if len(context_data[key]) > max_length:
                        warnings.append(f"컨텍스트 '{key}'의 길이가 권장 길이를 초과합니다.")

            return {
                "is_valid": len(warnings) == 0,
                "warnings": warnings
            }

        except Exception as e:
            self.logger.error(f"Error validating context data: {e}")
            return {
                "is_valid": False,
                "warnings": [f"검증 중 오류 발생: {str(e)}"]
            }


# 전역 인스턴스
prompt_template_manager = PromptTemplateManager()
