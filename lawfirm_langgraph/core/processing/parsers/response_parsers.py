# -*- coding: utf-8 -*-
"""
응답 파서 모듈
리팩토링: legal_workflow_enhanced.py에서 파서 메서드 분리
"""

import json
import re
import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class ResponseParser:
    """LLM 응답 파서 기본 클래스"""

    @staticmethod
    def extract_json(response: str) -> Optional[str]:
        """
        응답에서 JSON 부분 추출

        Args:
            response: LLM 응답 문자열

        Returns:
            JSON 문자열 또는 None
        """
        # 중첩 중괄호를 포함한 JSON 패턴
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return None

    @staticmethod
    def parse_json_safe(json_str: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """
        안전한 JSON 파싱

        Args:
            json_str: JSON 문자열
            default: 파싱 실패 시 기본값

        Returns:
            파싱된 딕셔너리
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"JSON 파싱 실패: {json_str[:100]}")
            return default

    @staticmethod
    def parse_json_optional(json_str: str) -> Optional[Dict[str, Any]]:
        """
        선택적 JSON 파싱 (파싱 실패 시 None 반환)

        Args:
            json_str: JSON 문자열

        Returns:
            파싱된 딕셔너리 또는 None
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None


class ClassificationParser(ResponseParser):
    """분류 관련 응답 파서"""

    @staticmethod
    def parse_question_type_response(response: str) -> Dict[str, Any]:
        """질문 유형 분류 응답 파싱"""
        json_str = ClassificationParser.extract_json(response)
        if json_str:
            parsed = ClassificationParser.parse_json_safe(
                json_str,
                {
                    "question_type": "general_question",
                    "confidence": 0.7,
                    "reasoning": "JSON 파싱 실패"
                }
            )
            if "question_type" in parsed:
                return parsed

        # 기본값
        return {
            "question_type": "general_question",
            "confidence": 0.7,
            "reasoning": "JSON 파싱 실패"
        }

    @staticmethod
    def parse_legal_field_response(response: str) -> Optional[Dict[str, Any]]:
        """법률 분야 추출 응답 파싱"""
        json_str = ClassificationParser.extract_json(response)
        if json_str:
            parsed = ClassificationParser.parse_json_optional(json_str)
            if parsed and "legal_field" in parsed:
                return parsed
        return None

    @staticmethod
    def parse_complexity_response(response: str) -> Dict[str, Any]:
        """복잡도 평가 응답 파싱"""
        json_str = ClassificationParser.extract_json(response)
        if json_str:
            parsed = ClassificationParser.parse_json_safe(
                json_str,
                {
                    "complexity": "moderate",
                    "confidence": 0.7,
                    "reasoning": "JSON 파싱 실패"
                }
            )
            if "complexity" in parsed:
                return parsed

        # 기본값
        return {
            "complexity": "moderate",
            "confidence": 0.7,
            "reasoning": "JSON 파싱 실패"
        }

    @staticmethod
    def parse_search_necessity_response(response: str) -> Optional[Dict[str, Any]]:
        """검색 필요성 판단 응답 파싱"""
        json_str = ClassificationParser.extract_json(response)
        if json_str:
            parsed = ClassificationParser.parse_json_optional(json_str)
            if parsed and "needs_search" in parsed:
                return parsed
        return None

    @staticmethod
    def parse_query_type_analysis_response(response: str) -> Dict[str, Any]:
        """질문 유형 분석 응답 파싱 (직접 답변 체인용)"""
        json_str = ClassificationParser.extract_json(response)
        if json_str:
            parsed = ClassificationParser.parse_json_safe(
                json_str,
                {
                    "query_type": "simple_question",
                    "confidence": 0.7,
                    "reasoning": "JSON 파싱 실패"
                }
            )
            if "query_type" in parsed:
                return parsed

        return {
            "query_type": "simple_question",
            "confidence": 0.7,
            "reasoning": "JSON 파싱 실패"
        }


class AnswerParser(ResponseParser):
    """답변 관련 응답 파서"""

    @staticmethod
    def parse_validation_response(response: str) -> Dict[str, Any]:
        """답변 검증 응답 파싱"""
        json_str = AnswerParser.extract_json(response)
        if json_str:
            parsed = AnswerParser.parse_json_safe(
                json_str,
                {
                    "is_valid": True,
                    "quality_score": 0.8,
                    "issues": [],
                    "strengths": [],
                    "recommendations": []
                }
            )
            if "is_valid" in parsed:
                return parsed

        return {
            "is_valid": True,
            "quality_score": 0.8,
            "issues": [],
            "strengths": [],
            "recommendations": []
        }

    @staticmethod
    def parse_improvement_instructions(response: str) -> Optional[Dict[str, Any]]:
        """개선 지시 응답 파싱"""
        json_str = AnswerParser.extract_json(response)
        if json_str:
            parsed = AnswerParser.parse_json_optional(json_str)
            if parsed and "improvement_instructions" in parsed:
                return parsed
        return None

    @staticmethod
    def parse_final_validation_response(response: str) -> Optional[Dict[str, Any]]:
        """최종 검증 응답 파싱"""
        json_str = AnswerParser.extract_json(response)
        if json_str:
            parsed = AnswerParser.parse_json_optional(json_str)
            if parsed and "final_score" in parsed:
                return parsed
        return None

    @staticmethod
    def parse_quality_validation_response(response: str) -> Dict[str, Any]:
        """답변 품질 검증 응답 파싱"""
        json_str = AnswerParser.extract_json(response)
        if json_str:
            parsed = AnswerParser.parse_json_safe(
                json_str,
                {
                    "is_valid": True,
                    "quality_score": 0.8,
                    "issues": [],
                    "needs_improvement": False
                }
            )
            if "is_valid" in parsed:
                return parsed

        return {
            "is_valid": True,
            "quality_score": 0.8,
            "issues": [],
            "needs_improvement": False
        }


class QueryParser(ResponseParser):
    """쿼리 관련 응답 파서"""

    @staticmethod
    def parse_query_analysis_response(response: str) -> Dict[str, Any]:
        """쿼리 분석 응답 파싱"""
        json_str = QueryParser.extract_json(response)
        if json_str:
            parsed = QueryParser.parse_json_safe(
                json_str,
                {
                    "core_keywords": [],
                    "query_intent": "",
                    "key_concepts": [],
                    "analysis": "JSON 파싱 실패"
                }
            )
            if "core_keywords" in parsed:
                return parsed

        return {
            "core_keywords": [],
            "query_intent": "",
            "key_concepts": [],
            "analysis": "JSON 파싱 실패"
        }

    @staticmethod
    def parse_keyword_expansion_response(response: str) -> Dict[str, Any]:
        """키워드 확장 응답 파싱"""
        json_str = QueryParser.extract_json(response)
        if json_str:
            parsed = QueryParser.parse_json_safe(
                json_str,
                {
                    "expanded_keywords": [],
                    "synonyms": [],
                    "related_terms": [],
                    "keyword_variants": [],
                    "reasoning": "JSON 파싱 실패"
                }
            )
            if "expanded_keywords" in parsed:
                return parsed

        return {
            "expanded_keywords": [],
            "synonyms": [],
            "related_terms": [],
            "keyword_variants": [],
            "reasoning": "JSON 파싱 실패"
        }

    @staticmethod
    def parse_query_optimization_response(response: str) -> Dict[str, Any]:
        """쿼리 최적화 응답 파싱"""
        json_str = QueryParser.extract_json(response)
        if json_str:
            parsed = QueryParser.parse_json_safe(
                json_str,
                {
                    "optimized_query": "",
                    "semantic_query": "",
                    "keyword_query": "",
                    "legal_terms": [],
                    "reasoning": "JSON 파싱 실패"
                }
            )
            if "optimized_query" in parsed:
                return parsed

        return {
            "optimized_query": "",
            "semantic_query": "",
            "keyword_query": "",
            "legal_terms": [],
            "reasoning": "JSON 파싱 실패"
        }

    @staticmethod
    def parse_query_validation_response(response: str) -> Optional[Dict[str, Any]]:
        """쿼리 검증 응답 파싱"""
        json_str = QueryParser.extract_json(response)
        if json_str:
            parsed = QueryParser.parse_json_optional(json_str)
            if parsed and "is_valid" in parsed:
                return parsed
        return None

    @staticmethod
    def parse_llm_query_enhancement(response: str) -> Optional[Dict[str, Any]]:
        """LLM 쿼리 강화 응답 파싱"""
        try:
            # JSON 추출 (코드 블록 내부 또는 직접 JSON)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 코드 블록 없이 JSON만 있는 경우
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return None

            # JSON 파싱
            result = json.loads(json_str)

            # 필수 필드 확인
            if not result.get("optimized_query"):
                return None

            # 기본값 설정
            enhanced = {
                "optimized_query": result.get("optimized_query", ""),
                "expanded_keywords": result.get("expanded_keywords", []),
                "keyword_variants": result.get("keyword_variants", []),
                "legal_terms": result.get("legal_terms", []),
                "reasoning": result.get("reasoning", "")
            }

            # 유효성 검사
            if not enhanced["optimized_query"] or len(enhanced["optimized_query"]) > 500:
                return None

            return enhanced

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM query enhancement response: {e}")
            return None


class DocumentParser(ResponseParser):
    """문서 분석 관련 응답 파서"""

    @staticmethod
    def parse_document_type_response(response: str) -> Dict[str, Any]:
        """문서 유형 확인 응답 파싱"""
        json_str = DocumentParser.extract_json(response)
        if json_str:
            parsed = DocumentParser.parse_json_safe(
                json_str,
                {
                    "document_type": "general_legal_document",
                    "confidence": 0.7,
                    "reasoning": "JSON 파싱 실패"
                }
            )
            if "document_type" in parsed:
                return parsed

        return {
            "document_type": "general_legal_document",
            "confidence": 0.7,
            "reasoning": "JSON 파싱 실패"
        }

    @staticmethod
    def parse_clause_extraction_response(response: str) -> Dict[str, Any]:
        """조항 추출 응답 파싱"""
        json_str = DocumentParser.extract_json(response)
        if json_str:
            parsed = DocumentParser.parse_json_optional(json_str)
            if parsed and "key_clauses" in parsed:
                return parsed

        return {
            "key_clauses": [],
            "clause_count": 0
        }

    @staticmethod
    def parse_issue_identification_response(response: str) -> Optional[Dict[str, Any]]:
        """문제점 식별 응답 파싱"""
        json_str = DocumentParser.extract_json(response)
        if json_str:
            parsed = DocumentParser.parse_json_optional(json_str)
            if parsed and "issues" in parsed:
                return parsed
        return None

    @staticmethod
    def parse_issue_identification_response_with_context(
        response: str,
        prev_output: Any
    ) -> Optional[Dict[str, Any]]:
        """문제점 식별 응답 파싱 (이전 단계 출력 통합)"""
        json_str = DocumentParser.extract_json(response)
        if json_str:
            parsed = DocumentParser.parse_json_optional(json_str)
            if parsed and "issues" in parsed:
                # 이전 단계 결과(key_clauses)도 포함
                if isinstance(prev_output, dict):
                    parsed["key_clauses"] = prev_output.get("key_clauses", [])
                    parsed["document_type"] = prev_output.get("document_type", "")
                return parsed
        return None

    @staticmethod
    def parse_improvement_recommendations_response(response: str) -> Optional[Dict[str, Any]]:
        """개선 권고 응답 파싱"""
        json_str = DocumentParser.extract_json(response)
        if json_str:
            parsed = DocumentParser.parse_json_optional(json_str)
            if parsed and "recommendations" in parsed:
                return parsed
        return None
