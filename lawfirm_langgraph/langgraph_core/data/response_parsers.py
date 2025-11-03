# -*- coding: utf-8 -*-
"""
?‘ë‹µ ?Œì„œ ëª¨ë“ˆ
ë¦¬íŒ©? ë§: legal_workflow_enhanced.py?ì„œ ?Œì„œ ë©”ì„œ??ë¶„ë¦¬
"""

import json
import re
import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class ResponseParser:
    """LLM ?‘ë‹µ ?Œì„œ ê¸°ë³¸ ?´ëž˜??""

    @staticmethod
    def extract_json(response: str) -> Optional[str]:
        """
        ?‘ë‹µ?ì„œ JSON ë¶€ë¶?ì¶”ì¶œ

        Args:
            response: LLM ?‘ë‹µ ë¬¸ìž??

        Returns:
            JSON ë¬¸ìž???ëŠ” None
        """
        # ì¤‘ì²© ì¤‘ê´„?¸ë? ?¬í•¨??JSON ?¨í„´
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return None

    @staticmethod
    def parse_json_safe(json_str: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """
        ?ˆì „??JSON ?Œì‹±

        Args:
            json_str: JSON ë¬¸ìž??
            default: ?Œì‹± ?¤íŒ¨ ??ê¸°ë³¸ê°?

        Returns:
            ?Œì‹±???•ì…”?ˆë¦¬
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"JSON ?Œì‹± ?¤íŒ¨: {json_str[:100]}")
            return default

    @staticmethod
    def parse_json_optional(json_str: str) -> Optional[Dict[str, Any]]:
        """
        ? íƒ??JSON ?Œì‹± (?Œì‹± ?¤íŒ¨ ??None ë°˜í™˜)

        Args:
            json_str: JSON ë¬¸ìž??

        Returns:
            ?Œì‹±???•ì…”?ˆë¦¬ ?ëŠ” None
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None


class ClassificationParser(ResponseParser):
    """ë¶„ë¥˜ ê´€???‘ë‹µ ?Œì„œ"""

    @staticmethod
    def parse_question_type_response(response: str) -> Dict[str, Any]:
        """ì§ˆë¬¸ ? í˜• ë¶„ë¥˜ ?‘ë‹µ ?Œì‹±"""
        json_str = ClassificationParser.extract_json(response)
        if json_str:
            parsed = ClassificationParser.parse_json_safe(
                json_str,
                {
                    "question_type": "general_question",
                    "confidence": 0.7,
                    "reasoning": "JSON ?Œì‹± ?¤íŒ¨"
                }
            )
            if "question_type" in parsed:
                return parsed

        # ê¸°ë³¸ê°?
        return {
            "question_type": "general_question",
            "confidence": 0.7,
            "reasoning": "JSON ?Œì‹± ?¤íŒ¨"
        }

    @staticmethod
    def parse_legal_field_response(response: str) -> Optional[Dict[str, Any]]:
        """ë²•ë¥  ë¶„ì•¼ ì¶”ì¶œ ?‘ë‹µ ?Œì‹±"""
        json_str = ClassificationParser.extract_json(response)
        if json_str:
            parsed = ClassificationParser.parse_json_optional(json_str)
            if parsed and "legal_field" in parsed:
                return parsed
        return None

    @staticmethod
    def parse_complexity_response(response: str) -> Dict[str, Any]:
        """ë³µìž¡???‰ê? ?‘ë‹µ ?Œì‹±"""
        json_str = ClassificationParser.extract_json(response)
        if json_str:
            parsed = ClassificationParser.parse_json_safe(
                json_str,
                {
                    "complexity": "moderate",
                    "confidence": 0.7,
                    "reasoning": "JSON ?Œì‹± ?¤íŒ¨"
                }
            )
            if "complexity" in parsed:
                return parsed

        # ê¸°ë³¸ê°?
        return {
            "complexity": "moderate",
            "confidence": 0.7,
            "reasoning": "JSON ?Œì‹± ?¤íŒ¨"
        }

    @staticmethod
    def parse_search_necessity_response(response: str) -> Optional[Dict[str, Any]]:
        """ê²€???„ìš”???ë‹¨ ?‘ë‹µ ?Œì‹±"""
        json_str = ClassificationParser.extract_json(response)
        if json_str:
            parsed = ClassificationParser.parse_json_optional(json_str)
            if parsed and "needs_search" in parsed:
                return parsed
        return None

    @staticmethod
    def parse_query_type_analysis_response(response: str) -> Dict[str, Any]:
        """ì§ˆë¬¸ ? í˜• ë¶„ì„ ?‘ë‹µ ?Œì‹± (ì§ì ‘ ?µë? ì²´ì¸??"""
        json_str = ClassificationParser.extract_json(response)
        if json_str:
            parsed = ClassificationParser.parse_json_safe(
                json_str,
                {
                    "query_type": "simple_question",
                    "confidence": 0.7,
                    "reasoning": "JSON ?Œì‹± ?¤íŒ¨"
                }
            )
            if "query_type" in parsed:
                return parsed

        return {
            "query_type": "simple_question",
            "confidence": 0.7,
            "reasoning": "JSON ?Œì‹± ?¤íŒ¨"
        }


class AnswerParser(ResponseParser):
    """?µë? ê´€???‘ë‹µ ?Œì„œ"""

    @staticmethod
    def parse_validation_response(response: str) -> Dict[str, Any]:
        """?µë? ê²€ì¦??‘ë‹µ ?Œì‹±"""
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
        """ê°œì„  ì§€???‘ë‹µ ?Œì‹±"""
        json_str = AnswerParser.extract_json(response)
        if json_str:
            parsed = AnswerParser.parse_json_optional(json_str)
            if parsed and "improvement_instructions" in parsed:
                return parsed
        return None

    @staticmethod
    def parse_final_validation_response(response: str) -> Optional[Dict[str, Any]]:
        """ìµœì¢… ê²€ì¦??‘ë‹µ ?Œì‹±"""
        json_str = AnswerParser.extract_json(response)
        if json_str:
            parsed = AnswerParser.parse_json_optional(json_str)
            if parsed and "final_score" in parsed:
                return parsed
        return None

    @staticmethod
    def parse_quality_validation_response(response: str) -> Dict[str, Any]:
        """?µë? ?ˆì§ˆ ê²€ì¦??‘ë‹µ ?Œì‹±"""
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
    """ì¿¼ë¦¬ ê´€???‘ë‹µ ?Œì„œ"""

    @staticmethod
    def parse_query_analysis_response(response: str) -> Dict[str, Any]:
        """ì¿¼ë¦¬ ë¶„ì„ ?‘ë‹µ ?Œì‹±"""
        json_str = QueryParser.extract_json(response)
        if json_str:
            parsed = QueryParser.parse_json_safe(
                json_str,
                {
                    "core_keywords": [],
                    "query_intent": "",
                    "key_concepts": [],
                    "analysis": "JSON ?Œì‹± ?¤íŒ¨"
                }
            )
            if "core_keywords" in parsed:
                return parsed

        return {
            "core_keywords": [],
            "query_intent": "",
            "key_concepts": [],
            "analysis": "JSON ?Œì‹± ?¤íŒ¨"
        }

    @staticmethod
    def parse_keyword_expansion_response(response: str) -> Dict[str, Any]:
        """?¤ì›Œ???•ìž¥ ?‘ë‹µ ?Œì‹±"""
        json_str = QueryParser.extract_json(response)
        if json_str:
            parsed = QueryParser.parse_json_safe(
                json_str,
                {
                    "expanded_keywords": [],
                    "synonyms": [],
                    "related_terms": [],
                    "keyword_variants": [],
                    "reasoning": "JSON ?Œì‹± ?¤íŒ¨"
                }
            )
            if "expanded_keywords" in parsed:
                return parsed

        return {
            "expanded_keywords": [],
            "synonyms": [],
            "related_terms": [],
            "keyword_variants": [],
            "reasoning": "JSON ?Œì‹± ?¤íŒ¨"
        }

    @staticmethod
    def parse_query_optimization_response(response: str) -> Dict[str, Any]:
        """ì¿¼ë¦¬ ìµœì ???‘ë‹µ ?Œì‹±"""
        json_str = QueryParser.extract_json(response)
        if json_str:
            parsed = QueryParser.parse_json_safe(
                json_str,
                {
                    "optimized_query": "",
                    "semantic_query": "",
                    "keyword_query": "",
                    "legal_terms": [],
                    "reasoning": "JSON ?Œì‹± ?¤íŒ¨"
                }
            )
            if "optimized_query" in parsed:
                return parsed

        return {
            "optimized_query": "",
            "semantic_query": "",
            "keyword_query": "",
            "legal_terms": [],
            "reasoning": "JSON ?Œì‹± ?¤íŒ¨"
        }

    @staticmethod
    def parse_query_validation_response(response: str) -> Optional[Dict[str, Any]]:
        """ì¿¼ë¦¬ ê²€ì¦??‘ë‹µ ?Œì‹±"""
        json_str = QueryParser.extract_json(response)
        if json_str:
            parsed = QueryParser.parse_json_optional(json_str)
            if parsed and "is_valid" in parsed:
                return parsed
        return None

    @staticmethod
    def parse_llm_query_enhancement(response: str) -> Optional[Dict[str, Any]]:
        """LLM ì¿¼ë¦¬ ê°•í™” ?‘ë‹µ ?Œì‹±"""
        try:
            # JSON ì¶”ì¶œ (ì½”ë“œ ë¸”ë¡ ?´ë? ?ëŠ” ì§ì ‘ JSON)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # ì½”ë“œ ë¸”ë¡ ?†ì´ JSONë§??ˆëŠ” ê²½ìš°
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return None

            # JSON ?Œì‹±
            result = json.loads(json_str)

            # ?„ìˆ˜ ?„ë“œ ?•ì¸
            if not result.get("optimized_query"):
                return None

            # ê¸°ë³¸ê°??¤ì •
            enhanced = {
                "optimized_query": result.get("optimized_query", ""),
                "expanded_keywords": result.get("expanded_keywords", []),
                "keyword_variants": result.get("keyword_variants", []),
                "legal_terms": result.get("legal_terms", []),
                "reasoning": result.get("reasoning", "")
            }

            # ? íš¨??ê²€??
            if not enhanced["optimized_query"] or len(enhanced["optimized_query"]) > 500:
                return None

            return enhanced

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM query enhancement response: {e}")
            return None


class DocumentParser(ResponseParser):
    """ë¬¸ì„œ ë¶„ì„ ê´€???‘ë‹µ ?Œì„œ"""

    @staticmethod
    def parse_document_type_response(response: str) -> Dict[str, Any]:
        """ë¬¸ì„œ ? í˜• ?•ì¸ ?‘ë‹µ ?Œì‹±"""
        json_str = DocumentParser.extract_json(response)
        if json_str:
            parsed = DocumentParser.parse_json_safe(
                json_str,
                {
                    "document_type": "general_legal_document",
                    "confidence": 0.7,
                    "reasoning": "JSON ?Œì‹± ?¤íŒ¨"
                }
            )
            if "document_type" in parsed:
                return parsed

        return {
            "document_type": "general_legal_document",
            "confidence": 0.7,
            "reasoning": "JSON ?Œì‹± ?¤íŒ¨"
        }

    @staticmethod
    def parse_clause_extraction_response(response: str) -> Dict[str, Any]:
        """ì¡°í•­ ì¶”ì¶œ ?‘ë‹µ ?Œì‹±"""
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
        """ë¬¸ì œ???ë³„ ?‘ë‹µ ?Œì‹±"""
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
        """ë¬¸ì œ???ë³„ ?‘ë‹µ ?Œì‹± (?´ì „ ?¨ê³„ ì¶œë ¥ ?µí•©)"""
        json_str = DocumentParser.extract_json(response)
        if json_str:
            parsed = DocumentParser.parse_json_optional(json_str)
            if parsed and "issues" in parsed:
                # ?´ì „ ?¨ê³„ ê²°ê³¼(key_clauses)???¬í•¨
                if isinstance(prev_output, dict):
                    parsed["key_clauses"] = prev_output.get("key_clauses", [])
                    parsed["document_type"] = prev_output.get("document_type", "")
                return parsed
        return None

    @staticmethod
    def parse_improvement_recommendations_response(response: str) -> Optional[Dict[str, Any]]:
        """ê°œì„  ê¶Œê³  ?‘ë‹µ ?Œì‹±"""
        json_str = DocumentParser.extract_json(response)
        if json_str:
            parsed = DocumentParser.parse_json_optional(json_str)
            if parsed and "recommendations" in parsed:
                return parsed
        return None
