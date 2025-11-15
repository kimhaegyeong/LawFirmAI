# -*- coding: utf-8 -*-
"""
워크플로우 검증기
답변 품질, 검색 품질, 컨텍스트 품질 등을 검증
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkflowValidator:
    """워크플로우 검증기"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def check_general_principle_first(self, answer: str) -> Dict[str, Any]:
        """일반 법적 원칙이 먼저 설명되었는지 검증"""
        if not answer or not isinstance(answer, str):
            return {
                "has_general_principle": False,
                "general_principle_position": -1,
                "specific_case_position": -1,
                "principle_first": False,
                "score": 0.0
            }
        
        general_principle_keywords = [
            r'일반적인\s*법적\s*원칙',
            r'일반적인\s*원칙',
            r'법적\s*원칙',
            r'주의사항',
            r'주의해야\s*할',
            r'일반적으로',
            r'원칙적으로',
            r'민법\s*제\d+조',
            r'형법\s*제\d+조',
            r'관련\s*법령',
            r'법률에\s*따르면',
            r'법적으로',
            r'법률상',
            r'법률에\s*의하면',
            r'계약서\s*작성\s*시',
            r'계약서\s*작성\s*과\s*관련하여',
            r'계약서\s*작성\s*시\s*주의',
            r'계약\s*작성\s*시',
            r'계약\s*체결\s*시',
        ]
        
        specific_case_patterns = [
            r'\d{4}[가나다라마바사아자차카타파하]\d+',
            r'피고\s+[가-힣]+',
            r'이\s*사건',
        ]
        
        first_200 = answer[:200] if len(answer) > 200 else answer
        first_500 = answer[:500] if len(answer) > 500 else answer
        
        general_principle_position = -1
        specific_case_position = -1
        
        for pattern in general_principle_keywords:
            match = re.search(pattern, first_500, re.IGNORECASE)
            if match:
                general_principle_position = match.start()
                break
        
        for pattern in specific_case_patterns:
            match = re.search(pattern, first_200, re.IGNORECASE)
            if match:
                specific_case_position = match.start()
                break
        
        if specific_case_position < 0:
            for pattern in specific_case_patterns:
                match = re.search(pattern, first_500, re.IGNORECASE)
                if match:
                    specific_case_position = match.start()
                    break
        
        has_general_principle = general_principle_position >= 0
        principle_first = (
            has_general_principle and 
            (specific_case_position < 0 or general_principle_position < specific_case_position)
        )
        
        score = 0.0
        if has_general_principle:
            score += 0.5
        if principle_first:
            score += 0.5
        elif specific_case_position >= 0 and general_principle_position < 0:
            score = 0.0
        
        if specific_case_position >= 0 and specific_case_position < 200:
            score = max(0.0, score - 0.3)
        
        return {
            "has_general_principle": has_general_principle,
            "general_principle_position": general_principle_position,
            "specific_case_position": specific_case_position,
            "principle_first": principle_first,
            "score": score
        }
    
    def check_answer_structure(self, answer: str) -> Dict[str, Any]:
        """답변 구조가 올바른지 검증"""
        if not answer or not isinstance(answer, str):
            return {
                "has_general_principle": False,
                "has_precautions": False,
                "has_law_citations": False,
                "has_precedents": False,
                "has_practical_advice": False,
                "structure_score": 0.0,
                "missing_sections": []
            }
        
        section_patterns = {
            "general_principle": [
                r'일반적인\s*법적\s*원칙',
                r'일반적인\s*원칙',
                r'법적\s*원칙',
                r'원칙적으로',
                r'법률에\s*따르면',
                r'법적으로',
                r'법률상',
                r'관련\s*법령',
                r'민법\s*제\d+조',
                r'형법\s*제\d+조',
            ],
            "precautions": [
                r'주의사항',
                r'주의해야\s*할',
                r'주의할\s*점',
                r'유의사항',
                r'주의',
                r'유의',
                r'경고',
            ],
            "law_citations": [
                r'[가-힣]+법\s*제?\s*\d+\s*조',
                r'관련\s*법령',
                r'법령\s*조문',
                r'제\d+조',
                r'조',
            ],
            "precedents": [
                r'판례',
                r'대법원.*?\d{4}',
                r'법원.*?판결',
                r'판결',
                r'선례',
            ],
            "practical_advice": [
                r'실무\s*조언',
                r'실무적으로',
                r'구체적으로',
                r'행동\s*방안',
                r'권장',
                r'제안',
                r'방법',
            ]
        }
        
        section_scores = {}
        missing_sections = []
        
        for section_name, patterns in section_patterns.items():
            found = False
            for pattern in patterns:
                if re.search(pattern, answer, re.IGNORECASE):
                    found = True
                    break
            section_scores[section_name] = found
            if not found:
                missing_sections.append(section_name)
        
        structure_score = sum(1.0 if found else 0.0 for found in section_scores.values()) / len(section_scores)
        
        return {
            **section_scores,
            "structure_score": structure_score,
            "missing_sections": missing_sections
        }
    
    def check_has_sources(
        self,
        sources: List[Any],
        retrieved_docs: List[Any],
        legal_references: List[Any],
        legal_citations: List[Any]
    ) -> bool:
        """소스 존재 여부 확인"""
        if sources and isinstance(sources, list) and len(sources) > 0:
            valid_sources = [s for s in sources if s and (isinstance(s, dict) or isinstance(s, str))]
            if len(valid_sources) > 0:
                return True
        
        if retrieved_docs and isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
            valid_docs = [d for d in retrieved_docs if d and (isinstance(d, dict) or isinstance(d, str))]
            if len(valid_docs) > 0:
                return True
        
        if legal_references and isinstance(legal_references, list) and len(legal_references) > 0:
            return True
        
        if legal_citations and isinstance(legal_citations, list) and len(legal_citations) > 0:
            return True
        
        return False
    
    def evaluate_search_quality(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        query: str,
        query_type_str: str,
        search_params: Dict[str, Any],
        evaluate_semantic_func=None,
        evaluate_keyword_func=None
    ) -> Dict[str, Any]:
        """검색 품질 평가"""
        total_results = len(semantic_results) + len(keyword_results)
        should_skip_quality_eval = total_results >= 10 and len(semantic_results) >= 5
        
        if should_skip_quality_eval:
            semantic_quality = {
                "score": 0.8,
                "result_count": len(semantic_results),
                "needs_retry": False
            }
            keyword_quality = {
                "score": 0.7,
                "result_count": len(keyword_results),
                "needs_retry": False
            }
            self.logger.debug(f"Skipping quality evaluation (results: {total_results} >= 10)")
        else:
            if evaluate_semantic_func:
                semantic_quality = evaluate_semantic_func(
                    semantic_results=semantic_results,
                    query=query,
                    query_type=query_type_str,
                    min_results=search_params.get("semantic_k", 10) // 2
                )
            else:
                semantic_quality = self.evaluate_semantic_search_quality(
                    semantic_results=semantic_results,
                    query=query,
                    query_type=query_type_str,
                    min_results=search_params.get("semantic_k", 10) // 2
                )
            
            if evaluate_keyword_func:
                keyword_quality = evaluate_keyword_func(
                    keyword_results=keyword_results,
                    query=query,
                    query_type=query_type_str,
                    min_results=search_params.get("keyword_limit", 20) // 2
                )
            else:
                keyword_quality = self.evaluate_keyword_search_quality(
                    keyword_results=keyword_results,
                    query=query,
                    query_type=query_type_str,
                    min_results=search_params.get("keyword_limit", 20) // 2
                )
        
        total_results = len(semantic_results) + len(keyword_results)
        if total_results == 0:
            overall_quality = 0.05
        else:
            if len(semantic_results) > 0 and len(keyword_results) > 0:
                overall_quality = (semantic_quality["score"] + keyword_quality["score"]) / 2.0
            elif len(semantic_results) > 0:
                overall_quality = semantic_quality["score"]
            else:
                overall_quality = keyword_quality["score"]
                if semantic_quality["result_count"] == 0:
                    bonus = min(0.2, keyword_quality["score"] * 0.3)
                    overall_quality = min(1.0, overall_quality + bonus)
                    self.logger.info(
                        f"📊 [SEARCH QUALITY] Semantic search failed, applying bonus to keyword score: "
                        f"{keyword_quality['score']:.3f} -> {overall_quality:.3f} (+{bonus:.3f})"
                    )
            if overall_quality == 0:
                overall_quality = 0.1
        
        needs_retry = semantic_quality["needs_retry"] or keyword_quality["needs_retry"]
        
        return {
            "semantic_quality": semantic_quality,
            "keyword_quality": keyword_quality,
            "overall_quality": overall_quality,
            "needs_retry": needs_retry
        }
    
    def evaluate_semantic_search_quality(
        self,
        semantic_results: List[Dict[str, Any]],
        query: str,
        query_type: str,
        min_results: int = 5
    ) -> Dict[str, Any]:
        """의미적 검색 품질 평가"""
        quality = {
            "score": 0.0,
            "result_count": len(semantic_results),
            "avg_relevance": 0.0,
            "diversity_ratio": 0.0,
            "query_match": 0.0,
            "needs_retry": False,
            "issues": []
        }
        
        if not semantic_results:
            quality["needs_retry"] = True
            quality["issues"].append("검색 결과가 없음")
            return quality
        
        result_count_score = min(1.0, len(semantic_results) / max(1, min_results))
        quality["result_count"] = len(semantic_results)
        
        relevance_scores = [
            doc.get("relevance_score", doc.get("score", 0.0))
            for doc in semantic_results
            if doc.get("relevance_score") or doc.get("score")
        ]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        quality["avg_relevance"] = avg_relevance
        
        seen_contents = set()
        unique_count = 0
        for doc in semantic_results:
            content_preview = doc.get("content", "")[:100]
            if content_preview and content_preview not in seen_contents:
                seen_contents.add(content_preview)
                unique_count += 1
        
        diversity_ratio = unique_count / len(semantic_results) if semantic_results else 0.0
        quality["diversity_ratio"] = diversity_ratio
        
        query_words = set(query.lower().split())
        match_count = 0
        for doc in semantic_results:
            doc_content = doc.get("content", "").lower()
            doc_words = set(doc_content.split())
            if query_words and doc_words:
                overlap = len(query_words.intersection(doc_words))
                if overlap > 0:
                    match_count += 1
        
        query_match = match_count / len(semantic_results) if semantic_results else 0.0
        quality["query_match"] = query_match
        
        quality_score = (
            result_count_score * 0.30 +
            avg_relevance * 0.35 +
            diversity_ratio * 0.15 +
            query_match * 0.20
        )
        if len(semantic_results) > 0:
            quality_score = max(quality_score, 0.1)
        if avg_relevance == 0 and len(semantic_results) > 0:
            quality_score = max(quality_score, 0.15)
        quality["score"] = quality_score
        
        needs_retry = (
            result_count_score < 0.5 or
            avg_relevance < 0.6 or
            query_match < 0.3
        )
        quality["needs_retry"] = needs_retry
        
        if needs_retry:
            if result_count_score < 0.5:
                quality["issues"].append(f"결과 수 부족: {len(semantic_results)}개")
            if avg_relevance < 0.6:
                quality["issues"].append(f"평균 관련성 점수 낮음: {avg_relevance:.2f}")
            if query_match < 0.3:
                quality["issues"].append(f"쿼리 일치도 낮음: {query_match:.2f}")
        
        return quality
    
    def evaluate_keyword_search_quality(
        self,
        keyword_results: List[Dict[str, Any]],
        query: str,
        query_type: str,
        min_results: int = 3
    ) -> Dict[str, Any]:
        """키워드 검색 품질 평가"""
        quality = {
            "score": 0.0,
            "result_count": len(keyword_results),
            "avg_relevance": 0.0,
            "category_match": 0.0,
            "legal_citation_ratio": 0.0,
            "needs_retry": False,
            "issues": []
        }
        
        if not keyword_results:
            quality["needs_retry"] = True
            quality["issues"].append("검색 결과가 없음")
            return quality
        
        result_count_score = min(1.0, len(keyword_results) / max(1, min_results))
        quality["result_count"] = len(keyword_results)
        
        relevance_scores = [
            doc.get("relevance_score", doc.get("score", 0.0))
            for doc in keyword_results
            if doc.get("relevance_score") or doc.get("score")
        ]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        quality["avg_relevance"] = avg_relevance
        
        category_match_count = sum(1 for doc in keyword_results if doc.get("category_boost", 1.0) > 1.0)
        category_match = category_match_count / len(keyword_results) if keyword_results else 0.0
        quality["category_match"] = category_match
        
        legal_citation_count = 0
        for doc in keyword_results:
            content = doc.get("content", "")
            if re.search(r'[가-힣]+법\s*제?\s*\d+\s*조', content):
                legal_citation_count += 1
        
        legal_citation_ratio = legal_citation_count / len(keyword_results) if keyword_results else 0.0
        quality["legal_citation_ratio"] = legal_citation_ratio
        
        quality_score = (
            result_count_score * 0.30 +
            avg_relevance * 0.30 +
            category_match * 0.20 +
            legal_citation_ratio * 0.20
        )
        if len(keyword_results) > 0:
            quality_score = max(quality_score, 0.1)
        if avg_relevance == 0 and len(keyword_results) > 0:
            quality_score = max(quality_score, 0.15)
        if legal_citation_ratio > 0.3:
            bonus = min(0.1, legal_citation_ratio * 0.2)
            quality_score = min(1.0, quality_score + bonus)
        quality["score"] = quality_score
        
        needs_retry = (
            result_count_score < 0.5 or
            avg_relevance < 0.6 or
            legal_citation_ratio < 0.2
        )
        quality["needs_retry"] = needs_retry
        
        if needs_retry:
            if result_count_score < 0.5:
                quality["issues"].append(f"결과 수 부족: {len(keyword_results)}개")
            if avg_relevance < 0.6:
                quality["issues"].append(f"평균 관련성 점수 낮음: {avg_relevance:.2f}")
            if legal_citation_ratio < 0.2:
                quality["issues"].append(f"법률 조항 포함도 낮음: {legal_citation_ratio:.2f}")
        
        return quality

