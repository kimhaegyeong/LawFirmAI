# -*- coding: utf-8 -*-
"""
?ˆì§ˆ ê²€ì¦?ëª¨ë“ˆ
ë¦¬íŒ©? ë§: legal_workflow_enhanced.py?ì„œ ê²€ì¦?ë¡œì§ ë¶„ë¦¬
"""

import re
import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class ContextValidator:
    """ì»¨í…?¤íŠ¸ ?ˆì§ˆ ê²€ì¦?""

    @staticmethod
    def calculate_relevance(context_text: str, query: str, semantic_calculator=None) -> float:
        """
        ì»¨í…?¤íŠ¸ ê´€?¨ì„± ê³„ì‚°

        Args:
            context_text: ì»¨í…?¤íŠ¸ ?ìŠ¤??
            query: ì§ˆë¬¸
            semantic_calculator: ?˜ë???? ì‚¬??ê³„ì‚° ?¨ìˆ˜ (? íƒ??

        Returns:
            ê´€?¨ì„± ?ìˆ˜ (0.0-1.0)
        """
        try:
            if not context_text:
                return 0.0

            # ?˜ë???? ì‚¬??ê³„ì‚° ?œë„
            if semantic_calculator and callable(semantic_calculator):
                try:
                    return semantic_calculator(query, context_text)
                except Exception as e:
                    logger.debug(f"Semantic relevance calculation failed: {e}")

            # ?´ë°±: ?¤ì›Œ??ê¸°ë°˜ ? ì‚¬??
            query_words = set(query.lower().split())
            context_words = set(context_text.lower().split())

            if not query_words or not context_words:
                return 0.0

            overlap = len(query_words.intersection(context_words))
            relevance = overlap / max(1, len(query_words))

            return min(1.0, relevance)

        except Exception as e:
            logger.warning(f"Context relevance calculation failed: {e}")
            return 0.5  # ê¸°ë³¸ê°?

    @staticmethod
    def calculate_coverage(
        context_text: str,
        extracted_keywords: List[str],
        legal_references: List[str],
        citations: List[Any]
    ) -> float:
        """
        ?•ë³´ ì»¤ë²„ë¦¬ì? ê³„ì‚° - ?µì‹¬ ?¤ì›Œ???¬í•¨??

        Args:
            context_text: ì»¨í…?¤íŠ¸ ?ìŠ¤??
            extracted_keywords: ì¶”ì¶œ???¤ì›Œ??ëª©ë¡
            legal_references: ë²•ë¥  ì°¸ì¡° ëª©ë¡
            citations: ?¸ìš© ëª©ë¡

        Returns:
            ì»¤ë²„ë¦¬ì? ?ìˆ˜ (0.0-1.0)
        """
        try:
            if not context_text and not legal_references and not citations:
                return 0.0

            coverage_scores = []

            # 1. ì¶”ì¶œ???¤ì›Œ??ì»¤ë²„ë¦¬ì?
            if extracted_keywords:
                context_lower = context_text.lower()
                keyword_matches = sum(1 for kw in extracted_keywords
                                    if isinstance(kw, str) and kw.lower() in context_lower)
                keyword_coverage = keyword_matches / max(1, len(extracted_keywords))
                coverage_scores.append(keyword_coverage)

            # 2. ì§ˆë¬¸ ?¤ì›Œ??ì»¤ë²„ë¦¬ì?
            if context_text:
                # ì§ˆë¬¸ ?¤ì›Œ?œëŠ” extracted_keywords???¬í•¨?˜ì–´ ?ˆì„ ???ˆìœ¼ë¯€ë¡?ë³„ë„ ê³„ì‚° ?ëµ
                pass

            # 3. ë²•ë¥  ì°¸ì¡° ?¬í•¨??
            if legal_references:
                ref_coverage = min(1.0, len(legal_references) / max(1, 5))  # ìµœë? 5ê°?ê¸°ì?
                coverage_scores.append(ref_coverage)

            # 4. ?¸ìš© ?¬í•¨??
            if citations:
                citation_coverage = min(1.0, len(citations) / max(1, 5))  # ìµœë? 5ê°?ê¸°ì?
                coverage_scores.append(citation_coverage)

            # ?‰ê·  ê³„ì‚°
            if coverage_scores:
                return sum(coverage_scores) / len(coverage_scores)
            else:
                return 0.5  # ê¸°ë³¸ê°?

        except Exception as e:
            logger.warning(f"Coverage calculation failed: {e}")
            return 0.5

    @staticmethod
    def validate_context_quality(
        context: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        calculate_relevance_func: callable = None,
        calculate_coverage_func: callable = None
    ) -> Dict[str, Any]:
        """
        ì»¨í…?¤íŠ¸ ?ˆì§ˆ ê²€ì¦?

        Args:
            context: ì»¨í…?¤íŠ¸ ?•ì…”?ˆë¦¬
            query: ì§ˆë¬¸
            query_type: ì§ˆë¬¸ ? í˜•
            extracted_keywords: ì¶”ì¶œ???¤ì›Œ??ëª©ë¡
            calculate_relevance_func: ê´€?¨ì„± ê³„ì‚° ?¨ìˆ˜ (? íƒ??
            calculate_coverage_func: ì»¤ë²„ë¦¬ì? ê³„ì‚° ?¨ìˆ˜ (? íƒ??

        Returns:
            ê²€ì¦?ê²°ê³¼ ?•ì…”?ˆë¦¬
        """
        try:
            context_text = context.get("context", "")
            legal_references = context.get("legal_references", [])
            citations = context.get("citations", [])

            # ê´€?¨ì„± ?ìˆ˜ ê³„ì‚°
            if calculate_relevance_func:
                relevance_score = calculate_relevance_func(context_text, query)
            else:
                relevance_score = ContextValidator.calculate_relevance(context_text, query)

            # ì»¤ë²„ë¦¬ì? ?ìˆ˜ ê³„ì‚°
            if calculate_coverage_func:
                coverage_score = calculate_coverage_func(context_text, extracted_keywords, legal_references, citations)
            else:
                coverage_score = ContextValidator.calculate_coverage(
                    context_text, extracted_keywords, legal_references, citations
                )

            # ì¶©ë¶„???ìˆ˜ ê³„ì‚° (ë¬¸ì„œ ê°œìˆ˜, ê¸¸ì´ ??
            document_count = context.get("document_count", 0)
            context_length = context.get("context_length", 0)

            # ìµœì†Œ ë¬¸ì„œ ê°œìˆ˜ ?•ì¸
            min_docs_required = 2 if query_type != "simple" else 1
            doc_sufficiency = min(1.0, document_count / max(1, min_docs_required))

            # ìµœì†Œ ì»¨í…?¤íŠ¸ ê¸¸ì´ ?•ì¸ (500???´ìƒ ê¶Œì¥)
            length_sufficiency = min(1.0, context_length / max(1, 500))

            sufficiency_score = (doc_sufficiency * 0.6 + length_sufficiency * 0.4)

            # ì¢…í•© ?ìˆ˜
            overall_score = (relevance_score * 0.4 + coverage_score * 0.4 + sufficiency_score * 0.2)

            # ?„ë½ ?•ë³´ ?•ì¸
            missing_info = []
            if coverage_score < 0.5:
                missing_info.append("?µì‹¬ ?¤ì›Œ??ì»¤ë²„ë¦¬ì? ë¶€ì¡?)
            if relevance_score < 0.5:
                missing_info.append("ì§ˆë¬¸ ê´€?¨ì„± ë¶€ì¡?)
            if sufficiency_score < 0.6:
                missing_info.append("ì»¨í…?¤íŠ¸ ì¶©ë¶„??ë¶€ì¡?)

            is_sufficient = overall_score >= 0.6
            needs_expansion = overall_score < 0.6 or len(missing_info) > 0

            validation_result = {
                "relevance_score": relevance_score,
                "coverage_score": coverage_score,
                "sufficiency_score": sufficiency_score,
                "overall_score": overall_score,
                "missing_information": missing_info,
                "is_sufficient": is_sufficient,
                "needs_expansion": needs_expansion,
                "document_count": document_count,
                "context_length": context_length
            }

            return validation_result

        except Exception as e:
            logger.warning(f"Context validation failed: {e}")
            return {
                "relevance_score": 0.5,
                "coverage_score": 0.5,
                "sufficiency_score": 0.5,
                "overall_score": 0.5,
                "missing_information": [],
                "is_sufficient": True,
                "needs_expansion": False
            }


class AnswerValidator:
    """?µë? ?ˆì§ˆ ê²€ì¦?""

    @staticmethod
    def validate_answer_uses_context(
        answer: str,
        context: Dict[str, Any],
        query: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        ?µë???ì»¨í…?¤íŠ¸ë¥??¬ìš©?˜ëŠ”ì§€ ê²€ì¦?

        Args:
            answer: ?µë? ?ìŠ¤??
            context: ì»¨í…?¤íŠ¸ ?•ì…”?ˆë¦¬
            query: ì§ˆë¬¸
            retrieved_docs: ê²€?‰ëœ ë¬¸ì„œ ëª©ë¡ (? íƒ??

        Returns:
            ê²€ì¦?ê²°ê³¼ ?•ì…”?ˆë¦¬
        """
        try:
            if not answer:
                return {
                    "uses_context": False,
                    "coverage_score": 0.0,
                    "citation_count": 0,
                    "has_document_references": False,
                    "needs_regeneration": True,
                    "missing_key_info": []
                }

            answer_lower = answer.lower()
            context_text = context.get("context", "").lower()
            legal_references = context.get("legal_references", [])
            citations = context.get("citations", [])

            # ê²€?‰ëœ ë¬¸ì„œ?ì„œ ì¶œì²˜ ì¶”ì¶œ
            document_sources = []
            if retrieved_docs:
                for doc in retrieved_docs[:10]:
                    if isinstance(doc, dict):
                        source = doc.get("source", "")
                        if source and source not in document_sources:
                            document_sources.append(source.lower())

            # 1. ì»¨í…?¤íŠ¸ ?¤ì›Œ???¬í•¨??ê³„ì‚°
            context_words = set(context_text.split())
            answer_words = set(answer_lower.split())

            keyword_coverage = 0.0
            if context_words and answer_words:
                overlap = len(context_words.intersection(answer_words))
                keyword_coverage = overlap / max(1, min(len(context_words), 100))

            # 2. ë²•ë¥  ì¡°í•­/?ë? ?¸ìš© ?¬í•¨ ?¬ë? ?•ì¸
            citation_pattern = r'[ê°€-??+ë²?s*??\s*\d+\s*ì¡?\[ë²•ë ¹:\s*[^\]]+\]'
            citations_in_answer = len(re.findall(citation_pattern, answer))

            precedent_pattern = r'?€ë²•ì›|ë²•ì›.*\d{4}[?¤ë‚˜ë§?\d+|\[?ë?:\s*[^\]]+\]'
            precedents_in_answer = len(re.findall(precedent_pattern, answer))

            # ë¬¸ì„œ ?¸ìš© ?¨í„´ ?•ì¸
            document_citation_pattern = r'\[ë¬¸ì„œ:\s*[^\]]+\]'
            document_citations = len(re.findall(document_citation_pattern, answer))

            total_citations_in_answer = citations_in_answer + precedents_in_answer + document_citations

            # 3. ê²€?‰ëœ ë¬¸ì„œ??ì¶œì²˜ê°€ ?µë????¬í•¨?˜ì–´ ?ˆëŠ”ì§€ ?•ì¸
            has_document_references = False
            if document_sources:
                for source in document_sources:
                    source_keywords = source.split()[:3]
                    if any(keyword in answer_lower for keyword in source_keywords if len(keyword) > 2):
                        has_document_references = True
                        break

            # ì»¨í…?¤íŠ¸?ì„œ ì¶”ì¶œ???¸ìš© ?•ë³´?€ ë¹„êµ
            expected_citations = []
            for ref in legal_references[:5]:
                if isinstance(ref, str):
                    expected_citations.append(ref)

            for cit in citations[:5]:
                if isinstance(cit, dict):
                    expected_citations.append(cit.get("text", ""))
                elif isinstance(cit, str):
                    expected_citations.append(cit)

            found_citations = 0
            missing_citations = []
            for expected in expected_citations:
                if expected and any(keyword in answer for keyword in expected.split()[:3]):
                    found_citations += 1
                else:
                    missing_citations.append(expected)

            citation_coverage = found_citations / max(1, len(expected_citations)) if expected_citations else 0.5

            # 4. ?µì‹¬ ê°œë… ?¬í•¨ ?¬ë?
            context_key_concepts = []
            if context_text:
                key_terms = ["ë²?, "ì¡?, "?ë?", "ê·œì •", "?ˆì°¨", "?”ê±´", "?¨ë ¥"]
                for term in key_terms:
                    if term in context_text:
                        context_key_concepts.append(term)

            concept_coverage = 0.0
            if context_key_concepts:
                found_concepts = sum(1 for concept in context_key_concepts if concept in answer_lower)
                concept_coverage = found_concepts / len(context_key_concepts)

            # 5. ì¢…í•© ?ìˆ˜
            coverage_score = (
                keyword_coverage * 0.4 +
                citation_coverage * 0.4 +
                concept_coverage * 0.2
            )

            uses_context = coverage_score >= 0.3
            needs_regeneration = coverage_score < 0.3 or (expected_citations and found_citations == 0)

            validation_result = {
                "uses_context": uses_context,
                "coverage_score": coverage_score,
                "keyword_coverage": keyword_coverage,
                "citation_coverage": citation_coverage,
                "concept_coverage": concept_coverage,
                "citations_found": found_citations,
                "citations_expected": len(expected_citations),
                "citation_count": total_citations_in_answer,
                "citations_in_answer": citations_in_answer,
                "precedents_in_answer": precedents_in_answer,
                "document_citations": document_citations,
                "total_citations_in_answer": total_citations_in_answer,
                "has_document_references": has_document_references,
                "document_sources_count": len(document_sources),
                "needs_regeneration": needs_regeneration,
                "missing_key_info": missing_citations[:5]
            }

            return validation_result

        except Exception as e:
            logger.warning(f"Answer-context validation failed: {e}")
            return {
                "uses_context": True,
                "coverage_score": 0.5,
                "needs_regeneration": False,
                "missing_key_info": []
            }

    @staticmethod
    def validate_answer_source_verification(
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        ?µë????´ìš©??ê²€?‰ëœ ë¬¸ì„œ??ê¸°ë°˜?˜ëŠ”ì§€ ê²€ì¦?(Hallucination ë°©ì?)

        Args:
            answer: ê²€ì¦í•  ?µë? ?ìŠ¤??
            retrieved_docs: ê²€?‰ëœ ë¬¸ì„œ ëª©ë¡
            query: ?ë³¸ ì§ˆì˜

        Returns:
            ê²€ì¦?ê²°ê³¼ ?•ì…”?ˆë¦¬
            {
                "is_grounded": bool,
                "grounding_score": float,
                "unverified_sections": List[str],
                "source_coverage": float,
                "needs_review": bool
            }
        """
        import re
        from difflib import SequenceMatcher

        if not answer or not retrieved_docs:
            return {
                "is_grounded": False,
                "grounding_score": 0.0,
                "unverified_sections": [answer] if answer else [],
                "source_coverage": 0.0,
                "needs_review": True,
                "error": "?µë? ?ëŠ” ê²€??ê²°ê³¼ê°€ ?†ìŠµ?ˆë‹¤."
            }

        # 1. ê²€?‰ëœ ë¬¸ì„œ?ì„œ ëª¨ë“  ?ìŠ¤??ì¶”ì¶œ
        source_texts = []
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                content = (
                    doc.get("content") or
                    doc.get("text") or
                    doc.get("content_text") or
                    ""
                )
                if content and len(content.strip()) > 50:
                    source_texts.append(content.lower())

        if not source_texts:
            return {
                "is_grounded": False,
                "grounding_score": 0.0,
                "unverified_sections": [],
                "source_coverage": 0.0,
                "needs_review": True,
                "error": "ê²€?‰ëœ ë¬¸ì„œ???´ìš©???†ìŠµ?ˆë‹¤."
            }

        # 2. ?µë???ë¬¸ì¥ ?¨ìœ„ë¡?ë¶„ë¦¬
        answer_sentences = re.split(r'[.!??‚ï¼ï¼?\s+', answer)
        answer_sentences = [s.strip() for s in answer_sentences if len(s.strip()) > 20]

        # 3. ê°?ë¬¸ì¥??ê²€?‰ëœ ë¬¸ì„œ??ê¸°ë°˜?˜ëŠ”ì§€ ê²€ì¦?
        verified_sentences = []
        unverified_sentences = []

        for sentence in answer_sentences:
            sentence_lower = sentence.lower()

            # ë¬¸ì¥???µì‹¬ ?¤ì›Œ??ì¶”ì¶œ (ë¶ˆìš©???œê±°)
            stopwords = {'??, '?€', '??, 'ê°€', '??, 'ë¥?, '??, '??, '?€', 'ê³?, 'ë¡?, '?¼ë¡œ', '?ì„œ', '??, 'ë§?, 'ë¶€??, 'ê¹Œì?'}
            sentence_words = [w for w in re.findall(r'[ê°€-??+', sentence_lower) if len(w) > 1 and w not in stopwords]

            if not sentence_words:
                continue

            # ê°??ŒìŠ¤ ?ìŠ¤?¸ì? ? ì‚¬??ê³„ì‚°
            max_similarity = 0.0
            best_match_source = None
            matched_keywords_count = 0

            for source_text in source_texts:
                # ?¤ì›Œ??ë§¤ì¹­ ?ìˆ˜
                matched_keywords = sum(1 for word in sentence_words if word in source_text)
                keyword_score = matched_keywords / len(sentence_words) if sentence_words else 0.0

                # ë¬¸ì¥ ? ì‚¬??(SequenceMatcher ?¬ìš©)
                similarity = SequenceMatcher(None, sentence_lower[:100], source_text[:1000]).ratio()

                # ì¢…í•© ?ìˆ˜ (?¤ì›Œ??ë§¤ì¹­ + ? ì‚¬??
                combined_score = (keyword_score * 0.6) + (similarity * 0.4)

                if combined_score > max_similarity:
                    max_similarity = combined_score
                    matched_keywords_count = matched_keywords
                    best_match_source = source_text[:100]  # ?”ë²„ê¹…ìš©

            # ê²€ì¦?ê¸°ì?: 30% ?´ìƒ ? ì‚¬?˜ê±°???µì‹¬ ?¤ì›Œ??50% ?´ìƒ ë§¤ì¹­
            keyword_coverage = matched_keywords_count / len(sentence_words) if sentence_words else 0.0
            if max_similarity >= 0.3 or keyword_coverage >= 0.5:
                verified_sentences.append({
                    "sentence": sentence,
                    "similarity": max_similarity,
                    "source_preview": best_match_source
                })
            else:
                # ë²•ë ¹ ?¸ìš©?´ë‚˜ ?¼ë°˜?ì¸ ë©´ì±… ì¡°í•­?€ ?œì™¸
                if not (re.search(r'\[ë²•ë ¹:\s*[^\]]+\]', sentence) or
                       re.search(r'ë³?s*?µë??€\s*?¼ë°˜?ì¸', sentence) or
                       re.search(r'ë³€?¸ì‚¬?€\s*ì§ì ‘\s*?ë‹´', sentence)):
                    unverified_sentences.append({
                        "sentence": sentence[:100],
                        "similarity": max_similarity,
                        "keywords": sentence_words[:5],
                        "keyword_coverage": keyword_coverage
                    })

        # 4. ì¢…í•© ê²€ì¦??ìˆ˜ ê³„ì‚°
        total_sentences = len(answer_sentences)
        verified_count = len(verified_sentences)

        grounding_score = verified_count / total_sentences if total_sentences > 0 else 0.0
        source_coverage = len(set([s["source_preview"] for s in verified_sentences if s.get("source_preview")])) / len(source_texts) if source_texts else 0.0

        # 5. ê²€ì¦??µê³¼ ê¸°ì?: 80% ?´ìƒ ë¬¸ì¥??ê²€ì¦ë¨
        is_grounded = grounding_score >= 0.8

        # 6. ? ë¢°??ì¡°ì • (ê²€ì¦ë˜ì§€ ?Šì? ë¬¸ì¥??ë§ìœ¼ë©?? ë¢°??ê°ì†Œ)
        confidence_penalty = len(unverified_sentences) * 0.05  # ë¬¸ì¥??5% ê°ì†Œ

        return {
            "is_grounded": is_grounded,
            "grounding_score": grounding_score,
            "verified_sentences": verified_sentences[:5],  # ?˜í”Œ
            "unverified_sentences": unverified_sentences,
            "unverified_count": len(unverified_sentences),
            "source_coverage": source_coverage,
            "needs_review": not is_grounded or len(unverified_sentences) > 3,
            "confidence_penalty": min(confidence_penalty, 0.3),  # ìµœë? 30% ê°ì†Œ
            "total_sentences": total_sentences,
            "verified_count": verified_count
        }


class SearchValidator:
    """ê²€???ˆì§ˆ ê²€ì¦?""

    @staticmethod
    def validate_search_quality(
        search_results: List[Dict[str, Any]],
        query: str,
        query_type: str
    ) -> Dict[str, Any]:
        """
        ê²€???ˆì§ˆ ê²€ì¦?

        Args:
            search_results: ê²€??ê²°ê³¼ ëª©ë¡
            query: ê²€??ì¿¼ë¦¬
            query_type: ì§ˆë¬¸ ? í˜•

        Returns:
            ê²€ì¦?ê²°ê³¼ ?•ì…”?ˆë¦¬
        """
        try:
            if not search_results:
                return {
                    "is_valid": False,
                    "quality_score": 0.0,
                    "doc_count": 0,
                    "avg_relevance": 0.0,
                    "issues": ["ê²€??ê²°ê³¼ê°€ ?†ìŠµ?ˆë‹¤"],
                    "recommendations": ["ê²€??ì¿¼ë¦¬ë¥??˜ì •?˜ê±°??ê²€??ë²”ìœ„ë¥??•ë??˜ì„¸??]
                }

            # ë¬¸ì„œ ê°œìˆ˜ ?•ì¸
            doc_count = len(search_results)
            min_docs_required = 2 if query_type != "simple" else 1

            # ?‰ê·  ê´€?¨ë„ ?ìˆ˜ ê³„ì‚°
            relevance_scores = []
            for doc in search_results:
                if isinstance(doc, dict):
                    score = doc.get("relevance_score") or doc.get("final_weighted_score", 0.0)
                    relevance_scores.append(score)

            avg_relevance = sum(relevance_scores) / max(1, len(relevance_scores)) if relevance_scores else 0.0

            # ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°
            doc_adequacy = min(1.0, doc_count / max(1, min_docs_required))
            relevance_adequacy = avg_relevance

            quality_score = (doc_adequacy * 0.4 + relevance_adequacy * 0.6)

            # ë¬¸ì œ???•ì¸
            issues = []
            if doc_count < min_docs_required:
                issues.append(f"ê²€??ê²°ê³¼ê°€ ë¶€ì¡±í•©?ˆë‹¤ ({doc_count}/{min_docs_required})")
            if avg_relevance < 0.3:
                issues.append(f"?‰ê·  ê´€?¨ë„ê°€ ??Šµ?ˆë‹¤ ({avg_relevance:.2f})")

            # ê¶Œê³ ?¬í•­ ?ì„±
            recommendations = []
            if doc_count < min_docs_required:
                recommendations.append("ê²€??ì¿¼ë¦¬ë¥??•ì¥?˜ê±°??ê²€??ë²”ìœ„ë¥??“íˆ?¸ìš”")
            if avg_relevance < 0.3:
                recommendations.append("ê²€??ì¿¼ë¦¬ë¥???êµ¬ì²´?ìœ¼ë¡??‘ì„±?˜ê±°???¤ë¥¸ ?¤ì›Œ?œë? ?œë„?˜ì„¸??)

            is_valid = doc_count >= min_docs_required and avg_relevance >= 0.3

            return {
                "is_valid": is_valid,
                "quality_score": quality_score,
                "doc_count": doc_count,
                "avg_relevance": avg_relevance,
                "min_docs_required": min_docs_required,
                "issues": issues,
                "recommendations": recommendations
            }

        except Exception as e:
            logger.warning(f"Search quality validation failed: {e}")
            return {
                "is_valid": False,
                "quality_score": 0.0,
                "doc_count": 0,
                "avg_relevance": 0.0,
                "issues": [f"ê²€ì¦?ì¤??¤ë¥˜ ë°œìƒ: {e}"],
                "recommendations": []
            }
