# -*- coding: utf-8 -*-
"""
품질 검증 모듈
리팩토링: legal_workflow_enhanced.py에서 검증 로직 분리
"""

import re
import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class ContextValidator:
    """컨텍스트 품질 검증"""

    @staticmethod
    def calculate_relevance(context_text: str, query: str, semantic_calculator=None) -> float:
        """
        컨텍스트 관련성 계산

        Args:
            context_text: 컨텍스트 텍스트
            query: 질문
            semantic_calculator: 의미적 유사도 계산 함수 (선택적)

        Returns:
            관련성 점수 (0.0-1.0)
        """
        try:
            if not context_text:
                return 0.0

            # 의미적 유사도 계산 시도
            if semantic_calculator and callable(semantic_calculator):
                try:
                    return semantic_calculator(query, context_text)
                except Exception as e:
                    logger.debug(f"Semantic relevance calculation failed: {e}")

            # 폴백: 키워드 기반 유사도
            query_words = set(query.lower().split())
            context_words = set(context_text.lower().split())

            if not query_words or not context_words:
                return 0.0

            overlap = len(query_words.intersection(context_words))
            relevance = overlap / max(1, len(query_words))

            return min(1.0, relevance)

        except Exception as e:
            logger.warning(f"Context relevance calculation failed: {e}")
            return 0.5  # 기본값

    @staticmethod
    def calculate_coverage(
        context_text: str,
        extracted_keywords: List[str],
        legal_references: List[str],
        citations: List[Any]
    ) -> float:
        """
        정보 커버리지 계산 - 핵심 키워드 포함도

        Args:
            context_text: 컨텍스트 텍스트
            extracted_keywords: 추출된 키워드 목록
            legal_references: 법률 참조 목록
            citations: 인용 목록

        Returns:
            커버리지 점수 (0.0-1.0)
        """
        try:
            if not context_text and not legal_references and not citations:
                return 0.0

            coverage_scores = []

            # 1. 추출된 키워드 커버리지
            if extracted_keywords:
                context_lower = context_text.lower()
                keyword_matches = sum(1 for kw in extracted_keywords
                                    if isinstance(kw, str) and kw.lower() in context_lower)
                keyword_coverage = keyword_matches / max(1, len(extracted_keywords))
                coverage_scores.append(keyword_coverage)

            # 2. 질문 키워드 커버리지
            if context_text:
                # 질문 키워드는 extracted_keywords에 포함되어 있을 수 있으므로 별도 계산 생략
                pass

            # 3. 법률 참조 포함도
            if legal_references:
                ref_coverage = min(1.0, len(legal_references) / max(1, 5))  # 최대 5개 기준
                coverage_scores.append(ref_coverage)

            # 4. 인용 포함도
            if citations:
                citation_coverage = min(1.0, len(citations) / max(1, 5))  # 최대 5개 기준
                coverage_scores.append(citation_coverage)

            # 평균 계산
            if coverage_scores:
                return sum(coverage_scores) / len(coverage_scores)
            else:
                return 0.5  # 기본값

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
        컨텍스트 품질 검증

        Args:
            context: 컨텍스트 딕셔너리
            query: 질문
            query_type: 질문 유형
            extracted_keywords: 추출된 키워드 목록
            calculate_relevance_func: 관련성 계산 함수 (선택적)
            calculate_coverage_func: 커버리지 계산 함수 (선택적)

        Returns:
            검증 결과 딕셔너리
        """
        try:
            context_text = context.get("context", "")
            legal_references = context.get("legal_references", [])
            citations = context.get("citations", [])

            # 관련성 점수 계산
            if calculate_relevance_func:
                relevance_score = calculate_relevance_func(context_text, query)
            else:
                relevance_score = ContextValidator.calculate_relevance(context_text, query)

            # 커버리지 점수 계산
            if calculate_coverage_func:
                coverage_score = calculate_coverage_func(context_text, extracted_keywords, legal_references, citations)
            else:
                coverage_score = ContextValidator.calculate_coverage(
                    context_text, extracted_keywords, legal_references, citations
                )

            # 충분성 점수 계산 (문서 개수, 길이 등)
            document_count = context.get("document_count", 0)
            context_length = context.get("context_length", 0)

            # 최소 문서 개수 확인
            min_docs_required = 2 if query_type != "simple" else 1
            doc_sufficiency = min(1.0, document_count / max(1, min_docs_required))

            # 최소 컨텍스트 길이 확인 (500자 이상 권장)
            length_sufficiency = min(1.0, context_length / max(1, 500))

            sufficiency_score = (doc_sufficiency * 0.6 + length_sufficiency * 0.4)

            # 종합 점수
            overall_score = (relevance_score * 0.4 + coverage_score * 0.4 + sufficiency_score * 0.2)

            # 누락 정보 확인
            missing_info = []
            if coverage_score < 0.5:
                missing_info.append("핵심 키워드 커버리지 부족")
            if relevance_score < 0.5:
                missing_info.append("질문 관련성 부족")
            if sufficiency_score < 0.6:
                missing_info.append("컨텍스트 충분성 부족")

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
    """답변 품질 검증"""

    @staticmethod
    def validate_answer_uses_context(
        answer: str,
        context: Dict[str, Any],
        query: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        답변이 컨텍스트를 사용하는지 검증

        Args:
            answer: 답변 텍스트
            context: 컨텍스트 딕셔너리
            query: 질문
            retrieved_docs: 검색된 문서 목록 (선택적)

        Returns:
            검증 결과 딕셔너리
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

            # 검색된 문서에서 출처 추출
            document_sources = []
            if retrieved_docs:
                for doc in retrieved_docs[:10]:
                    if isinstance(doc, dict):
                        source = doc.get("source", "")
                        if source and source not in document_sources:
                            document_sources.append(source.lower())

            # 1. 컨텍스트 키워드 포함도 계산
            context_words = set(context_text.split())
            answer_words = set(answer_lower.split())

            keyword_coverage = 0.0
            if context_words and answer_words:
                overlap = len(context_words.intersection(answer_words))
                keyword_coverage = overlap / max(1, min(len(context_words), 100))

            # 2. 법률 조항/판례 인용 포함 여부 확인
            citation_pattern = r'[가-힣]+법\s*제?\s*\d+\s*조|\[법령:\s*[^\]]+\]'
            citations_in_answer = len(re.findall(citation_pattern, answer))

            precedent_pattern = r'대법원|법원.*\d{4}[다나마]\d+|\[판례:\s*[^\]]+\]'
            precedents_in_answer = len(re.findall(precedent_pattern, answer))

            # 문서 인용 패턴 확인
            document_citation_pattern = r'\[문서:\s*[^\]]+\]'
            document_citations = len(re.findall(document_citation_pattern, answer))

            total_citations_in_answer = citations_in_answer + precedents_in_answer + document_citations

            # 3. 검색된 문서의 출처가 답변에 포함되어 있는지 확인
            has_document_references = False
            if document_sources:
                for source in document_sources:
                    source_keywords = source.split()[:3]
                    if any(keyword in answer_lower for keyword in source_keywords if len(keyword) > 2):
                        has_document_references = True
                        break

            # 컨텍스트에서 추출한 인용 정보와 비교
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

            # 4. 핵심 개념 포함 여부
            context_key_concepts = []
            if context_text:
                key_terms = ["법", "조", "판례", "규정", "절차", "요건", "효력"]
                for term in key_terms:
                    if term in context_text:
                        context_key_concepts.append(term)

            concept_coverage = 0.0
            if context_key_concepts:
                found_concepts = sum(1 for concept in context_key_concepts if concept in answer_lower)
                concept_coverage = found_concepts / len(context_key_concepts)

            # 5. 종합 점수
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
        답변의 내용이 검색된 문서에 기반하는지 검증 (Hallucination 방지)

        Args:
            answer: 검증할 답변 텍스트
            retrieved_docs: 검색된 문서 목록
            query: 원본 질의

        Returns:
            검증 결과 딕셔너리
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
                "error": "답변 또는 검색 결과가 없습니다."
            }

        # 1. 검색된 문서에서 모든 텍스트 추출
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
                "error": "검색된 문서의 내용이 없습니다."
            }

        # 2. 답변을 문장 단위로 분리
        answer_sentences = re.split(r'[.!?。！？]\s+', answer)
        answer_sentences = [s.strip() for s in answer_sentences if len(s.strip()) > 20]

        # 3. 각 문장이 검색된 문서에 기반하는지 검증
        verified_sentences = []
        unverified_sentences = []

        for sentence in answer_sentences:
            sentence_lower = sentence.lower()

            # 문장의 핵심 키워드 추출 (불용어 제거)
            stopwords = {'는', '은', '이', '가', '을', '를', '에', '의', '와', '과', '로', '으로', '에서', '도', '만', '부터', '까지'}
            sentence_words = [w for w in re.findall(r'[가-힣]+', sentence_lower) if len(w) > 1 and w not in stopwords]

            if not sentence_words:
                continue

            # 각 소스 텍스트와 유사도 계산
            max_similarity = 0.0
            best_match_source = None
            matched_keywords_count = 0

            for source_text in source_texts:
                # 키워드 매칭 점수
                matched_keywords = sum(1 for word in sentence_words if word in source_text)
                keyword_score = matched_keywords / len(sentence_words) if sentence_words else 0.0

                # 문장 유사도 (SequenceMatcher 사용)
                similarity = SequenceMatcher(None, sentence_lower[:100], source_text[:1000]).ratio()

                # 종합 점수 (키워드 매칭 + 유사도)
                combined_score = (keyword_score * 0.6) + (similarity * 0.4)

                if combined_score > max_similarity:
                    max_similarity = combined_score
                    matched_keywords_count = matched_keywords
                    best_match_source = source_text[:100]  # 디버깅용

            # 검증 기준: 30% 이상 유사하거나 핵심 키워드 50% 이상 매칭
            keyword_coverage = matched_keywords_count / len(sentence_words) if sentence_words else 0.0
            if max_similarity >= 0.3 or keyword_coverage >= 0.5:
                verified_sentences.append({
                    "sentence": sentence,
                    "similarity": max_similarity,
                    "source_preview": best_match_source
                })
            else:
                # 법령 인용이나 일반적인 면책 조항은 제외
                if not (re.search(r'\[법령:\s*[^\]]+\]', sentence) or
                       re.search(r'본\s*답변은\s*일반적인', sentence) or
                       re.search(r'변호사와\s*직접\s*상담', sentence)):
                    unverified_sentences.append({
                        "sentence": sentence[:100],
                        "similarity": max_similarity,
                        "keywords": sentence_words[:5],
                        "keyword_coverage": keyword_coverage
                    })

        # 4. 종합 검증 점수 계산
        total_sentences = len(answer_sentences)
        verified_count = len(verified_sentences)

        grounding_score = verified_count / total_sentences if total_sentences > 0 else 0.0
        source_coverage = len(set([s["source_preview"] for s in verified_sentences if s.get("source_preview")])) / len(source_texts) if source_texts else 0.0

        # 5. 검증 통과 기준: 80% 이상 문장이 검증됨
        is_grounded = grounding_score >= 0.8

        # 6. 신뢰도 조정 (검증되지 않은 문장이 많으면 신뢰도 감소)
        confidence_penalty = len(unverified_sentences) * 0.05  # 문장당 5% 감소

        return {
            "is_grounded": is_grounded,
            "grounding_score": grounding_score,
            "verified_sentences": verified_sentences[:5],  # 샘플
            "unverified_sentences": unverified_sentences,
            "unverified_count": len(unverified_sentences),
            "source_coverage": source_coverage,
            "needs_review": not is_grounded or len(unverified_sentences) > 3,
            "confidence_penalty": min(confidence_penalty, 0.3),  # 최대 30% 감소
            "total_sentences": total_sentences,
            "verified_count": verified_count
        }


class SearchValidator:
    """검색 품질 검증"""

    @staticmethod
    def validate_search_quality(
        search_results: List[Dict[str, Any]],
        query: str,
        query_type: str
    ) -> Dict[str, Any]:
        """
        검색 품질 검증

        Args:
            search_results: 검색 결과 목록
            query: 검색 쿼리
            query_type: 질문 유형

        Returns:
            검증 결과 딕셔너리
        """
        try:
            if not search_results:
                return {
                    "is_valid": False,
                    "quality_score": 0.0,
                    "doc_count": 0,
                    "avg_relevance": 0.0,
                    "issues": ["검색 결과가 없습니다"],
                    "recommendations": ["검색 쿼리를 수정하거나 검색 범위를 확대하세요"]
                }

            # 문서 개수 확인
            doc_count = len(search_results)
            min_docs_required = 2 if query_type != "simple" else 1

            # 평균 관련도 점수 계산
            relevance_scores = []
            for doc in search_results:
                if isinstance(doc, dict):
                    score = doc.get("relevance_score") or doc.get("final_weighted_score", 0.0)
                    relevance_scores.append(score)

            avg_relevance = sum(relevance_scores) / max(1, len(relevance_scores)) if relevance_scores else 0.0

            # 품질 점수 계산
            doc_adequacy = min(1.0, doc_count / max(1, min_docs_required))
            relevance_adequacy = avg_relevance

            quality_score = (doc_adequacy * 0.4 + relevance_adequacy * 0.6)

            # 문제점 확인
            issues = []
            if doc_count < min_docs_required:
                issues.append(f"검색 결과가 부족합니다 ({doc_count}/{min_docs_required})")
            if avg_relevance < 0.3:
                issues.append(f"평균 관련도가 낮습니다 ({avg_relevance:.2f})")

            # 권고사항 생성
            recommendations = []
            if doc_count < min_docs_required:
                recommendations.append("검색 쿼리를 확장하거나 검색 범위를 넓히세요")
            if avg_relevance < 0.3:
                recommendations.append("검색 쿼리를 더 구체적으로 작성하거나 다른 키워드를 시도하세요")

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
                "issues": [f"검증 중 오류 발생: {e}"],
                "recommendations": []
            }
