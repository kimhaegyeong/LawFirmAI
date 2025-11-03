# -*- coding: utf-8 -*-
"""
컨텍스트 빌더 모듈
검색된 문서로부터 답변 생성에 필요한 컨텍스트를 구축하는 로직을 독립 모듈로 분리
"""

import logging
import re
from typing import Any, Dict, List, Optional

from langgraph_core.data.extractors import DocumentExtractor
from langgraph_core.utils.state_definitions import LegalWorkflowState
from langgraph_core.utils.workflow_utils import WorkflowUtils


class ContextBuilder:
    """
    컨텍스트 구축 클래스

    검색된 문서들을 재랭킹하고 선별하여, 답변 생성에 필요한 최적의 컨텍스트를 구성합니다.
    """

    def __init__(
        self,
        semantic_search: Any,
        config: Any,
        logger: Optional[logging.Logger] = None
    ):
        """
        ContextBuilder 초기화

        Args:
            semantic_search: 의미적 검색 엔진 인스턴스
            config: 설정 객체
            logger: 로거 (없으면 자동 생성)
        """
        self.semantic_search = semantic_search
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def build_context(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """기본 컨텍스트 구성 (길이 제한 관리)"""
        max_length = self.config.max_context_length
        context_parts = []
        current_length = 0
        docs_truncated = 0

        retrieved_docs = WorkflowUtils.get_state_value(state, "retrieved_docs", [])
        for doc in retrieved_docs:
            doc_content = doc.get("content", "")
            doc_length = len(doc_content)

            # 컨텍스트 길이 확인
            if current_length + doc_length > max_length:
                # 가능한 만큼만 추가
                remaining_length = max_length - current_length - 200  # 여유 공간
                if remaining_length > 100:  # 최소 100자
                    truncated_content = doc_content[:remaining_length] + "..."
                    context_parts.append(f"[문서: {doc.get('source', 'unknown')}]\n{truncated_content}")
                    docs_truncated += 1
                    self.logger.warning("Document truncated due to context length limit")
                break

            context_part = f"[문서: {doc.get('source', 'unknown')}]\n{doc_content}"
            context_parts.append(context_part)
            current_length += len(context_part)

        context_text = "\n\n".join(context_parts)

        if docs_truncated > 0:
            self.logger.info(f"Context length management: {current_length}/{max_length} chars, {docs_truncated} docs truncated")

        # structured_documents 생성 (폴백 경로에서도 문서 포함)
        structured_documents = {
            "total_count": len(retrieved_docs),
            "documents": []
        }

        for idx, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "") or doc.get("text", "")
            if content and len(content.strip()) >= 10:
                structured_documents["documents"].append({
                    "document_id": idx,
                    "source": doc.get("source", "Unknown"),
                    "relevance_score": doc.get("relevance_score", doc.get("final_weighted_score", 0.0)),
                    "content": content[:2000]  # 최대 2000자로 제한
                })

        return {
            "context": context_text,
            "structured_documents": structured_documents,
            "document_count": len(structured_documents["documents"]),
            "legal_references": WorkflowUtils.get_state_value(state, "legal_references", []),
            "query_type": WorkflowUtils.get_state_value(state, "query_type", ""),
            "context_length": current_length,
            "docs_included": len(structured_documents["documents"]),
            "docs_truncated": docs_truncated
        }

    def build_intelligent_context(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """지능형 컨텍스트 구축"""
        try:
            max_length = self.config.max_context_length

            retrieved_docs = WorkflowUtils.get_state_value(state, "retrieved_docs", [])
            query = WorkflowUtils.get_state_value(state, "query", "")
            query_type = WorkflowUtils.get_state_value(state, "query_type", "")
            extracted_keywords = WorkflowUtils.get_state_value(state, "extracted_keywords", [])

            if not retrieved_docs:
                self.logger.warning("No documents retrieved for context building")
                return {
                    "context": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "legal_references": [],
                    "query_type": query_type,
                    "context_length": 0,
                    "docs_included": 0,
                    "insights": [],
                    "citations": []
                }

            # 1. 문서 재랭킹
            reranked_docs = self.rerank_documents_by_relevance(
                retrieved_docs,
                query,
                extracted_keywords
            )

            # 2. 고품질 문서 선별
            high_value_docs = self.select_high_value_documents(
                reranked_docs,
                query,
                min_relevance=0.5,
                max_docs=8
            )

            # 3. 컨텍스트 조합 최적화
            optimized_composition = self.optimize_context_composition(
                high_value_docs,
                query,
                max_length
            )

            # 4. 핵심 정보 추출
            key_insights = self.extract_key_insights(high_value_docs, query)

            # 5. 법률 인용 정보 추출
            legal_citations = self.extract_legal_citations(high_value_docs)

            # 6. 최종 컨텍스트 구성
            context_text = "\n\n".join(optimized_composition["context_parts"])

            # context_parts가 비어있으면 최소한의 context 생성 (폴백)
            if not context_text and high_value_docs:
                self.logger.warning(
                    f"⚠️ [INTELLIGENT CONTEXT] context_parts is empty, "
                    f"creating minimal context from {len(high_value_docs)} docs"
                )
                # 최소한의 context 생성 (문서 요약)
                context_parts = []
                for doc in high_value_docs[:5]:
                    content = doc.get("content", "") or doc.get("text", "") or doc.get("content_text", "")
                    source = doc.get("source", "Unknown")
                    if content and len(content.strip()) > 20:
                        # 문서 내용 일부 포함
                        content_preview = content[:500]
                        context_parts.append(f"[문서: {source}]\n{content_preview}")

                if context_parts:
                    context_text = "\n\n".join(context_parts)
                    self.logger.info(
                        f"✅ [INTELLIGENT CONTEXT] Created minimal context with {len(context_parts)} docs "
                        f"({len(context_text)} chars)"
                    )

            # 인사이트 추가
            if key_insights and len(context_text) < max_length - 300:
                insights_text = "\n\n## 핵심 정보\n" + "\n".join([f"- {insight}" for insight in key_insights[:5]])
                if len(context_text) + len(insights_text) < max_length:
                    context_text += insights_text

            # 인용 정보 추가
            if legal_citations and len(context_text) < max_length - 200:
                citations_text = "\n\n## 법률 인용\n" + "\n".join([f"- {cit['text']}" for cit in legal_citations[:5]])
                if len(context_text) + len(citations_text) < max_length:
                    context_text += citations_text

            self.logger.info(
                f"🧠 [INTELLIGENT CONTEXT] Built context with {optimized_composition['docs_included']} docs, "
                f"{len(key_insights)} insights, {len(legal_citations)} citations, "
                f"length: {len(context_text)}/{max_length}"
            )

            # structured_documents 생성 (high_value_docs를 구조화)
            structured_documents = {
                "total_count": len(high_value_docs),
                "documents": []
            }

            for idx, doc in enumerate(high_value_docs[:8], 1):
                content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                relevance_score = doc.get("final_weighted_score") or doc.get("relevance_score", 0.0)

                if content and len(content.strip()) >= 10:
                    structured_documents["documents"].append({
                        "document_id": idx,
                        "source": doc.get("source", "Unknown"),
                        "relevance_score": relevance_score,
                        "content": content[:2000]  # 최대 2000자로 제한
                    })

            return {
                "context": context_text,
                "structured_documents": structured_documents,
                "document_count": len(structured_documents["documents"]),
                "legal_references": [cit["text"] for cit in legal_citations],
                "query_type": query_type,
                "context_length": len(context_text),
                "docs_included": optimized_composition["docs_included"],
                "docs_truncated": optimized_composition["docs_truncated"],
                "insights": key_insights[:5],
                "citations": legal_citations[:5]
            }

        except Exception as e:
            self.logger.error(f"Intelligent context building failed: {e}, falling back to simple context")
            return self.build_context(state)

    def rerank_documents_by_relevance(
        self,
        documents: List[Dict],
        query: str,
        extracted_keywords: List[str]
    ) -> List[Dict]:
        """문서 관련성 재랭킹 - 질문과 문서의 직접 유사도 재계산"""
        if not documents:
            return documents

        try:
            reranked_docs = []

            for doc in documents:
                doc_content = doc.get("content", "")
                if not doc_content:
                    continue

                # 1. 의미적 유사도 재계산 (semantic_search 활용)
                semantic_score = 0.0
                try:
                    if self.semantic_search:
                        existing_score = doc.get("relevance_score", 0.0) or doc.get("combined_score", 0.0)
                        semantic_score = existing_score

                        # 직접 유사도 계산 시도
                        if hasattr(self.semantic_search, '_calculate_semantic_score'):
                            direct_score = self.semantic_search._calculate_semantic_score(query, doc_content)
                            # 기존 점수와 직접 계산 점수 가중 평균
                            semantic_score = 0.6 * existing_score + 0.4 * direct_score
                except Exception as e:
                    self.logger.debug(f"Semantic score calculation failed: {e}")
                    semantic_score = doc.get("relevance_score", 0.0) or doc.get("combined_score", 0.0)

                # 2. 키워드 매칭 점수 계산
                keyword_score = 0.0
                if extracted_keywords:
                    content_lower = doc_content.lower()
                    query_lower = query.lower()

                    # 추출된 키워드 매칭
                    keyword_matches = sum(1 for kw in extracted_keywords if isinstance(kw, str) and kw.lower() in content_lower)
                    keyword_score = min(1.0, keyword_matches / max(1, len(extracted_keywords)))

                    # 질문 키워드 매칭
                    query_words = set(query_lower.split())
                    content_words = set(content_lower.split())
                    if query_words and content_words:
                        query_match_ratio = len(query_words.intersection(content_words)) / len(query_words)
                        keyword_score = (keyword_score + query_match_ratio) / 2

                # 3. 법률 용어 매칭 점수
                legal_term_score = 0.0
                if extracted_keywords:
                    # 법률 용어 패턴 확인
                    legal_patterns = ['법', '조', '조문', '판례', '대법원', '법원', '규정', '법령']
                    content_lower = doc_content.lower()
                    term_matches = sum(1 for pattern in legal_patterns if pattern in content_lower)
                    legal_term_score = min(1.0, term_matches / max(1, len(legal_patterns) // 2))

                # 4. 종합 관련성 점수
                existing_combined = doc.get("combined_score", 0.0)

                # 가중치: 의미적 50%, 키워드 30%, 법률 용어 20%
                new_relevance_score = (
                    0.5 * semantic_score +
                    0.3 * keyword_score +
                    0.2 * legal_term_score
                )

                # 기존 점수와 새 점수 결합
                final_relevance = 0.7 * max(existing_combined, new_relevance_score) + 0.3 * new_relevance_score

                doc["final_relevance_score"] = final_relevance
                doc["query_direct_similarity"] = semantic_score
                doc["keyword_match_score"] = keyword_score
                doc["legal_term_score"] = legal_term_score

                reranked_docs.append(doc)

            # 최종 관련성 점수로 정렬
            reranked_docs.sort(key=lambda x: x.get("final_relevance_score", 0.0), reverse=True)

            self.logger.info(
                f"📊 [RERANK] Re-ranked {len(reranked_docs)} documents. "
                f"Top score: {reranked_docs[0].get('final_relevance_score', 0.0):.3f} if reranked_docs else 0.0"
            )

            return reranked_docs

        except Exception as e:
            self.logger.warning(f"Document reranking failed: {e}, using original order")
            return documents

    def select_high_value_documents(
        self,
        documents: List[Dict],
        query: str,
        min_relevance: float = 0.7,
        max_docs: int = 5
    ) -> List[Dict]:
        """정보 밀도 기반 문서 선택"""
        if not documents:
            return documents

        try:
            high_value_docs = []

            for doc in documents:
                doc_content = doc.get("content", "")
                if not doc_content or len(doc_content) < 20:
                    continue

                # 1. 법률 조항 인용 수 계산
                citation_pattern = r'[가-힣]+법\s*제?\s*\d+\s*조'
                citations = re.findall(citation_pattern, doc_content)
                citation_count = len(citations)
                citation_score = min(1.0, citation_count / 5.0)

                # 2. 핵심 개념 설명 완성도 평가
                query_words = set(query.lower().split())
                content_words = set(doc_content.lower().split())
                explanation_completeness = 0.0
                if query_words and content_words:
                    overlap = len(query_words.intersection(content_words))
                    explanation_completeness = min(1.0, overlap / max(1, len(query_words)))

                sentences = doc_content.split('。') or doc_content.split('.')
                avg_sentence_length = sum(len(s.strip()) for s in sentences if s.strip()) / max(1, len(sentences))

                descriptive_score_bonus = 0.0
                if 20 <= avg_sentence_length <= 100:
                    descriptive_score_bonus = 0.2
                elif avg_sentence_length > 100:
                    descriptive_score_bonus = 0.1

                explanation_completeness = min(1.0, explanation_completeness + descriptive_score_bonus)

                # 3. 질문 키워드 포함도
                keyword_coverage = 0.0
                if query_words and content_words:
                    keyword_coverage = len(query_words.intersection(content_words)) / max(1, len(query_words))

                # 4. 정보 밀도 종합 점수
                relevance_score = doc.get("final_relevance_score") or doc.get("combined_score", 0.0) or doc.get("relevance_score", 0.0)

                information_density = (
                    0.3 * citation_score +
                    0.3 * explanation_completeness +
                    0.2 * keyword_coverage +
                    0.2 * min(1.0, relevance_score)
                )

                doc["information_density_score"] = information_density
                doc["citation_count"] = citation_count
                doc["explanation_completeness"] = explanation_completeness

                # 관련성 점수와 정보 밀도 점수 가중 평균
                combined_value_score = 0.6 * relevance_score + 0.4 * information_density
                doc["combined_value_score"] = combined_value_score

                # 임계값 체크
                if combined_value_score >= min_relevance:
                    high_value_docs.append(doc)

            # combined_value_score로 정렬
            high_value_docs.sort(key=lambda x: x.get("combined_value_score", 0.0), reverse=True)

            # 최대 문서 수 제한
            selected_docs = high_value_docs[:max_docs]

            self.logger.info(
                f"📚 [HIGH VALUE SELECTION] Selected {len(selected_docs)}/{len(documents)} documents. "
                f"Avg density: {sum(d.get('information_density_score', 0.0) for d in selected_docs) / max(1, len(selected_docs)):.3f}"
            )

            return selected_docs

        except Exception as e:
            self.logger.warning(f"High value document selection failed: {e}, using first {max_docs} documents")
            return documents[:max_docs]

    def optimize_context_composition(
        self,
        high_value_docs: List[Dict],
        query: str,
        max_length: int
    ) -> Dict[str, Any]:
        """컨텍스트 조합 최적화 (길이 관리)"""
        try:
            optimized_context = {
                "context_parts": [],
                "citations": [],
                "insights": [],
                "total_length": 0,
                "docs_included": 0,
                "docs_truncated": 0
            }

            current_length = 0
            reserved_space = 500

            for doc in high_value_docs:
                doc_content = doc.get("content", "")
                if not doc_content:
                    continue

                doc_length = len(doc_content)
                doc_source = doc.get("source", "unknown")

                available_space = max_length - current_length - reserved_space

                if available_space <= 100:
                    break

                if doc_length <= available_space:
                    context_part = f"[문서: {doc_source}]\n{doc_content}"
                    optimized_context["context_parts"].append(context_part)
                    current_length += len(context_part)
                    optimized_context["docs_included"] += 1
                elif available_space > 200:
                    sentences = re.split(r'[。.！!?？]\s*', doc_content)
                    included_text = ""

                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) < 10:
                            continue

                        if len(included_text) + len(sentence) + 10 <= available_space:
                            included_text += sentence + ". "
                        else:
                            break

                    if included_text:
                        truncated_content = included_text.strip() + "..."
                        context_part = f"[문서: {doc_source}]\n{truncated_content}"
                        optimized_context["context_parts"].append(context_part)
                        current_length += len(context_part)
                        optimized_context["docs_included"] += 1
                        optimized_context["docs_truncated"] += 1
                    else:
                        continue
                else:
                    continue

            optimized_context["total_length"] = current_length

            self.logger.info(
                f"📦 [CONTEXT OPTIMIZATION] Included {optimized_context['docs_included']} docs, "
                f"truncated {optimized_context['docs_truncated']}, "
                f"total length: {current_length}/{max_length} chars"
            )

            return optimized_context

        except Exception as e:
            self.logger.warning(f"Context composition optimization failed: {e}")
            context_parts = []
            current_length = 0

            for doc in high_value_docs[:5]:
                if current_length >= max_length - 500:
                    break
                doc_content = doc.get("content", "")[:500]
                doc_source = doc.get("source", "unknown")
                context_part = f"[문서: {doc_source}]\n{doc_content}"
                context_parts.append(context_part)
                current_length += len(context_part)

            return {
                "context_parts": context_parts,
                "citations": [],
                "insights": [],
                "total_length": current_length,
                "docs_included": len(context_parts),
                "docs_truncated": 0
            }

    def extract_key_insights(
        self,
        documents: List[Dict],
        query: str
    ) -> List[str]:
        """핵심 정보 추출 - 질문과 직접 관련된 핵심 문장 추출"""
        insights = DocumentExtractor.extract_key_insights(documents, query)
        self.logger.debug(f"📝 [KEY INSIGHTS] Extracted {len(insights)} key insights")
        return insights

    def extract_legal_citations(
        self,
        documents: List[Dict]
    ) -> List[Dict[str, str]]:
        """법률 인용 정보 추출"""
        citations = DocumentExtractor.extract_legal_citations(documents)
        self.logger.debug(f"⚖️ [LEGAL CITATIONS] Extracted {len(citations)} citations")
        return citations

    def calculate_context_relevance(
        self,
        context: Dict[str, Any],
        query: str
    ) -> float:
        """컨텍스트 관련성 계산 - 질문과 각 문서의 유사도 계산"""
        try:
            context_text = context.get("context", "")
            if not context_text:
                return 0.0

            # 의미적 유사도 계산 시도
            try:
                if self.semantic_search and hasattr(self.semantic_search, '_calculate_semantic_score'):
                    relevance_score = self.semantic_search._calculate_semantic_score(query, context_text)
                    return relevance_score
            except Exception as e:
                self.logger.debug(f"Semantic relevance calculation failed: {e}")

            # 폴백: 키워드 기반 유사도
            query_words = set(query.lower().split())
            context_words = set(context_text.lower().split())

            if not query_words or not context_words:
                return 0.0

            overlap = len(query_words.intersection(context_words))
            relevance = overlap / max(1, len(query_words))

            return min(1.0, relevance)

        except Exception as e:
            self.logger.warning(f"Context relevance calculation failed: {e}")
            return 0.5  # 기본값

    def calculate_information_coverage(
        self,
        context: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str]
    ) -> float:
        """정보 커버리지 계산 - 핵심 키워드 포함도"""
        try:
            context_text = context.get("context", "").lower()
            legal_references = context.get("legal_references", [])
            citations = context.get("citations", [])

            if not context_text and not legal_references and not citations:
                return 0.0

            coverage_scores = []

            # 1. 추출된 키워드 커버리지
            if extracted_keywords:
                keyword_matches = sum(1 for kw in extracted_keywords
                                    if isinstance(kw, str) and kw.lower() in context_text)
                keyword_coverage = keyword_matches / max(1, len(extracted_keywords))
                coverage_scores.append(keyword_coverage)

            # 2. 질문 키워드 커버리지
            query_words = set(query.lower().split())
            context_words = set(context_text.split())
            if query_words and context_words:
                query_coverage = len(query_words.intersection(context_words)) / max(1, len(query_words))
                coverage_scores.append(query_coverage)

            # 3. 질문 유형별 필수 정보 포함 여부
            type_coverage = 0.0
            if query_type:
                type_lower = query_type.lower()
                if "precedent" in type_lower or "판례" in type_lower:
                    # 판례 정보 포함 여부
                    precedent_indicators = ["판례", "대법원", "법원", "선고", "판결"]
                    type_coverage = 1.0 if any(ind in context_text for ind in precedent_indicators) else 0.3
                elif "law" in type_lower or "법령" in type_lower or "조문" in type_lower:
                    # 법률 조문 포함 여부
                    law_indicators = ["법", "조", "조문", "규정"]
                    type_coverage = 1.0 if any(ind in context_text for ind in law_indicators) else 0.3
                elif "advice" in type_lower or "조언" in type_lower:
                    # 실무 조언 포함 여부
                    advice_indicators = ["해야", "해야", "권장", "주의", "방법"]
                    type_coverage = 1.0 if any(ind in context_text for ind in advice_indicators) else 0.5
                else:
                    type_coverage = 0.7  # 일반 질문

            coverage_scores.append(type_coverage)

            # 평균 커버리지
            return sum(coverage_scores) / max(1, len(coverage_scores)) if coverage_scores else 0.0

        except Exception as e:
            self.logger.warning(f"Information coverage calculation failed: {e}")
            return 0.5

    def calculate_context_sufficiency(
        self,
        context: Dict[str, Any],
        query_type: str
    ) -> float:
        """컨텍스트 충분성 평가 - 질문 유형별 최소 요구사항 충족 여부"""
        try:
            context_text = context.get("context", "")
            legal_references = context.get("legal_references", [])
            citations = context.get("citations", [])

            # 질문 유형별 최소 요구사항
            if not query_type:
                return 0.5

            type_lower = query_type.lower()

            # 판례 검색
            if "precedent" in type_lower or "판례" in type_lower:
                # 판례 정보 최소 1개
                precedent_count = len([c for c in citations if isinstance(c, dict) and c.get("type") == "precedent"])
                if precedent_count > 0:
                    return 1.0
                elif "판례" in context_text or "대법원" in context_text or "법원" in context_text:
                    return 0.7
                else:
                    return 0.3

            # 법령 조회
            elif "law" in type_lower or "법령" in type_lower or "조문" in type_lower:
                # 법률 조문 최소 1개
                law_citation_count = len([c for c in citations if isinstance(c, dict) and c.get("type") == "law_article"])
                if law_citation_count > 0 or legal_references:
                    return 1.0
                elif "법" in context_text and "조" in context_text:
                    return 0.7
                else:
                    return 0.3

            # 법률 조언
            elif "advice" in type_lower or "조언" in type_lower:
                # 법령 + 실무 조언
                has_law = bool(legal_references) or "법" in context_text
                has_advice = any(word in context_text for word in ["해야", "권장", "주의", "방법", "절차"])

                if has_law and has_advice:
                    return 1.0
                elif has_law or has_advice:
                    return 0.6
                else:
                    return 0.3

            # 일반 질문
            else:
                # 최소한의 정보라도 있어야 함
                if len(context_text) > 100:
                    return 0.8
                elif len(context_text) > 50:
                    return 0.5
                else:
                    return 0.3

        except Exception as e:
            self.logger.warning(f"Context sufficiency calculation failed: {e}")
            return 0.5

    def identify_missing_information(
        self,
        context: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str]
    ) -> List[str]:
        """부족한 정보 식별"""
        missing = []

        try:
            context_text = context.get("context", "").lower()
            legal_references = context.get("legal_references", [])
            citations = context.get("citations", [])

            # 1. 누락된 키워드 확인
            if extracted_keywords:
                for kw in extracted_keywords:
                    if isinstance(kw, str) and kw.lower() not in context_text:
                        missing.append(kw)

            # 2. 질문 유형별 필수 정보 누락 확인
            type_lower = query_type.lower() if query_type else ""

            if "precedent" in type_lower or "판례" in type_lower:
                if not any(c.get("type") == "precedent" for c in citations if isinstance(c, dict)):
                    if "판례" not in context_text and "대법원" not in context_text:
                        missing.append("판례 정보")

            elif "law" in type_lower or "법령" in type_lower:
                if not legal_references and not any(c.get("type") == "law_article" for c in citations if isinstance(c, dict)):
                    if "법" not in context_text or "조" not in context_text:
                        missing.append("법률 조문")

            elif "advice" in type_lower or "조언" in type_lower:
                if not any(word in context_text for word in ["해야", "권장", "주의", "방법", "절차"]):
                    missing.append("실무 조언")

            return missing[:5]  # 최대 5개

        except Exception as e:
            self.logger.warning(f"Missing information identification failed: {e}")
            return []
