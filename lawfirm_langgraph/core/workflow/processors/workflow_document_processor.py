# -*- coding: utf-8 -*-
"""
워크플로우 문서 처리 프로세서
검색 결과 문서 선택, 컨텍스트 빌딩, 프롬프트 최적화 등을 담당
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkflowDocumentProcessor:
    """워크플로우 문서 처리 프로세서"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, query_enhancer=None):
        self.logger = logger or logging.getLogger(__name__)
        self.query_enhancer = query_enhancer
    
    def build_prompt_optimized_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        query: str,
        extracted_keywords: List[str],
        query_type: str,
        legal_field: str,
        select_balanced_documents_func=None,
        extract_query_relevant_sentences_func=None,
        generate_document_based_instructions_func=None
    ) -> Dict[str, Any]:
        """프롬프트에 최대한 반영되도록 최적화된 컨텍스트 구축"""
        try:
            if not retrieved_docs:
                self.logger.warning("build_prompt_optimized_context: retrieved_docs is empty")
                return {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                }
            
            valid_docs = []
            invalid_docs_count = 0
            
            # 개선 1, 4: 문서 타입별 필터링 기준 차등화 (실제 점수 범위에 맞게 완화)
            # 검색 결과 평균 점수: 0.458, 범위: 0.373~0.732
            min_relevance_score_semantic = 0.35
            min_relevance_score_keyword = 0.35
            min_relevance_score_statute_article = 0.30
            min_relevance_score_precedent = 0.35
            min_relevance_score_general = 0.40
            
            # 개선 6: 키워드 매칭 점수 최소 기준
            min_keyword_match_score = 0.01
            
            # 개선 7: 질문 핵심 키워드 추출 (간단한 버전)
            query_lower = query.lower()
            query_keywords = []
            for keyword in extracted_keywords:
                if keyword and len(keyword) > 1:
                    query_keywords.append(keyword.lower())
            
            for doc in retrieved_docs:
                if not isinstance(doc, dict):
                    invalid_docs_count += 1
                    continue
                
                content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                
                # content가 없거나 너무 짧은 경우 복원 시도 (최소 길이 완화: 10자 → 5자)
                if not content or len(content.strip()) < 5:
                    # metadata에서 복원 시도
                    metadata = doc.get("metadata", {})
                    if isinstance(metadata, dict):
                        content = metadata.get("content") or metadata.get("text") or content
                    
                    # 여전히 없으면 최소한의 정보라도 유지 (필터링하지 않음, 최소 길이 완화: 10자 → 5자)
                    if not content or len(content.strip()) < 5:
                        # doc의 다른 필드에서 정보 추출
                        title = doc.get("title") or doc.get("name") or ""
                        if title:
                            content = title
                        else:
                            # 최후의 수단: doc_id나 다른 식별자 사용
                            doc_id = doc.get("doc_id") or doc.get("id") or ""
                            if doc_id:
                                content = f"Document {doc_id}"
                    
                    # content를 doc에 다시 설정
                    if content:
                        doc["content"] = content
                        doc["text"] = content
                
                # 최소 길이 검증 (더욱 완화된 기준)
                doc_type = doc.get("type", "").lower() if doc.get("type") else ""
                source_type = doc.get("source_type", "").lower() if doc.get("source_type") else ""
                is_legal_doc = "statute" in doc_type or "statute" in source_type or "case" in doc_type or "case" in source_type
                # 법령/판례 문서는 3자, 기타는 5자로 더 완화
                min_content_length = 3 if is_legal_doc else 5
                
                if not content or len(content.strip()) < min_content_length:
                    invalid_docs_count += 1
                    self.logger.debug(f"Document filtered: content too short or empty (length: {len(content) if content else 0}, min_required: {min_content_length}, source: {doc.get('source', 'Unknown')})")
                    continue
                
                search_type = doc.get("search_type", "semantic")
                relevance_score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
                keyword_match_score = doc.get("keyword_match_score", 0.0)
                matched_keywords = doc.get("matched_keywords", [])
                has_keyword_match = keyword_match_score > 0.0 or len(matched_keywords) > 0
                
                # 개선 6: 키워드 매칭 점수 기반 필터링 (더욱 완화)
                if keyword_match_score == 0.0 and not matched_keywords:
                    content_lower = content.lower()
                    has_query_keyword = False
                    for qkw in query_keywords:
                        if qkw in content_lower:
                            has_query_keyword = True
                            break
                    
                    # 관련성 임계값 더욱 완화 (법령/판례 문서는 0.30, 기타는 0.40)
                    relevance_threshold = 0.30 if is_legal_doc else 0.40
                    if not has_query_keyword and relevance_score < relevance_threshold:
                        invalid_docs_count += 1
                        self.logger.debug(
                            f"Document filtered: no keyword match and low relevance "
                            f"(relevance: {relevance_score:.3f}, threshold: {relevance_threshold}, source: {doc.get('source', 'Unknown')})"
                        )
                        continue
                
                # 문서 타입 확인 (이미 위에서 정의됨, 추가 확인만 수행)
                is_statute_article = (
                    doc_type == "statute_article" or 
                    source_type == "statute_article" or
                    "statute_article" in doc_type or
                    "statute_article" in source_type or
                    doc.get("direct_match", False) or
                    search_type == "direct_statute"
                )
                is_precedent = (
                    doc_type == "precedent" or
                    source_type == "precedent" or
                    "precedent" in doc_type or
                    "precedent" in source_type or
                    "case_paragraph" in doc_type or
                    "case_paragraph" in source_type or
                    "판례" in content[:200] or
                    "대법원" in content[:200]
                )
                # is_legal_doc는 이미 위에서 정의됨
                
                # 개선 4: 문서 타입별 필터링 기준 차등화 (키워드 매칭이 있으면 완화)
                if is_statute_article:
                    min_score = min_relevance_score_statute_article
                elif is_precedent:
                    min_score = min_relevance_score_precedent
                elif search_type == "keyword" and has_keyword_match:
                    min_score = min_relevance_score_keyword
                elif search_type == "semantic":
                    min_score = min_relevance_score_semantic
                else:
                    min_score = min_relevance_score_general
                
                # 키워드 매칭이 있으면 기준을 더 완화 (0.10 감소)
                if has_keyword_match or has_query_keyword:
                    min_score = max(0.20, min_score - 0.10)
                
                # 첫 번째 필터링(키워드 매칭 없을 때)을 통과한 경우, 두 번째 필터링은 더 완화
                if not has_keyword_match and not has_query_keyword and relevance_score >= 0.30:
                    # 이미 첫 번째 필터링을 통과했으므로 두 번째 필터링은 더 완화
                    min_score = max(0.25, min_score - 0.15)
                
                if relevance_score < min_score:
                    invalid_docs_count += 1
                    self.logger.debug(
                        f"Document filtered: relevance score too low ({relevance_score:.3f} < {min_score:.3f}) "
                        f"(source: {doc.get('source', 'Unknown')}, type: {search_type}, doc_type: {doc_type}, "
                        f"has_keyword: {has_keyword_match or has_query_keyword})"
                    )
                    continue
                
                valid_docs.append(doc)
            
            if invalid_docs_count > 0:
                self.logger.warning(
                    f"build_prompt_optimized_context: Filtered {invalid_docs_count} invalid documents "
                    f"(no content, content too short, or relevance < threshold). Valid docs: {len(valid_docs)}"
                )
            
            if not valid_docs:
                self.logger.error("build_prompt_optimized_context: No valid documents with content found")
                return {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                }
            
            sorted_docs = sorted(
                valid_docs,
                key=lambda x: (
                    x.get("final_weighted_score", x.get("relevance_score", 0.0)),
                    x.get("keyword_match_score", 0.0)
                ),
                reverse=True
            )
            
            # 개선 8: 프롬프트에 포함할 문서 수 제한 (5-7개)
            max_docs_for_prompt = 7
            
            # 개선 12: 문서 선택 로직 개선 (관련성 우선)
            if select_balanced_documents_func:
                balanced_docs = select_balanced_documents_func(sorted_docs, max_docs=max_docs_for_prompt)
            else:
                balanced_docs = self.select_balanced_documents_relevance_first(
                    sorted_docs, 
                    query=query,
                    extracted_keywords=extracted_keywords,
                    query_type=query_type,
                    max_docs=max_docs_for_prompt
                )
            
            if not balanced_docs and sorted_docs:
                balanced_docs = sorted_docs[:min(max_docs_for_prompt, len(sorted_docs))]
            
            sorted_docs = balanced_docs
            
            if not sorted_docs:
                self.logger.error("build_prompt_optimized_context: sorted_docs is empty after filtering")
                return {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                }
            
            if generate_document_based_instructions_func:
                document_instructions = generate_document_based_instructions_func(
                    documents=sorted_docs,
                    query=query,
                    query_type=query_type
                )
            else:
                document_instructions = self.generate_document_based_instructions(
                    documents=sorted_docs,
                    query=query,
                    query_type=query_type
                )
            
            semantic_count = sum(1 for doc in sorted_docs if doc.get("search_type") == "semantic")
            keyword_count = sum(1 for doc in sorted_docs if doc.get("search_type") == "keyword")
            hybrid_count = len(sorted_docs) - semantic_count - keyword_count
            
            prompt_section = f"""## 답변 생성 지시사항

{document_instructions}

## 참고 문서 목록

다음 {len(sorted_docs)}개의 문서를 반드시 참고하여 답변을 생성하세요.
각 문서는 관련성 점수와 핵심 내용이 표시되어 있습니다.

**검색 결과 통계:**
- 의미적 검색 결과: {semantic_count}개
- 키워드 검색 결과: {keyword_count}개
- 하이브리드 검색 결과: {hybrid_count}개
- 총 문서 수: {len(sorted_docs)}개

**참고:** 의미적 검색 결과는 의미적 유사도를, 키워드 검색 결과는 키워드 매칭 정도를 나타냅니다.
두 검색 방식의 결과를 종합하여 정확하고 포괄적인 답변을 생성하세요.

"""
            
            for idx, doc in enumerate(sorted_docs, 1):
                relevance_score = doc.get("final_weighted_score") or doc.get("relevance_score", 0.0)
                source = doc.get("source", "Unknown")
                content = doc.get("content", "")
                
                if extract_query_relevant_sentences_func:
                    relevant_sentences = extract_query_relevant_sentences_func(
                        doc_content=content,
                        query=query,
                        extracted_keywords=extracted_keywords
                    )
                elif self.query_enhancer:
                    relevant_sentences = self.query_enhancer.extract_query_relevant_sentences(
                        content, query, extracted_keywords
                    )
                else:
                    relevant_sentences = []
                
                search_type = doc.get("search_type", "hybrid")
                search_method = doc.get("search_method", "hybrid_search")
                keyword_match_score = doc.get("keyword_match_score", 0.0)
                matched_keywords = doc.get("matched_keywords", [])
                
                doc_section = f"""
### 문서 {idx}: {source} (관련성 점수: {relevance_score:.2f})

**검색 정보:**
- 검색 방식: {search_type} ({search_method})
- 키워드 매칭 점수: {keyword_match_score:.2f}
- 매칭된 키워드: {', '.join(matched_keywords[:5]) if matched_keywords else '없음'}

**핵심 내용:**
"""
                
                if relevant_sentences:
                    doc_section += "\n".join([
                        f"- [중요] {sent['sentence']}"
                        for sent in relevant_sentences[:3]
                    ])
                    doc_section += "\n\n"
                
                # 컨텍스트 길이 최적화: 토큰 수 기반 동적 조정
                # 한글 기준 대략 1토큰 = 2-3자, 영어 기준 1토큰 = 4자
                # 안전하게 1토큰 = 2.5자로 계산
                max_tokens_per_doc = 600  # 문서당 최대 토큰 수
                max_content_length = int(max_tokens_per_doc * 2.5)  # 약 1500자
                
                # 질문 타입별 동적 조정
                if query_type == "law_inquiry":
                    max_tokens_per_doc = 800  # 법령 조회: 더 긴 컨텍스트 허용
                    max_content_length = int(max_tokens_per_doc * 2.5)  # 약 2000자
                elif query_type == "complex_question":
                    max_tokens_per_doc = 1000  # 복잡한 질문: 더 긴 컨텍스트 허용
                    max_content_length = int(max_tokens_per_doc * 2.5)  # 약 2500자
                
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                
                doc_section += f"""**전체 내용:**
{content}

---
"""
                
                prompt_section += doc_section
            
            prompt_section += """
## 문서 인용 규칙

답변에서 위 문서를 인용할 때는 다음과 같이 명시하세요:
- "문서 {0}에 따르면..." 또는 "[{0}] 인용 내용"
- 각 문서의 출처를 명확히 표시

## 중요 사항

- 위 문서의 내용을 바탕으로 답변을 생성하세요
- 문서에서 추론하거나 추측하지 말고, 문서에 명시된 내용만 사용하세요
- 문서에 없는 정보는 포함하지 마세요
- 여러 문서의 내용을 종합하여 일관된 답변을 구성하세요
""".format("n")
            
            content_validation = {
                "has_document_content": False,
                "total_content_length": 0,
                "documents_with_content": 0
            }
            
            for doc in sorted_docs:
                content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                if content and len(content.strip()) >= 10:
                    content_preview = content[:100]
                    if content_preview in prompt_section:
                        content_validation["has_document_content"] = True
                        content_validation["total_content_length"] += len(content)
                        content_validation["documents_with_content"] += 1
            
            if not content_validation["has_document_content"]:
                self.logger.error(
                    f"build_prompt_optimized_context: WARNING - prompt_section does not contain actual document content! "
                    f"Documents processed: {len(sorted_docs)}, "
                    f"Prompt length: {len(prompt_section)}"
                )
            else:
                self.logger.info(
                    f"build_prompt_optimized_context: Successfully included content from {content_validation['documents_with_content']} documents "
                    f"(total content length: {content_validation['total_content_length']} chars, "
                    f"prompt length: {len(prompt_section)} chars)"
                )
            
            if not content_validation["has_document_content"] and len(sorted_docs) > 0:
                self.logger.warning(
                    f"build_prompt_optimized_context: Content validation failed, but returning prompt anyway "
                    f"(may contain instructions only without actual document content)"
                )
            
            # 개선 10: 프롬프트 생성 후 최종 검증
            final_validation = self._validate_final_documents(
                sorted_docs=sorted_docs,
                query=query,
                extracted_keywords=extracted_keywords,
                query_type=query_type
            )
            
            if final_validation.get("low_relevance_warning"):
                self.logger.warning(
                    f"build_prompt_optimized_context: {final_validation['low_relevance_warning']} "
                    f"(low_relevance_count: {final_validation.get('low_relevance_count', 0)})"
                )
            
            return {
                "prompt_optimized_text": prompt_section,
                "structured_documents": {
                    "total_count": len(sorted_docs),
                    "documents": [{
                        "document_id": idx,
                        "source": doc.get("source", "Unknown"),
                        "relevance_score": doc.get("final_weighted_score") or doc.get("relevance_score", 0.0),
                        "content": (doc.get("content") or doc.get("text") or doc.get("content_text", ""))[:2000]
                    } for idx, doc in enumerate(sorted_docs, 1)]
                },
                "document_count": len(sorted_docs),
                "total_context_length": len(prompt_section),
                "content_validation": content_validation,
                "final_validation": final_validation
            }
        
        except Exception as e:
            self.logger.error(f"Prompt optimized context building failed: {e}")
            return {
                "prompt_optimized_text": "",
                "structured_documents": {},
                "document_count": 0,
                "total_context_length": 0
            }
    
    def select_balanced_documents(
        self,
        sorted_docs: List[Dict[str, Any]],
        max_docs: int = 10
    ) -> List[Dict[str, Any]]:
        """의미적 검색과 키워드 검색 결과의 균형을 맞춰서 문서 선택"""
        if not sorted_docs:
            return []
        
        semantic_docs = [doc for doc in sorted_docs if doc.get("search_type") == "semantic"]
        keyword_docs = [doc for doc in sorted_docs if doc.get("search_type") == "keyword"]
        hybrid_docs = [doc for doc in sorted_docs if doc.get("search_type") not in ["semantic", "keyword"]]
        
        selected_docs = []
        
        top_count = max(1, max_docs // 2)
        selected_docs.extend(sorted_docs[:top_count])
        
        remaining_slots = max_docs - len(selected_docs)
        
        if remaining_slots > 0:
            semantic_to_add = []
            for doc in semantic_docs:
                if doc not in selected_docs:
                    semantic_to_add.append(doc)
            
            keyword_to_add = []
            for doc in keyword_docs:
                if doc not in selected_docs:
                    keyword_to_add.append(doc)
            
            max_alternate = remaining_slots // 2
            for i in range(min(max_alternate, max(len(semantic_to_add), len(keyword_to_add)))):
                if i < len(semantic_to_add) and len(selected_docs) < max_docs:
                    if semantic_to_add[i] not in selected_docs:
                        selected_docs.append(semantic_to_add[i])
                if i < len(keyword_to_add) and len(selected_docs) < max_docs:
                    if keyword_to_add[i] not in selected_docs:
                        selected_docs.append(keyword_to_add[i])
            
            if len(selected_docs) < max_docs:
                for doc in hybrid_docs:
                    if doc not in selected_docs and len(selected_docs) < max_docs:
                        selected_docs.append(doc)
            
            if len(selected_docs) < max_docs:
                for doc in sorted_docs:
                    if doc not in selected_docs and len(selected_docs) < max_docs:
                        selected_docs.append(doc)
        
        selected_docs = sorted(
            selected_docs,
            key=lambda x: (
                x.get("final_weighted_score", x.get("relevance_score", 0.0)),
                x.get("keyword_match_score", 0.0)
            ),
            reverse=True
        )
        
        return selected_docs[:max_docs]
    
    def select_balanced_documents_relevance_first(
        self,
        sorted_docs: List[Dict[str, Any]],
        query: str,
        extracted_keywords: List[str],
        query_type: str,
        max_docs: int = 7
    ) -> List[Dict[str, Any]]:
        """
        개선 12: 관련성 우선 문서 선택 (다양성보다 관련성 우선)
        
        Args:
            sorted_docs: 점수로 정렬된 문서 리스트
            query: 사용자 질문
            extracted_keywords: 추출된 키워드 리스트
            query_type: 질문 유형
            max_docs: 선택할 최대 문서 수
        
        Returns:
            관련성이 높은 문서 리스트
        """
        if not sorted_docs:
            return []
        
        # 개선 9: 질문 유형별 문서 필터링 기준 적용
        query_lower = query.lower()
        query_keywords_lower = [kw.lower() for kw in extracted_keywords if kw and len(kw) > 1]
        
        selected_docs = []
        seen_sources = set()
        
        # 개선: Citation 가능성이 높은 문서 식별 (법령 조문, 판례 등)
        import re
        citation_pattern = re.compile(r'[가-힣]+법\s*제?\s*\d+\s*조')
        precedent_pattern = re.compile(r'[가-힣]+(?:지방)?법원|대법원|판결|사건')
        
        # 문서에 citation 점수 부여
        for doc in sorted_docs:
            content = (doc.get("content") or doc.get("text") or "").lower()
            doc_type = doc.get("type") or doc.get("source_type") or ""
            
            citation_score = 0.0
            # 법령 조문 타입이면 높은 점수
            if doc_type in ["statute_article", "statute"]:
                citation_score += 0.5
            # 판례 타입이면 높은 점수
            elif doc_type in ["case_paragraph", "precedent", "decision_paragraph"]:
                citation_score += 0.4
            # 내용에서 법령 조문 발견
            if citation_pattern.search(content):
                citation_score += 0.3
            # 내용에서 판례 발견
            if precedent_pattern.search(content):
                citation_score += 0.2
            
            doc["citation_potential_score"] = min(1.0, citation_score)
        
        # 1단계: 관련도가 높고 citation 가능성이 높은 문서 우선 선택
        high_relevance_docs = [
            doc for doc in sorted_docs 
            if (doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)) >= 0.65
        ]
        
        # citation 가능성 순으로 정렬
        high_relevance_docs.sort(
            key=lambda x: (
                x.get("citation_potential_score", 0.0),
                x.get("relevance_score", 0.0) or x.get("final_weighted_score", 0.0)
            ),
            reverse=True
        )
        
        for doc in high_relevance_docs:
            if len(selected_docs) >= max_docs:
                break
            
            source = doc.get("source", "")
            if source and source not in seen_sources:
                selected_docs.append(doc)
                seen_sources.add(source)
        
        # 2단계: 관련도가 중간이지만 citation 가능성이 높은 문서 우선 선택
        if len(selected_docs) < max_docs:
            medium_relevance_docs = [
                doc for doc in sorted_docs 
                if 0.55 <= (doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)) < 0.65
                and doc not in selected_docs
            ]
            
            # citation 가능성 순으로 정렬
            medium_relevance_docs.sort(
                key=lambda x: (
                    x.get("citation_potential_score", 0.0),
                    x.get("relevance_score", 0.0) or x.get("final_weighted_score", 0.0)
                ),
                reverse=True
            )
            
            for doc in medium_relevance_docs:
                if len(selected_docs) >= max_docs:
                    break
                
                # 개선: citation 가능성이 높거나 키워드 매칭이 있는 문서 우선
                content = (doc.get("content") or doc.get("text") or "").lower()
                has_relevant_keyword = False
                
                for qkw in query_keywords_lower:
                    if qkw in content or qkw in query_lower:
                        has_relevant_keyword = True
                        break
                
                citation_potential = doc.get("citation_potential_score", 0.0)
                keyword_match = doc.get("keyword_match_score", 0.0)
                
                # citation 가능성이 높거나 키워드 매칭이 있으면 선택
                if citation_potential >= 0.3 or has_relevant_keyword or keyword_match > 0.0:
                    source = doc.get("source", "")
                    if not source or source not in seen_sources:
                        selected_docs.append(doc)
                        if source:
                            seen_sources.add(source)
        
        # 3단계: 부족하면 상위 문서로 채우기
        if len(selected_docs) < max_docs:
            for doc in sorted_docs:
                if len(selected_docs) >= max_docs:
                    break
                if doc not in selected_docs:
                    selected_docs.append(doc)
        
        self.logger.info(
            f"select_balanced_documents_relevance_first: Selected {len(selected_docs)}/{len(sorted_docs)} documents "
            f"(high_relevance: {len([d for d in selected_docs if (d.get('relevance_score', 0.0) or d.get('final_weighted_score', 0.0)) >= 0.65])})"
        )
        
        return selected_docs[:max_docs]
    
    def _validate_final_documents(
        self,
        sorted_docs: List[Dict[str, Any]],
        query: str,
        extracted_keywords: List[str],
        query_type: str
    ) -> Dict[str, Any]:
        """
        개선 10: 프롬프트 생성 후 최종 검증
        
        Args:
            sorted_docs: 최종 선택된 문서 리스트
            query: 사용자 질문
            extracted_keywords: 추출된 키워드 리스트
            query_type: 질문 유형
        
        Returns:
            검증 결과 딕셔너리
        """
        validation_result = {
            "total_docs": len(sorted_docs),
            "high_relevance_count": 0,
            "medium_relevance_count": 0,
            "low_relevance_count": 0,
            "low_relevance_warning": None,
            "avg_relevance_score": 0.0,
            "min_relevance_score": 0.0,
            "max_relevance_score": 0.0
        }
        
        if not sorted_docs:
            return validation_result
        
        relevance_scores = []
        query_lower = query.lower()
        query_keywords_lower = [kw.lower() for kw in extracted_keywords if kw and len(kw) > 1]
        
        for doc in sorted_docs:
            relevance_score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
            relevance_scores.append(relevance_score)
            
            if relevance_score >= 0.65:
                validation_result["high_relevance_count"] += 1
            elif relevance_score >= 0.55:
                validation_result["medium_relevance_count"] += 1
            else:
                validation_result["low_relevance_count"] += 1
                
                # 관련도가 낮은 문서의 경우 키워드 매칭 확인
                content = (doc.get("content") or doc.get("text") or "").lower()
                has_keyword = False
                for qkw in query_keywords_lower:
                    if qkw in content:
                        has_keyword = True
                        break
                
                if not has_keyword and doc.get("keyword_match_score", 0.0) == 0.0:
                    source = doc.get("source", "Unknown")
                    if not validation_result["low_relevance_warning"]:
                        validation_result["low_relevance_warning"] = f"Low relevance documents detected: {source}"
                    else:
                        validation_result["low_relevance_warning"] += f", {source}"
        
        if relevance_scores:
            validation_result["avg_relevance_score"] = sum(relevance_scores) / len(relevance_scores)
            validation_result["min_relevance_score"] = min(relevance_scores)
            validation_result["max_relevance_score"] = max(relevance_scores)
        
        # 경고 조건: 관련도가 낮은 문서가 전체의 30% 이상이거나 평균 관련도가 0.60 미만
        if validation_result["low_relevance_count"] > 0:
            low_relevance_ratio = validation_result["low_relevance_count"] / validation_result["total_docs"]
            if low_relevance_ratio >= 0.3 or validation_result["avg_relevance_score"] < 0.60:
                if not validation_result["low_relevance_warning"]:
                    validation_result["low_relevance_warning"] = (
                        f"Low relevance ratio: {low_relevance_ratio:.1%}, "
                        f"avg_score: {validation_result['avg_relevance_score']:.3f}"
                    )
        
        self.logger.info(
            f"_validate_final_documents: Validation complete - "
            f"total: {validation_result['total_docs']}, "
            f"high: {validation_result['high_relevance_count']}, "
            f"medium: {validation_result['medium_relevance_count']}, "
            f"low: {validation_result['low_relevance_count']}, "
            f"avg_score: {validation_result['avg_relevance_score']:.3f}"
        )
        
        return validation_result
    
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
                
                citation_pattern = r'[가-힣]+법\s*제?\s*\d+\s*조'
                citations = re.findall(citation_pattern, doc_content)
                citation_count = len(citations)
                citation_score = min(1.0, citation_count / 5.0)
                
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
                
                keyword_coverage = 0.0
                if query_words and content_words:
                    keyword_coverage = len(query_words.intersection(content_words)) / max(1, len(query_words))
                
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
                
                combined_value_score = 0.6 * relevance_score + 0.4 * information_density
                doc["combined_value_score"] = combined_value_score
                
                if combined_value_score >= min_relevance:
                    high_value_docs.append(doc)
            
            high_value_docs.sort(key=lambda x: x.get("combined_value_score", 0.0), reverse=True)
            
            selected_docs = high_value_docs[:max_docs]
            
            self.logger.info(
                f"📚 [HIGH VALUE SELECTION] Selected {len(selected_docs)}/{len(documents)} documents. "
                f"Avg density: {sum(d.get('information_density_score', 0.0) for d in selected_docs) / max(1, len(selected_docs)):.3f}"
            )
            
            return selected_docs
        
        except Exception as e:
            self.logger.warning(f"High value document selection failed: {e}, using first {max_docs} documents")
            return documents[:max_docs]
    
    def generate_document_based_instructions(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        query_type: str
    ) -> str:
        """문서를 기반으로 답변 생성하라는 명시적 지시사항 생성"""
        instructions = f"""당신은 법률 전문가입니다. 아래 제공된 문서들을 반드시 참고하여 다음 질문에 답변하세요.

**질문**: {query}
**질문 유형**: {query_type}

**답변 생성 규칙**:
1. **문서 기반 답변**: 제공된 문서의 내용을 바탕으로 답변을 생성하세요.
2. **문서 인용 필수**: 답변에서 문서를 인용할 때는 "문서 [번호]에 따르면..." 형식으로 명시하세요.
3. **정확성**: 문서에 명시된 내용만 사용하고, 추론하거나 추측하지 마세요.
4. **구조화**: 답변은 다음 구조로 작성하세요:
   - 핵심 답변
   - 관련 법령 및 조항
   - 실무 적용 시 주의사항
   - 참고할 만한 판례 (있는 경우)
5. **출처 명시**: 각 인용문에 대해 문서 번호를 명시하세요.
"""
        
        return instructions

