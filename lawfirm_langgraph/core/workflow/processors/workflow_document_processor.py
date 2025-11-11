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
            min_relevance_score_semantic = 0.2
            min_relevance_score_keyword = 0.15
            
            for doc in retrieved_docs:
                if not isinstance(doc, dict):
                    invalid_docs_count += 1
                    continue
                
                content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                if not content or len(content.strip()) < 10:
                    invalid_docs_count += 1
                    self.logger.debug(f"Document filtered: content too short or empty (source: {doc.get('source', 'Unknown')})")
                    continue
                
                search_type = doc.get("search_type", "semantic")
                relevance_score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
                keyword_match_score = doc.get("keyword_match_score", 0.0)
                has_keyword_match = keyword_match_score > 0.0 or len(doc.get("matched_keywords", [])) > 0
                
                min_score = min_relevance_score_keyword if search_type == "keyword" else min_relevance_score_semantic
                
                if search_type == "keyword" and has_keyword_match:
                    min_score = min_relevance_score_keyword
                elif search_type == "semantic":
                    min_score = min_relevance_score_semantic
                
                if relevance_score < min_score:
                    invalid_docs_count += 1
                    self.logger.debug(
                        f"Document filtered: relevance score too low ({relevance_score:.3f} < {min_score:.3f}) "
                        f"(source: {doc.get('source', 'Unknown')}, type: {search_type})"
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
            
            if select_balanced_documents_func:
                balanced_docs = select_balanced_documents_func(sorted_docs, max_docs=10)
            else:
                balanced_docs = self.select_balanced_documents(sorted_docs, max_docs=10)
            
            if not balanced_docs and sorted_docs:
                balanced_docs = sorted_docs[:min(8, len(sorted_docs))]
            
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
                
                max_content_length = 1500
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
                "content_validation": content_validation
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

