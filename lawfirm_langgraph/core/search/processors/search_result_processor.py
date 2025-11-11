# -*- coding: utf-8 -*-
"""
검색 결과 처리 프로세서
검색 결과 병합, 가중치 적용, 필터링, 재정렬 등을 담당
"""

import logging
import os
import re
import math
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 성능 최적화: 정규식 패턴 컴파일 (모듈 레벨)
LAW_PATTERN = re.compile(r'[가-힣]+법\s*제?\s*\d+\s*조')
PRECEDENT_PATTERN = re.compile(r'대법원|법원.*\d{4}[다나마]\d+')


class SearchResultProcessor:
    """검색 결과 처리 프로세서"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, result_merger=None, result_ranker=None):
        self.logger = logger or logging.getLogger(__name__)
        self.result_merger = result_merger
        self.result_ranker = result_ranker
    
    def merge_search_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        result_merger=None
    ) -> List[Dict[str, Any]]:
        """검색 결과 병합"""
        merger = result_merger or self.result_merger
        
        if merger:
            exact_results_dict = {
                "semantic": semantic_results if isinstance(semantic_results, list) else [],
                "keyword": keyword_results if isinstance(keyword_results, list) else []
            } if keyword_results else {}
            
            merged_results = merger.merge_results(
                exact_results=exact_results_dict,
                semantic_results=semantic_results if isinstance(semantic_results, list) else [],
                weights={"exact": 0.7, "semantic": 0.3}
            )
        else:
            merged_results = semantic_results + keyword_results
        
        merged_docs = []
        for merged_result in merged_results:
            if hasattr(merged_result, 'text'):
                text_value = merged_result.text
                if not text_value or len(str(text_value).strip()) == 0:
                    if hasattr(merged_result, 'content'):
                        text_value = merged_result.content
                    elif hasattr(merged_result, 'metadata') and isinstance(merged_result.metadata, dict):
                        text_value = (
                            merged_result.metadata.get('content') or
                            merged_result.metadata.get('text') or
                            merged_result.metadata.get('document') or
                            ''
                        )
                
                if not text_value or len(str(text_value).strip()) == 0:
                    source_name = getattr(merged_result, 'source', 'Unknown')
                    score_value = getattr(merged_result, 'score', 0.0)
                    self.logger.warning(f"⚠️ [DEBUG] MergedResult text가 비어있음 - source: {source_name}, score: {score_value:.3f}")
                
                merged_docs.append({
                    "content": str(text_value) if text_value else "",
                    "text": str(text_value) if text_value else "",
                    "relevance_score": getattr(merged_result, 'score', 0.0),
                    "source": getattr(merged_result, 'source', 'Unknown'),
                    "metadata": getattr(merged_result, 'metadata', {}) if hasattr(merged_result, 'metadata') else {}
                })
            elif isinstance(merged_result, dict):
                doc = merged_result.copy()
                if "content" not in doc and "text" in doc:
                    doc["content"] = doc["text"]
                elif "text" not in doc and "content" in doc:
                    doc["text"] = doc["content"]
                elif "content" not in doc and "text" not in doc:
                    doc["content"] = ""
                    doc["text"] = ""
                merged_docs.append(doc)
        
        return merged_docs
    
    def calculate_keyword_weights(
        self,
        extracted_keywords: List[str],
        query: str,
        query_type: str,
        legal_field: str
    ) -> Dict[str, float]:
        """키워드별 중요도 가중치 계산"""
        keyword_weights = {}
        
        if not extracted_keywords:
            return keyword_weights
        
        query_lower = query.lower()
        
        legal_term_patterns = [
            re.compile(r'[가-힣]+법'),
            re.compile(r'[가-힣]+규정'),
            re.compile(r'[가-힣]+조항'),
            re.compile(r'판례'),
            re.compile(r'대법원'),
            re.compile(r'법원'),
            re.compile(r'판결'),
            re.compile(r'계약'),
            re.compile(r'손해배상'),
            re.compile(r'소송'),
            re.compile(r'청구')
        ]
        
        query_type_keywords = {
            "precedent_search": ["판례", "사건", "판결", "대법원"],
            "law_inquiry": ["법률", "조문", "법령", "규정", "조항"],
            "legal_advice": ["조언", "해석", "권리", "의무", "책임"],
            "procedure_guide": ["절차", "방법", "대응", "소송"],
            "term_explanation": ["의미", "정의", "개념", "해석"]
        }
        
        field_keywords = {
            "family": ["가족", "이혼", "양육", "상속", "부부"],
            "civil": ["민사", "계약", "손해배상", "채권", "채무"],
            "criminal": ["형사", "범죄", "처벌", "형량"],
            "labor": ["노동", "근로", "해고", "임금", "근로자"],
            "corporate": ["기업", "회사", "주주", "법인"]
        }
        
        important_keywords_for_type = query_type_keywords.get(query_type, [])
        important_keywords_for_field = field_keywords.get(legal_field, [])
        
        for keyword in extracted_keywords:
            if not keyword or not isinstance(keyword, str):
                continue
            
            keyword_lower = keyword.lower()
            weight = 0.0
            
            query_frequency = query_lower.count(keyword_lower)
            query_weight = min(0.3, (query_frequency / max(1, len(query.split()))) * 0.3)
            weight += query_weight
            
            is_legal_term = any(pattern.search(keyword) for pattern in legal_term_patterns)
            if is_legal_term:
                weight += 0.3
            
            if any(imp_kw in keyword_lower for imp_kw in important_keywords_for_type):
                weight += 0.2
            
            if any(imp_kw in keyword_lower for imp_kw in important_keywords_for_field):
                weight += 0.2
            
            if weight == 0.0:
                weight = 0.1
            
            keyword_weights[keyword] = min(1.0, weight)
        
        total_weight = sum(keyword_weights.values())
        if total_weight > 0:
            max_weight = max(keyword_weights.values()) if keyword_weights else 1.0
            if max_weight > 0:
                for kw in keyword_weights:
                    keyword_weights[kw] = keyword_weights[kw] / max_weight
        
        return keyword_weights
    
    def calculate_keyword_match_score(
        self,
        document: Dict[str, Any],
        keyword_weights: Dict[str, float],
        query: str
    ) -> Dict[str, float]:
        """문서에 대한 키워드 매칭 점수 계산"""
        doc_content = document.get("content", "")
        if not doc_content:
            return {
                "keyword_match_score": 0.0,
                "keyword_coverage": 0.0,
                "matched_keywords": [],
                "weighted_keyword_score": 0.0
            }
        
        doc_content_lower = doc_content.lower()
        
        matched_keywords = []
        total_weight = 0.0
        matched_weight = 0.0
        
        for keyword, weight in keyword_weights.items():
            if not keyword:
                continue
            
            total_weight += weight
            keyword_lower = keyword.lower()
            
            if keyword_lower in doc_content_lower:
                matched_keywords.append(keyword)
                matched_weight += weight
                
                keyword_count = doc_content_lower.count(keyword_lower)
                if keyword_count > 1:
                    matched_weight += weight * 0.1 * min(2, keyword_count - 1)
        
        keyword_coverage = len(matched_keywords) / max(1, len(keyword_weights))
        keyword_match_score = matched_weight / max(0.1, total_weight) if total_weight > 0 else 0.0
        weighted_keyword_score = min(1.0, matched_weight / max(1, len(keyword_weights)))
        
        return {
            "keyword_match_score": keyword_match_score,
            "keyword_coverage": keyword_coverage,
            "matched_keywords": matched_keywords,
            "weighted_keyword_score": weighted_keyword_score
        }
    
    def calculate_weighted_final_score(
        self,
        document: Dict[str, Any],
        keyword_scores: Dict[str, float],
        search_params: Dict[str, Any],
        query_type: Optional[str] = None
    ) -> float:
        """가중치를 적용한 최종 점수 계산"""
        base_relevance = (
            document.get("relevance_score", 0.0) or
            document.get("combined_score", 0.0) or
            document.get("score", 0.0)
        )
        
        keyword_match = keyword_scores.get("weighted_keyword_score", 0.0)
        
        search_type = document.get("search_type", "")
        type_weight = 1.4 if search_type == "semantic" else 0.9
        
        doc_type = document.get("type", "").lower() if document.get("type") else ""
        doc_type_weight = 1.0
        if "법령" in doc_type or "law" in doc_type:
            doc_type_weight = 1.3
        elif "판례" in doc_type or "precedent" in doc_type:
            doc_type_weight = 1.15
        else:
            doc_type_weight = 0.85
        
        query_type_weight = 1.0
        if query_type:
            if query_type == "precedent_search" and ("판례" in doc_type or "precedent" in doc_type):
                query_type_weight = 1.4
            elif query_type == "law_inquiry" and ("법령" in doc_type or "law" in doc_type):
                query_type_weight = 1.4
        
        category_boost = document.get("category_boost", 1.0)
        field_match_score = document.get("field_match_score", 0.5)
        category_bonus = (category_boost * 0.7 + field_match_score * 0.3)
        
        normalized_relevance = base_relevance
        if normalized_relevance < 0:
            normalized_relevance = 0.0
        elif normalized_relevance > 1.0:
            normalized_relevance = 1.0 + (math.log1p(normalized_relevance - 1.0) / 10.0)
            normalized_relevance = min(1.5, normalized_relevance)
        
        dynamic_weights = self.calculate_dynamic_weights(
            query_type=query_type,
            search_quality=search_params.get("overall_quality", 0.7),
            document_count=search_params.get("document_count", 10)
        )
        
        final_score = (
            normalized_relevance * dynamic_weights["relevance"] +
            keyword_match * dynamic_weights["keyword"] +
            (normalized_relevance * doc_type_weight * query_type_weight) * dynamic_weights["type"] +
            (type_weight - 1.0) * dynamic_weights["search_type"] +
            category_bonus * dynamic_weights["category"]
        )
        
        if normalized_relevance <= 0.0 and keyword_match <= 0.0:
            final_score = 0.15
        else:
            final_score = max(0.0, final_score)
        
        return min(1.5, max(0.0, final_score))
    
    def calculate_dynamic_weights(
        self,
        query_type: str = "",
        search_quality: float = 0.7,
        document_count: int = 10
    ) -> Dict[str, float]:
        """동적 가중치 계산"""
        base_weights = {
            "relevance": 0.40,
            "keyword": 0.35,
            "type": 0.15,
            "search_type": 0.05,
            "category": 0.05
        }
        
        if query_type == "law_inquiry":
            base_weights["keyword"] += 0.05
            base_weights["relevance"] -= 0.05
        elif query_type == "precedent_search":
            base_weights["relevance"] += 0.05
            base_weights["keyword"] -= 0.05
        
        if search_quality < 0.5:
            base_weights["keyword"] += 0.1
            base_weights["relevance"] -= 0.1
        elif search_quality > 0.8:
            base_weights["relevance"] += 0.05
            base_weights["keyword"] -= 0.05
        
        if document_count < 5:
            base_weights["relevance"] += 0.05
            base_weights["keyword"] -= 0.05
        
        total = sum(base_weights.values())
        if total > 0:
            base_weights = {k: v / total for k, v in base_weights.items()}
        
        return base_weights
    
    def apply_keyword_weights_to_docs(
        self,
        merged_docs: List[Dict[str, Any]],
        keyword_weights: Dict[str, float],
        query: str,
        query_type_str: str,
        search_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """키워드 가중치 적용"""
        def process_doc(doc):
            doc_content = doc.get("content", "") or doc.get("text", "")
            if not doc_content or not isinstance(doc_content, str) or len(doc_content.strip()) < 5:
                doc["keyword_match_score"] = 0.0
                doc["keyword_coverage"] = 0.0
                doc["matched_keywords"] = []
                doc["weighted_keyword_score"] = 0.0
                doc["final_weighted_score"] = doc.get("relevance_score", 0.0) * 0.5
                return doc
            
            keyword_scores = self.calculate_keyword_match_score(
                document=doc,
                keyword_weights=keyword_weights,
                query=query
            )
            
            final_score = self.calculate_weighted_final_score(
                document=doc,
                keyword_scores=keyword_scores,
                search_params=search_params,
                query_type=query_type_str
            )
            
            doc["keyword_match_score"] = keyword_scores.get("keyword_match_score", 0.0)
            doc["keyword_coverage"] = keyword_scores.get("keyword_coverage", 0.0)
            doc["matched_keywords"] = keyword_scores.get("matched_keywords", [])
            doc["weighted_keyword_score"] = keyword_scores.get("weighted_keyword_score", 0.0)
            doc["final_weighted_score"] = final_score
            
            return doc
        
        if len(merged_docs) > 10:
            weighted_docs = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_doc, doc): doc for doc in merged_docs}
                for future in as_completed(futures):
                    try:
                        weighted_docs.append(future.result(timeout=2))
                    except Exception as e:
                        self.logger.warning(f"Document processing failed: {e}")
                        weighted_docs.append(futures[future])
        else:
            weighted_docs = [process_doc(doc) for doc in merged_docs]
        
        weighted_docs.sort(key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)), reverse=True)
        return weighted_docs
    
    def apply_citation_boost(
        self,
        weighted_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Citation 부스트 적용"""
        citation_boosted = []
        non_citation = []
        
        for doc in weighted_docs:
            content = doc.get("content", "") or doc.get("text", "") or ""
            if not isinstance(content, str):
                content = str(content) if content else ""
            
            if len(content) < 50:
                non_citation.append(doc)
                continue
            
            content_sample = content[:500]
            has_law = bool(LAW_PATTERN.search(content_sample))
            has_precedent = bool(PRECEDENT_PATTERN.search(content_sample))
            
            if has_law or has_precedent:
                current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                boosted_score = current_score * 1.2
                doc["final_weighted_score"] = boosted_score
                doc["relevance_score"] = boosted_score
                citation_boosted.append(doc)
            else:
                non_citation.append(doc)
        
        citation_boosted.sort(key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)), reverse=True)
        non_citation.sort(key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)), reverse=True)
        
        if citation_boosted:
            self.logger.info(f"🔍 [SEARCH FILTERING] Citation boost applied: {len(citation_boosted)} documents with citations prioritized")
        
        return citation_boosted + non_citation
    
    def filter_documents(
        self,
        weighted_docs: List[Dict[str, Any]],
        max_docs: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """문서 필터링"""
        debug_mode = os.getenv("DEBUG_SEARCH_RESULTS", "false").lower() == "true"
        
        filtered_docs = []
        skipped_content = 0
        skipped_score = 0
        skipped_content_details = []
        
        for doc in weighted_docs:
            content = (
                doc.get("content", "") or
                doc.get("text", "") or
                doc.get("content_text", "") or
                doc.get("document", "") or
                str(doc.get("metadata", {}).get("content", "")) or
                str(doc.get("metadata", {}).get("text", "")) or
                ""
            )
            
            if not isinstance(content, str):
                content = str(content) if content else ""
            
            if not content or len(content.strip()) < 5:
                skipped_content += 1
                if skipped_content <= 3:
                    skipped_content_details.append({
                        "keys": list(doc.keys()),
                        "content_type": type(doc.get("content", None)).__name__,
                        "text_type": type(doc.get("text", None)).__name__,
                        "content_len": len(str(doc.get("content", ""))),
                        "text_len": len(str(doc.get("text", "")))
                    })
                continue
            
            score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
            if score < 0.05:
                skipped_score += 1
                continue
            
            filtered_docs.append(doc)
        
        if debug_mode:
            self.logger.info(f"📊 [SEARCH RESULTS] Filtering statistics - Weighted: {len(weighted_docs)}, Filtered: {len(filtered_docs)}, Skipped (content): {skipped_content}, Skipped (score): {skipped_score}")
            
            if skipped_content > 0 and skipped_content_details:
                self.logger.warning(f"⚠️ [SEARCH RESULTS] Content 필터링 제외 상세 (상위 {len(skipped_content_details)}개): {skipped_content_details}")
        
        final_docs = filtered_docs[:max_docs]
        
        return final_docs, {
            "skipped_content": skipped_content,
            "skipped_score": skipped_score,
            "filtered_count": len(filtered_docs)
        }
    
    def rerank_with_keyword_weights(
        self,
        results: List[Dict[str, Any]],
        keyword_weights: Dict[str, float],
        rerank_params: Dict[str, Any],
        result_ranker=None
    ) -> List[Dict[str, Any]]:
        """키워드 가중치를 적용한 Reranking"""
        ranker = result_ranker or self.result_ranker
        
        try:
            sorted_results = sorted(
                results,
                key=lambda x: (
                    x.get("final_weighted_score", 0.0),
                    x.get("keyword_match_score", 0.0),
                    x.get("keyword_coverage", 0.0)
                ),
                reverse=True
            )
            
            for doc in sorted_results:
                coverage = doc.get("keyword_coverage", 0.0)
                if coverage > 0.7:
                    doc["final_weighted_score"] *= 1.1
                elif coverage > 0.5:
                    doc["final_weighted_score"] *= 1.05
            
            sorted_results = sorted(
                sorted_results,
                key=lambda x: x.get("final_weighted_score", 0.0),
                reverse=True
            )
            
            top_k = rerank_params.get("top_k", 20)
            if ranker and len(sorted_results) > 0:
                try:
                    reranked_results = ranker.rank_results(
                        sorted_results[:top_k * 2],
                        top_k=top_k
                    )
                    if reranked_results and hasattr(reranked_results[0], 'score'):
                        reranked_dicts = []
                        for result in reranked_results:
                            doc = {
                                "content": result.text,
                                "relevance_score": result.score,
                                "source": result.source,
                                "id": f"{result.source}_{hash(result.text)}",
                                "final_weighted_score": result.score
                            }
                            if isinstance(result.metadata, dict):
                                doc.update(result.metadata)
                            reranked_dicts.append(doc)
                        sorted_results = reranked_dicts[:top_k]
                    else:
                        sorted_results = sorted_results[:top_k]
                except Exception as e:
                    self.logger.warning(f"Reranker failed, using keyword-weighted scores: {e}")
                    sorted_results = sorted_results[:top_k]
            else:
                sorted_results = sorted_results[:top_k]
            
            try:
                if ranker and hasattr(ranker, 'apply_diversity_filter'):
                    diverse_results = ranker.apply_diversity_filter(
                        sorted_results,
                        max_per_type=5,
                        diversity_weight=rerank_params.get("diversity_weight", 0.3)
                    )
                else:
                    diverse_results = sorted_results
            except Exception as e:
                self.logger.warning(f"Diversity filter failed: {e}")
                diverse_results = sorted_results
            
            return diverse_results
        
        except Exception as e:
            self.logger.warning(f"Reranking with keyword weights failed: {e}")
            return sorted(
                results,
                key=lambda x: x.get("final_weighted_score", 0.0),
                reverse=True
            )[:rerank_params.get("top_k", 20)]

