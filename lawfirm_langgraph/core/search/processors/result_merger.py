# -*- coding: utf-8 -*-
"""
Result Merger and Ranker
검색 결과 병합 및 순위 결정
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MergedResult:
    """병합된 검색 결과"""
    text: str
    score: float
    source: str
    metadata: Dict[str, Any]


class ResultMerger:
    """검색 결과 병합기"""
    
    def __init__(self):
        """병합기 초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("ResultMerger initialized")
    
    def merge_results(self, 
                     exact_results: Dict[str, List[Dict[str, Any]]], 
                     semantic_results: List[Dict[str, Any]],
                     weights: Dict[str, float] = None) -> List[MergedResult]:
        """
        검색 결과 병합
        
        Args:
            exact_results: 정확한 검색 결과 (딕셔너리)
            semantic_results: 의미적 검색 결과 (리스트)
            weights: 가중치 딕셔너리
            
        Returns:
            List[MergedResult]: 병합된 결과
        """
        if weights is None:
            weights = {"exact": 0.7, "semantic": 0.3}
        
        merged_results = []
        
        # 정확한 검색 결과 처리 (딕셔너리 형태)
        for search_type, results in exact_results.items():
            for result in results:
                if isinstance(result, dict):
                    # content 필드도 text로 매핑 (검색 결과가 content 필드를 사용하는 경우)
                    text_content = (
                        result.get('text', '') or
                        result.get('content', '') or
                        str(result.get('metadata', {}).get('content', '')) or
                        str(result.get('metadata', {}).get('text', '')) or
                        ''
                    )
                    
                    # text가 비어있으면 건너뛰기 (필터링)
                    if not text_content or len(text_content.strip()) == 0:
                        self.logger.warning(f"Skipping MergedResult with empty text from exact_{search_type} (id: {result.get('id', 'unknown')})")
                        continue
                    
                    # metadata에 content 필드 명시적으로 저장 (향후 딕셔너리 변환 시 복원용)
                    metadata = result.get('metadata', {})
                    if not isinstance(metadata, dict):
                        metadata = result if isinstance(result, dict) else {}
                    metadata['content'] = text_content
                    metadata['text'] = text_content
                    
                    merged_result = MergedResult(
                        text=text_content,
                        score=result.get('similarity', result.get('relevance_score', result.get('score', 0.0))) * weights["exact"],
                        source=f"exact_{search_type}",
                        metadata=metadata
                    )
                    merged_results.append(merged_result)
        
        # 의미적 검색 결과 처리 (리스트 형태)
        for result in semantic_results:
            if isinstance(result, dict):
                # content 필드도 text로 매핑
                text_content = (
                    result.get('text', '') or
                    result.get('content', '') or
                    str(result.get('metadata', {}).get('content', '')) or
                    str(result.get('metadata', {}).get('text', '')) or
                    ''
                )
                
                # text가 비어있으면 건너뛰기 (필터링)
                if not text_content or len(text_content.strip()) == 0:
                    self.logger.warning(f"Skipping MergedResult with empty text from semantic (id: {result.get('id', 'unknown')})")
                    continue
                
                # metadata에 content 필드 명시적으로 저장 (향후 딕셔너리 변환 시 복원용)
                metadata = result.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = result if isinstance(result, dict) else {}
                metadata['content'] = text_content
                metadata['text'] = text_content
                
                merged_result = MergedResult(
                    text=text_content,
                    score=result.get('similarity', result.get('relevance_score', result.get('score', 0.0))) * weights["semantic"],
                    source="semantic",
                    metadata=metadata
                )
                merged_results.append(merged_result)
        
        return merged_results


class ResultRanker:
    """검색 결과 순위 결정기"""
    
    def __init__(self):
        """순위 결정기 초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("ResultRanker initialized")
    
    def rank_results(self, results: List[Any], top_k: int = 10) -> List[Any]:
        """
        검색 결과 순위 결정
        
        Args:
            results: 병합된 검색 결과 (MergedResult 또는 Dict)
            top_k: 반환할 결과 수
            
        Returns:
            List[MergedResult] 또는 List[Dict]: 순위가 매겨진 결과
        """
        if not results:
            return []
        
        # Dict를 MergedResult로 변환
        converted_results = []
        for result in results:
            if isinstance(result, dict):
                # Dict를 MergedResult로 변환
                text = result.get("text") or result.get("content") or result.get("chunk_text") or ""
                score = result.get("score") or result.get("relevance_score") or result.get("similarity", 0.0)
                source = result.get("source") or result.get("title") or result.get("document_id") or ""
                metadata = result.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = result if isinstance(result, dict) else {}
                
                converted_result = MergedResult(
                    text=text,
                    score=score,
                    source=source,
                    metadata=metadata
                )
                converted_results.append(converted_result)
            elif isinstance(result, MergedResult):
                converted_results.append(result)
            else:
                # 기타 타입은 건너뛰기
                continue
        
        # 중복 제거 (텍스트 기준)
        unique_results = {}
        for result in converted_results:
            if result.text not in unique_results:
                unique_results[result.text] = result
            else:
                # 더 높은 점수로 업데이트
                if result.score > unique_results[result.text].score:
                    unique_results[result.text] = result
        
        # 점수순 정렬
        ranked_results = list(unique_results.values())
        ranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Dict로 변환하여 반환 (호환성 유지)
        return [self._merged_result_to_dict(r) for r in ranked_results[:top_k]]
    
    def _merged_result_to_dict(self, result: MergedResult) -> Dict[str, Any]:
        """MergedResult를 Dict로 변환"""
        return {
            "text": result.text,
            "content": result.text,
            "score": result.score,
            "relevance_score": result.score,
            "similarity": result.score,
            "source": result.source,
            "metadata": result.metadata
        }
    
    def multi_stage_rerank(
        self,
        documents: List[Dict[str, Any]],
        query: str = "",
        query_type: str = "",
        extracted_keywords: List[str] = None,
        search_quality: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        다단계 재정렬 전략 (개선: 키워드 매칭, Citation 매칭, 질문 유형별 특화, MMR 다양성, 검색 품질 기반 조정, 정보 밀도, 최신성)
        
        Args:
            documents: 재정렬할 문서 리스트
            query: 검색 쿼리
            query_type: 질문 유형
            extracted_keywords: 추출된 키워드
            search_quality: 검색 품질 점수
            
        Returns:
            List[Dict]: 재정렬된 문서 리스트
        """
        if not documents:
            return []
        
        if extracted_keywords is None:
            extracted_keywords = []
        
        # Stage 1: 관련성 점수로 초기 정렬
        stage1 = sorted(
            documents,
            key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)),
            reverse=True
        )
        
        # Stage 1.5: 키워드 매칭 점수 직접 반영 (개선: 키워드 매칭 점수 직접 반영)
        for doc in stage1:
            keyword_score = doc.get("weighted_keyword_score", 0.0)
            keyword_coverage = doc.get("keyword_coverage", 0.0)
            matched_keywords = doc.get("matched_keywords", [])
            
            # 핵심 키워드 매칭 보너스 계산
            core_keyword_bonus = 0.0
            if extracted_keywords and matched_keywords:
                # 핵심 키워드 매칭 비율
                core_keywords_matched = sum(1 for kw in extracted_keywords[:3] if kw in matched_keywords)
                core_keyword_ratio = core_keywords_matched / min(3, len(extracted_keywords)) if extracted_keywords else 0.0
                core_keyword_bonus = core_keyword_ratio * 0.25  # 최대 25% 증가
            
            # 키워드 커버리지 보너스
            coverage_bonus = keyword_coverage * 0.15  # 최대 15% 증가
            
            # 키워드 매칭 점수 보너스
            keyword_bonus = keyword_score * 0.2  # 최대 20% 증가
            
            # 총 키워드 보너스 적용
            current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
            total_keyword_bonus = core_keyword_bonus + coverage_bonus + keyword_bonus
            doc["final_weighted_score"] = current_score * (1.0 + total_keyword_bonus)
            doc["keyword_bonus"] = total_keyword_bonus
        
        # Stage 2: Citation 포함 문서 우선순위 (개선: Citation 매칭 정확도 개선)
        citation_docs = []
        non_citation_docs = []
        for doc in stage1:
            if self._has_citation(doc):
                # Citation 매칭 점수 계산
                citation_match_score = self._calculate_citation_match_score(
                    document=doc,
                    query=query,
                    extracted_keywords=extracted_keywords
                )
                
                # Citation 매칭 점수에 따른 보너스 적용
                current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                if citation_match_score >= 0.5:
                    # 정확 일치: 30-50% 증가
                    citation_bonus = 0.3 + (citation_match_score - 0.5) * 0.4
                elif citation_match_score > 0.0:
                    # 부분 일치: 10-30% 증가
                    citation_bonus = 0.1 + citation_match_score * 0.4
                else:
                    # Citation만 있음: 10% 증가
                    citation_bonus = 0.1
                
                doc["final_weighted_score"] = current_score * (1.0 + citation_bonus)
                doc["citation_match_score"] = citation_match_score
                doc["citation_bonus"] = citation_bonus
                citation_docs.append(doc)
            else:
                non_citation_docs.append(doc)
        
        # Stage 3: 신뢰도 점수 적용
        for doc in citation_docs + non_citation_docs:
            trust_score = self._calculate_trust_score(doc)
            current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
            doc["trust_score"] = trust_score
            doc["final_weighted_score"] = current_score * (1.0 + trust_score * 0.1)
        
        # Stage 3.5: 질문 유형별 특화 재정렬 (개선: 질문 유형별 특화 재정렬)
        if query_type == "law_inquiry":
            # 법령 문의: 법령 문서 우선순위 강화
            for doc in citation_docs + non_citation_docs:
                doc_type = doc.get("type", "").lower() if doc.get("type") else ""
                source = doc.get("source", "").lower()
                
                if "법령" in doc_type or "statute" in doc_type or "법령" in source:
                    current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                    doc["final_weighted_score"] = current_score * 1.3  # 30% 증가
                    doc["query_type_boost"] = 0.3
                elif "판례" in doc_type or "precedent" in doc_type:
                    current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                    doc["final_weighted_score"] = current_score * 0.9  # 10% 감소
        
        elif query_type == "precedent_search":
            # 판례 검색: 판례 문서 우선순위 강화
            for doc in citation_docs + non_citation_docs:
                doc_type = doc.get("type", "").lower() if doc.get("type") else ""
                source = doc.get("source", "").lower()
                
                if "판례" in doc_type or "precedent" in doc_type or "대법원" in source:
                    current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                    doc["final_weighted_score"] = current_score * 1.3  # 30% 증가
                    doc["query_type_boost"] = 0.3
                elif "법령" in doc_type or "statute" in doc_type:
                    current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                    doc["final_weighted_score"] = current_score * 0.9  # 10% 감소
        
        # Stage 3.6: 정보 밀도 점수 적용 (개선: 문서 길이/정보 밀도 고려)
        for doc in citation_docs + non_citation_docs:
            density_score = self._calculate_information_density(doc)
            current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
            doc["final_weighted_score"] = current_score * (1.0 + density_score * 0.1)  # 최대 10% 증가
            doc["information_density"] = density_score
        
        # Stage 3.7: 최신성 점수 적용 (개선: 시간적 관련성 고려)
        for doc in citation_docs + non_citation_docs:
            doc_type = doc.get("type", "").lower() if doc.get("type") else ""
            if "판례" in doc_type or "precedent" in doc_type:
                # 판례만 최신성 고려
                recency_score = self._calculate_recency_score(doc)
                current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                doc["final_weighted_score"] = current_score * (1.0 + recency_score * 0.15)  # 최대 15% 증가
                doc["recency_score"] = recency_score
        
        # Stage 4: MMR 기반 다양성 적용 (개선: 검색 결과 품질 및 질문 타입 기반 동적 조정)
        # 검색 품질에 따른 lambda_score 조정
        if search_quality < 0.5:
            # 품질 낮음: 키워드 매칭 강화 (다양성 감소)
            lambda_score = 0.8
        elif search_quality > 0.8:
            # 품질 높음: 관련성 우선 (다양성 증가)
            lambda_score = 0.6
        else:
            # 품질 중간: 균형잡힌 재정렬
            lambda_score = 0.7
        
        # 질문 타입별 다양성 조정
        if query_type == "law_inquiry":
            # 법령 조회: 관련성 우선 (다양성 약간 감소)
            lambda_score = min(0.85, lambda_score + 0.1)
        elif query_type == "precedent_search":
            # 판례 검색: 다양성 강화 (다양한 판례 포함)
            lambda_score = max(0.5, lambda_score - 0.1)
        
        diverse_docs = self._apply_mmr_diversity(
            citation_docs + non_citation_docs,
            query,
            lambda_score=lambda_score
        )
        
        return diverse_docs
    
    def _has_citation(self, document: Dict[str, Any]) -> bool:
        """문서에 Citation이 있는지 확인"""
        import re
        content = document.get("content", "") or document.get("text", "") or ""
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        law_pattern = r'[가-힣]+법\s*제?\s*\d+\s*조'
        precedent_pattern = r'대법원|법원.*\d{4}[다나마]\d+'
        
        has_law = bool(re.search(law_pattern, content))
        has_precedent = bool(re.search(precedent_pattern, content))
        
        return has_law or has_precedent
    
    def _calculate_citation_match_score(
        self,
        document: Dict[str, Any],
        query: str = "",
        extracted_keywords: List[str] = None
    ) -> float:
        """질문의 법령/판례와 문서 Citation 일치도 계산 (개선: Citation 매칭 정확도 개선)"""
        import re
        
        if extracted_keywords is None:
            extracted_keywords = []
        
        content = document.get("content", "") or document.get("text", "") or ""
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        if not content:
            return 0.0
        
        # 질문에서 법령/판례 추출
        query_laws = re.findall(r'([가-힣]+법)\s*제?\s*(\d+)\s*조', query)
        query_precedents = re.findall(r'대법원.*?(\d{4}[다나마]\d+)', query)
        
        # 문서에서 법령/판례 추출
        doc_laws = re.findall(r'([가-힣]+법)\s*제?\s*(\d+)\s*조', content)
        doc_precedents = re.findall(r'대법원.*?(\d{4}[다나마]\d+)', content)
        
        match_score = 0.0
        
        # 법령 일치도 계산
        if query_laws and doc_laws:
            for q_law, q_article in query_laws:
                for d_law, d_article in doc_laws:
                    if q_law == d_law and q_article == d_article:
                        match_score += 0.5  # 정확 일치
                    elif q_law == d_law:
                        match_score += 0.2  # 법령명만 일치
        
        # 판례 일치도 계산
        if query_precedents and doc_precedents:
            for q_precedent in query_precedents:
                if q_precedent in doc_precedents:
                    match_score += 0.5  # 정확 일치
        
        # 키워드에서 법령/판례 추출
        for keyword in extracted_keywords:
            keyword_laws = re.findall(r'([가-힣]+법)\s*제?\s*(\d+)\s*조', keyword)
            keyword_precedents = re.findall(r'대법원.*?(\d{4}[다나마]\d+)', keyword)
            
            if keyword_laws and doc_laws:
                for k_law, k_article in keyword_laws:
                    for d_law, d_article in doc_laws:
                        if k_law == d_law and k_article == d_article:
                            match_score += 0.3  # 키워드 일치
        
        return min(1.0, match_score)
    
    def _calculate_trust_score(self, document: Dict[str, Any]) -> float:
        """문서 신뢰도 점수 계산"""
        trust_score = 0.5
        
        source = document.get("source", "").lower()
        doc_type = document.get("type", "").lower() if document.get("type") else ""
        
        if "법령" in source or "statute" in source or "법령" in doc_type:
            trust_score += 0.3
        elif "판례" in source or "precedent" in source or "대법원" in source or "판례" in doc_type:
            trust_score += 0.2
        elif "해석" in source or "interpretation" in source:
            trust_score += 0.1
        
        metadata = document.get("metadata", {})
        if isinstance(metadata, dict):
            citation_count = metadata.get("citation_count", 0)
            if citation_count > 0:
                trust_score += min(0.2, citation_count / 100.0)
        
        return min(1.0, trust_score)
    
    def _apply_mmr_diversity(
        self,
        documents: List[Dict[str, Any]],
        query: str = "",
        lambda_score: float = 0.7,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        MMR (Maximal Marginal Relevance) 기반 다양성 적용
        
        Args:
            documents: 문서 리스트
            query: 검색 쿼리
            lambda_score: 관련성 가중치 (0.0-1.0)
            top_k: 반환할 결과 수
            
        Returns:
            List[Dict]: 다양성이 적용된 문서 리스트
        """
        if not documents:
            return []
        
        selected = []
        remaining = documents.copy()
        
        if not remaining:
            return []
        
        selected.append(remaining.pop(0))
        
        while remaining and len(selected) < top_k:
            best_doc = None
            best_score = -1
            
            for doc in remaining:
                relevance = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                
                diversity = 1.0
                if selected:
                    max_similarity = max([
                        self._calculate_doc_similarity(doc, sel_doc)
                        for sel_doc in selected
                    ])
                    diversity = 1.0 - max_similarity
                
                mmr_score = lambda_score * relevance + (1 - lambda_score) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = doc
            
            if best_doc:
                selected.append(best_doc)
                remaining.remove(best_doc)
            else:
                break
        
        return selected
    
    def _calculate_doc_similarity(
        self,
        doc1: Dict[str, Any],
        doc2: Dict[str, Any]
    ) -> float:
        """두 문서 간의 유사도 계산 (개선: MMR 다양성 계산 개선)"""
        content1 = doc1.get("content", "") or doc1.get("text", "") or ""
        content2 = doc2.get("content", "") or doc2.get("text", "") or ""
        
        if not isinstance(content1, str):
            content1 = str(content1) if content1 else ""
        if not isinstance(content2, str):
            content2 = str(content2) if content2 else ""
        
        if not content1 or not content2:
            return 0.0
        
        # 1. Jaccard 유사도 (단어 기반)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            jaccard_sim = 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard_sim = intersection / union if union > 0 else 0.0
        
        # 2. 문서 타입 유사도
        type1 = doc1.get("type", "").lower() if doc1.get("type") else ""
        type2 = doc2.get("type", "").lower() if doc2.get("type") else ""
        type_sim = 1.0 if type1 == type2 and type1 else 0.0
        
        # 3. Citation 유사도
        citations1 = self._extract_citations(doc1)
        citations2 = self._extract_citations(doc2)
        
        if citations1 and citations2:
            citation_intersection = len(set(citations1).intersection(set(citations2)))
            citation_union = len(set(citations1).union(set(citations2)))
            citation_sim = citation_intersection / citation_union if citation_union > 0 else 0.0
        else:
            citation_sim = 0.0
        
        # 가중 평균 유사도
        combined_sim = (
            0.5 * jaccard_sim +  # 단어 유사도
            0.2 * type_sim +     # 타입 유사도
            0.3 * citation_sim    # Citation 유사도
        )
        
        return min(1.0, combined_sim)
    
    def _extract_citations(self, document: Dict[str, Any]) -> List[str]:
        """문서에서 Citation 추출 (개선: MMR 다양성 계산 개선)"""
        import re
        content = document.get("content", "") or document.get("text", "") or ""
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        citations = []
        
        # 법령 추출
        laws = re.findall(r'([가-힣]+법)\s*제?\s*(\d+)\s*조', content)
        citations.extend([f"{law} 제{article}조" for law, article in laws])
        
        # 판례 추출
        precedents = re.findall(r'대법원.*?(\d{4}[다나마]\d+)', content)
        citations.extend(precedents)
        
        return citations
    
    def _calculate_information_density(self, document: Dict[str, Any]) -> float:
        """문서 정보 밀도 계산 (개선: 문서 길이/정보 밀도 고려)"""
        import re
        
        content = document.get("content", "") or document.get("text", "") or ""
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        if not content:
            return 0.0
        
        content_length = len(content)
        
        # 최적 길이 범위 (200-2000자)
        if 200 <= content_length <= 2000:
            length_score = 1.0
        elif content_length < 200:
            # 너무 짧음: 페널티
            length_score = max(0.3, content_length / 200.0)
        else:
            # 너무 김: 페널티
            length_score = max(0.5, 1.0 - (content_length - 2000) / 2000.0)
        
        # 법률 용어 밀도
        legal_terms = re.findall(r'[가-힣]+법|제\d+조|대법원|판례|법령', content)
        legal_term_density = len(legal_terms) / max(1, content_length / 100.0)
        legal_term_score = min(1.0, legal_term_density / 5.0)  # 최적 밀도: 5개/100자
        
        # Citation 밀도
        citations = self._extract_citations(document)
        citation_density = len(citations) / max(1, content_length / 500.0)
        citation_score = min(1.0, citation_density / 2.0)  # 최적 밀도: 2개/500자
        
        # 정보 밀도 점수
        density_score = (
            0.4 * length_score +
            0.3 * legal_term_score +
            0.3 * citation_score
        )
        
        return min(1.0, density_score)
    
    def _calculate_recency_score(self, document: Dict[str, Any]) -> float:
        """문서 최신성 점수 계산 (개선: 시간적 관련성 고려)"""
        from datetime import datetime
        
        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        
        # 판례 날짜 추출
        date_str = metadata.get("date") or metadata.get("decision_date") or ""
        if not date_str:
            return 0.5  # 날짜 정보 없음: 중간 점수
        
        try:
            # 날짜 파싱 (형식: "2020-01-01" 또는 "2020.01.01")
            if "." in date_str:
                date_obj = datetime.strptime(date_str.split()[0], "%Y.%m.%d")
            else:
                date_obj = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
            
            # 현재 날짜와의 차이 계산
            current_date = datetime.now()
            days_diff = (current_date - date_obj).days
            
            # 최신성 점수 (최근일수록 높은 점수)
            if days_diff <= 365:
                recency_score = 1.0  # 1년 이내
            elif days_diff <= 1825:
                recency_score = 0.8 - (days_diff - 365) / 1825.0 * 0.3  # 5년 이내
            else:
                recency_score = max(0.3, 0.5 - (days_diff - 1825) / 3650.0 * 0.2)  # 5년 이상
            
            return min(1.0, recency_score)
        except:
            return 0.5
    
    def apply_diversity_filter(self, results: List[Any], max_per_type: int = 5, diversity_weight: float = None) -> List[Any]:
        """
        다양성 필터 적용
        
        Args:
            results: 순위가 매겨진 결과 (MergedResult 또는 Dict)
            max_per_type: 타입별 최대 결과 수
            diversity_weight: 다양성 가중치 (사용하지 않지만 호환성을 위해 유지)
            
        Returns:
            List[Dict]: 다양성이 적용된 결과
        """
        if not results:
            return []
        
        # Dict를 MergedResult로 변환
        converted_results = []
        for result in results:
            if isinstance(result, dict):
                # Dict를 MergedResult로 변환
                text = result.get("text") or result.get("content") or result.get("chunk_text") or ""
                score = result.get("score") or result.get("relevance_score") or result.get("similarity", 0.0)
                source = result.get("source") or result.get("title") or result.get("document_id") or ""
                metadata = result.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = result if isinstance(result, dict) else {}
                
                converted_result = MergedResult(
                    text=text,
                    score=score,
                    source=source,
                    metadata=metadata
                )
                converted_results.append(converted_result)
            elif isinstance(result, MergedResult):
                converted_results.append(result)
            else:
                # 기타 타입은 건너뛰기
                continue
        
        # 타입별 카운터 (source를 타입으로 사용)
        type_counts = {}
        filtered_results = []
        
        for result in converted_results:
            result_type = result.source
            if result_type not in type_counts:
                type_counts[result_type] = 0
            
            if type_counts[result_type] < max_per_type:
                filtered_results.append(result)
                type_counts[result_type] += 1
        
        # Dict로 변환하여 반환 (호환성 유지)
        return [self._merged_result_to_dict(r) for r in filtered_results]


# 기본 인스턴스 생성
def create_result_merger() -> ResultMerger:
    """기본 결과 병합기 생성"""
    return ResultMerger()


def create_result_ranker() -> ResultRanker:
    """기본 결과 순위 결정기 생성"""
    return ResultRanker()


if __name__ == "__main__":
    # 테스트 코드
    merger = create_result_merger()
    ranker = create_result_ranker()
    
    # 샘플 결과
    exact_results = [
        type('obj', (object,), {'text': '민법 제543조', 'score': 0.9, 'metadata': {}})()
    ]
    semantic_results = [
        type('obj', (object,), {'text': '계약 해지 규정', 'score': 0.7, 'metadata': {}})()
    ]
    
    # 병합 및 순위 결정
    merged = merger.merge_results(exact_results, semantic_results)
    ranked = ranker.rank_results(merged)
    
    print(f"Ranked results: {len(ranked)}")
    for result in ranked:
        print(f"  Score: {result.score:.3f}, Source: {result.source}")
        print(f"  Text: {result.text}")