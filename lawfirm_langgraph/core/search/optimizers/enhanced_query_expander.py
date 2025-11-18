# -*- coding: utf-8 -*-
"""
Enhanced Query Expander
강화된 쿼리 확장 모듈
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExpandedQuery:
    """확장된 쿼리 정보"""
    original: str
    expanded_keywords: List[str]
    synonyms: List[str]
    related_terms: List[str]
    negative_keywords: List[str]
    query_type: str


class EnhancedQueryExpander:
    """강화된 쿼리 확장기"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        self._legal_synonyms = self._load_legal_synonyms()
        self.logger.info("EnhancedQueryExpander initialized")
    
    def _load_legal_synonyms(self) -> Dict[str, List[str]]:
        """법률 용어 동의어 사전 로드"""
        return {
            "계약": ["계약서", "계약관계", "계약당사자", "계약조건", "계약체결", "계약해지"],
            "손해배상": ["손해 배상", "손해보상", "배상책임", "손해전보", "손해보상청구", "배상"],
            "불법행위": ["불법", "위법행위", "불법적 행위", "불법행위책임"],
            "이혼": ["이혼소송", "이혼절차", "이혼신고", "협의이혼", "재판이혼"],
            "상속": ["상속분", "상속재산", "상속인", "상속포기", "상속세"],
            "임대차": ["임대", "임차", "임대인", "임차인", "보증금", "전세", "월세"],
            "소유권": ["소유", "소유자", "소유권이전", "소유권보존등기"],
            "채권": ["채권자", "채권채무", "채권양도", "채권압류"],
            "채무": ["채무자", "채무불이행", "채무면제", "채무인수"],
            "계약해지": ["계약해제", "계약취소", "계약무효", "계약철회"],
            "계약위반": ["계약불이행", "계약위반책임", "계약위반손해"],
            "손해": ["손실", "피해", "재산상 손해", "정신적 손해"],
            "배상": ["보상", "전보", "배상금", "배상책임"],
            "과실": ["과실책임", "과실상계", "과실비율"],
            "인과관계": ["인과관계입증", "인과관계단절"],
        }
    
    def expand_query(
        self,
        query: str,
        query_type: str = "general_question",
        extracted_keywords: Optional[List[str]] = None,
        use_llm: bool = False,
        llm_client: Optional[Any] = None
    ) -> ExpandedQuery:
        """
        쿼리 확장 실행
        
        Args:
            query: 원본 쿼리
            query_type: 질문 유형
            extracted_keywords: 추출된 키워드
            use_llm: LLM 기반 확장 사용 여부
            llm_client: LLM 클라이언트 (use_llm=True일 때 필요)
        
        Returns:
            ExpandedQuery: 확장된 쿼리 정보
        """
        try:
            self.logger.debug(f"Expanding query: {query[:50]}... (type: {query_type})")
            
            # 1. 동의어 확장
            synonyms = self._expand_synonyms(query, extracted_keywords)
            
            # 2. 관련 용어 확장
            related_terms = self._expand_related_terms(query, query_type, extracted_keywords)
            
            # 3. LLM 기반 확장 (선택적)
            llm_expanded = []
            if use_llm and llm_client:
                llm_expanded = self._llm_expand(query, query_type, llm_client)
            
            # 4. Negative keywords 추출
            negative_keywords = self._extract_negative_keywords(query)
            
            # 모든 확장 키워드 통합
            expanded_keywords = list(set(synonyms + related_terms + llm_expanded))
            
            # 원본 키워드 제거 (중복 방지)
            if extracted_keywords:
                expanded_keywords = [kw for kw in expanded_keywords if kw not in extracted_keywords]
            
            self.logger.info(
                f"Query expansion completed: {len(expanded_keywords)} keywords "
                f"(synonyms: {len(synonyms)}, related: {len(related_terms)}, "
                f"llm: {len(llm_expanded)})"
            )
            
            return ExpandedQuery(
                original=query,
                expanded_keywords=expanded_keywords,
                synonyms=synonyms,
                related_terms=related_terms,
                negative_keywords=negative_keywords,
                query_type=query_type
            )
        
        except Exception as e:
            self.logger.error(f"Query expansion failed: {e}")
            return ExpandedQuery(
                original=query,
                expanded_keywords=[],
                synonyms=[],
                related_terms=[],
                negative_keywords=[],
                query_type=query_type
            )
    
    def _expand_synonyms(
        self,
        query: str,
        extracted_keywords: Optional[List[str]] = None
    ) -> List[str]:
        """동의어 확장"""
        synonyms = []
        
        # 추출된 키워드에서 동의어 찾기
        keywords_to_check = extracted_keywords or [query]
        
        for keyword in keywords_to_check:
            keyword_lower = keyword.lower().strip()
            
            # 동의어 사전에서 찾기
            for term, term_synonyms in self._legal_synonyms.items():
                if term in keyword_lower or keyword_lower in term:
                    synonyms.extend(term_synonyms)
                    break
        
        return list(set(synonyms))
    
    def _expand_related_terms(
        self,
        query: str,
        query_type: str,
        extracted_keywords: Optional[List[str]] = None
    ) -> List[str]:
        """관련 용어 확장"""
        related_terms = []
        
        # 질문 유형별 관련 용어
        type_specific_terms = {
            "law_inquiry": ["법령", "조문", "법률", "규정", "법조문"],
            "precedent_search": ["판례", "판결", "대법원", "법원", "사건"],
            "procedure_inquiry": ["절차", "신청", "제출", "처리", "절차"],
            "legal_advice": ["법률자문", "법률상담", "법률검토", "법률의견"],
        }
        
        if query_type in type_specific_terms:
            related_terms.extend(type_specific_terms[query_type])
        
        # 복합어 분리
        if extracted_keywords:
            for keyword in extracted_keywords:
                # 2글자 이상인 경우 분리 시도
                if len(keyword) >= 4:
                    # 예: "손해배상" -> "손해", "배상"
                    mid = len(keyword) // 2
                    related_terms.extend([keyword[:mid], keyword[mid:]])
        
        return list(set(related_terms))
    
    def _llm_expand(
        self,
        query: str,
        query_type: str,
        llm_client: Any
    ) -> List[str]:
        """LLM 기반 쿼리 확장"""
        try:
            prompt = f"""다음 법률 질문과 관련된 검색 키워드를 5개 이내로 제시해주세요.

질문: {query}
유형: {query_type}

응답 형식: 키워드1, 키워드2, 키워드3, ...
응답만 제공하고 설명은 생략해주세요."""
            
            response = llm_client.generate(prompt)
            
            # 응답 파싱
            keywords = [kw.strip() for kw in response.split(",") if kw.strip()]
            return keywords[:5]  # 최대 5개
        
        except Exception as e:
            self.logger.warning(f"LLM expansion failed: {e}")
            return []
    
    def _extract_negative_keywords(self, query: str) -> List[str]:
        """제외 키워드 추출"""
        negative_keywords = []
        
        # 부정 표현 패턴
        negative_patterns = [
            r"제외\s*[하고외]",
            r"빼고",
            r"아닌",
            r"제외하고",
        ]
        
        import re
        for pattern in negative_patterns:
            matches = re.findall(pattern, query)
            if matches:
                # 부정 표현 이후의 키워드 추출
                parts = re.split(pattern, query)
                if len(parts) > 1:
                    negative_keywords.extend([kw.strip() for kw in parts[1].split() if kw.strip()])
        
        return list(set(negative_keywords))

