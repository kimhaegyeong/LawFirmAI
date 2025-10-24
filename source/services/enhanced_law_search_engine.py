# -*- coding: utf-8 -*-
"""
Enhanced Law Search Engine
법령 테이블과 벡터 스토어를 활용한 향상된 조문 검색 엔진
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ArticleSearchResult:
    """조문 검색 결과 데이터 클래스"""
    content: str
    law_name: str
    article_number: str
    article_title: Optional[str] = None
    similarity: float = 1.0
    source: str = "exact_article"
    type: str = "current_law"
    metadata: Dict[str, Any] = None


class EnhancedLawSearchEngine:
    """법령 테이블과 벡터 스토어를 활용한 향상된 조문 검색 엔진"""
    
    def __init__(self, db_manager, vector_store):
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
        
        # 조문 패턴 매칭 강화
        self.article_patterns = [
            r'(\w+법)\s*제\s*(\d+)조\s*제\s*(\d+)항',  # 민법 제750조 제1항
            r'(\w+법)\s*제\s*(\d+)조',                 # 민법 제750조
            r'제\s*(\d+)조\s*제\s*(\d+)항',            # 제750조 제1항
            r'제\s*(\d+)조',                           # 제750조
            r'(\w+법)\s*(\d+)조',                     # 민법 750조
        ]
        
        # 법령명 매핑
        self.law_name_mapping = {
            '민법': '민법',
            '형법': '형법',
            '상법': '상법',
            '행정법': '행정법',
            '민사소송법': '민사소송법',
            '형사소송법': '형사소송법',
            '노동법': '근로기준법',
            '근로기준법': '근로기준법',
            '가족법': '가족법',
            '부동산법': '부동산법'
        }
        
        self.logger.info("Enhanced Law Search Engine 초기화 완료")
    
    async def search_specific_article(self, query: str) -> Optional[ArticleSearchResult]:
        """특정 조문 검색 (정확도 최우선)"""
        try:
            # 1. 조문 패턴 분석
            article_info = self._extract_article_info(query)
            
            if not article_info:
                self.logger.debug(f"No article pattern found in query: {query}")
                return None
            
            self.logger.info(f"Extracted article info: {article_info}")
            
            # 2. 정확한 조문 검색
            exact_result = await self._search_exact_article(article_info)
            
            if exact_result:
                return ArticleSearchResult(
                    content=exact_result['article_content'],
                    law_name=exact_result['law_name'],
                    article_number=str(exact_result['article_number']),
                    article_title=exact_result.get('article_title', ''),
                    similarity=1.0,
                    source='exact_article',
                    type='current_law',
                    metadata={
                        'law_id': exact_result['law_id'],
                        'article_id': exact_result['article_id'],
                        'is_supplementary': exact_result.get('is_supplementary', False),
                        'parsing_quality_score': exact_result.get('parsing_quality_score', 0.0)
                    }
                )
            
            # 3. 유사 조문 검색 (패턴 매칭 실패 시)
            return await self._search_similar_article(query, article_info)
            
        except Exception as e:
            self.logger.error(f"Specific article search failed: {e}")
            return None
    
    def _extract_article_info(self, query: str) -> Optional[Dict[str, str]]:
        """쿼리에서 법령명과 조문번호 추출"""
        for pattern in self.article_patterns:
            match = re.search(pattern, query)
            if match:
                groups = match.groups()
                
                if len(groups) == 3:  # 법령명 + 조문번호 + 항번호
                    return {
                        'law_name': groups[0],
                        'article_number': groups[1],
                        'paragraph_number': groups[2]
                    }
                elif len(groups) == 2:  # 법령명 + 조문번호 또는 조문번호 + 항번호
                    if groups[0].endswith('법'):
                        return {
                            'law_name': groups[0],
                            'article_number': groups[1]
                        }
                    else:
                        return {
                            'law_name': '민법',  # 기본값
                            'article_number': groups[0],
                            'paragraph_number': groups[1]
                        }
                elif len(groups) == 1:  # 조문번호만
                    return {
                        'law_name': '민법',  # 기본값
                        'article_number': groups[0]
                    }
        
        return None
    
    async def _search_exact_article(self, article_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """정확한 조문 검색 (현행법령 우선)"""
        law_name = article_info['law_name']
        article_number = int(article_info['article_number'])
        
        # 1. 현행법령 조문에서 검색 (우선순위)
        try:
            current_laws_results = self.db_manager.search_current_laws_articles(law_name, article_number)
            if current_laws_results:
                return self._format_current_laws_result(current_laws_results[0])
        except Exception as e:
            self.logger.warning(f"현행법령 조문 검색 실패: {e}")
        
        # 2. Assembly 조문 테이블에서 검색 (폴백)
        query = """
            SELECT aa.*, al.law_name, al.law_id
            FROM assembly_articles aa
            JOIN assembly_laws al ON aa.law_id = al.law_id
            WHERE al.law_name = ? AND aa.article_number = ?
            ORDER BY aa.parsing_quality_score DESC, aa.word_count DESC
            LIMIT 1
        """
        
        try:
            results = self.db_manager.execute_query(query, (law_name, article_number))
            
            if results:
                result = results[0]
                
                # 항번호가 있는 경우 해당 항만 추출
                if 'paragraph_number' in article_info:
                    paragraph_content = self._extract_paragraph_content(
                        result['article_content'], 
                        int(article_info['paragraph_number'])
                    )
                    if paragraph_content:
                        result['article_content'] = paragraph_content
                
                self.logger.info(f"Found exact article: {law_name} 제{article_number}조")
                return result
            
            self.logger.debug(f"No exact article found for {law_name} 제{article_number}조")
            return None
            
        except Exception as e:
            self.logger.error(f"Exact article search failed: {e}")
            return None
    
    async def _search_similar_article(self, query: str, article_info: Dict[str, str]) -> Optional[ArticleSearchResult]:
        """유사 조문 검색 (벡터 검색 활용)"""
        try:
            # 벡터 검색으로 유사한 조문 찾기
            vector_results = self.vector_store.search(query, top_k=5)
            
            # 법령명과 조문번호로 필터링
            filtered_results = []
            for result in vector_results:
                metadata = result.get('metadata', {})
                if (metadata.get('law_name') == article_info['law_name'] and
                    str(metadata.get('article_number')) == article_info['article_number']):
                    filtered_results.append(result)
            
            if filtered_results:
                best_result = filtered_results[0]
                return ArticleSearchResult(
                    content=best_result['content'],
                    law_name=article_info['law_name'],
                    article_number=article_info['article_number'],
                    similarity=best_result.get('similarity', 0.8),
                    source='similar_article',
                    type='current_law',
                    metadata=best_result.get('metadata', {})
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Similar article search failed: {e}")
            return None
    
    def _format_current_laws_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """현행법령 조문 결과 포맷팅"""
        # 조문 내용 구성
        content_parts = [result['article_content']]
        
        if result.get('paragraph_content'):
            content_parts.append(f"항: {result['paragraph_content']}")
        
        if result.get('sub_paragraph_content'):
            content_parts.append(f"호: {result['sub_paragraph_content']}")
        
        full_content = "\n".join(content_parts)
        
        return {
            'content': full_content,
            'law_name': result['law_name_korean'],
            'article_number': result['article_number'],
            'article_title': result.get('article_title', ''),
            'similarity': 1.0,
            'source': 'current_laws_articles',
            'type': 'current_law',
            'metadata': {
                'article_id': result['article_id'],
                'law_id': result['law_id'],
                'paragraph_number': result.get('paragraph_number'),
                'sub_paragraph_number': result.get('sub_paragraph_number'),
                'quality_score': result.get('quality_score', 0.9),
                'ministry_name': result.get('ministry_name', ''),
                'effective_date': result.get('effective_date', ''),
                'parsing_method': result.get('parsing_method', 'batch_parser')
            }
        }
    
    def _extract_paragraph_content(self, article_content: str, paragraph_number: int) -> Optional[str]:
        """조문 내용에서 특정 항의 내용 추출"""
        try:
            # 항 번호 패턴 매칭
            patterns = [
                rf'제{paragraph_number}항\s*([^제]+?)(?=제\d+항|$)',
                rf'{paragraph_number}항\s*([^제]+?)(?=제\d+항|$)',
                rf'\( {paragraph_number} \)\s*([^\(]+?)(?=\( \d+ \)|$)',
                rf'\({paragraph_number}\)\s*([^\(]+?)(?=\(\d+\)|$)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, article_content, re.DOTALL)
                if match:
                    return match.group(1).strip()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Paragraph extraction failed: {e}")
            return None
    
    async def search_by_keywords(self, keywords: List[str], law_name: str = None) -> List[ArticleSearchResult]:
        """키워드 기반 조문 검색"""
        try:
            results = []
            
            # 키워드 조합으로 검색 쿼리 생성
            search_query = " ".join(keywords)
            
            # 벡터 검색 실행
            vector_results = self.vector_store.search(search_query, top_k=10)
            
            # 법령명 필터링 (지정된 경우)
            for result in vector_results:
                metadata = result.get('metadata', {})
                
                if law_name and metadata.get('law_name') != law_name:
                    continue
                
                results.append(ArticleSearchResult(
                    content=result['content'],
                    law_name=metadata.get('law_name', ''),
                    article_number=str(metadata.get('article_number', '')),
                    article_title=metadata.get('article_title', ''),
                    similarity=result.get('similarity', 0.7),
                    source='keyword_search',
                    type='current_law',
                    metadata=metadata
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            return []
    
    async def get_article_statistics(self, law_name: str = None) -> Dict[str, Any]:
        """조문 통계 정보 조회"""
        try:
            stats = {}
            
            if law_name:
                # 특정 법령의 조문 통계
                query = """
                    SELECT 
                        COUNT(*) as total_articles,
                        COUNT(CASE WHEN is_supplementary = 0 THEN 1 END) as main_articles,
                        COUNT(CASE WHEN is_supplementary = 1 THEN 1 END) as supplementary_articles,
                        AVG(parsing_quality_score) as avg_quality_score,
                        AVG(word_count) as avg_word_count
                    FROM assembly_articles aa
                    JOIN assembly_laws al ON aa.law_id = al.law_id
                    WHERE al.law_name = ?
                """
                results = self.db_manager.execute_query(query, (law_name,))
            else:
                # 전체 조문 통계
                query = """
                    SELECT 
                        COUNT(*) as total_articles,
                        COUNT(CASE WHEN is_supplementary = 0 THEN 1 END) as main_articles,
                        COUNT(CASE WHEN is_supplementary = 1 THEN 1 END) as supplementary_articles,
                        AVG(parsing_quality_score) as avg_quality_score,
                        AVG(word_count) as avg_word_count
                    FROM assembly_articles
                """
                results = self.db_manager.execute_query(query)
            
            if results:
                stats = results[0]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Article statistics failed: {e}")
            return {}
