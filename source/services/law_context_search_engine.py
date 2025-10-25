# -*- coding: utf-8 -*-
"""
Law Context Search Engine
법령 컨텍스트 검색 시스템
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RelatedArticle:
    """관련 조문 데이터 클래스"""
    article_number: int
    article_title: str
    article_content: str
    is_target: bool
    distance: int


@dataclass
class LawDefinition:
    """법령 정의 데이터 클래스"""
    law_name: str
    law_type: str
    ministry: str
    promulgation_date: str
    enforcement_date: str
    summary: str
    keywords: str
    main_article_count: int
    supplementary_article_count: int


class LawContextSearchEngine:
    """법령 컨텍스트 검색 시스템"""
    
    def __init__(self, db_manager, vector_store):
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Law Context Search Engine 초기화 완료")
    
    async def search_related_articles(self, law_name: str, article_number: int, context_range: int = 3) -> List[RelatedArticle]:
        """관련 조문 검색 (전후 조문 포함)"""
        try:
            # 해당 법령의 조문 범위 검색
            query = """
                SELECT cla.*
                FROM current_laws_articles cla
                WHERE cla.law_name_korean = ? 
                AND cla.article_number BETWEEN ? AND ?
                ORDER BY cla.article_number
            """
            
            start_article = max(1, article_number - context_range)
            end_article = article_number + context_range
            
            results = self.db_manager.execute_query(query, (law_name, start_article, end_article))
            
            related_articles = []
            for result in results:
                related_articles.append(RelatedArticle(
                    article_number=result['article_number'],
                    article_title=result.get('article_title', ''),
                    article_content=result['article_content'],
                    is_target=result['article_number'] == article_number,
                    distance=abs(result['article_number'] - article_number)
                ))
            
            self.logger.info(f"Found {len(related_articles)} related articles for {law_name} 제{article_number}조")
            return related_articles
            
        except Exception as e:
            self.logger.error(f"Related articles search failed: {e}")
            return []
    
    async def search_law_definition(self, law_name: str) -> Optional[LawDefinition]:
        """법령 정의 및 기본 정보 검색"""
        try:
            # 법령 기본 정보 검색
            query = """
                SELECT * FROM assembly_laws 
                WHERE law_name = ?
                ORDER BY parsing_quality_score DESC
                LIMIT 1
            """
            
            results = self.db_manager.execute_query(query, (law_name,))
            
            if results:
                law_info = results[0]
                return LawDefinition(
                    law_name=law_info['law_name'],
                    law_type=law_info.get('law_type', ''),
                    ministry=law_info.get('ministry', ''),
                    promulgation_date=law_info.get('promulgation_date', ''),
                    enforcement_date=law_info.get('enforcement_date', ''),
                    summary=law_info.get('summary', ''),
                    keywords=law_info.get('keywords', ''),
                    main_article_count=law_info.get('main_article_count', 0),
                    supplementary_article_count=law_info.get('supplementary_article_count', 0)
                )
            
            self.logger.debug(f"No law definition found for {law_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"Law definition search failed: {e}")
            return None
    
    async def search_similar_laws(self, law_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """유사한 법령 검색"""
        try:
            # 법령명으로 벡터 검색
            vector_results = self.vector_store.search(law_name, top_k=top_k)
            
            similar_laws = []
            for result in vector_results:
                metadata = result.get('metadata', {})
                if metadata.get('law_name') != law_name:  # 자기 자신 제외
                    similar_laws.append({
                        'law_name': metadata.get('law_name', ''),
                        'similarity': result.get('similarity', 0.0),
                        'law_type': metadata.get('law_type', ''),
                        'ministry': metadata.get('ministry', ''),
                        'summary': metadata.get('summary', '')
                    })
            
            return similar_laws
            
        except Exception as e:
            self.logger.error(f"Similar laws search failed: {e}")
            return []
    
    async def search_articles_by_keywords(self, keywords: List[str], law_name: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """키워드 기반 조문 검색"""
        try:
            search_query = " ".join(keywords)
            
            # 벡터 검색 실행
            vector_results = self.vector_store.search(search_query, top_k=top_k)
            
            articles = []
            for result in vector_results:
                metadata = result.get('metadata', {})
                
                # 법령명 필터링 (지정된 경우)
                if law_name and metadata.get('law_name') != law_name:
                    continue
                
                articles.append({
                    'law_name': metadata.get('law_name', ''),
                    'article_number': metadata.get('article_number', ''),
                    'article_title': metadata.get('article_title', ''),
                    'content': result['content'],
                    'similarity': result.get('similarity', 0.0),
                    'metadata': metadata
                })
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Articles by keywords search failed: {e}")
            return []
    
    async def get_law_hierarchy(self, law_name: str) -> Dict[str, Any]:
        """법령 계층 구조 조회"""
        try:
            # 부모 법령 검색
            parent_query = """
                SELECT * FROM assembly_laws 
                WHERE law_name = ? AND parent_law IS NOT NULL
                LIMIT 1
            """
            parent_results = self.db_manager.execute_query(parent_query, (law_name,))
            
            # 자식 법령 검색
            child_query = """
                SELECT * FROM assembly_laws 
                WHERE parent_law = ?
                ORDER BY promulgation_date DESC
            """
            child_results = self.db_manager.execute_query(child_query, (law_name,))
            
            # 관련 법령 검색
            related_query = """
                SELECT * FROM assembly_laws 
                WHERE related_laws LIKE ? AND law_name != ?
                ORDER BY promulgation_date DESC
                LIMIT 5
            """
            related_results = self.db_manager.execute_query(related_query, (f"%{law_name}%", law_name))
            
            hierarchy = {
                'current_law': law_name,
                'parent_laws': parent_results if parent_results else [],
                'child_laws': child_results if child_results else [],
                'related_laws': related_results if related_results else []
            }
            
            return hierarchy
            
        except Exception as e:
            self.logger.error(f"Law hierarchy search failed: {e}")
            return {'current_law': law_name, 'parent_laws': [], 'child_laws': [], 'related_laws': []}
    
    async def search_articles_by_date_range(self, law_name: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """날짜 범위로 조문 검색"""
        try:
            query = """
                SELECT cla.*
                FROM current_laws_articles cla
                WHERE cla.law_name_korean = ? 
                AND cla.effective_date BETWEEN ? AND ?
                ORDER BY cla.article_number
            """
            
            results = self.db_manager.execute_query(query, (law_name, start_date, end_date))
            
            articles = []
            for result in results:
                articles.append({
                    'article_number': result['article_number'],
                    'article_title': result.get('article_title', ''),
                    'article_content': result['article_content'],
                    'promulgation_date': result.get('promulgation_date', ''),
                    'enforcement_date': result.get('enforcement_date', '')
                })
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Articles by date range search failed: {e}")
            return []
    
    async def get_law_statistics(self, law_name: str = None) -> Dict[str, Any]:
        """법령 통계 정보 조회"""
        try:
            stats = {}
            
            if law_name:
                # 특정 법령의 통계
                query = """
                    SELECT 
                        COUNT(cla.article_id) as total_articles,
                        COUNT(CASE WHEN cla.is_supplementary = 0 THEN 1 END) as main_articles,
                        COUNT(CASE WHEN cla.is_supplementary = 1 THEN 1 END) as supplementary_articles,
                        AVG(cla.quality_score) as avg_quality_score,
                        COUNT(CASE WHEN cla.paragraph_content IS NOT NULL THEN 1 END) as articles_with_paragraphs,
                        cla.law_name_korean,
                        cla.effective_date
                    FROM current_laws_articles cla
                    WHERE cla.law_name_korean = ?
                    GROUP BY cla.law_name_korean
                """
                results = self.db_manager.execute_query(query, (law_name,))
            else:
                # 전체 법령 통계
                query = """
                    SELECT 
                        COUNT(DISTINCT cla.law_name_korean) as total_laws,
                        COUNT(cla.article_id) as total_articles,
                        COUNT(CASE WHEN cla.is_supplementary = 0 THEN 1 END) as main_articles,
                        COUNT(CASE WHEN cla.is_supplementary = 1 THEN 1 END) as supplementary_articles,
                        AVG(cla.quality_score) as avg_quality_score,
                        COUNT(CASE WHEN cla.paragraph_content IS NOT NULL THEN 1 END) as articles_with_paragraphs
                    FROM current_laws_articles cla
                """
                results = self.db_manager.execute_query(query)
            
            if results:
                stats = results[0]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Law statistics failed: {e}")
            return {}
    
    async def search_articles_by_ministry(self, ministry_name: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """소관부처별 조문 검색"""
        try:
            query = """
                SELECT cla.*
                FROM current_laws_articles cla
                WHERE cla.law_name_korean LIKE ?
                ORDER BY cla.quality_score DESC
                LIMIT ?
            """
            
            results = self.db_manager.execute_query(query, (f"%{ministry_name}%", top_k))
            
            articles = []
            for result in results:
                articles.append({
                    'law_name': result['law_name'],
                    'article_number': result['article_number'],
                    'article_title': result.get('article_title', ''),
                    'article_content': result['article_content'],
                    'ministry': result.get('ministry', ''),
                    'parsing_quality_score': result.get('parsing_quality_score', 0.0)
                })
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Articles by ministry search failed: {e}")
            return []

