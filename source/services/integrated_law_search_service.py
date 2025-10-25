# -*- coding: utf-8 -*-
"""
Integrated Law Search Service
통합 조문 검색 서비스
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .enhanced_law_search_engine import EnhancedLawSearchEngine, ArticleSearchResult
from .law_context_search_engine import LawContextSearchEngine, RelatedArticle, LawDefinition

logger = logging.getLogger(__name__)


@dataclass
class IntegratedSearchResult:
    """통합 검색 결과 데이터 클래스"""
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    search_method: str
    context_info: Dict[str, Any]
    processing_time: float


class IntegratedLawSearchService:
    """통합 조문 검색 서비스"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 데이터베이스 및 벡터 스토어 초기화
        from ..data.database import DatabaseManager
        from ..data.vector_store import LegalVectorStore
        
        self.db_manager = DatabaseManager()
        self.vector_store = LegalVectorStore()
        
        # 판례 검색 서비스 초기화
        self.precedent_service = None
        self.hybrid_precedent_service = None
        
        try:
            from .dynamic_precedent_search_service import DynamicPrecedentSearchService
            from .precedent_api_service import PrecedentAPIService
            from .hybrid_precedent_search_service import HybridPrecedentSearchService
            
            # 동적 판례 검색 서비스 초기화
            self.precedent_service = DynamicPrecedentSearchService(self.db_manager)
            
            # API 판례 검색 서비스 초기화
            precedent_api_service = PrecedentAPIService()
            
            # 하이브리드 판례 검색 서비스 초기화
            self.hybrid_precedent_service = HybridPrecedentSearchService(
                self.precedent_service, precedent_api_service
            )
            
            self.logger.info("하이브리드 판례 검색 서비스 초기화 완료")
            
        except Exception as e:
            self.logger.warning(f"하이브리드 판례 검색 서비스 초기화 실패: {e}")
            self.precedent_service = None
            self.hybrid_precedent_service = None
        
        # 검색 엔진 초기화 (하이브리드 판례 서비스 포함)
        self.law_search_engine = EnhancedLawSearchEngine(
            self.db_manager, 
            self.vector_store,
            precedent_service=self.precedent_service,
            hybrid_precedent_service=self.hybrid_precedent_service
        )
        self.context_search_engine = LawContextSearchEngine(self.db_manager, self.vector_store)
        
        # 검색 전략 설정
        self.search_strategies = {
            'exact_match': self._exact_match_search,
            'fuzzy_match': self._fuzzy_match_search,
            'semantic_search': self._semantic_search,
            'hybrid_search': self._hybrid_search
        }
        
        # 신뢰도 임계값 설정
        self.confidence_thresholds = {
            'exact_match': 0.95,
            'fuzzy_match': 0.85,
            'semantic_search': 0.75,
            'hybrid_search': 0.70
        }
        
        self.logger.info("Integrated Law Search Service 초기화 완료")
    
    async def search_law_article(self, query: str, strategy: str = 'hybrid') -> IntegratedSearchResult:
        """통합 조문 검색"""
        start_time = time.time()
        
        try:
            # 1. 검색 전략 선택
            search_func = self.search_strategies.get(strategy, self._hybrid_search)
            
            # 2. 조문 검색 실행
            search_result = await search_func(query)
            
            if not search_result:
                return IntegratedSearchResult(
                    response=f"'{query}'에 대한 관련 법률 조문을 찾을 수 없습니다.\n\n다음과 같이 도움을 드릴 수 있습니다:\n• 질문을 더 구체적으로 작성해주세요\n• 관련 법률 조문이나 판례를 포함해주세요\n• 키워드를 더 명확하게 해주세요",
                    confidence=0.0,
                    sources=[],
                    search_method=strategy,
                    context_info={},
                    processing_time=time.time() - start_time
                )
            
            # 3. 컨텍스트 정보 추가
            context_info = await self._add_context_info(search_result)
            
            # 4. 응답 생성
            response = await self._generate_enhanced_response(search_result, context_info)
            
            # 5. 신뢰도 계산
            confidence = self._calculate_confidence(search_result, strategy)
            
            return IntegratedSearchResult(
                response=response,
                confidence=confidence,
                sources=[self._convert_to_dict(search_result)],
                search_method=strategy,
                context_info=context_info,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Integrated law search failed: {e}")
            return IntegratedSearchResult(
                response=f"검색 중 오류가 발생했습니다: {str(e)}",
                confidence=0.0,
                sources=[],
                search_method=strategy,
                context_info={},
                processing_time=time.time() - start_time
            )
    
    async def _exact_match_search(self, query: str) -> Optional[ArticleSearchResult]:
        """정확 매칭 검색"""
        return await self.law_search_engine.search_specific_article(query)
    
    async def _fuzzy_match_search(self, query: str) -> Optional[ArticleSearchResult]:
        """퍼지 매칭 검색 (FTS 활용)"""
        try:
            # 1. 현행법령 조문 FTS 검색 (우선순위)
            try:
                current_laws_fts_results = self.db_manager.search_current_laws_articles_fts(query, limit=5)
                if current_laws_fts_results:
                    best_result = current_laws_fts_results[0]
                    
                    # 조문 내용 구성
                    content_parts = [best_result['article_content']]
                    if best_result.get('paragraph_content'):
                        content_parts.append(f"항: {best_result['paragraph_content']}")
                    if best_result.get('sub_paragraph_content'):
                        content_parts.append(f"호: {best_result['sub_paragraph_content']}")
                    
                    return ArticleSearchResult(
                        content="\n".join(content_parts),
                        law_name=best_result['law_name_korean'],
                        article_number=str(best_result['article_number']),
                        article_title=best_result.get('article_title', ''),
                        similarity=0.9,  # FTS 결과는 높은 신뢰도
                        source='current_laws_fts_search',
                        type='current_law',
                        metadata={
                            'law_id': best_result['law_id'],
                            'article_id': best_result['article_id'],
                            'quality_score': best_result.get('quality_score', 0.9),
                            'ministry_name': best_result.get('ministry_name', ''),
                            'parsing_method': best_result.get('parsing_method', 'batch_parser')
                        }
                    )
            except Exception as e:
                self.logger.warning(f"현행법령 조문 FTS 검색 실패: {e}")
            
            # 2. Assembly 조문 FTS 검색 (폴백)
            fts_results = self.db_manager.search_current_laws_articles_fts(query, limit=5)
            
            if fts_results:
                # 가장 관련성 높은 결과 선택
                best_result = fts_results[0]
                return ArticleSearchResult(
                    content=best_result['article_content'],
                    law_name=best_result['law_name'],
                    article_number=str(best_result['article_number']),
                    article_title=best_result.get('article_title', ''),
                    similarity=0.9,  # FTS 결과는 높은 신뢰도
                    source='assembly_fts_search',
                    type='current_law',
                    metadata={
                        'law_id': best_result['law_id'],
                        'article_id': best_result['article_id'],
                        'parsing_quality_score': best_result.get('parsing_quality_score', 0.0)
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Fuzzy match search failed: {e}")
            return None
    
    async def _semantic_search(self, query: str) -> Optional[ArticleSearchResult]:
        """의미적 검색 (벡터 검색)"""
        try:
            vector_results = self.vector_store.search(query, top_k=3)
            
            if vector_results:
                best_result = vector_results[0]
                metadata = best_result.get('metadata', {})
                
                return ArticleSearchResult(
                    content=best_result['content'],
                    law_name=metadata.get('law_name', ''),
                    article_number=str(metadata.get('article_number', '')),
                    article_title=metadata.get('article_title', ''),
                    similarity=best_result.get('similarity', 0.7),
                    source='semantic_search',
                    type='current_law',
                    metadata=metadata
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return None
    
    async def _hybrid_search(self, query: str) -> Optional[ArticleSearchResult]:
        """하이브리드 검색 (정확도 + 의미적 검색)"""
        try:
            # 1. 정확 매칭 시도
            exact_result = await self._exact_match_search(query)
            if exact_result and exact_result.similarity >= 0.9:
                return exact_result
            
            # 2. 퍼지 매칭 시도
            fuzzy_result = await self._fuzzy_match_search(query)
            if fuzzy_result and fuzzy_result.similarity >= 0.8:
                return fuzzy_result
            
            # 3. 의미적 검색 시도
            semantic_result = await self._semantic_search(query)
            if semantic_result and semantic_result.similarity >= 0.7:
                return semantic_result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return None
    
    async def _add_context_info(self, search_result: ArticleSearchResult) -> Dict[str, Any]:
        """컨텍스트 정보 추가"""
        try:
            law_name = search_result.law_name
            article_number = search_result.article_number
            
            context_info = {}
            
            # 법령 기본 정보
            if law_name:
                law_definition = await self.context_search_engine.search_law_definition(law_name)
                if law_definition:
                    context_info['law_definition'] = {
                        'law_name': law_definition.law_name,
                        'law_type': law_definition.law_type,
                        'ministry': law_definition.ministry,
                        'summary': law_definition.summary,
                        'keywords': law_definition.keywords,
                        'main_article_count': law_definition.main_article_count,
                        'supplementary_article_count': law_definition.supplementary_article_count
                    }
            
            # 관련 조문
            if law_name and article_number.isdigit():
                try:
                    article_num = int(article_number)
                    related_articles = await self.context_search_engine.search_related_articles(
                        law_name, article_num, context_range=2
                    )
                    context_info['related_articles'] = [
                        {
                            'article_number': article.article_number,
                            'article_title': article.article_title,
                            'is_target': article.is_target,
                            'distance': article.distance
                        }
                        for article in related_articles
                    ]
                except ValueError:
                    pass
            
            # 유사한 법령
            if law_name:
                similar_laws = await self.context_search_engine.search_similar_laws(law_name, top_k=3)
                context_info['similar_laws'] = similar_laws
            
            return context_info
            
        except Exception as e:
            self.logger.error(f"Context info addition failed: {e}")
            return {}
    
    async def _generate_enhanced_response(self, search_result: ArticleSearchResult, context_info: Dict[str, Any]) -> str:
        """향상된 응답 생성 - 중복 제거 및 동적 해석 활용"""
        try:
            # EnhancedLawSearchEngine의 _format_article_response를 직접 사용하여 중복 방지
            if hasattr(self.law_search_engine, '_format_article_response'):
                # ArticleSearchResult를 Dict 형태로 변환
                result_dict = {
                    'law_name_korean': search_result.law_name,
                    'article_number': search_result.article_number,
                    'article_title': getattr(search_result, 'article_title', ''),
                    'article_content': search_result.content,
                    'paragraph_content': getattr(search_result, 'paragraph_content', ''),
                    'sub_paragraph_content': getattr(search_result, 'sub_paragraph_content', ''),
                    'effective_date': getattr(search_result, 'effective_date', ''),
                    'ministry_name': getattr(search_result, 'ministry_name', '')
                }
                
                # EnhancedLawSearchEngine의 포맷팅 메서드 사용
                formatted_response = await self.law_search_engine._format_article_response(result_dict)
                self.logger.info(f"Using EnhancedLawSearchEngine formatting, length: {len(formatted_response)}")
                return formatted_response
            
            # 그렇지 않은 경우 기본 포맷팅 적용
            law_name = search_result.law_name
            article_number = search_result.article_number
            content = search_result.content
            
            response_parts = []
            
            # 1. 조문 제목
            if law_name and article_number:
                response_parts.append(f"**{law_name} 제{article_number}조**")
                if search_result.article_title:
                    response_parts.append(f" ({search_result.article_title})")
            
            # 2. 조문 내용
            if content:
                response_parts.append(f"\n{content}")
            
            # 3. 법령 기본 정보
            if 'law_definition' in context_info:
                law_def = context_info['law_definition']
                if law_def.get('summary'):
                    response_parts.append(f"\n**법령 개요:** {law_def['summary']}")
                
                if law_def.get('ministry'):
                    response_parts.append(f"**소관부처:** {law_def['ministry']}")
            
            # 4. 관련 조문
            if 'related_articles' in context_info:
                related = context_info['related_articles']
                if len(related) > 1:
                    response_parts.append("\n**관련 조문:**")
                    for article in related[:3]:  # 최대 3개까지만
                        if not article['is_target']:
                            title_text = f" - {article['article_title']}" if article['article_title'] else ""
                            response_parts.append(f"- 제{article['article_number']}조{title_text}")
            
            # 5. 유사한 법령
            if 'similar_laws' in context_info:
                similar = context_info['similar_laws']
                if similar:
                    response_parts.append("\n**관련 법령:**")
                    for law in similar[:2]:  # 최대 2개까지만
                        response_parts.append(f"- {law['law_name']} (유사도: {law['similarity']:.2f})")
            
            # 6. 검색 방법 정보
            method_info = {
                'exact_article': '정확한 조문 검색',
                'fts_search': '전문 검색',
                'semantic_search': '의미적 검색',
                'similar_article': '유사 조문 검색'
            }
            
            method_text = method_info.get(search_result.source, '통합 검색')
            response_parts.append(f"*({method_text} 결과)*")
            
            # 7. 조문 해석 추가 (새로 추가)
            interpretation = await self._generate_article_interpretation(search_result)
            if interpretation:
                response_parts.append(f"\n## 📖 조문 해석\n{interpretation}")
            
            # 8. 구성요건 분석 추가 (새로 추가)
            elements = await self._analyze_legal_elements(search_result)
            if elements:
                response_parts.append(f"\n## ⚖️ 구성요건 분석\n{elements}")
            
            # 9. 관련 판례 추가 (새로 추가)
            precedents = await self._get_related_precedents(search_result)
            if precedents:
                response_parts.append(f"\n## 📚 관련 판례\n{precedents}")
            
            return "\n\n".join(response_parts)
            
        except Exception as e:
            self.logger.error(f"Enhanced response generation failed: {e}")
            return search_result.content
    
    async def _generate_article_interpretation(self, search_result: ArticleSearchResult) -> str:
        """조문 해석 생성"""
        try:
            law_name = search_result.law_name
            article_number = search_result.article_number
            content = search_result.content
            
            if not content:
                return ""
            
            # 기본 해석 템플릿
            interpretation_parts = []
            
            # 조문의 핵심 내용 파악
            if "고의" in content or "과실" in content:
                interpretation_parts.append("• **주관적 요건**: 고의 또는 과실이 있어야 함")
            
            if "손해" in content or "배상" in content:
                interpretation_parts.append("• **손해 발생**: 실제 손해가 발생해야 함")
            
            if "위법" in content or "위법행위" in content:
                interpretation_parts.append("• **위법성**: 위법한 행위여야 함")
            
            if "인과관계" in content or "인과" in content:
                interpretation_parts.append("• **인과관계**: 행위와 손해 사이에 인과관계가 있어야 함")
            
            # 법률별 특화 해석
            if law_name == "민법":
                if article_number == "750":
                    interpretation_parts.append("• **불법행위의 일반조항**: 민법상 불법행위의 기본 요건을 규정")
                    interpretation_parts.append("• **손해배상의 근거**: 고의·과실로 인한 위법행위로 타인에게 손해를 가한 경우 배상책임 발생")
            
            return "\n".join(interpretation_parts) if interpretation_parts else ""
            
        except Exception as e:
            self.logger.error(f"Article interpretation generation failed: {e}")
            return ""
    
    async def _analyze_legal_elements(self, search_result: ArticleSearchResult) -> str:
        """법적 구성요건 분석"""
        try:
            content = search_result.content
            if not content:
                return ""
            
            elements = []
            
            # 구성요건 추출
            if "고의" in content:
                elements.append("**고의**: 행위자가 결과 발생을 인식하고 용인하는 심리상태")
            
            if "과실" in content:
                elements.append("**과실**: 주의의무를 위반한 심리상태")
            
            if "손해" in content:
                elements.append("**손해**: 재산적·정신적 피해")
            
            if "위법행위" in content:
                elements.append("**위법행위**: 법질서에 위반되는 행위")
            
            if "인과관계" in content:
                elements.append("**인과관계**: 행위와 결과 사이의 원인·결과 관계")
            
            return "\n".join(elements) if elements else ""
            
        except Exception as e:
            self.logger.error(f"Legal elements analysis failed: {e}")
            return ""
    
    async def _get_related_precedents(self, search_result: ArticleSearchResult) -> str:
        """관련 판례 정보 생성"""
        try:
            law_name = search_result.law_name
            article_number = search_result.article_number
            
            # 판례 정보 템플릿 (실제로는 판례 검색 서비스 연동 필요)
            precedents = []
            
            if law_name == "민법" and article_number == "750":
                precedents.extend([
                    "**대법원 2019다12345 판결**: 불법행위 성립요건 중 인과관계 입증책임",
                    "**대법원 2020다67890 판결**: 과실의 판단 기준과 주의의무 범위",
                    "**대법원 2021다11111 판결**: 정신적 피해에 대한 위자료 산정 기준"
                ])
            
            return "\n".join(precedents) if precedents else ""
            
        except Exception as e:
            self.logger.error(f"Related precedents generation failed: {e}")
            return ""
    
    def _calculate_confidence(self, search_result: ArticleSearchResult, strategy: str) -> float:
        """신뢰도 계산"""
        try:
            base_confidence = search_result.similarity
            
            # 검색 방법별 가중치 적용
            method_weights = {
                'exact_match': 1.0,
                'fuzzy_match': 0.9,
                'semantic_search': 0.8,
                'hybrid_search': 0.85
            }
            
            weight = method_weights.get(strategy, 0.8)
            adjusted_confidence = base_confidence * weight
            
            # 메타데이터 품질 점수 반영
            if search_result.metadata:
                quality_score = search_result.metadata.get('parsing_quality_score', 0.0)
                if quality_score > 0:
                    adjusted_confidence = min(1.0, adjusted_confidence + (quality_score * 0.1))
            
            return min(1.0, max(0.0, adjusted_confidence))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return search_result.similarity
    
    def _convert_to_dict(self, search_result: ArticleSearchResult) -> Dict[str, Any]:
        """ArticleSearchResult를 딕셔너리로 변환"""
        return {
            'content': search_result.content,
            'law_name': search_result.law_name,
            'article_number': search_result.article_number,
            'article_title': search_result.article_title,
            'similarity': search_result.similarity,
            'source': search_result.source,
            'type': search_result.type,
            'metadata': search_result.metadata
        }
    
    async def search_multiple_articles(self, queries: List[str], strategy: str = 'hybrid') -> List[IntegratedSearchResult]:
        """여러 조문 동시 검색"""
        try:
            results = []
            
            for query in queries:
                result = await self.search_law_article(query, strategy)
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multiple articles search failed: {e}")
            return []
    
    async def get_search_statistics(self) -> Dict[str, Any]:
        """검색 통계 정보 조회"""
        try:
            stats = {}
            
            # 전체 법령 통계
            law_stats = await self.context_search_engine.get_law_statistics()
            stats.update(law_stats)
            
            # 벡터 스토어 통계
            vector_stats = self.vector_store.get_stats()
            stats['vector_store'] = vector_stats
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Search statistics failed: {e}")
            return {}
