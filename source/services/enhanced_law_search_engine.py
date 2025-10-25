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
    
    def __init__(self, db_manager, vector_store, precedent_service=None, hybrid_precedent_service=None):
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.precedent_service = precedent_service
        self.hybrid_precedent_service = hybrid_precedent_service
        self.logger = logging.getLogger(__name__)
        
        # LawContextSearchEngine 초기화 (관련 조문 검색용)
        try:
            from .law_context_search_engine import LawContextSearchEngine
            self.context_search_engine = LawContextSearchEngine(db_manager, vector_store)
        except ImportError as e:
            self.logger.warning(f"LawContextSearchEngine 초기화 실패: {e}")
            self.context_search_engine = None
        
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
            'article_content': full_content,  # search_specific_article에서 기대하는 필드명
            'law_name': result['law_name_korean'],
            'article_number': result['article_number'],
            'article_title': result.get('article_title', ''),
            'law_id': result['law_id'],
            'article_id': result['article_id'],
            'is_supplementary': result.get('is_supplementary', False),
            'parsing_quality_score': result.get('quality_score', 0.9),
            'paragraph_content': result.get('paragraph_content', ''),
            'sub_paragraph_content': result.get('sub_paragraph_content', ''),
            'effective_date': result.get('effective_date', ''),
            'ministry_name': result.get('ministry_name', ''),
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
    
    def _extract_key_elements(self, content: str, law_name: str = "", article_number: str = "") -> List[str]:
        """조문에서 핵심 구성요건 추출 (개선된 통합 버전)"""
        if not content:
            return ["구성요건 정보 없음"]
        
        elements = []
        
        try:
            # 1. 하이브리드 접근법: 여러 방법론 조합
            # NLP 기반 추출
            nlp_elements = self._extract_key_elements_nlp(content, law_name, article_number)
            elements.extend(nlp_elements)
            
            # 패턴 기반 추출
            pattern_elements = self._extract_key_elements_pattern(content, law_name, article_number)
            elements.extend(pattern_elements)
            
            # ML 기반 추출 (충분한 텍스트가 있을 때만)
            if len(content) > 50:
                ml_elements = self._extract_key_elements_ml(content, law_name, article_number)
                elements.extend(ml_elements)
            
        except Exception as e:
            self.logger.warning(f"Advanced extraction failed, falling back to basic method: {e}")
            # 기본 방법으로 폴백
            elements.extend(self._extract_key_elements_basic(content))
        
        # 2. 중복 제거 및 신뢰도 기반 정렬
        unique_elements = self._deduplicate_and_rank_elements(elements)
        
        # 3. 최종 결과 반환 (최대 5개)
        return unique_elements[:5]
    
    def _extract_key_elements_basic(self, content: str) -> List[str]:
        """기본 구성요건 추출 (기존 로직)"""
        elements = []
        
        # 고의/과실 관련
        if any(word in content for word in ["고의", "과실"]):
            elements.append("고의/과실")
        
        # 행위 관련
        if any(word in content for word in ["행위", "위법", "불법"]):
            elements.append("위법행위")
        
        # 손해 관련
        if any(word in content for word in ["손해", "배상"]):
            elements.append("손해발생")
        
        # 인과관계 관련
        if any(word in content for word in ["인과", "관계", "원인"]):
            elements.append("인과관계")
        
        # 책임 관련
        if any(word in content for word in ["책임", "의무"]):
            elements.append("법적 책임")
        
        # 기본 구성요건이 없으면 일반적인 요소들
        if not elements:
            elements = ["법적 요건", "적용 범위", "법적 효과"]
        
        return elements
    
    def _extract_key_elements_nlp(self, content: str, law_name: str = "", article_number: str = "") -> List[str]:
        """자연어처리 기반 구성요건 추출"""
        if not content:
            return ["구성요건 정보 없음"]
        
        elements = []
        
        # 1. 문장 구조 분석을 통한 구성요건 추출
        sentences = self._split_legal_sentences(content)
        
        for sentence in sentences:
            # 조건문 패턴 분석
            condition_elements = self._extract_condition_elements(sentence)
            elements.extend(condition_elements)
            
            # 효과문 패턴 분석
            effect_elements = self._extract_effect_elements(sentence)
            elements.extend(effect_elements)
        
        # 2. 법령별 특화된 구성요건 추출
        specialized_elements = self._extract_specialized_elements(content, law_name, article_number)
        elements.extend(specialized_elements)
        
        # 3. 중복 제거 및 우선순위 정렬
        unique_elements = self._prioritize_elements(elements)
        
        return unique_elements[:5]
    
    def _split_legal_sentences(self, content: str) -> List[str]:
        """법률 문장을 의미 단위로 분할"""
        sentences = []
        
        # 문장 끝 패턴
        sentence_endings = ['.', '다.', '한다.', '된다.', '아야 한다.', '어야 한다.']
        
        current_sentence = ""
        for char in content:
            current_sentence += char
            
            # 문장 끝 확인
            if any(current_sentence.endswith(ending) for ending in sentence_endings):
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # 마지막 문장 처리
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def _extract_condition_elements(self, sentence: str) -> List[str]:
        """조건문에서 구성요건 추출"""
        elements = []
        
        # 조건문 패턴들
        condition_patterns = [
            r'([가-힣]+(?:이|가|을|를|에|에서|로|으로))\s*([^다]+다)',
            r'([가-힣]+(?:이|가|을|를|에|에서|로|으로))\s*([^면]+면)',
            r'([가-힣]+(?:이|가|을|를|에|에서|로|으로))\s*([^때]+때)',
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                condition = match[0].strip()
                if len(condition) > 1 and len(condition) < 10:  # 적절한 길이 필터링
                    elements.append(f"조건: {condition}")
        
        return elements
    
    def _extract_effect_elements(self, sentence: str) -> List[str]:
        """효과문에서 구성요건 추출"""
        elements = []
        
        # 효과문 패턴들
        effect_patterns = [
            r'([가-힣]+(?:한다|된다|아야 한다|어야 한다))',
            r'([가-힣]+(?:의무|책임|권리|효력|효과))',
        ]
        
        for pattern in effect_patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                effect = match.strip()
                if len(effect) > 1 and len(effect) < 15:
                    elements.append(f"효과: {effect}")
        
        return elements
    
    def _extract_specialized_elements(self, content: str, law_name: str, article_number: str) -> List[str]:
        """법령별 특화된 구성요건 추출"""
        elements = []
        
        # 법령별 특화 규칙
        specialized_rules = {
            "민법": {
                "750": ["고의/과실", "위법행위", "손해발생", "인과관계"],
                "751": ["정신적 피해", "고의/과실", "위법행위"],
                "752": ["생명침해", "재산이외 손해", "유족의 피해"],
                "753": ["미성년자", "책임능력", "법정대리인"]
            },
            "형법": {
                "250": ["살인의 고의", "생명침해", "범의"],
                "257": ["상해의 고의", "신체상해", "범의"],
                "329": ["절도의 고의", "타인재물", "절취행위"]
            },
            "상법": {
                "382": ["이사의 선관주의의무", "회사이익", "충실의무"],
                "400": ["감사의 독립성", "회계감사", "감사보고서"]
            }
        }
        
        # 해당 법령의 특화 규칙 적용
        if law_name in specialized_rules and article_number in specialized_rules[law_name]:
            elements.extend(specialized_rules[law_name][article_number])
        
        # 일반적인 법령별 특화 요소
        elif law_name in specialized_rules:
            elements.extend(self._extract_general_law_elements(content, law_name))
        
        return elements
    
    def _extract_general_law_elements(self, content: str, law_name: str) -> List[str]:
        """일반적인 법령별 구성요건 추출"""
        elements = []
        
        law_specific_patterns = {
            "민법": [
                (r'[가-힣]+(?:권|의무)', "민법적 권리/의무"),
                (r'[가-힣]+(?:계약|합의)', "계약 관련 요소"),
                (r'[가-힣]+(?:손해|배상)', "손해배상 요소")
            ],
            "형법": [
                (r'[가-힣]+(?:범죄|처벌)', "범죄 구성요소"),
                (r'[가-힣]+(?:고의|과실)', "주관적 요소"),
                (r'[가-힣]+(?:벌금|징역)', "형벌 요소")
            ],
            "상법": [
                (r'[가-힣]+(?:회사|주식)', "회사 관련 요소"),
                (r'[가-힣]+(?:이사|감사)', "기관 관련 요소"),
                (r'[가-힣]+(?:주주|총회)', "주주 관련 요소")
            ]
        }
        
        if law_name in law_specific_patterns:
            for pattern, element_type in law_specific_patterns[law_name]:
                matches = re.findall(pattern, content)
                if matches:
                    elements.append(element_type)
        
        return elements
    
    def _prioritize_elements(self, elements: List[str]) -> List[str]:
        """구성요건 우선순위 정렬"""
        # 우선순위 기준
        priority_keywords = {
            "고의/과실": 10,
            "위법행위": 9,
            "손해발생": 8,
            "인과관계": 7,
            "법적 책임": 6,
            "조건": 5,
            "효과": 4,
            "권리": 3,
            "의무": 2,
            "범죄": 1
        }
        
        # 우선순위별 정렬
        def get_priority(element):
            for keyword, priority in priority_keywords.items():
                if keyword in element:
                    return priority
            return 0
        
        # 중복 제거 후 우선순위 정렬
        unique_elements = list(dict.fromkeys(elements))
        return sorted(unique_elements, key=get_priority, reverse=True)
    
    def _extract_key_elements_pattern(self, content: str, law_name: str = "", article_number: str = "") -> List[str]:
        """패턴 기반 구성요건 추출 시스템"""
        if not content:
            return ["구성요건 정보 없음"]
        
        elements = []
        
        # 1. 법률 문장 구조 패턴 분석
        structural_elements = self._analyze_structural_patterns(content)
        elements.extend(structural_elements)
        
        # 2. 법률 용어 패턴 매칭
        legal_term_elements = self._extract_legal_term_patterns(content)
        elements.extend(legal_term_elements)
        
        # 3. 문법적 패턴 분석
        grammatical_elements = self._analyze_grammatical_patterns(content)
        elements.extend(grammatical_elements)
        
        # 4. 법령별 특화 패턴 적용
        specialized_elements = self._apply_specialized_patterns(content, law_name, article_number)
        elements.extend(specialized_elements)
        
        # 5. 신뢰도 기반 필터링 및 정렬
        filtered_elements = self._filter_and_rank_elements(elements)
        
        return filtered_elements[:5]
    
    def _analyze_structural_patterns(self, content: str) -> List[str]:
        """법률 문장의 구조적 패턴 분석"""
        elements = []
        
        # 조건-결과 구조 패턴
        condition_result_patterns = [
            r'([^다]+다)\s*([^다]+다)',  # 조건문 + 결과문
            r'([^면]+면)\s*([^다]+다)',  # 조건문 + 결과문
            r'([^때]+때)\s*([^다]+다)',  # 조건문 + 결과문
        ]
        
        for pattern in condition_result_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match[0]) > 3 and len(match[0]) < 20:
                    elements.append(f"구조적 조건: {match[0].strip()}")
                if len(match[1]) > 3 and len(match[1]) < 20:
                    elements.append(f"구조적 결과: {match[1].strip()}")
        
        # 항목별 구조 패턴
        item_patterns = [
            r'①\s*([^②]+)',  # ① 항목
            r'②\s*([^③]+)',  # ② 항목
            r'③\s*([^④]+)',   # ③ 항목
        ]
        
        for pattern in item_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) > 5 and len(match) < 30:
                    elements.append(f"구조적 항목: {match.strip()}")
        
        return elements
    
    def _extract_legal_term_patterns(self, content: str) -> List[str]:
        """법률 용어 패턴 추출"""
        elements = []
        
        # 법률 용어 패턴 정의
        legal_term_patterns = {
            "주관적 요소": [
                r'[가-힣]*(?:고의|과실|의도|의식)[가-힣]*',
                r'[가-힣]*(?:인식|예견|희망)[가-힣]*'
            ],
            "객관적 요소": [
                r'[가-힣]*(?:행위|위법|불법)[가-힣]*',
                r'[가-힣]*(?:결과|손해|피해)[가-힣]*'
            ],
            "법적 효과": [
                r'[가-힣]*(?:책임|의무|권리)[가-힣]*',
                r'[가-힣]*(?:효력|효과|결과)[가-힣]*'
            ],
            "절차적 요소": [
                r'[가-힣]*(?:절차|방법|과정)[가-힣]*',
                r'[가-힣]*(?:신청|제출|처리)[가-힣]*'
            ]
        }
        
        for category, patterns in legal_term_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # 가장 자주 나타나는 용어 선택
                    most_common = max(set(matches), key=matches.count)
                    elements.append(f"{category}: {most_common}")
        
        return elements
    
    def _analyze_grammatical_patterns(self, content: str) -> List[str]:
        """문법적 패턴 분석"""
        elements = []
        
        # 조동사 패턴 분석
        auxiliary_patterns = [
            r'([가-힣]+)\s*(?:아야|어야)\s*(?:한다|된다)',  # 의무 표현
            r'([가-힣]+)\s*(?:할|될)\s*(?:수|것)\s*(?:있다|없다)',  # 가능성 표현
            r'([가-힣]+)\s*(?:하여야|되어야)\s*(?:한다|된다)',  # 의무 표현
        ]
        
        for pattern in auxiliary_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) > 2 and len(match) < 15:
                    elements.append(f"문법적 의무: {match.strip()}")
        
        # 조건문 패턴 분석
        conditional_patterns = [
            r'([가-힣]+)\s*(?:이면|가면|으면|면)\s*([가-힣]+)',  # 조건문
            r'([가-힣]+)\s*(?:인|한)\s*(?:경우|때)\s*([가-힣]+)',  # 조건문
        ]
        
        for pattern in conditional_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match[0]) > 2 and len(match[0]) < 10:
                    elements.append(f"문법적 조건: {match[0].strip()}")
        
        return elements
    
    def _apply_specialized_patterns(self, content: str, law_name: str, article_number: str) -> List[str]:
        """법령별 특화 패턴 적용"""
        elements = []
        
        # 법령별 특화 패턴 정의
        specialized_patterns = {
            "민법": {
                "불법행위": [
                    r'[가-힣]*(?:고의|과실)[가-힣]*(?:로|으로)\s*[가-힣]*(?:인한|한)\s*[가-힣]*(?:위법|불법)[가-힣]*(?:행위|행동)',
                    r'[가-힣]*(?:타인|다른\s*사람)[가-힣]*(?:에게|에)\s*[가-힣]*(?:손해|피해)[가-힣]*(?:를|을)\s*[가-힣]*(?:가하는|입히는)'
                ],
                "계약": [
                    r'[가-힣]*(?:계약|합의|약정)[가-힣]*(?:의|이)\s*[가-힣]*(?:성립|효력|해제)',
                    r'[가-힣]*(?:당사자|계약자)[가-힣]*(?:간|사이)\s*[가-힣]*(?:합의|약정)'
                ]
            },
            "형법": {
                "범죄": [
                    r'[가-힣]*(?:범죄|죄)[가-힣]*(?:를|을)\s*[가-힣]*(?:범한|저지른)\s*[가-힣]*(?:자|자)',
                    r'[가-힣]*(?:고의|과실)[가-힣]*(?:로|으로)\s*[가-힣]*(?:범한|저지른)'
                ]
            },
            "상법": {
                "회사": [
                    r'[가-힣]*(?:회사|법인)[가-힣]*(?:의|이)\s*[가-힣]*(?:이사|감사|주주)',
                    r'[가-힣]*(?:이사|감사)[가-힣]*(?:의|이)\s*[가-힣]*(?:의무|책임)'
                ]
            }
        }
        
        # 해당 법령의 특화 패턴 적용
        if law_name in specialized_patterns:
            for category, patterns in specialized_patterns[law_name].items():
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        elements.append(f"{law_name} {category}: {matches[0]}")
        
        return elements
    
    def _filter_and_rank_elements(self, elements: List[str]) -> List[str]:
        """구성요건 필터링 및 순위 결정"""
        if not elements:
            return ["구성요건 정보 없음"]
        
        # 신뢰도 점수 계산
        element_scores = {}
        for element in elements:
            score = self._calculate_element_confidence(element)
            element_scores[element] = score
        
        # 신뢰도 기준 필터링 (0.3 이상)
        filtered_elements = [
            element for element, score in element_scores.items() 
            if score >= 0.3
        ]
        
        # 신뢰도 기준 정렬
        sorted_elements = sorted(
            filtered_elements, 
            key=lambda x: element_scores[x], 
            reverse=True
        )
        
        return sorted_elements
    
    def _calculate_element_confidence(self, element: str) -> float:
        """구성요건 신뢰도 점수 계산"""
        score = 0.0
        
        # 키워드 기반 점수
        high_confidence_keywords = {
            "고의": 0.9, "과실": 0.9, "위법행위": 0.8, "손해": 0.8,
            "인과관계": 0.7, "책임": 0.7, "의무": 0.6, "권리": 0.6
        }
        
        for keyword, keyword_score in high_confidence_keywords.items():
            if keyword in element:
                score += keyword_score
        
        # 길이 기반 점수 (적절한 길이일수록 높은 점수)
        length = len(element)
        if 5 <= length <= 20:
            score += 0.2
        elif length < 5 or length > 30:
            score -= 0.1
        
        # 패턴 기반 점수
        if ":" in element:  # 구조화된 패턴
            score += 0.1
        
        # 법령별 특화 점수
        if any(law in element for law in ["민법", "형법", "상법"]):
            score += 0.1
        
        # 최종 점수 정규화 (0.0 ~ 1.0)
        return min(max(score, 0.0), 1.0)
    
    def _extract_key_elements_ml(self, content: str, law_name: str = "", article_number: str = "") -> List[str]:
        """머신러닝 기반 구성요건 분류 시스템"""
        if not content:
            return ["구성요건 정보 없음"]
        
        elements = []
        
        # 1. 특성 추출
        features = self._extract_ml_features(content, law_name, article_number)
        
        # 2. 구성요건 분류
        classified_elements = self._classify_legal_elements(features)
        elements.extend(classified_elements)
        
        # 3. 신뢰도 기반 필터링
        filtered_elements = self._filter_by_confidence(elements)
        
        return filtered_elements[:5]
    
    def _extract_ml_features(self, content: str, law_name: str, article_number: str) -> Dict[str, Any]:
        """머신러닝을 위한 특성 추출"""
        features = {
            "text_length": len(content),
            "sentence_count": len(re.split(r'[.!?]', content)),
            "has_conditions": bool(re.search(r'[가-힣]+(?:이면|가면|으면|면)', content)),
            "has_effects": bool(re.search(r'[가-힣]+(?:한다|된다|아야 한다|어야 한다)', content)),
            "has_penalties": bool(re.search(r'[가-힣]*(?:벌금|징역|과태료)', content)),
            "has_deadlines": bool(re.search(r'\d+(?:일|개월|년)', content)),
            "has_multiple_items": bool(re.search(r'[①②③④⑤]', content)),
            "law_type": law_name,
            "article_number": article_number,
            "keyword_density": self._calculate_keyword_density(content),
            "legal_term_count": len(re.findall(r'[가-힣]{2,6}(?:권|의무|책임|효력)', content)),
            "conditional_structure": self._analyze_conditional_structure(content),
            "effect_structure": self._analyze_effect_structure(content)
        }
        
        return features
    
    def _calculate_keyword_density(self, content: str) -> float:
        """법률 키워드 밀도 계산"""
        legal_keywords = [
            "고의", "과실", "위법", "불법", "손해", "배상", "책임", "의무",
            "권리", "효력", "효과", "계약", "합의", "소유권", "상속", "이혼"
        ]
        
        total_words = len(content.split())
        keyword_count = sum(1 for keyword in legal_keywords if keyword in content)
        
        return keyword_count / total_words if total_words > 0 else 0.0
    
    def _analyze_conditional_structure(self, content: str) -> Dict[str, int]:
        """조건문 구조 분석"""
        conditional_patterns = {
            "if_then": len(re.findall(r'[가-힣]+(?:이면|가면|으면|면)\s*[가-힣]+', content)),
            "when": len(re.findall(r'[가-힣]+(?:인|한)\s*(?:경우|때)\s*[가-힣]+', content)),
            "unless": len(re.findall(r'[가-힣]+(?:이|가|을|를)\s*(?:아니면|아닌|아닐)\s*[가-힣]+', content))
        }
        
        return conditional_patterns
    
    def _analyze_effect_structure(self, content: str) -> Dict[str, int]:
        """효과문 구조 분석"""
        effect_patterns = {
            "obligation": len(re.findall(r'[가-힣]+(?:아야|어야)\s*(?:한다|된다)', content)),
            "prohibition": len(re.findall(r'[가-힣]+(?:하여서는|되어서는)\s*(?:안|못)\s*(?:된다|한다)', content)),
            "permission": len(re.findall(r'[가-힣]+(?:할|될)\s*(?:수|것)\s*(?:있다|없다)', content)),
            "consequence": len(re.findall(r'[가-힣]+(?:인|한)\s*(?:경우|때)\s*[가-힣]+', content))
        }
        
        return effect_patterns
    
    def _classify_legal_elements(self, features: Dict[str, Any]) -> List[str]:
        """법률 구성요건 분류"""
        elements = []
        
        # 규칙 기반 분류 (실제 ML 모델로 대체 가능)
        if features["has_conditions"] and features["has_effects"]:
            elements.append("요건-효과 구조")
        
        if features["keyword_density"] > 0.1:
            elements.append("법률 용어 집중")
        
        if features["has_penalties"]:
            elements.append("벌칙 조항")
        
        if features["has_deadlines"]:
            elements.append("기한 관련 조항")
        
        if features["has_multiple_items"]:
            elements.append("다항 조항")
        
        # 법령별 특화 분류
        law_type = features.get("law_type", "")
        if law_type == "민법":
            elements.extend(self._classify_civil_law_elements(features))
        elif law_type == "형법":
            elements.extend(self._classify_criminal_law_elements(features))
        elif law_type == "상법":
            elements.extend(self._classify_commercial_law_elements(features))
        
        return elements
    
    def _classify_civil_law_elements(self, features: Dict[str, Any]) -> List[str]:
        """민법 구성요건 분류"""
        elements = []
        
        if features["legal_term_count"] > 3:
            elements.append("민법적 권리/의무")
        
        if features["conditional_structure"]["if_then"] > 0:
            elements.append("민법적 조건")
        
        if features["effect_structure"]["obligation"] > 0:
            elements.append("민법적 의무")
        
        return elements
    
    def _classify_criminal_law_elements(self, features: Dict[str, Any]) -> List[str]:
        """형법 구성요건 분류"""
        elements = []
        
        if features["has_penalties"]:
            elements.append("형사적 처벌")
        
        if features["keyword_density"] > 0.15:
            elements.append("형사적 구성요건")
        
        return elements
    
    def _classify_commercial_law_elements(self, features: Dict[str, Any]) -> List[str]:
        """상법 구성요건 분류"""
        elements = []
        
        if features["has_multiple_items"]:
            elements.append("상법적 절차")
        
        if features["effect_structure"]["permission"] > 0:
            elements.append("상법적 권한")
        
        return elements
    
    def _filter_by_confidence(self, elements: List[str]) -> List[str]:
        """신뢰도 기반 필터링"""
        # 신뢰도 점수 계산
        confidence_scores = {}
        for element in elements:
            score = self._calculate_ml_confidence(element)
            confidence_scores[element] = score
        
        # 신뢰도 0.5 이상만 필터링
        filtered_elements = [
            element for element, score in confidence_scores.items() 
            if score >= 0.5
        ]
        
        # 신뢰도 기준 정렬
        return sorted(filtered_elements, key=lambda x: confidence_scores[x], reverse=True)
    
    def _calculate_ml_confidence(self, element: str) -> float:
        """ML 기반 신뢰도 계산"""
        score = 0.0
        
        # 구조적 신뢰도
        if "구조" in element:
            score += 0.3
        
        # 법령별 특화 신뢰도
        if any(law in element for law in ["민법", "형법", "상법"]):
            score += 0.2
        
        # 용어 신뢰도
        if any(term in element for term in ["권리", "의무", "책임", "효력"]):
            score += 0.2
        
        # 길이 신뢰도
        if 5 <= len(element) <= 20:
            score += 0.1
        
        return min(score, 1.0)
    
    def _deduplicate_and_rank_elements(self, elements: List[str]) -> List[str]:
        """구성요건 중복 제거 및 순위 결정"""
        if not elements:
            return ["구성요건 정보 없음"]
        
        # 1. 유사도 기반 중복 제거
        unique_elements = []
        for element in elements:
            is_duplicate = False
            for existing in unique_elements:
                if self._calculate_similarity(element, existing) > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_elements.append(element)
        
        # 2. 신뢰도 기반 순위 결정
        element_scores = {}
        for element in unique_elements:
            score = self._calculate_comprehensive_confidence(element)
            element_scores[element] = score
        
        # 3. 신뢰도 기준 정렬
        sorted_elements = sorted(
            unique_elements, 
            key=lambda x: element_scores[x], 
            reverse=True
        )
        
        return sorted_elements
    
    def _calculate_similarity(self, element1: str, element2: str) -> float:
        """두 구성요건 간의 유사도 계산"""
        # 간단한 유사도 계산 (실제로는 더 정교한 방법 사용 가능)
        words1 = set(element1.split())
        words2 = set(element2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_comprehensive_confidence(self, element: str) -> float:
        """종합적인 신뢰도 점수 계산"""
        score = 0.0
        
        # 1. 키워드 기반 점수
        high_confidence_keywords = {
            "고의": 0.9, "과실": 0.9, "위법행위": 0.8, "손해": 0.8,
            "인과관계": 0.7, "책임": 0.7, "의무": 0.6, "권리": 0.6,
            "구조적": 0.5, "문법적": 0.4, "ML": 0.3
        }
        
        for keyword, keyword_score in high_confidence_keywords.items():
            if keyword in element:
                score += keyword_score
        
        # 2. 길이 기반 점수
        length = len(element)
        if 5 <= length <= 20:
            score += 0.2
        elif length < 5 or length > 30:
            score -= 0.1
        
        # 3. 구조적 패턴 점수
        if ":" in element:  # 구조화된 패턴
            score += 0.1
        
        # 4. 법령별 특화 점수
        if any(law in element for law in ["민법", "형법", "상법"]):
            score += 0.1
        
        # 최종 점수 정규화 (0.0 ~ 1.0)
        return min(max(score, 0.0), 1.0)
    
    async def _format_article_response(self, result: Dict[str, Any], user_query: str = "") -> str:
        """간결하고 실용적인 조문 응답 포맷팅 (비동기)"""
        law_name = result.get('law_name_korean', '')
        article_number = result.get('article_number', '')
        article_title = result.get('article_title', '')
        article_content = result.get('article_content', '')
        paragraph_content = result.get('paragraph_content', '')
        sub_paragraph_content = result.get('sub_paragraph_content', '')
        effective_date = result.get('effective_date', '')
        ministry_name = result.get('ministry_name', '')
        
        self.logger.info(f"Formatting concise article response for {law_name} 제{article_number}조")
        
        # 응답 초기화 (중복 방지)
        response_parts = []
        
        # 제목
        title = f"**{law_name} 제{article_number}조"
        if article_title:
            title += f" ({article_title})"
        title += "**"
        response_parts.append(title)
        response_parts.append("")  # 빈 줄
        
        # 조문 내용 (인용문 스타일)
        main_content = paragraph_content if paragraph_content and paragraph_content.strip() else article_content
        if main_content and main_content.strip():
            response_parts.append(f"> {main_content}")
            response_parts.append("")  # 빈 줄
        
        # 핵심 구성요건 (간결하게)
        response_parts.append("**📋 핵심 구성요건**")
        key_elements = self._extract_key_elements(main_content or article_content)
        response_parts.append(f"- {', '.join(key_elements[:3])}")
        response_parts.append("")  # 빈 줄
        
        # 관련 판례 (하이브리드 검색 - 비동기)
        precedents = []
        self.logger.info(f"판례 검색 시작 - 하이브리드 서비스: {self.hybrid_precedent_service is not None}, DB 서비스: {self.precedent_service is not None}")
        
        if self.hybrid_precedent_service:
            # 하이브리드 서비스 사용 (DB + API)
            try:
                self.logger.info(f"하이브리드 판례 검색 호출: {law_name} 제{article_number}조")
                precedents = await self.hybrid_precedent_service.get_related_precedents(
                    law_name, str(article_number), main_content or article_content, limit=3
                )
                self.logger.info(f"하이브리드 판례 검색 결과: {len(precedents)}개")
            except Exception as e:
                self.logger.error(f"하이브리드 판례 검색 실패: {e}")
                # 폴백: 기존 DB 서비스 사용
                if self.precedent_service:
                    self.logger.info("폴백: DB 판례 검색 사용")
                    precedents = self.precedent_service.get_related_precedents(
                        law_name, str(article_number), main_content or article_content, limit=3
                    )
                    self.logger.info(f"DB 판례 검색 결과: {len(precedents)}개")
        elif self.precedent_service:
            # 기존 DB 서비스만 사용
            self.logger.info(f"DB 판례 검색 호출: {law_name} 제{article_number}조")
            precedents = self.precedent_service.get_related_precedents(
                law_name, str(article_number), main_content or article_content, limit=3
            )
            self.logger.info(f"DB 판례 검색 결과: {len(precedents)}개")
        else:
            self.logger.warning("판례 검색 서비스가 초기화되지 않음")
        
        if precedents:
            response_parts.append("**⚖️ 주요 판례**")
            for precedent in precedents:
                source_indicator = "🌐" if hasattr(precedent, 'source') and precedent.source == "api" else "📚"
                response_parts.append(f"- {source_indicator} **{precedent.case_number}**: {precedent.summary}")
            response_parts.append("")  # 빈 줄
        
        # 실무 적용 (핵심만)
        response_parts.append("**💼 실무 적용**")
        practical_tips = self._get_concise_practical_tips(law_name, article_number, main_content or article_content)
        for tip in practical_tips[:2]:  # 최대 2개
            response_parts.append(f"- {tip}")
        response_parts.append("")  # 빈 줄
        
        # 관련 조문 (간단히)
        related_articles = await self._get_related_articles(law_name, article_number)
        if related_articles:
            response_parts.append("**🔗 관련 조문**")
            response_parts.append(f"- {', '.join(related_articles[:3])}")
        
        # 최종 응답 조합 (중복 제거)
        final_response = "\n".join(response_parts)
        return self._validate_response_quality(final_response)
    
    def _validate_response_quality(self, response: str) -> str:
        """응답 품질 검증 및 개선"""
        if not response:
            return response
        
        # 중복 내용 제거
        lines = response.split('\n')
        unique_lines = []
        seen_content = set()
        
        for line in lines:
            content_key = line.strip().lower()
            if content_key and content_key not in seen_content:
                unique_lines.append(line)
                seen_content.add(content_key)
            elif not content_key:  # 빈 줄은 유지
                unique_lines.append(line)
        
        # 연속된 빈 줄 제거 (최대 2개까지만 허용)
        cleaned_lines = []
        empty_count = 0
        for line in unique_lines:
            if line.strip() == "":
                empty_count += 1
                if empty_count <= 2:
                    cleaned_lines.append(line)
            else:
                empty_count = 0
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _get_concise_practical_tips(self, law_name: str, article_number: str, content: str) -> List[str]:
        """간결한 실무 적용 팁 생성"""
        tips = []
        
        try:
            # 법령별 특화된 실무 팁
            if law_name == "민법":
                if "750" in str(article_number):  # 불법행위
                    tips.extend([
                        "입증책임: 원고(피해자)가 과실 입증",
                        "배상범위: 민법 제393조 (통상손해 + 특별손해)"
                    ])
                elif "565" in str(article_number):  # 계약금
                    tips.extend([
                        "계약금 포기 시 계약 해제 가능",
                        "배액 상환 시 매도인도 해제 가능"
                    ])
                elif "615" in str(article_number):  # 임대차
                    tips.extend([
                        "보증금 반환 의무: 계약 종료 시",
                        "임차권 등기명령으로 대항력 확보"
                    ])
            
            elif law_name == "형법":
                if "250" in str(article_number):  # 살인
                    tips.extend([
                        "구성요건: 고의 + 생명침해 + 인과관계",
                        "미수범도 처벌 대상"
                    ])
                elif "257" in str(article_number):  # 상해
                    tips.extend([
                        "신체상해 정도에 따라 형량 차등",
                        "상해치사는 결과적 가중범"
                    ])
            
            elif law_name == "가족법" or "이혼" in content:
                tips.extend([
                    "협의이혼: 숙려기간 필수",
                    "재판상 이혼: 유책사유 필요"
                ])
            
            # 일반적인 실무 팁
            if not tips:
                if "손해배상" in content:
                    tips.append("손해액 산정: 객관적 증거 필요")
                elif "계약" in content:
                    tips.append("계약서 작성: 명확한 조건 기재")
                elif "소송" in content:
                    tips.append("소송 제기: 관할 법원 확인")
                else:
                    tips.append("법률 자문: 전문가 상담 권장")
            
            # 최대 2개까지만 반환
            return tips[:2]
            
        except Exception as e:
            self.logger.error(f"실무 팁 생성 실패: {e}")
            return ["법률 자문: 전문가 상담 권장"]
    
    async def _get_related_articles(self, law_name: str, article_number: str) -> List[str]:
        """관련 조문 목록 생성 (DB 기반)"""
        try:
            # LawContextSearchEngine을 사용하여 관련 조문 검색
            if self.context_search_engine:
                related_articles = await self.context_search_engine.search_related_articles(
                    law_name, int(article_number), context_range=3
                )
                
                # 결과 포맷팅 (대상 조문 제외)
                formatted_articles = []
                for article in related_articles:
                    if not article.is_target:  # 대상 조문 제외
                        title_suffix = f"({article.article_title})" if article.article_title else ""
                        formatted_articles.append(f"{law_name} 제{article.article_number}조{title_suffix}")
                
                if formatted_articles:
                    self.logger.info(f"Found {len(formatted_articles)} related articles for {law_name} 제{article_number}조")
                    return formatted_articles[:3]  # 최대 3개
            
            # LawContextSearchEngine이 없으면 빈 리스트 반환
            self.logger.warning(f"LawContextSearchEngine not available for {law_name} 제{article_number}조")
            return []
            
        except Exception as e:
            self.logger.error(f"Related articles search failed: {e}")
            return []
    
