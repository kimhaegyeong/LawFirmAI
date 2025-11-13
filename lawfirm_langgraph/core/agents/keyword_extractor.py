# -*- coding: utf-8 -*-
"""
키워드 추출 유틸리티 클래스
형태소 분석을 활용한 한국어 키워드 추출 및 최적화
"""

import logging
import re
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """
    한국어 키워드 추출 클래스
    형태소 분석을 활용하여 조사/어미를 자동 제거하고 핵심 키워드를 추출합니다.
    """
    
    # 기본 불용어 목록 (클래스 변수)
    BASIC_STOPWORDS: Set[str] = {
        '에', '대해', '설명해주세요', '설명', '의', '을', '를', '이', '가', '는', '은',
        '으로', '로', '에서', '에게', '한테', '께', '와', '과', '하고', '그리고',
        '또는', '또한', '때문에', '위해', '통해', '관련', '및', '등', '등등',
        '어떻게', '무엇', '언제', '어디', '어떤', '무엇인가', '요청', '질문',
        '답변', '알려주세요', '알려주시기', '바랍니다'
    }
    
    # 조사 패턴 (정규식)
    JOSA_PATTERN = re.compile(
        r'(에|에서|에게|한테|께|으로|로|의|을|를|이|가|는|은|와|과|도|만|부터|까지|만큼|처럼|같이|따라|대신|더불어|대하여|관하여)$'
    )
    
    # 법령명 패턴
    LAW_NAME_PATTERN = re.compile(
        r'([가-힣]+법|민법|형법|상법|행정법|헌법|노동법|가족법|특허법|상표법|저작권법)'
    )
    
    # 조문 번호 패턴
    ARTICLE_PATTERNS = [
        re.compile(r'제\s*\d+\s*조'),
        re.compile(r'\d+\s*조'),
    ]
    
    # 허용할 품사 태그 (형태소 분석용)
    ALLOWED_POS_TAGS = {
        'NNG', 'NNP', 'NNB', 'NR', 'NP',  # 명사류 (일반명사, 고유명사, 의존명사, 수사, 대명사)
        'VV', 'VA', 'VX', 'VCP', 'VCN',   # 용언 (동사, 형용사, 보조용언, 긍정지정사, 부정지정사)
        'MM', 'MDT', 'MDN',                # 관형사, 수 관형사, 명사 관형사
        'SL', 'SH', 'SN', 'XR'             # 외국어, 한자, 숫자, 어근
    }
    
    def __init__(self, use_morphology: bool = True, logger_instance: Optional[logging.Logger] = None):
        """
        KeywordExtractor 초기화
        
        Args:
            use_morphology: 형태소 분석 사용 여부 (기본값: True)
            logger_instance: 로거 인스턴스 (없으면 자동 생성)
        """
        self.use_morphology = use_morphology
        self.logger = logger_instance or logging.getLogger(__name__)
        
        # KoNLPy 형태소 분석기 초기화 (선택적)
        self._okt = None
        if use_morphology:
            try:
                from konlpy.tag import Okt
                self._okt = Okt()
                self.logger.debug("KoNLPy Okt initialized successfully")
            except ImportError:
                self.logger.debug("KoNLPy not available, will use fallback method")
                self.use_morphology = False
            except Exception as e:
                self.logger.warning(f"Error initializing KoNLPy: {e}, will use fallback method")
                self.use_morphology = False
    
    def extract_keywords(
        self,
        query: str,
        max_keywords: int = 5,
        prefer_morphology: bool = True
    ) -> List[str]:
        """
        키워드 추출 (형태소 분석 우선, 폴백 사용)
        
        Args:
            query: 추출할 쿼리 문자열
            max_keywords: 최대 키워드 수
            prefer_morphology: 형태소 분석 우선 사용 여부
            
        Returns:
            추출된 키워드 리스트
        """
        if not query or not query.strip():
            return []
        
        # 형태소 분석 사용 가능하고 우선 사용 설정이면 형태소 분석 사용
        if prefer_morphology and self.use_morphology and self._okt is not None:
            try:
                keywords = self._extract_with_morphology(query, max_keywords)
                if keywords:
                    return keywords
            except Exception as e:
                self.logger.debug(f"Morphological analysis failed: {e}, using fallback")
        
        # 폴백: 정규식 기반 키워드 추출
        return self._extract_with_fallback(query, max_keywords)
    
    def _extract_with_morphology(self, query: str, max_keywords: int) -> List[str]:
        """
        형태소 분석을 사용한 키워드 추출
        
        Args:
            query: 추출할 쿼리 문자열
            max_keywords: 최대 키워드 수
            
        Returns:
            추출된 키워드 리스트
        """
        if not self._okt:
            return []
        
        keywords: List[str] = []
        
        # 형태소 분석으로 품사 태깅
        pos_tags = self._okt.pos(query)
        
        # 법령명 추출 (우선순위 최고)
        law_match = self.LAW_NAME_PATTERN.search(query)
        if law_match:
            law_name = law_match.group(1)
            if law_name not in keywords:
                keywords.append(law_name)
        
        # 조문 번호 추출 (우선순위 높음)
        for pattern in self.ARTICLE_PATTERNS:
            article_match = pattern.search(query)
            if article_match:
                article_text = article_match.group().replace(' ', '').strip()
                if not article_text.startswith('제'):
                    article_text = '제' + article_text
                if article_text not in keywords:
                    keywords.append(article_text)
                    break
        
        # 형태소 분석으로 핵심 키워드 추출
        for word, pos in pos_tags:
            # 허용된 품사만 선택
            if pos in self.ALLOWED_POS_TAGS:
                # 2자 이상이고 중복이 아닌 경우
                if len(word) >= 2 and word not in keywords:
                    # 법령명이나 조문번호 패턴도 포함
                    if (word.endswith('법') or 
                        re.match(r'^제\d+조$', word) or
                        re.match(r'^[가-힣a-zA-Z]+$', word)):
                        keywords.append(word)
                        if len(keywords) >= max_keywords:
                            break
        
        if keywords:
            self.logger.debug(f"Extracted keywords using morphology: {keywords}")
            return keywords
        
        return []
    
    def _extract_with_fallback(self, query: str, max_keywords: int) -> List[str]:
        """
        폴백 방식: 정규식 기반 키워드 추출
        
        Args:
            query: 추출할 쿼리 문자열
            max_keywords: 최대 키워드 수
            
        Returns:
            추출된 키워드 리스트
        """
        keywords: List[str] = []
        words = query.split()
        
        # 법령명 추출
        law_match = self.LAW_NAME_PATTERN.search(query)
        if law_match:
            law_name = law_match.group(1)
            keywords.append(law_name)
        
        # 조문 번호 추출
        for pattern in self.ARTICLE_PATTERNS:
            article_match = pattern.search(query)
            if article_match:
                article_text = article_match.group().replace(' ', '').strip()
                if not article_text.startswith('제'):
                    article_text = '제' + article_text
                if article_text not in keywords:
                    keywords.append(article_text)
                    break
        
        # 나머지 키워드 추출
        for w in words[:10]:  # 상위 10개 단어 검토
            w_clean = self.JOSA_PATTERN.sub('', w.strip())
            
            if not w_clean or len(w_clean) < 2:
                continue
            
            if w_clean in self.BASIC_STOPWORDS or w_clean in keywords:
                continue
            
            # 한글/영문 포함 단어만
            if re.match(r'^[가-힣a-zA-Z]+$', w_clean):
                keywords.append(w_clean)
                if len(keywords) >= max_keywords:
                    break
        
        # 핵심 키워드가 없으면 원본 단어 사용 (불용어만 제거)
        if not keywords:
            for w in words[:10]:
                w_clean = self.JOSA_PATTERN.sub('', w.strip())
                if w_clean and len(w_clean) >= 2 and w_clean not in self.BASIC_STOPWORDS:
                    keywords.append(w_clean)
                    if len(keywords) >= max_keywords:
                        break
        
        self.logger.debug(f"Extracted keywords using fallback: {keywords}")
        return keywords
    
    def extract_keywords_from_words(
        self,
        words: List[str],
        max_keywords: int = 3
    ) -> List[str]:
        """
        단어 리스트에서 키워드 추출
        
        Args:
            words: 추출할 단어 리스트
            max_keywords: 최대 키워드 수
            
        Returns:
            추출된 키워드 리스트
        """
        if not words:
            return []
        
        query_text = " ".join(words)
        return self.extract_keywords(query_text, max_keywords=max_keywords)
    
    def optimize_query(self, query: str, max_keywords: int = 5) -> str:
        """
        쿼리 최적화 (키워드 추출 후 공백으로 연결)
        
        Args:
            query: 원본 쿼리
            max_keywords: 최대 키워드 수
            
        Returns:
            최적화된 쿼리 문자열
        """
        keywords = self.extract_keywords(query, max_keywords=max_keywords)
        if keywords:
            return " ".join(keywords)
        return query

