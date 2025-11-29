# -*- coding: utf-8 -*-
"""
키워드 추출 유틸리티 클래스
형태소 분석을 활용한 한국어 키워드 추출 및 최적화
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import re
from typing import List, Optional, Set

try:
    from lawfirm_langgraph.core.utils.korean_stopword_processor import KoreanStopwordProcessor
except ImportError:
    try:
        from core.utils.korean_stopword_processor import KoreanStopwordProcessor
    except ImportError:
        KoreanStopwordProcessor = None

logger = get_logger(__name__)


class KeywordExtractor:
    """
    한국어 키워드 추출 클래스
    형태소 분석을 활용하여 조사/어미를 자동 제거하고 핵심 키워드를 추출합니다.
    """
    
    _okt_logged: bool = False  # KoNLPy Okt 초기화 로그 출력 여부 (최초 1회만)
    
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
    
    # 법률 용어 복합어 패턴 (개선: 복합어 인식 강화)
    LEGAL_COMPOUND_PATTERNS = [
        re.compile(r'소멸시효'),
        re.compile(r'손해배상'),
        re.compile(r'불법행위'),
        re.compile(r'계약해지'),
        re.compile(r'계약해제'),
        re.compile(r'손해배상청구권'),
        re.compile(r'원상회복'),
        re.compile(r'이행지체'),
        re.compile(r'이행불능'),
        re.compile(r'채무불이행'),
        re.compile(r'동시이행항변권'),
        re.compile(r'계약위반'),
        re.compile(r'계약불이행'),
        re.compile(r'해지권'),
        re.compile(r'해제권'),
        re.compile(r'약정해제권'),
        re.compile(r'법정해제권'),
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
        
        # KoreanStopwordProcessor 초기화 (KoNLPy 우선 사용, 싱글톤)
        self.stopword_processor = None
        if KoreanStopwordProcessor:
            try:
                self.stopword_processor = KoreanStopwordProcessor.get_instance()
            except Exception as e:
                self.logger.warning(f"Error initializing KoreanStopwordProcessor: {e}")
        
        # KoNLPy 형태소 분석기 초기화 (싱글톤 사용)
        self._okt = None
        if use_morphology:
            try:
                from lawfirm_langgraph.core.utils.konlpy_singleton import get_okt_instance
                self._okt = get_okt_instance()
                if self._okt is None:
                    self.use_morphology = False
            except ImportError:
                try:
                    from core.utils.konlpy_singleton import get_okt_instance
                    self._okt = get_okt_instance()
                    if self._okt is None:
                        self.use_morphology = False
                except ImportError:
                    # 폴백: 직접 초기화 (싱글톤 유틸리티가 없는 경우)
                    try:
                        from konlpy.tag import Okt
                        self._okt = Okt()
                    except (ImportError, Exception):
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
        
        # 법률 용어 복합어 추출 (개선: 복합어 인식 강화)
        for pattern in self.LEGAL_COMPOUND_PATTERNS:
            compound_match = pattern.search(query)
            if compound_match:
                compound_term = compound_match.group().replace(' ', '').strip()
                if compound_term not in keywords:
                    keywords.append(compound_term)
        
        # 형태소 분석으로 핵심 키워드 추출
        for word, pos in pos_tags:
            # 불용어 필터링 (KoreanStopwordProcessor 사용)
            if self.stopword_processor and self.stopword_processor.is_stopword(word):
                continue
            
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
        
        # 법률 용어 복합어 추출 (개선: 복합어 인식 강화)
        for pattern in self.LEGAL_COMPOUND_PATTERNS:
            compound_match = pattern.search(query)
            if compound_match:
                compound_term = compound_match.group().replace(' ', '').strip()
                if compound_term not in keywords:
                    keywords.append(compound_term)
        
        # 나머지 키워드 추출
        for w in words[:10]:  # 상위 10개 단어 검토
            w_clean = self.JOSA_PATTERN.sub('', w.strip())
            
            if not w_clean or len(w_clean) < 2:
                continue
            
            # 불용어 필터링 (KoreanStopwordProcessor 사용)
            if (self.stopword_processor and self.stopword_processor.is_stopword(w_clean)) or w_clean in keywords:
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
                if w_clean and len(w_clean) >= 2:
                    if self.stopword_processor and not self.stopword_processor.is_stopword(w_clean):
                        keywords.append(w_clean)
                    elif not self.stopword_processor:
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

