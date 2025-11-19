# -*- coding: utf-8 -*-
"""
한국어 불용어 처리 유틸리티 클래스
KoNLPy를 우선적으로 사용하여 형태소 분석 기반 불용어 처리를 수행합니다.
"""

import logging
import re
from typing import List, Set, Optional

logger = logging.getLogger(__name__)


class KoreanStopwordProcessor:
    """
    KoNLPy 기반 한국어 불용어 처리 클래스
    
    형태소 분석을 통해 조사와 어미를 자동으로 제거하고,
    법률 도메인 특화 불용어를 추가로 필터링합니다.
    
    사용 예:
        processor = KoreanStopwordProcessor()
        keywords = processor.remove_stopwords("계약 해지에 대해 알려주세요")
        # 결과: ['계약', '해지']
    """
    
    # 법률 도메인 특화 불용어 (KoNLPy로 자동 제거되지 않는 것들)
    DOMAIN_STOPWORDS: Set[str] = {
        # 질문/요청 표현
        '설명해주세요', '설명', '알려주세요', '알려주시기', '알려줘', '설명해줘',
        '요청', '질문', '답변', '바랍니다', '부탁', '드립니다', '해주세요', '해주시기',
        '보여줘', '보여주세요', '찾아줘', '찾아주세요',
        # 법률 도메인 일반 불용어
        '법률', '법', '조문', '항', '호', '목', '단', '절', '장', '편',
        '규정', '조항', '법령', '규칙',
        # 기타
        '무엇인가요', '무엇인가', '어떤', '어떻게', '왜', '언제', '어디서', '누가',
        '입니다', '이에요', '예요'
    }
    
    # 허용된 품사 태그 (명사, 동사, 형용사만 추출)
    ALLOWED_POS_TAGS = ['Noun', 'Verb', 'Adjective']
    
    def __init__(self):
        """KoNLPy 형태소 분석기 초기화 (선택적)"""
        self._okt = None
        try:
            from konlpy.tag import Okt
            self._okt = Okt()
            logger.debug("KoNLPy Okt initialized successfully")
        except ImportError:
            logger.debug("KoNLPy not available, will use fallback method")
        except Exception as e:
            logger.warning(f"Error initializing KoNLPy: {e}, will use fallback method")
    
    def remove_stopwords(self, text: str) -> List[str]:
        """
        KoNLPy 형태소 분석으로 불용어 제거
        
        Args:
            text: 처리할 텍스트
            
        Returns:
            불용어가 제거된 키워드 리스트
        """
        if not text or not text.strip():
            return []
        
        if self._okt:
            try:
                pos_tags = self._okt.pos(text)
                filtered = [
                    word for word, pos in pos_tags
                    if pos in self.ALLOWED_POS_TAGS
                    and word not in self.DOMAIN_STOPWORDS
                    and len(word) >= 2
                ]
                return filtered
            except Exception as e:
                logger.warning(f"KoNLPy processing error: {e}, using fallback")
                return self._remove_stopwords_fallback(text)
        else:
            return self._remove_stopwords_fallback(text)
    
    def _remove_stopwords_fallback(self, text: str) -> List[str]:
        """
        폴백 방식: 정규식 기반 불용어 제거
        
        Args:
            text: 처리할 텍스트
            
        Returns:
            불용어가 제거된 키워드 리스트
        """
        # 조사 패턴 (정규식으로 제거)
        josa_pattern = re.compile(
            r'(에|에서|에게|한테|께|으로|로|의|을|를|이|가|는|은|와|과|도|만|부터|까지|만큼|처럼|같이|따라|대신|더불어|대하여|관하여)$'
        )
        
        # 한글 단어 추출
        words = re.findall(r'[가-힣]+', text)
        
        filtered = []
        for word in words:
            # 조사 제거
            cleaned = josa_pattern.sub('', word.strip())
            if cleaned and len(cleaned) >= 2 and cleaned not in self.DOMAIN_STOPWORDS:
                filtered.append(cleaned)
        
        return filtered
    
    def is_stopword(self, word: str) -> bool:
        """
        단어가 불용어인지 확인
        
        Args:
            word: 확인할 단어
            
        Returns:
            불용어 여부
        """
        if not word or len(word) < 2:
            return True
        
        if word in self.DOMAIN_STOPWORDS:
            return True
        
        if self._okt:
            try:
                pos_tags = self._okt.pos(word)
                if pos_tags and pos_tags[0][1] not in self.ALLOWED_POS_TAGS:
                    return True
            except Exception:
                pass
        
        return False
    
    def filter_stopwords(self, words: List[str]) -> List[str]:
        """
        단어 리스트에서 불용어 제거
        
        Args:
            words: 처리할 단어 리스트
            
        Returns:
            불용어가 제거된 단어 리스트
        """
        if not words:
            return []
        
        filtered = []
        for word in words:
            if not self.is_stopword(word):
                filtered.append(word)
        
        return filtered
    
    def extract_keywords(self, text: str, max_keywords: Optional[int] = None) -> List[str]:
        """
        텍스트에서 키워드 추출 (불용어 제거 포함)
        
        Args:
            text: 처리할 텍스트
            max_keywords: 최대 키워드 수 (None이면 제한 없음)
            
        Returns:
            추출된 키워드 리스트
        """
        keywords = self.remove_stopwords(text)
        
        if max_keywords and len(keywords) > max_keywords:
            return keywords[:max_keywords]
        
        return keywords

