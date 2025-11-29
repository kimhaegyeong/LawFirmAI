# -*- coding: utf-8 -*-
"""
한국어 불용어 처리 유틸리티 클래스
KoNLPy를 우선적으로 사용하여 형태소 분석 기반 불용어 처리를 수행합니다.
"""

import logging
import re
from typing import List, Set, Optional

# Global logger 사용
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

logger = get_logger(__name__)

# 싱글톤 인스턴스 (전역 공유)
_global_korean_stopword_processor: Optional['KoreanStopwordProcessor'] = None


class KoreanStopwordProcessor:
    """싱글톤 패턴으로 구현된 한국어 불용어 처리 클래스"""
    
    _instance: Optional['KoreanStopwordProcessor'] = None
    _initialized: bool = False  # 클래스 레벨 초기화 플래그
    _okt_logged: bool = False  # KoNLPy Okt 초기화 로그 출력 여부 (최초 1회만)
    _init_logged: bool = False  # KoreanStopwordProcessor 초기화 로그 출력 여부 (최초 1회만)
    
    def __new__(cls, force_new: bool = False):
        """
        싱글톤 인스턴스 생성
        
        Args:
            force_new: True이면 새 인스턴스 생성 (기본값: False, 싱글톤 사용)
        """
        if force_new:
            # 새 인스턴스 강제 생성
            instance = super().__new__(cls)
            return instance
        
        # 싱글톤 인스턴스 반환
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
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
        # 연결어/부사
        '또한', '그리고', '그러나', '하지만', '따라서', '그러므로', '그런데',
        '또는', '혹은', '및', '그래서', '그럼', '그렇다면',
        # 기타
        '무엇인가요', '무엇인가', '어떤', '어떻게', '왜', '언제', '어디서', '누가',
        '입니다', '이에요', '예요', '것', '이', '그', '때문', '위해', '대해', '관련', '등'
    }
    
    # 허용된 품사 태그 (명사, 동사, 형용사만 추출)
    ALLOWED_POS_TAGS = ['Noun', 'Verb', 'Adjective']
    
    def __init__(self, force_new: bool = False):
        """
        KoNLPy 형태소 분석기 초기화 (선택적)
        
        Args:
            force_new: True이면 새 인스턴스 생성 (기본값: False, 싱글톤 사용)
        """
        # 싱글톤 패턴: 클래스 레벨에서 이미 초기화된 경우 재초기화하지 않음
        if not force_new and KoreanStopwordProcessor._initialized:
            return
        
        self._okt = None
        try:
            # Okt 싱글톤 사용
            from lawfirm_langgraph.core.utils.konlpy_singleton import get_okt_instance
            self._okt = get_okt_instance()
        except ImportError:
            try:
                from core.utils.konlpy_singleton import get_okt_instance
                self._okt = get_okt_instance()
            except ImportError:
                # 폴백: 직접 초기화 (싱글톤 유틸리티가 없는 경우)
                try:
                    from konlpy.tag import Okt
                    self._okt = Okt()
                    # 최초 초기화 시에만 로그 출력
                    if not KoreanStopwordProcessor._okt_logged:
                        logger.debug("KoNLPy Okt initialized successfully")
                        KoreanStopwordProcessor._okt_logged = True
                except ImportError as e:
                    if not KoreanStopwordProcessor._okt_logged:
                        logger.debug(f"KoNLPy not available (ImportError: {e}), will use fallback method")
                        logger.info(
                            "💡 KoNLPy를 사용하려면 다음을 설치하세요:\n"
                            "   1. Java JDK 설치 (KoNLPy는 Java가 필요합니다)\n"
                            "   2. pip install konlpy\n"
                            "   자세한 내용: https://konlpy.org/ko/latest/install/"
                        )
                        KoreanStopwordProcessor._okt_logged = True
                except Exception as e:
                    error_msg = str(e)
                    # Java 관련 에러인지 확인
                    if "java" in error_msg.lower() or "jvm" in error_msg.lower():
                        if not KoreanStopwordProcessor._okt_logged:
                            logger.warning(
                                f"KoNLPy 초기화 실패 (Java 관련): {e}\n"
                                "💡 Java JDK가 설치되어 있는지 확인하세요.\n"
                                "   Windows: https://adoptium.net/ 에서 JDK 다운로드\n"
                                "   환경 변수 JAVA_HOME이 설정되어 있는지 확인하세요."
                            )
                            KoreanStopwordProcessor._okt_logged = True
                    else:
                        if not KoreanStopwordProcessor._okt_logged:
                            logger.warning(f"Error initializing KoNLPy: {e}, will use fallback method")
                            KoreanStopwordProcessor._okt_logged = True
        
        # 초기화 완료 표시 (클래스 레벨)
        if not force_new:
            KoreanStopwordProcessor._initialized = True
            KoreanStopwordProcessor._instance = self
            # 최초 초기화 시에만 로그 출력
            if not KoreanStopwordProcessor._init_logged:
                logger.debug("KoreanStopwordProcessor initialized successfully")
                KoreanStopwordProcessor._init_logged = True
    
    @classmethod
    def get_instance(cls) -> 'KoreanStopwordProcessor':
        """
        싱글톤 인스턴스 가져오기
        
        Returns:
            KoreanStopwordProcessor 인스턴스
        """
        global _global_korean_stopword_processor
        if _global_korean_stopword_processor is None:
            _global_korean_stopword_processor = cls()
        return _global_korean_stopword_processor
    
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

