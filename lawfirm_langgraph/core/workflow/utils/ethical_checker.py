# -*- coding: utf-8 -*-
"""
윤리적 검사 모듈
법률 AI 시스템에서 윤리적으로 문제되는 질의를 감지하고 차단
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EthicalChecker:
    """윤리적 검사 클래스"""
    
    # 불법 행위 조장 키워드
    ILLEGAL_ACTIVITY_KEYWORDS = [
        # 범죄 관련
        "살인", "강도", "절도", "사기", "횡령", "배임", "뇌물", "공갈", "협박",
        "마약", "도박", "성매매", "인신매매", "아동학대", "가정폭력",
        # 해킹/사이버 범죄
        "해킹", "크래킹", "디도스", "피싱", "스미싱", "악성코드", "바이러스",
        "계정 탈취", "비밀번호 해킹", "개인정보 유출", "신용카드 도용",
        # 불법 거래
        "불법 거래", "암거래", "세금 탈루", "탈세", "조세 회피",
        # 기타 불법 행위
        "위조", "변조", "도주", "증거 인멸", "증인 매수", "재판 조작"
    ]
    
    # 윤리적으로 문제되는 질문 패턴
    UNETHICAL_PATTERNS = [
        # 불법 행위 방법 묻기
        r"어떻게.*(불법|범죄|해킹|탈세|위조).*하는",
        r"(불법|범죄|해킹|탈세|위조).*방법",
        r"어떻게.*(살인|강도|절도|사기).*하는",
        # 불법 행위 도움 요청
        r"(불법|범죄|해킹|탈세|위조).*도와",
        r"(불법|범죄|해킹|탈세|위조).*도움",
        r"(불법|범죄|해킹|탈세|위조).*방법.*알려",
        # 법적 책임 회피
        r"법적.*책임.*회피",
        r"법적.*책임.*피하는",
        r"법적.*책임.*없는",
        # 증거 인멸/조작
        r"증거.*인멸",
        r"증거.*조작",
        r"증거.*숨기",
        # 재판 조작
        r"재판.*조작",
        r"판사.*매수",
        r"증인.*매수"
    ]
    
    # 경계선상의 질문 (경고만)
    BORDERLINE_PATTERNS = [
        r"법적.*공백",
        r"법적.*회피",
        r"세금.*절감",
        r"세금.*최소화"
    ]
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        EthicalChecker 초기화
        
        Args:
            logger_instance: 로거 인스턴스 (없으면 자동 생성)
        """
        self.logger = logger_instance or logger
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """정규식 패턴 컴파일"""
        self.unethical_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.UNETHICAL_PATTERNS]
        self.borderline_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.BORDERLINE_PATTERNS]
    
    def check_query(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        질의에 대한 윤리적 검사 수행
        
        Args:
            query: 검사할 질의
        
        Returns:
            (is_problematic, rejection_reason, severity)
            - is_problematic: 윤리적 문제가 있는지 여부
            - rejection_reason: 거부 사유 (문제가 있을 경우)
            - severity: 심각도 ("high", "medium", "low")
        """
        if not query or not isinstance(query, str):
            return False, None, None
        
        query_lower = query.lower()
        query_stripped = query.strip()
        
        # 1. 불법 행위 조장 키워드 체크
        for keyword in self.ILLEGAL_ACTIVITY_KEYWORDS:
            if keyword in query_stripped:
                # 키워드가 문맥상 불법 행위를 묻는지 확인
                if self._is_asking_how_to_commit_illegal_act(query_stripped, keyword):
                    reason = f"불법 행위 조장: '{keyword}' 관련 불법 행위 방법을 묻는 질문이 감지되었습니다."
                    self.logger.warning(f"윤리적 문제 감지: {reason}")
                    return True, reason, "high"
        
        # 2. 윤리적으로 문제되는 패턴 체크
        for pattern in self.unethical_patterns:
            if pattern.search(query_stripped):
                reason = "윤리적으로 문제되는 질문이 감지되었습니다. 불법 행위를 조장하거나 법적 책임을 회피하려는 의도로 보입니다."
                self.logger.warning(f"윤리적 문제 감지: {reason}")
                return True, reason, "high"
        
        # 3. 경계선상의 질문 체크 (경고만, 거부하지 않음)
        for pattern in self.borderline_patterns:
            if pattern.search(query_stripped):
                self.logger.info(f"경계선상의 질문 감지: {query_stripped[:100]}")
                # 경계선상의 질문은 거부하지 않지만 로깅
        
        return False, None, None
    
    def _is_asking_how_to_commit_illegal_act(self, query: str, keyword: str) -> bool:
        """
        질문이 불법 행위 방법을 묻는지 확인
        
        Args:
            query: 질의
            keyword: 감지된 키워드
        
        Returns:
            불법 행위 방법을 묻는지 여부
        """
        # 불법 행위 방법을 묻는 패턴
        how_patterns = [
            r"어떻게.*" + re.escape(keyword),
            r"어떤.*방법.*" + re.escape(keyword),
            r"어떤.*수단.*" + re.escape(keyword),
            r"방법.*알려",
            r"방법.*가르쳐",
            r"방법.*설명",
            r"어떻게.*하는",
            r"어떻게.*할.*수",
            r"어떻게.*하면"
        ]
        
        for pattern in how_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # 단순히 키워드가 언급된 것만으로는 문제가 아닐 수 있음
        # 예: "해킹에 대한 법적 처벌은?" 같은 질문은 괜찮음
        legal_context_patterns = [
            r"법적.*처벌",
            r"법적.*책임",
            r"법적.*규제",
            r"법률.*규정",
            r"법률.*조항",
            r"법률.*처벌",
            r"법률.*제재",
            r"불법.*여부",
            r"불법.*판단",
            r"불법.*확인"
        ]
        
        # 법적 맥락에서 묻는 질문은 허용
        for pattern in legal_context_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False
        
        # 키워드가 있지만 명확한 불법 행위 방법을 묻지 않는 경우
        # 기본적으로는 허용하되, 패턴 매칭으로 재확인
        return False
    
    def check_with_llm(
        self,
        query: str,
        llm=None,
        use_llm_check: bool = False
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        LLM을 사용한 윤리적 검사 (선택적)
        
        Args:
            query: 검사할 질의
            llm: LLM 인스턴스 (선택적)
            use_llm_check: LLM 검사 사용 여부
        
        Returns:
            (is_problematic, rejection_reason, severity)
        """
        # 기본 키워드/패턴 기반 검사 먼저 수행
        is_problematic, reason, severity = self.check_query(query)
        
        if is_problematic:
            return is_problematic, reason, severity
        
        # LLM 검사가 활성화되어 있고 LLM이 제공된 경우
        if use_llm_check and llm:
            try:
                prompt = f"""다음 질문이 윤리적으로 문제되는지 판단해주세요.
법률 AI 어시스턴트로서 불법 행위를 조장하거나 법적 책임을 회피하려는 의도가 있는 질문인지 확인해주세요.

질문: {query}

다음 형식으로 응답해주세요:
- 문제없음: "OK"
- 문제있음: "PROBLEM: [거부 사유]"

응답:"""
                
                response = llm.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                if "PROBLEM:" in response_text:
                    reason = response_text.split("PROBLEM:")[1].strip()
                    self.logger.warning(f"LLM 기반 윤리적 문제 감지: {reason}")
                    return True, reason, "medium"
                
            except Exception as e:
                self.logger.error(f"LLM 기반 윤리 검사 중 오류 발생: {e}")
                # LLM 검사 실패 시 기본 검사 결과 반환
        
        return False, None, None

