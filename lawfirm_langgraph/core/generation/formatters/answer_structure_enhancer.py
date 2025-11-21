# -*- coding: utf-8 -*-
"""
답변 구조화 향상 시스템
질문 유형별 맞춤형 답변 구조 템플릿 적용
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import re
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    ChatGoogleGenerativeAI = None

from ..validators.legal_basis_validator import LegalBasisValidator
from .legal_citation_enhancer import LegalCitationEnhancer

try:
    from .enhancement.classifiers.question_type_classifier import QuestionTypeClassifier, QuestionType as NewQuestionType  # type: ignore
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    QuestionTypeClassifier = None
    NewQuestionType = None

# 안전한 로깅 유틸리티 import (멀티스레딩 안전)
# 먼저 폴백 함수를 정의 (항상 사용 가능하도록)
def _safe_log_fallback_debug(logger, message):
    """폴백 디버그 로깅 함수"""
    try:
        logger.debug(message)
    except (ValueError, AttributeError, RuntimeError, OSError):
        pass

def _safe_log_fallback_info(logger, message):
    """폴백 정보 로깅 함수"""
    try:
        logger.info(message)
    except (ValueError, AttributeError, RuntimeError, OSError):
        pass

def _safe_log_fallback_warning(logger, message):
    """폴백 경고 로깅 함수"""
    try:
        logger.warning(message)
    except (ValueError, AttributeError, RuntimeError, OSError):
        pass

def _safe_log_fallback_error(logger, message, exc_info=False):
    """폴백 오류 로깅 함수"""
    try:
        logger.error(message, exc_info=exc_info)
    except (ValueError, AttributeError, RuntimeError, OSError):
        pass

# 여러 경로 시도하여 safe_log_* 함수 import
SAFE_LOGGING_AVAILABLE = False
try:
    from core.utils.safe_logging_utils import (
        safe_log_debug,
        safe_log_info,
        safe_log_warning,
        safe_log_error
    )
    SAFE_LOGGING_AVAILABLE = True
except ImportError:
    try:
        # lawfirm_langgraph 경로에서 시도
        from lawfirm_langgraph.core.utils.safe_logging_utils import (
            safe_log_debug,
            safe_log_info,
            safe_log_warning,
            safe_log_error
        )
        SAFE_LOGGING_AVAILABLE = True
    except ImportError:
        # Import 실패 시 폴백 함수 사용
        safe_log_debug = _safe_log_fallback_debug
        safe_log_info = _safe_log_fallback_info
        safe_log_warning = _safe_log_fallback_warning
        safe_log_error = _safe_log_fallback_error

# 최종 확인: safe_log_debug가 정의되지 않았다면 폴백 함수 사용
try:
    _ = safe_log_debug
except NameError:
    safe_log_debug = _safe_log_fallback_debug
try:
    _ = safe_log_info
except NameError:
    safe_log_info = _safe_log_fallback_info
try:
    _ = safe_log_warning
except NameError:
    safe_log_warning = _safe_log_fallback_warning
try:
    _ = safe_log_error
except NameError:
    safe_log_error = _safe_log_fallback_error

# 로거 설정
logger = get_logger(__name__)


# QuestionType은 새로운 분류기에서 import 시도, 실패 시 기존 정의 사용
if CLASSIFIER_AVAILABLE and NewQuestionType:
    QuestionType = NewQuestionType
else:
    class QuestionType(Enum):
        """질문 유형 분류"""
        PRECEDENT_SEARCH = "precedent_search"
        LAW_INQUIRY = "law_inquiry"
        LEGAL_ADVICE = "legal_advice"
        PROCEDURE_GUIDE = "procedure_guide"
        TERM_EXPLANATION = "term_explanation"
        GENERAL_QUESTION = "general_question"
        CONTRACT_REVIEW = "contract_review"
        DIVORCE_PROCEDURE = "divorce_procedure"
        INHERITANCE_PROCEDURE = "inheritance_procedure"
        CRIMINAL_CASE = "criminal_case"
        LABOR_DISPUTE = "labor_dispute"


class AnswerStructureEnhancer:
    """답변 구조화 향상 시스템"""

    def __init__(self, llm=None, max_few_shot_examples: int = 2,
                 enable_few_shot: bool = True, enable_cot: bool = True):
        """
        초기화

        Args:
            llm: LangChain LLM 인스턴스 (없으면 자동 초기화)
                - Google Gemini 지원
            max_few_shot_examples: Few-Shot 예시 최대 개수 (기본값: 2)
                - 프롬프트 길이 제한에 따라 조정 가능
            enable_few_shot: Few-Shot 예시 사용 여부 (기본값: True)
                - False로 설정 시 예시 섹션 제외
            enable_cot: Chain-of-Thought 사용 여부 (기본값: True)
                - False로 설정 시 간단한 Step 1,2,3 가이드 사용

        Raises:
            FileNotFoundError: Few-Shot 예시 파일을 찾을 수 없는 경우 (경고만 발생)

        Note:
            Few-Shot 예시는 data/training/few_shot_examples.json 파일에서 로드됩니다.
            캐싱이 적용되어 여러 번 호출 시 파일 I/O가 발생하지 않습니다.
        """
        # 설정 저장
        self.max_few_shot_examples = max_few_shot_examples
        self.enable_few_shot = enable_few_shot
        self.enable_cot = enable_cot

        try:
            from core.generation.formatters.enhancement.loaders import (
                StructureTemplateLoader,
                QualityIndicatorLoader,
                FewShotExampleLoader
            )
            
            template_loader = StructureTemplateLoader()
            quality_loader = QualityIndicatorLoader()
            few_shot_loader = FewShotExampleLoader()
            
            self.structure_templates = template_loader.load()
            self.quality_indicators = quality_loader.load()
            self.few_shot_examples = few_shot_loader.load() if enable_few_shot else {}
            self._few_shot_loader = few_shot_loader  # 나중에 사용하기 위해 저장
            self._few_shot_examples_cache = None  # 호환성 유지
        except ImportError:
            # 폴백: 기존 메서드 사용
            logger.debug("New loaders not available, using legacy methods")
            self.structure_templates = self._load_structure_templates()
            self.quality_indicators = self._load_quality_indicators()
            self._few_shot_examples_cache = None
            self.few_shot_examples = self._load_few_shot_examples() if enable_few_shot else {}
            self._few_shot_loader = None

        # 법적 근거 강화 시스템 초기화
        self.citation_enhancer = LegalCitationEnhancer()
        self.basis_validator = LegalBasisValidator()

        # LLM 초기화 (LLM 기반 구조화를 위해)
        self.llm = llm or self._initialize_llm()
        self.use_llm = LLM_AVAILABLE and self.llm is not None
        
        # 질문 유형 분류기 초기화 (새로운 모듈 사용)
        if CLASSIFIER_AVAILABLE and QuestionTypeClassifier:
            try:
                self.question_classifier = QuestionTypeClassifier()
            except Exception as e:
                logger.warning(f"Failed to initialize QuestionTypeClassifier: {e}")
                self.question_classifier = None
        else:
            self.question_classifier = None

    def classify_question_type(self, question: str) -> QuestionType:
        """질문 유형 분류 (개선된 키워드 우선순위)"""
        try:
            question_lower = question.lower()

            # 법조문 패턴 우선 체크
            if re.search(r'제\d+조|제\d+항|제\d+호', question):
                # 법령명과 함께 나타나는지 확인
                law_names = ['민법', '형법', '근로기준법', '상법', '행정법', '헌법', '특허법', '부동산등기법']
                if any(law_name in question_lower for law_name in law_names):
                    return QuestionType.LAW_INQUIRY

            # 판례 관련 (패턴 기반 개선)
            if any(keyword in question_lower for keyword in ['판례', '대법원', '고등법원', '지방법원', '판결']):
                # 판례 검색 패턴
                precedent_patterns = [
                    r'판례를?\s+찾아주세요',           # "판례를 찾아주세요"
                    r'관련\s+판례',                   # "관련 판례"
                    r'유사한\s+판례',                 # "유사한 판례"
                    r'최근\s+판례',                   # "최근 판례"
                    r'대법원\s+판례',                 # "대법원 판례"
                    r'고등법원\s+판례',               # "고등법원 판례"
                    r'지방법원\s+판례',               # "지방법원 판례"
                    r'판례\s+검색',                   # "판례 검색"
                    r'판례\s+찾기',                   # "판례 찾기"
                ]

                if any(re.search(pattern, question_lower) for pattern in precedent_patterns):
                    return QuestionType.PRECEDENT_SEARCH

                return QuestionType.PRECEDENT_SEARCH

            # 이혼 관련 (패턴 기반 개선)
            if any(keyword in question_lower for keyword in ['이혼', '협의이혼', '재판이혼', '이혼절차']):
                # 이혼 절차 패턴
                divorce_patterns = [
                    r'이혼\s+절차',                   # "이혼 절차"
                    r'이혼\s+방법',                   # "이혼 방법"
                    r'이혼\s+신청',                   # "이혼 신청"
                    r'협의이혼\s+절차',               # "협의이혼 절차"
                    r'재판이혼\s+절차',               # "재판이혼 절차"
                    r'이혼\s+어떻게',                 # "이혼 어떻게"
                    r'이혼\s+어디서',                 # "이혼 어디서"
                    r'이혼\s+비용',                   # "이혼 비용"
                ]

                if any(re.search(pattern, question_lower) for pattern in divorce_patterns):
                    return QuestionType.DIVORCE_PROCEDURE

                return QuestionType.DIVORCE_PROCEDURE

            # 상속 관련 (패턴 기반 개선)
            if any(keyword in question_lower for keyword in ['상속', '유산', '상속인', '상속세', '유언', '상속포기']):
                # 상속 절차 패턴
                inheritance_patterns = [
                    r'상속\s+절차',                   # "상속 절차"
                    r'상속\s+신청',                   # "상속 신청"
                    r'상속\s+방법',                   # "상속 방법"
                    r'유산\s+분할',                   # "유산 분할"
                    r'상속인\s+확인',                 # "상속인 확인"
                    r'상속세\s+신고',                 # "상속세 신고"
                    r'유언\s+검인',                   # "유언 검인"
                    r'상속포기\s+절차',               # "상속포기 절차"
                ]

                if any(re.search(pattern, question_lower) for pattern in inheritance_patterns):
                    return QuestionType.INHERITANCE_PROCEDURE

                return QuestionType.INHERITANCE_PROCEDURE

            # 형사 관련 (패턴 기반 개선)
            if any(keyword in question_lower for keyword in ['사기', '절도', '강도', '살인', '형사', '범죄', '구성요건']):
                # 형사 사건 패턴
                criminal_patterns = [
                    r'\w+죄\s+구성요건',              # "사기죄 구성요건"
                    r'\w+죄\s+처벌',                  # "사기죄 처벌"
                    r'\w+죄\s+형량',                  # "사기죄 형량"
                    r'\w+범죄\s+처벌',                # "절도범죄 처벌"
                    r'형사\s+사건',                  # "형사 사건"
                    r'범죄\s+구성요건',               # "범죄 구성요건"
                    r'\w+사건\s+대응',                # "사기사건 대응"
                ]

                if any(re.search(pattern, question_lower) for pattern in criminal_patterns):
                    return QuestionType.CRIMINAL_CASE

                return QuestionType.CRIMINAL_CASE

            # 노동 관련 (패턴 기반 개선)
            if any(keyword in question_lower for keyword in ['노동', '근로', '임금', '해고', '부당해고', '임금체불', '근로시간', '노동위원회']):
                # 노동 분쟁 패턴
                labor_patterns = [
                    r'노동\s+분쟁',                   # "노동 분쟁"
                    r'근로\s+분쟁',                   # "근로 분쟁"
                    r'임금\s+체불',                   # "임금 체불"
                    r'부당해고\s+구제',               # "부당해고 구제"
                    r'해고\s+대응',                   # "해고 대응"
                    r'근로시간\s+규정',               # "근로시간 규정"
                    r'노동위원회\s+신청',             # "노동위원회 신청"
                    r'임금\s+지급',                   # "임금 지급"
                ]

                if any(re.search(pattern, question_lower) for pattern in labor_patterns):
                    return QuestionType.LABOR_DISPUTE

                return QuestionType.LABOR_DISPUTE

            # 법률 용어 설명 관련 (패턴 기반 개선)
            if any(keyword in question_lower for keyword in ['의미', '정의', '개념', '설명', '무엇', '뜻']):
                # 용어 설명 패턴 감지
                term_patterns = [
                    r'무엇이\s+\w+인가요?',           # "무엇이 계약인가요?"
                    r'무엇이\s+\w+인가\?',           # "무엇이 계약인가?"
                    r'\w+의\s+의미는?',              # "계약의 의미는?"
                    r'\w+의\s+정의는?',              # "계약의 정의는?"
                    r'\w+의\s+개념은?',              # "계약의 개념은?"
                    r'\w+이\s+무엇인가요?',          # "계약이 무엇인가요?"
                    r'\w+이\s+무엇인가\?',           # "계약이 무엇인가?"
                    r'\w+란\s+무엇인가요?',          # "계약이란 무엇인가요?"
                    r'\w+란\s+무엇인가\?',           # "계약이란 무엇인가?"
                    r'\w+의\s+뜻은?',               # "계약의 뜻은?"
                    r'\w+이\s+뜻하는\s+바는?',       # "계약이 뜻하는 바는?"
                    r'\w+의\s+내용은?',             # "계약의 내용은?" (용어 설명)
                    r'\w+이\s+어떤\s+것인가요?',     # "계약이 어떤 것인가요?"
                    r'\w+이\s+어떤\s+것인가\?',      # "계약이 어떤 것인가?"
                    r'\w+에\s+대해\s+설명해주세요',   # "계약에 대해 설명해주세요"
                    r'\w+에\s+대한\s+설명',          # "계약에 대한 설명"
                    r'\w+이\s+무엇을\s+의미하나요?', # "계약이 무엇을 의미하나요?"
                    r'\w+이\s+무엇을\s+의미하나\?',  # "계약이 무엇을 의미하나?"
                    r'\w+이\s+어떤\s+것인가요?',     # "계약이 어떤 것인가요?" (중복 제거)
                    r'\w+이\s+어떤\s+것인가\?',      # "계약이 어떤 것인가?" (중복 제거)
                ]

                # 용어 설명 패턴 매칭
                if any(re.search(pattern, question_lower) for pattern in term_patterns):
                    return QuestionType.TERM_EXPLANATION

                # 계약서 검토 의도가 명확한 경우 (구체적 행동 키워드)
                contract_action_keywords = [
                    '계약서', '조항', '검토', '수정', '불리한', '작성', '체결',
                    '서명', '계약서를', '계약서에', '계약서의', '계약서가',
                    '계약 조건', '계약 조항', '계약서 검토',
                    '계약서 작성', '계약서 수정', '계약서 체결'
                ]

                # "계약의 내용은?" 같은 용어 설명은 제외
                if any(keyword in question_lower for keyword in contract_action_keywords):
                    # 용어 설명 패턴이 아닌 경우에만 계약서 검토로 분류
                    if not any(re.search(pattern, question_lower) for pattern in [
                        r'\w+의\s+내용은?',  # "계약의 내용은?"
                        r'\w+이\s+어떤\s+것인가요?',  # "계약이 어떤 것인가요?"
                    ]):
                        return QuestionType.CONTRACT_REVIEW

                return QuestionType.TERM_EXPLANATION

            # 계약서 검토 관련 (패턴 기반 개선)
            if any(keyword in question_lower for keyword in ['계약서', '계약', '조항', '검토', '수정', '불리한']):
                # 판례 키워드가 함께 있으면 판례 검색 우선
                if any(keyword in question_lower for keyword in ['판례', '대법원', '고등법원', '지방법원', '판결']):
                    return QuestionType.PRECEDENT_SEARCH

                # 계약서 검토 패턴
                contract_patterns = [
                    r'계약서를?\s+검토',               # "계약서를 검토"
                    r'계약서\s+검토',                 # "계약서 검토"
                    r'계약서를?\s+수정',               # "계약서를 수정"
                    r'계약서\s+수정',                 # "계약서 수정"
                    r'계약서를?\s+작성',               # "계약서를 작성"
                    r'계약서\s+작성',                 # "계약서 작성"
                    r'계약서를?\s+체결',               # "계약서를 체결"
                    r'계약서\s+체결',                 # "계약서 체결"
                    r'계약\s+조항',                   # "계약 조항"
                    r'계약\s+조건',                   # "계약 조건"
                    r'계약\s+내용',                   # "계약 내용"
                    r'불리한\s+조항',                 # "불리한 조항"
                    r'계약서의?\s+문제점',             # "계약서의 문제점"
                    r'계약서를?\s+확인',               # "계약서를 확인"
                ]

                if any(re.search(pattern, question_lower) for pattern in contract_patterns):
                    return QuestionType.CONTRACT_REVIEW

                return QuestionType.CONTRACT_REVIEW

            # 법률 자문 관련 (패턴 기반 개선)
            if any(keyword in question_lower for keyword in ['대응', '권리', '의무', '구제', '상담', '자문', '해야', '조언', '도움', '지원']):
                # 법률 자문 패턴
                advice_patterns = [
                    r'어떻게\s+대응해야',               # "어떻게 대응해야"
                    r'어떻게\s+해야',                  # "어떻게 해야"
                    r'권리\s+구제',                    # "권리 구제"
                    r'의무\s+이행',                    # "의무 이행"
                    r'법률\s+상담',                    # "법률 상담"
                    r'법률\s+자문',                    # "법률 자문"
                    r'변호사\s+상담',                  # "변호사 상담"
                    r'변호사\s+자문',                  # "변호사 자문"
                    r'법적\s+대응',                    # "법적 대응"
                    r'법적\s+조언',                    # "법적 조언"
                    r'법적\s+상담',                    # "법적 상담"
                    r'법적\s+자문',                    # "법적 자문"
                    r'법적\s+도움',                    # "법적 도움"
                    r'법적\s+지원',                    # "법적 지원"
                    r'법적\s+보호',                    # "법적 보호"
                    r'법적\s+해결',                    # "법적 해결"
                    r'법적\s+구제',                    # "법적 구제"
                    r'해야\s+할\s+일',                 # "해야 할 일"
                    r'어떤\s+조치',                   # "어떤 조치"
                    r'어떤\s+방법',                   # "어떤 방법"
                    r'조언을?\s+구하고',               # "조언을 구하고"
                    r'도움이?\s+필요',                 # "도움이 필요"
                    r'지원을?\s+받고',                 # "지원을 받고"
                    r'상담을?\s+받고',                 # "상담을 받고"
                    r'자문을?\s+받고',                 # "자문을 받고"
                ]

                if any(re.search(pattern, question_lower) for pattern in advice_patterns):
                    return QuestionType.LEGAL_ADVICE

                # 절차 안내 키워드와 충돌하는 경우 법률 자문 우선
                if any(keyword in question_lower for keyword in ['어떻게', '방법']):
                    # 구체적인 법률 자문 키워드가 있으면 법률 자문 우선
                    if any(keyword in question_lower for keyword in ['대응', '권리', '의무', '구제', '상담', '자문', '조언', '도움', '지원']):
                        return QuestionType.LEGAL_ADVICE

                return QuestionType.LEGAL_ADVICE

            # 절차 안내 관련 (패턴 기반 개선)
            if any(keyword in question_lower for keyword in ['절차', '신청', '소액사건', '민사조정', '소송']):
                # 절차 안내 패턴
                procedure_patterns = [
                    r'\w+\s+절차',                    # "소송 절차"
                    r'\w+\s+신청',                    # "소송 신청"
                    r'\w+\s+방법',                    # "소송 방법"
                    r'소액사건\s+절차',               # "소액사건 절차"
                    r'민사조정\s+신청',               # "민사조정 신청"
                    r'소송\s+제기',                    # "소송 제기"
                    r'어떻게\s+신청',                 # "어떻게 신청"
                    r'어디서\s+신청',                 # "어디서 신청"
                    r'신청\s+방법',                   # "신청 방법"
                    r'신청\s+절차',                   # "신청 절차"
                    r'처리\s+절차',                   # "처리 절차"
                    r'진행\s+절차',                   # "진행 절차"
                ]

                if any(re.search(pattern, question_lower) for pattern in procedure_patterns):
                    return QuestionType.PROCEDURE_GUIDE

                return QuestionType.PROCEDURE_GUIDE

            # 일반적인 방법/어떻게 질문 (마지막에 체크)
            if any(keyword in question_lower for keyword in ['어떻게', '방법', '해야']):
                # 다른 구체적 키워드가 없으면 절차 안내
                return QuestionType.PROCEDURE_GUIDE

            # 일반적인 도움 요청 (구체적 법률 키워드가 없는 경우)
            if any(keyword in question_lower for keyword in ['도움', '지원', '필요']):
                # 구체적인 법률 키워드가 없으면 일반 질문
                if not any(keyword in question_lower for keyword in ['법률', '법적', '변호사', '상담', '자문', '조언', '대응', '권리', '의무', '구제']):
                    return QuestionType.GENERAL_QUESTION

            # 기본값
            return QuestionType.GENERAL_QUESTION

        except Exception as e:
            safe_log_error(logger, f"Error in classify_question_type: {e}")
            return QuestionType.GENERAL_QUESTION

    def _load_structure_templates(self) -> Dict[QuestionType, Dict[str, Any]]:
        """구조 템플릿 로드"""
        try:
            templates = {}

            # 판례 검색 템플릿
            templates[QuestionType.PRECEDENT_SEARCH] = {
                "title": "판례 검색 결과",
                "sections": [
                    {"name": "관련 판례", "priority": "high", "template": "다음과 같은 관련 판례를 찾았습니다:", "content_guide": "판례 번호, 사건명, 핵심 판결요지 포함", "legal_citations": True},
                    {"name": "판례 분석", "priority": "high", "template": "해당 판례의 주요 쟁점과 법원의 판단:", "content_guide": "법리적 분석과 실무적 시사점"},
                    {"name": "적용 가능성", "priority": "medium", "template": "귀하의 사안에의 적용 가능성:", "content_guide": "유사점과 차이점 분석"},
                    {"name": "실무 조언", "priority": "medium", "template": "실무적 권장사항:", "content_guide": "구체적 행동 방안"}
                ]
            }

            # 법령 문의 템플릿
            templates[QuestionType.LAW_INQUIRY] = {
                "title": "법률 문의 답변",
                "sections": [
                    {"name": "관련 법령", "priority": "high", "template": "관련 법령:", "content_guide": "정확한 조문 번호와 내용", "legal_citations": True},
                    {"name": "법령 해설", "priority": "high", "template": "법령 해설:", "content_guide": "쉬운 말로 풀어서 설명"},
                    {"name": "적용 사례", "priority": "medium", "template": "실제 적용 사례:", "content_guide": "구체적 예시와 설명"},
                    {"name": "주의사항", "priority": "medium", "template": "주의사항:", "content_guide": "법적 리스크와 제한사항"}
                ]
            }

            # 법률 상담 템플릿
            templates[QuestionType.LEGAL_ADVICE] = {
                "title": "법률 상담 답변",
                "sections": [
                    {"name": "상황 정리", "priority": "high", "template": "말씀하신 상황을 정리하면:", "content_guide": "핵심 사실 관계 정리"},
                    {"name": "법적 분석", "priority": "high", "template": "법적 분석:", "content_guide": "적용 법률과 법리 분석", "legal_citations": True},
                    {"name": "권리 구제 방법", "priority": "high", "template": "권리 구제 방법:", "content_guide": "단계별 구체적 방안"},
                    {"name": "필요 증거", "priority": "medium", "template": "필요한 증거 자료:", "content_guide": "구체적 증거 목록"},
                    {"name": "전문가 상담", "priority": "low", "template": "전문가 상담 권유:", "content_guide": "변호사 상담 필요성"}
                ]
            }

            # 절차 안내 템플릿
            templates[QuestionType.PROCEDURE_GUIDE] = {
                "title": "절차 안내",
                "sections": [
                    {"name": "절차 개요", "priority": "high", "template": "전체 절차 개요:", "content_guide": "절차의 전체적인 흐름"},
                    {"name": "단계별 절차", "priority": "high", "template": "단계별 절차:", "content_guide": "구체적 단계별 설명"},
                    {"name": "필요 서류", "priority": "high", "template": "필요한 서류:", "content_guide": "구체적 서류 목록"},
                    {"name": "처리 기간", "priority": "medium", "template": "처리 기간 및 비용:", "content_guide": "예상 소요시간과 비용"},
                    {"name": "주의사항", "priority": "medium", "template": "주의사항:", "content_guide": "절차 진행 시 주의할 점"}
                ]
            }

            # 용어 해설 템플릿
            templates[QuestionType.TERM_EXPLANATION] = {
                "title": "법률 용어 해설",
                "sections": [
                    {"name": "용어 정의", "priority": "high", "template": "용어 정의:", "content_guide": "정확한 법률적 정의"},
                    {"name": "법적 근거", "priority": "high", "template": "법적 근거:", "content_guide": "관련 법조문과 판례", "legal_citations": True},
                    {"name": "실제 적용", "priority": "medium", "template": "실제 적용 사례:", "content_guide": "구체적 적용 예시"},
                    {"name": "관련 용어", "priority": "low", "template": "관련 용어:", "content_guide": "비슷하거나 관련된 용어들"}
                ]
            }

            # 계약서 검토 템플릿
            templates[QuestionType.CONTRACT_REVIEW] = {
                "title": "계약서 검토 결과",
                "sections": [
                    {"name": "계약서 분석", "priority": "high", "template": "계약서 주요 내용 분석:", "content_guide": "계약의 핵심 조항 분석"},
                    {"name": "법적 검토", "priority": "high", "template": "법적 검토 결과:", "content_guide": "법적 유효성과 문제점", "legal_citations": True},
                    {"name": "주의사항", "priority": "high", "template": "주의해야 할 사항:", "content_guide": "불리한 조항과 리스크"},
                    {"name": "개선 제안", "priority": "medium", "template": "개선 제안:", "content_guide": "구체적 수정 권장사항"}
                ]
            }

            # 이혼 절차 템플릿
            templates[QuestionType.DIVORCE_PROCEDURE] = {
                "title": "이혼 절차 안내",
                "sections": [
                    {"name": "이혼 방법", "priority": "high", "template": "이혼 방법 선택:", "content_guide": "협의이혼, 조정이혼, 재판이혼 비교"},
                    {"name": "절차 단계", "priority": "high", "template": "구체적 절차:", "content_guide": "단계별 상세 절차"},
                    {"name": "필요 서류", "priority": "high", "template": "필요한 서류:", "content_guide": "구체적 서류 목록"},
                    {"name": "재산분할", "priority": "medium", "template": "재산분할 및 위자료:", "content_guide": "재산분할 기준과 위자료 산정"},
                    {"name": "양육권", "priority": "medium", "template": "양육권 및 면접교섭권:", "content_guide": "자녀 양육 관련 사항"}
                ]
            }

            # 상속 절차 템플릿
            templates[QuestionType.INHERITANCE_PROCEDURE] = {
                "title": "상속 절차 안내",
                "sections": [
                    {"name": "상속인 확인", "priority": "high", "template": "상속인 및 상속분:", "content_guide": "법정상속인과 상속분 계산"},
                    {"name": "상속 절차", "priority": "high", "template": "상속 절차:", "content_guide": "단계별 상속 절차"},
                    {"name": "필요 서류", "priority": "high", "template": "필요한 서류:", "content_guide": "상속 관련 서류 목록"},
                    {"name": "세금 문제", "priority": "medium", "template": "상속세 및 증여세:", "content_guide": "세금 관련 주의사항"},
                    {"name": "유언 검인", "priority": "low", "template": "유언 검인 절차:", "content_guide": "유언이 있는 경우 절차"}
                ]
            }

            # 형사 사건 템플릿
            templates[QuestionType.CRIMINAL_CASE] = {
                "title": "형사 사건 안내",
                "sections": [
                    {"name": "범죄 분석", "priority": "high", "template": "해당 범죄의 구성요건:", "content_guide": "범죄 성립요건 분석", "legal_citations": True},
                    {"name": "법정형", "priority": "high", "template": "법정형 및 형량:", "content_guide": "처벌 기준과 형량"},
                    {"name": "수사 절차", "priority": "medium", "template": "수사 및 재판 절차:", "content_guide": "수사부터 재판까지 절차"},
                    {"name": "변호인 조력", "priority": "high", "template": "변호인 조력권:", "content_guide": "변호인 선임과 조력권"},
                    {"name": "구제 방법", "priority": "medium", "template": "권리 구제 방법:", "content_guide": "항소, 상고 등 구제 절차"}
                ]
            }

            # 노동 분쟁 템플릿
            templates[QuestionType.LABOR_DISPUTE] = {
                "title": "노동 분쟁 안내",
                "sections": [
                    {"name": "분쟁 분석", "priority": "high", "template": "노동 분쟁 분석:", "content_guide": "분쟁의 성격과 쟁점"},
                    {"name": "적용 법령", "priority": "high", "template": "적용 법령:", "content_guide": "근로기준법 등 관련 법령", "legal_citations": True},
                    {"name": "구제 절차", "priority": "high", "template": "구제 절차:", "content_guide": "노동위원회, 법원 절차"},
                    {"name": "필요 증거", "priority": "medium", "template": "필요한 증거:", "content_guide": "임금대장, 근로계약서 등"},
                    {"name": "시효 문제", "priority": "medium", "template": "시효 및 제한:", "content_guide": "신청 기한과 제한사항"}
                ]
            }

            # 일반 질문 템플릿
            templates[QuestionType.GENERAL_QUESTION] = {
                "title": "법률 질문 답변",
                "sections": [
                    {"name": "질문 분석", "priority": "high", "template": "질문 내용 분석:", "content_guide": "질문의 핵심 파악"},
                    {"name": "관련 법령", "priority": "high", "template": "관련 법령:", "content_guide": "적용 가능한 법령", "legal_citations": True},
                    {"name": "법적 해설", "priority": "medium", "template": "법적 해설:", "content_guide": "쉬운 말로 설명"},
                    {"name": "실무 조언", "priority": "medium", "template": "실무적 조언:", "content_guide": "구체적 행동 방안"}
                ]
            }

            return templates

        except Exception as e:
            safe_log_error(logger, f"Failed to load templates: {e}")
            return self._get_fallback_templates()

    def _get_fallback_templates(self) -> Dict[QuestionType, Dict[str, Any]]:
        """폴백 템플릿 생성"""
        return {
            QuestionType.GENERAL_QUESTION: {
                "title": "법률 질문 답변",
                "sections": [
                    {
                        "name": "질문 분석",
                        "priority": "high",
                        "template": "질문 내용 분석:",
                        "content_guide": "질문의 핵심 파악"
                    },
                    {
                        "name": "관련 법령",
                        "priority": "high",
                        "template": "관련 법령:",
                        "content_guide": "적용 가능한 법령"
                    },
                    {
                        "name": "법적 해설",
                        "priority": "medium",
                        "template": "법적 해설:",
                        "content_guide": "쉬운 말로 설명"
                    },
                    {
                        "name": "실무 조언",
                        "priority": "medium",
                        "template": "실무적 조언:",
                        "content_guide": "구체적 행동 방안"
                    }
                ]
            }
        }

    def _load_few_shot_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Few-Shot 예시 데이터 로드 (캐싱 적용)

        Returns:
            Dict[str, List[Dict[str, Any]]]: 질문 유형별 Few-Shot 예시 데이터
        """
        # 캐시 확인
        if hasattr(self, '_few_shot_examples_cache') and self._few_shot_examples_cache is not None:
            return self._few_shot_examples_cache

        import json
        import os

        # 파일 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        examples_file = os.path.join(
            current_dir,
            '..',
            '..',
            'data',
            'training',
            'few_shot_examples.json'
        )

        try:
            if os.path.exists(examples_file):
                with open(examples_file, 'r', encoding='utf-8') as f:
                    examples = json.load(f)
                    # 캐시에 저장
                    self._few_shot_examples_cache = examples
                    safe_log_debug(logger, f"Few-shot examples loaded and cached: {len(examples)} question types")
                    return examples
            else:
                # 파일이 없으면 빈 딕셔너리 반환
                safe_log_warning(logger, f"Few-shot examples file not found: {examples_file}")
                return {}
        except Exception as e:
            # 에러 발생 시 빈 딕셔너리 반환
            safe_log_warning(logger, f"Failed to load few-shot examples: {e}")
            return {}

    def _get_few_shot_examples(self, question_type: QuestionType, question: str = "") -> List[Dict[str, Any]]:
        """
        질문 유형별 Few-Shot 예시 반환 (리팩토링: 새로운 로더 사용)
        
        Args:
            question_type: 질문 유형 (QuestionType enum)
            question: 질문 텍스트 (유사도 계산용, 선택적)
            
        Returns:
            List[Dict[str, Any]]: 질문 유형별 Few-Shot 예시 리스트
        """
        # 새로운 로더 사용 시도
        if hasattr(self, '_few_shot_loader') and self._few_shot_loader is not None:
            try:
                max_examples = getattr(self, 'max_few_shot_examples', 2)
                return self._few_shot_loader.get_examples(question_type, question, max_examples)
            except Exception as e:
                logger.warning(f"Failed to get examples from new loader: {e}, falling back to legacy method")
        
        # 폴백: 기존 메서드 사용
        if not hasattr(self, 'few_shot_examples') or not self.few_shot_examples:
            return []

        # 질문 유형을 문자열로 변환
        question_type_str = question_type.value if isinstance(question_type, QuestionType) else str(question_type)

        # 해당 질문 유형의 예시 가져오기
        examples = self.few_shot_examples.get(question_type_str, [])

        # 검증 통과한 예시만 필터링 (품질 메트릭 포함)
        valid_examples = []
        for ex in examples:
            if hasattr(self, '_validate_few_shot_example') and self._validate_few_shot_example(ex):
                valid_examples.append(ex)
            elif not hasattr(self, '_validate_few_shot_example'):
                # 검증 메서드가 없으면 기본 검증만 수행
                if all(key in ex for key in ['question', 'original_answer', 'enhanced_answer', 'improvements']):
                    valid_examples.append(ex)

        # 검증 실패한 예시가 있으면 경고
        if len(valid_examples) < len(examples):
            invalid_count = len(examples) - len(valid_examples)
            safe_log_warning(logger, f"{question_type_str}: {invalid_count}개 예시가 검증 실패했습니다.")

        # 질문이 제공되고 예시가 여러 개인 경우 유사도 기반 정렬 시도
        if question and len(valid_examples) > 1:
            try:
                if hasattr(self, '_sort_examples_by_similarity'):
                    valid_examples = self._sort_examples_by_similarity(valid_examples, question)
            except Exception as e:
                safe_log_debug(logger, f"Failed to sort examples by similarity: {e}")

        # 설정된 최대 개수까지만 반환 (프롬프트 길이 제한)
        max_examples = getattr(self, 'max_few_shot_examples', 2)
        return valid_examples[:max_examples]

    def _load_quality_indicators(self) -> Dict[str, List[str]]:
        """품질 지표 로드"""
        return self._get_fallback_quality_indicators()

    def _get_fallback_quality_indicators(self) -> Dict[str, List[str]]:
        """폴백 품질 지표"""
        return {
            "legal_accuracy": [
                "법령", "조문", "조항", "항목", "법원", "판례", "대법원", "하급심"
            ],
            "practical_guidance": [
                "구체적", "실행", "단계별", "절차", "방법", "조치", "권장", "고려"
            ],
            "structure_quality": [
                "##", "###", "**", "1.", "2.", "3.", "•", "-", "첫째", "둘째", "셋째"
            ],
            "completeness": [
                "따라서", "결론적으로", "요약하면", "종합하면", "판단컨대"
            ],
            "risk_management": [
                "주의", "주의사항", "리스크", "제한", "한계", "전문가", "상담"
            ]
        }

    def enhance_answer_structure(self, answer: str, question_type: str,
                               question: str = "", domain: str = "general",
                               retrieved_docs: Optional[List[Dict[str, Any]]] = None,
                               legal_references: Optional[List[str]] = None,
                               legal_citations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """답변 구조화 향상 (안전한 버전) - 법적 근거 정보 포함"""
        try:
            # 입력 검증
            if not answer or not isinstance(answer, str):
                return {"error": "Invalid answer input"}

            # 법적 근거 정보 준비 (None 체크 및 타입 안전성 보장)
            retrieved_docs = retrieved_docs if retrieved_docs is not None else []
            legal_references = legal_references if legal_references is not None else []
            legal_citations = legal_citations if legal_citations is not None else []

            # 타입 검증
            if not isinstance(retrieved_docs, list):
                retrieved_docs = []
            if not isinstance(legal_references, list):
                legal_references = []
            if not isinstance(legal_citations, list):
                legal_citations = []

            # 질문 유형 매핑
            mapped_question_type = self._map_question_type(question_type, question)

            # LLM 기반 구조화 시도 (권장)
            if self.use_llm:
                try:
                    return self._enhance_with_llm(
                        answer, question, mapped_question_type,
                        retrieved_docs, legal_references, legal_citations
                    )
                except Exception as e:
                    safe_log_warning(logger, f"LLM 기반 구조화 실패, 템플릿 방식으로 폴백: {e}")
                    # 폴백: 템플릿 기반 구조화

            # 템플릿 기반 구조화 (폴백)
            return self._enhance_with_template(
                answer, mapped_question_type, question,
                retrieved_docs, legal_references, legal_citations
            )

        except Exception as e:
            safe_log_error(logger, f"답변 구조화 향상 실패: {e}")
            return {"error": str(e)}

    def _map_question_type(self, question_type: any, question: str) -> QuestionType:
        """질문 유형 매핑"""
        try:
            # 명시적 질문 유형 처리
            explicit_result = self._handle_explicit_question_type(question_type)
            if explicit_result != QuestionType.GENERAL_QUESTION:
                return explicit_result

            # 질문 구조 분석
            structure_result = self._analyze_question_structure(question)
            if structure_result != QuestionType.GENERAL_QUESTION:
                return structure_result

            # 최종 폴백
            return QuestionType.GENERAL_QUESTION

        except Exception as e:
            logger.warning(f"Question type mapping failed: {e}", exc_info=True)
            # 폴백: 기존 방식 사용
            return self._map_question_type_fallback(question_type, question)

    def _handle_explicit_question_type(self, question_type: any) -> QuestionType:
        """명시적 질문 유형 처리"""
        # QuestionType enum인 경우 value 사용
        if isinstance(question_type, QuestionType):
            return question_type

        # 문자열인 경우 처리
        if not question_type or (isinstance(question_type, str) and question_type.lower() == "general"):
            return QuestionType.GENERAL_QUESTION

        # 명시적 매핑
        explicit_mapping = {
            'precedent_search': QuestionType.PRECEDENT_SEARCH,
            'contract_review': QuestionType.CONTRACT_REVIEW,
            'divorce_procedure': QuestionType.DIVORCE_PROCEDURE,
            'inheritance_procedure': QuestionType.INHERITANCE_PROCEDURE,
            'criminal_case': QuestionType.CRIMINAL_CASE,
            'labor_dispute': QuestionType.LABOR_DISPUTE,
            'procedure_guide': QuestionType.PROCEDURE_GUIDE,
            'term_explanation': QuestionType.TERM_EXPLANATION,
            'legal_advice': QuestionType.LEGAL_ADVICE,
            'law_inquiry': QuestionType.LAW_INQUIRY,
            'general_question': QuestionType.GENERAL_QUESTION
        }

        # 문자열 변환
        if isinstance(question_type, str):
            return explicit_mapping.get(question_type.lower(), QuestionType.GENERAL_QUESTION)

            return QuestionType.GENERAL_QUESTION

    def _initialize_llm(self):
        """LLM 초기화"""
        if not LLM_AVAILABLE:
            return None

        try:
            # LangGraphConfig에서 LLM 설정 가져오기
            # 여러 경로 시도 (상대/절대 경로 모두 지원)
            try:
                from core.utils.langgraph_config import LangGraphConfig
            except ImportError:
                try:
                    from ...utils.langgraph_config import LangGraphConfig
                except ImportError:
                    # 최종 폴백: sys.path를 이용한 동적 import
                    import os
                    import sys
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                    if project_root not in sys.path:
                        sys.path.insert(0, project_root)
                    from core.utils.langgraph_config import LangGraphConfig

            config = LangGraphConfig.from_env()

            if config.llm_provider == "google" and ChatGoogleGenerativeAI:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=config.google_model or "gemini-2.5-flash-lite",
                        temperature=0.3,
                        max_output_tokens=4000,
                        timeout=30,
                        api_key=config.google_api_key
                    )
                    logger.info(f"LLM initialized: Google Gemini ({config.google_model})")
                    return llm
                except Exception as e:
                    logger.warning(f"Failed to initialize Google Gemini (optional): {e}")
                    # 선택적 의존성이므로 오류를 무시하고 계속 진행

        except Exception as e:
            safe_log_error(logger, f"LLM initialization error: {e}")

        return None

    def _enhance_with_llm(
        self,
        answer: str,
        question: str,
        question_type: QuestionType,
        retrieved_docs: List[Dict[str, Any]],
        legal_references: List[str],
        legal_citations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """LLM을 활용한 구조화된 답변 생성"""

        # None 체크 및 타입 안전성 보장
        retrieved_docs = retrieved_docs if retrieved_docs is not None else []
        legal_references = legal_references if legal_references is not None else []
        legal_citations = legal_citations if legal_citations is not None else []

        if not isinstance(retrieved_docs, list):
            retrieved_docs = []
        if not isinstance(legal_references, list):
            legal_references = []
        if not isinstance(legal_citations, list):
            legal_citations = []

        # 프롬프트 구성
        prompt = self._build_llm_enhancement_prompt(
            answer, question, question_type,
            retrieved_docs, legal_references, legal_citations
        )

        # LLM 호출
        try:
            response = self.llm.invoke(prompt)
            # content 속성이 있으면 사용 (ChatModel), 없으면 직접 문자열 변환 (BaseLLM)
            structured_answer = response.content if hasattr(response, 'content') else str(response)
        except Exception:
            # 예외 발생 시 재시도
            structured_answer = str(self.llm.invoke(prompt))

        # LLM 응답 후처리 - 원본 내용 보존 검증 및 출처 형식 통일
        structured_answer = self._post_process_llm_response(
            structured_answer, answer, question_type, retrieved_docs
        )

        # 품질 메트릭 계산
        quality_metrics = self._calculate_quality_metrics(structured_answer)

        return {
            "original_answer": answer,
            "structured_answer": structured_answer,
            "question_type": question_type.value,
            "template_used": "LLM 기반 구조화",
            "method": "llm_based",
            "analysis": {
                "has_title": bool(re.search(r'^#+\s+', structured_answer, re.MULTILINE)),
                "section_count": len(re.findall(r'^###\s+', structured_answer, re.MULTILINE))
            },
            "improvements": [],
            "quality_metrics": quality_metrics,
            "enhancement_timestamp": datetime.now().isoformat()
        }

    def _build_llm_enhancement_prompt(
        self,
        answer: str,
        question: str,
        question_type: QuestionType,
        retrieved_docs: List[Dict[str, Any]],
        legal_references: List[str],
        legal_citations: List[Dict[str, Any]]
    ) -> str:
        """LLM 구조화를 위한 프롬프트 구성 (개선된 버전)"""

        # None 체크
        retrieved_docs = retrieved_docs if retrieved_docs is not None else []
        legal_references = legal_references if legal_references is not None else []
        legal_citations = legal_citations if legal_citations is not None else []



        # 법적 문서 포맷팅
        legal_docs_text = self._format_docs_for_prompt(retrieved_docs)

        # 원본 답변의 핵심 키워드 추출 (내용 보존 확인용)
        answer_keywords = set()
        if answer:
            keywords = re.findall(r'[\w가-힣]{2,}', answer.lower())
            # 법률 관련 키워드 우선 추출
            legal_keywords = [kw for kw in keywords if any(term in kw for term in ['법', '조', '항', '판례', '법원', '판결', '소송', '계약', '권리', '의무'])]
            answer_keywords.update(legal_keywords[:15])
            answer_keywords.update(keywords[:30])  # 일반 키워드도 추가

        keywords_preview = ", ".join(list(answer_keywords)[:10]) if answer_keywords else "없음"

        # Few-Shot 예시 섹션 생성 (Phase 1.1: 선택적 포함 - term_explanation일 때만)
        few_shot_examples_section = ""
        if self.enable_few_shot and question_type == QuestionType.TERM_EXPLANATION:
            few_shot_examples = self._get_few_shot_examples(question_type, question)
            # Phase 1.1: 최대 1개만 포함 (프롬프트 길이 축소)
            if few_shot_examples:
                example = few_shot_examples[0]  # 첫 번째 예시만 사용
                few_shot_examples_section = "\n## 📚 개선 예시 (참고용)\n\n"
                few_shot_examples_section += f"""**질문**: {example.get('question', '')}

**원본 답변**: {example.get('original_answer', '')[:200]}...

**개선된 답변**: {example.get('enhanced_answer', '')[:200]}...

**주요 개선 사항**: {', '.join(example.get('improvements', [])[:3])}
"""

        prompt = f"""당신은 법률 답변 포맷팅 전문가입니다. 아래 변환 규칙을 정확히 적용하세요.

**중요**: 최종 답변에는 작업 과정(STEP, 평가, 체크리스트 등)을 포함하지 마세요. 오직 변환된 답변 내용만 작성하세요.

## 🔧 변환 규칙 (반드시 순서대로 적용)

### 규칙 1: 출처 형식 변환

다음 패턴을 찾아 변환하세요:

```
패턴 1: [출처: 문서 N] → [문서 N]
패턴 2: [출처: 문서명] → [문서 N] (문서명을 번호로 매핑)
패턴 3: **출처**: [문서명] → [문서 N]
패턴 4: (출처: 문서명) → [문서 N]
패턴 5: [1], [2], [3] → [문서 1], [문서 2], [문서 3]
패턴 6: 문서[1], 문서[2] → [문서 1], [문서 2]
```

**변환 예시**:
- "민법 제543조에 따르면..." [출처: 문서 1] → "민법 제543조에 따르면..." [문서 1]
- "대법원 판결에 의하면..." [출처: 문서 2] → "대법원 판결에 의하면..." [문서 2]
- "민법 제543조에 따르면..." [1] → "민법 제543조에 따르면..." [문서 1]

### 규칙 2: 표 형식 제거

"문서별 근거 비교" 표를 찾아 다음 규칙으로 변환:

```
표 행 형식:
| [문서 N] | 출처 | 핵심 근거 | 관련 내용 |

또는

| [N] | 출처 | 핵심 근거 | 관련 내용 |

변환 후 형식:
[문서 N]에 따르면 (출처) 핵심 근거. 관련 내용.
```

**변환 예시**:
```
변환 전:
| [1] | 민법 제543조 | 해지권 발생 | 계약 또는 법률 규정에 의한 해지권 |

변환 후:
[문서 1]에 따르면 (민법 제543조) 해지권이 발생합니다. 계약 또는 법률 규정에 의한 해지권입니다.
```

### 규칙 3: 인사말 제거

다음 패턴을 찾아 제거:
- "말씀하신...", "참고로...", "본 답변은..."
- "구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다"
- "일반적인 법률 정보 제공을 목적으로 하며..."

### 규칙 4: 구조 정리

- 연속된 빈 줄을 2개 이하로 제한
- 섹션 제목은 `##` 또는 `###` 형식 유지
- 불필요한 마커 제거

## 📄 원본 답변

{answer}

## 📋 문서 매핑 정보

{self._build_document_mapping(retrieved_docs) if retrieved_docs else "문서 정보 없음"}

## 🔄 변환 작업

위 규칙을 순서대로 적용하여 원본 답변을 변환하세요.

**작업 순서**:
1. 규칙 1 적용: 출처 형식 변환
2. 규칙 2 적용: 표 형식 제거
3. 규칙 3 적용: 인사말 제거
4. 규칙 4 적용: 구조 정리

## ✅ 변환 검증

변환 후 다음을 확인하세요:
- [ ] 모든 `[출처: 문서 N]`가 `[문서 N]`으로 변환되었는가?
- [ ] 모든 `[1]`, `[2]` 등이 `[문서 1]`, `[문서 2]`로 변환되었는가?
- [ ] 모든 표가 제거되고 텍스트로 변환되었는가?
- [ ] 인사말이 제거되었는가?
- [ ] 원본의 법적 정보가 보존되었는가?

## 🎯 핵심 원칙

1. **정보 보존 우선**: 모든 법적 정보(법조문 번호 및 내용, 판례 정보, 법적 해설, 실무 조언, 구체적 예시)를 정확히 보존하세요. 절대 요약하거나 간소화하지 마세요.

2. **최소 침습 원칙**: 원본 구조, 설명 방식, 예시를 최대한 존중하세요. 구조가 명확하면 그대로 유지하고, 섹션 제목을 함부로 추가하지 마세요.

3. **형식 개선만**: 인사말 제거, 불필요한 반복 통합, 어투 통일(전문적 어조)만 수행하세요.


원본의 핵심 키워드 확인: {keywords_preview}

{few_shot_examples_section}

## 📝 질문 정보

**질문**: {question}
**질문 유형**: {question_type.value}

## 📄 원본 답변

{answer}

"""

        # 법적 문서 정보 (있는 경우 - 보완용)
        if legal_docs_text and legal_docs_text.strip() != "검색된 문서가 없습니다.":
            prompt += f"""

## 🔍 참고: 검색된 법률 문서 (보완용)

{legal_docs_text}

**사용 규칙**:
- 원본 답변에 이미 포함된 내용이면 추가하지 마세요
- 원본에 빠진 중요한 법적 정보가 있을 때만 자연스럽게 통합하세요
- 문서 인용 시 `[문서 N]` 형식으로 표시하세요 (예: `[문서 1]`, `[문서 2]`)
"""

        if legal_references:
            refs_text = "\n".join([f"- {ref}" for ref in legal_references[:8]])
            prompt += f"""

## ⚖️ 참고 법령 (보완용)

{refs_text}

**사용 규칙**:
- 원본 답변에 이미 언급된 법령이면 중복하지 마세요
- 원본에 빠진 중요한 법령이 있을 때만 자연스럽게 추가하세요
- 예: "이에 대해서는 **민법 제111조**에서 규정하고 있습니다."
"""

        if legal_citations:
            citations_text = "\n".join([
                f"- {cite.get('text', cite.get('citation', str(cite)))}"
                for cite in legal_citations[:8]
            ])
            prompt += f"""

## 📚 참고 법적 인용 (보완용)

{citations_text}

**사용 규칙**:
- 원본에 이미 포함된 인용이면 추가하지 마세요
- 중요한 인용이 누락된 경우에만 자연스럽게 추가하세요
- 판례나 법령 인용 시 정확한 형식으로 표기하세요
"""

        prompt += """

## 📐 출력 형식

- 제목: `## 제목` (원본 구조 존중)
- 섹션: `### 섹션명` (표시 문구 금지)
- 강조: `**텍스트**`
- 리스트: `- 항목` 또는 `1. 항목`
- **표 형식 사용 금지**: 모든 내용을 텍스트로만 표현

## 📤 출력

위 변환 규칙을 순서대로 적용하여 변환된 답변을 작성하세요.
설명 없이 바로 변환된 답변을 시작하세요:

"""

        return prompt

    def _normalize_titles(self, text: str) -> str:
        """제목 중복 제거 및 정규화"""
        lines = text.split('\n')
        normalized_lines = []
        seen_titles = set()

        for i, line in enumerate(lines):
            # 제목 라인 감지
            title_match = re.match(r'^(#{1,6})\s+(.+)', line)

            if title_match:
                level = len(title_match.group(1))
                title_text = title_match.group(2).strip()

                # 이모지 제거한 순수 제목 텍스트 (개선: 더 많은 이모지 제거)
                clean_title = re.sub(r'[📖⚖️💼💡📚📋⭐📌🔍💬🎯📊📝📄⏰🔗⚠️❗✅🚨]+\s*', '', title_text).strip()

                # 동일한 레벨의 중복 제목 제거
                if level == 2:  # ## 제목
                    # 중복 제목 제거 로직 강화
                    title_key = clean_title.lower()

                    # "답변", "법률질문답변" 등 유사 제목 처리
                    is_similar_to_answer = any(pattern in clean_title for pattern in ["답변", "법률질문답변", "법률 질문 답변"])

                    # 중복이거나 유사한 제목 제거
                    if title_key in seen_titles or (is_similar_to_answer and "답변" in seen_titles):
                        continue  # 중복 제목 스킵

                    # "답변" 계열 제목은 하나만 허용
                    if is_similar_to_answer:
                        seen_titles.add("답변")

                    seen_titles.add(title_key)

                    # 이모지가 있으면 제거하고 순수 제목만 사용
                    if re.search(r'[📖⚖️💼💡📚📋⭐📌🔍💬🎯📊📝📄⏰🔗⚠️❗✅🚨]', title_text):
                        normalized_lines.append(f"## {clean_title}")
                    else:
                        normalized_lines.append(f"## {clean_title}")

                elif level == 3:  # ### 제목
                    # 개선: "💬 답변" 같은 하위 헤더가 "답변" 계열이면 제거
                    is_answer_subtitle = any(pattern in clean_title for pattern in ["답변", "법률질문답변", "법률 질문 답변"])
                    if is_answer_subtitle and ("답변" in seen_titles or any(st in seen_titles for st in ["답변", "법률질문답변", "법률 질문 답변"])):
                        continue  # "답변" 계열 하위 헤더는 제거

                    # 이모지 제거
                    if re.search(r'[📖⚖️💼💡📚📋⭐📌🔍💬🎯📊📝📄⏰🔗⚠️❗✅🚨]', title_text):
                        normalized_lines.append(f"### {clean_title}")
                    else:
                        normalized_lines.append(line)
                else:
                    normalized_lines.append(line)
            else:
                normalized_lines.append(line)

        return '\n'.join(normalized_lines)

    def _remove_empty_sections(self, text: str) -> str:
        """빈 섹션 제거"""
        lines = text.split('\n')
        result_lines = []
        current_section_lines = []
        current_section_title = ""
        in_section = False

        i = 0
        while i < len(lines):
            line = lines[i]

            # 섹션 시작 감지 (### 또는 ####)
            section_match = re.match(r'^(#{3,4})\s+(.+)', line)

            if section_match:
                # 이전 섹션 처리
                if in_section:
                    section_content = '\n'.join(current_section_lines)
                    # 빈 섹션인지 확인
                    if self._validate_section_content(section_content):
                        # 유효한 섹션이면 추가
                        result_lines.append(f"### {current_section_title}")
                        result_lines.extend(current_section_lines)

                # 새 섹션 시작
                in_section = True
                current_section_title = section_match.group(2).strip()
                current_section_lines = []
                i += 1
                continue

            if in_section:
                current_section_lines.append(line)
            else:
                result_lines.append(line)

            i += 1

        # 마지막 섹션 처리
        if in_section:
            section_content = '\n'.join(current_section_lines)
            if self._validate_section_content(section_content):
                result_lines.append(f"### {current_section_title}")
                result_lines.extend(current_section_lines)

        return '\n'.join(result_lines)

    def _validate_section_content(self, content: str) -> bool:
        """섹션 내용이 유효한지 확인"""
        if not content or not content.strip():
            return False

        # 빈 내용 패턴 확인
        empty_patterns = [
            r'^관련\s*법률을?\s*찾을\s*수\s*없습니다?\.?\s*$',
            r'^관련\s*법률\s*예시를?\s*찾을\s*수\s*없습니다?\.?\s*$',
            r'^찾을\s*수\s*없습니다?\.?\s*$',
            r'^알\s*수\s*없습니다?\.?\s*$',
            r'^없습니다?\.?\s*$',
            r'^정보를?\s*찾을\s*수\s*없습니다?\.?\s*$',
            r'^관련\s*법령을?\s*찾을\s*수\s*없습니다?\.?\s*$',
        ]

        content_clean = content.strip()
        for pattern in empty_patterns:
            if re.match(pattern, content_clean, re.IGNORECASE):
                return False

        # 너무 짧고 의미 없는 내용 (50자 미만이고 "없습니다"로 끝나는 경우)
        if len(content_clean) < 50 and re.search(r'없습니다?\.?\s*$', content_clean):
            return False

        # 최소 길이 체크 (의미 있는 내용은 최소 20자)
        if len(content_clean) < 20:
            return False

        return True

    def _remove_quality_metrics(self, text: str) -> str:
        """품질 지표 및 신뢰도 정보 제거"""
        lines = text.split('\n')
        result_lines = []
        skip_section = False

        i = 0
        while i < len(lines):
            line = lines[i]

            # 신뢰도 섹션 시작 감지
            if re.search(r'신뢰도|품질\s*점수|품질\s*지표|confidence|quality\s*score', line, re.IGNORECASE):
                # 해당 섹션 전체 제거
                skip_section = True
                i += 1
                continue

            # 신뢰도 패턴이 포함된 라인 제거
            if re.search(r'🟠.*신뢰도|신뢰도.*\d+%|🟢.*신뢰도|🔴.*신뢰도', line):
                i += 1
                continue

            # "신뢰도: XX%" 패턴 제거
            if re.search(r'신뢰도\s*:\s*\d+\.?\d*%', line):
                # 라인에서 신뢰도 부분만 제거
                line = re.sub(r'신뢰도\s*:\s*\d+\.?\d*%[^\n]*', '', line)
                line = re.sub(r'\(신뢰도[^\)]+\)', '', line)

            # 면책 조항 섹션은 유지하되 신뢰도 정보만 제거
            if '면책' in line and '조항' in line:
                skip_section = False

            if skip_section:
                # 섹션 끝까지 스킵 (다음 ### 또는 ## 만날 때까지)
                if re.match(r'^#{2,3}\s+', line):
                    skip_section = False
                    result_lines.append(line)
            else:
                # 신뢰도 숫자만 제거
                line = re.sub(r'\s*신뢰도\s*:\s*\d+\.?\d*%', '', line)
                line = re.sub(r'\(신뢰도[^\)]+\)', '', line)
                result_lines.append(line)

            i += 1

        return '\n'.join(result_lines)

    def _remove_decorative_emojis(self, text: str) -> str:
        """장식용 이모지 제거 (섹션명 및 본문에서) - 개선: 모든 이모지 제거"""
        lines = text.split('\n')
        result_lines = []

        # 모든 이모지 패턴 (더 많은 이모지 포함)
        emoji_pattern = r'[📖⚖️💼💡📚📋⭐📌🔍💬🎯📊📝📄⏰🔗⚠️❗✅🚨📑📌🎓🔬⚡🌟💫]+\s*'

        for line in lines:
            # 제목 라인에서 이모지 제거 (### 제목 형식)
            title_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if title_match:
                level = title_match.group(1)
                title_text = title_match.group(2)

                # 모든 이모지 제거 (개선: 챗봇 친화적)
                title_text = re.sub(emoji_pattern, '', title_text).strip()
                line = f"{level} {title_text}"
            else:
                # 본문에서도 모든 이모지 제거 (개선)
                line = re.sub(emoji_pattern, '', line)

            result_lines.append(line)

        return '\n'.join(result_lines)

    def _normalize_structure(self, text: str) -> str:
        """Markdown 구조 정규화"""
        lines = text.split('\n')
        result_lines = []
        last_level = 0

        for i, line in enumerate(lines):
            title_match = re.match(r'^(#{1,6})\s+(.+)', line)

            if title_match:
                level = len(title_match.group(1))

                # 계층 구조 검증 및 수정
                if level > 2 and last_level == 0:
                    # 첫 제목이 ###이면 ##로 변경
                    if level == 3:
                        line = f"## {title_match.group(2)}"
                        level = 2

                # ## 다음에 바로 #### 오는 경우 ###로 조정
                if last_level == 2 and level == 4:
                    line = f"### {title_match.group(2)}"
                    level = 3

                last_level = level

            result_lines.append(line)

        # 빈 줄 정리 (섹션 사이에 빈 줄 1개만)
        result = '\n'.join(result_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)

        return result

    def _post_process_llm_response(
        self,
        structured_answer: str,
        original_answer: str,
        question_type: QuestionType,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """LLM 응답 후처리 - 원본 내용 보존 검증 및 개선 (개선된 버전)"""

        if not structured_answer or not original_answer:
            return structured_answer if structured_answer else original_answer

        try:
            # 1. 문서별 근거 비교 표 제거 (CRITICAL)
            structured_answer = self._remove_comparison_table(structured_answer)
            
            # 2. 통합 정리 함수 사용
            structured_answer = self._clean_structured_answer(structured_answer, question_type)

            # 3. 원본의 중요 법률 정보 추출 및 검증
            original_lower = original_answer.lower()
            structured_lower = structured_answer.lower()

            # 법조문 패턴 (제X조, 제X항 등)
            legal_article_patterns = re.findall(r'제\d+조|제\d+항|제\d+호', original_answer)
            missing_articles = [
                article for article in legal_article_patterns
                if article not in structured_lower
            ]

            # 판례 패턴
            precedent_patterns = re.findall(
                r'(대법원|고등법원|지방법원|법원)\s+[\d가-힣]+',
                original_answer
            )
            missing_precedents = [
                prec for prec in precedent_patterns
                if prec not in structured_lower
            ]

            # 3. 누락된 중요 정보 복원 시도 (Phase 3.2: 후처리 로직 개선)
            if missing_articles:
                logger.warning(f"누락된 법조문: {missing_articles[:5]}")
                # 원본에서 해당 법조문 부분 복원 시도
                for article in missing_articles[:3]:  # 최대 3개만 복원
                    article_pattern = article.replace("제", r"\s*제").replace("조", r"\s*조")
                    article_match = re.search(article_pattern, original_answer, re.IGNORECASE)
                    if article_match:
                        # 법조문과 관련된 문단 찾기
                        start_pos = max(0, article_match.start() - 100)
                        end_pos = min(len(original_answer), article_match.end() + 500)
                        article_context = original_answer[start_pos:end_pos]
                        # 구조화된 답변에 해당 내용이 포함되어 있지 않으면 복원
                        if article.lower() not in structured_lower:
                            # 구조화된 답변의 적절한 위치에 삽입 (법적 근거 섹션이나 관련 섹션)
                            structured_answer = self._restore_missing_content(
                                structured_answer, article_context, article
                            )
                            logger.info(f"법조문 복원 시도: {article}")

            if missing_precedents:
                logger.warning(f"누락된 판례: {missing_precedents[:3]}")
                # 원본에서 해당 판례 부분 복원 시도
                for precedent in missing_precedents[:2]:  # 최대 2개만 복원
                    precedent_match = re.search(re.escape(precedent), original_answer, re.IGNORECASE)
                    if precedent_match:
                        start_pos = max(0, precedent_match.start() - 100)
                        end_pos = min(len(original_answer), precedent_match.end() + 300)
                        precedent_context = original_answer[start_pos:end_pos]
                        if precedent.lower() not in structured_lower:
                            structured_answer = self._restore_missing_content(
                                structured_answer, precedent_context, precedent
                            )
                            logger.info(f"판례 복원 시도: {precedent}")

            # 5. 출처 정보를 [문서 N] 형식으로 통일 (CRITICAL)
            structured_answer = self._normalize_source_citations(
                structured_answer, retrieved_docs
            )
            
            # 원본 답변의 출처 정보도 추출하여 누락 확인
            source_citation_patterns = [
                r'\*\*출처\*\*:\s*\[[^\]]+\]',  # **출처**: [문서명]
                r'\[출처:\s*[^\]]+\]',  # [출처: 문서명]
                r'\(출처:\s*[^\)]+\)',  # (출처: 문서명)
                r'출처:\s*\[[^\]]+\]',  # 출처: [문서명]
            ]
            
            original_sources = []
            for pattern in source_citation_patterns:
                matches = re.findall(pattern, original_answer, re.IGNORECASE)
                original_sources.extend(matches)
            
            # 문서 번호 패턴 추출 ([1], [2], [문서 1] 등)
            document_number_patterns = [
                r'\[문서\s*\d+\]',  # [문서 1]
                r'\[문서\s*번호\s*\d+\]',  # [문서 번호 1]
                r'\[[1-9]\d*\]',  # [1], [2], [3] 등
            ]
            
            original_doc_numbers = []
            for pattern in document_number_patterns:
                matches = re.findall(pattern, original_answer, re.IGNORECASE)
                original_doc_numbers.extend(matches)
            
            # 구조화된 답변에서 문서 번호 확인
            structured_doc_numbers = []
            for pattern in document_number_patterns:
                matches = re.findall(pattern, structured_answer, re.IGNORECASE)
                structured_doc_numbers.extend(matches)
            
            # 누락된 문서 번호 복원
            missing_doc_numbers = [d for d in original_doc_numbers if d.lower() not in [dd.lower() for dd in structured_doc_numbers]]
            
            if missing_doc_numbers:
                logger.warning(f"누락된 문서 번호 감지: {len(missing_doc_numbers)}개")
                
                # 문서 번호 복원
                for doc_num in missing_doc_numbers[:5]:  # 최대 5개만 복원
                    doc_num_escaped = re.escape(doc_num)
                    doc_match = re.search(doc_num_escaped, original_answer, re.IGNORECASE)
                    if doc_match:
                        start_pos = max(0, doc_match.start() - 100)
                        end_pos = min(len(original_answer), doc_match.end() + 200)
                        doc_context = original_answer[start_pos:end_pos]
                        
                        if doc_num.lower() not in structured_lower:
                            structured_answer = self._restore_missing_source(
                                structured_answer, doc_context, doc_num
                            )
                            logger.info(f"문서 번호 복원: {doc_num}")

            # 6. 핵심 키워드 보존률 확인
            original_keywords = set(re.findall(r'[\w가-힣]{3,}', original_lower))
            # 법률 관련 키워드 필터링
            important_keywords = {
                kw for kw in original_keywords
                if any(term in kw for term in ['법', '조', '항', '판례', '법원', '판결', '권리', '의무', '계약', '소송'])
            }

            preserved_keywords = {
                kw for kw in important_keywords
                if kw in structured_lower
            }

            preservation_rate = len(preserved_keywords) / len(important_keywords) if important_keywords else 1.0

            # Phase 3.2: 핵심 키워드 보존률이 70% 미만일 때 원본 부분 복원 시도
            if preservation_rate < 0.7 and important_keywords:
                logger.warning(f"핵심 키워드 보존률이 낮습니다 ({preservation_rate:.2%})")
                # 누락된 키워드가 포함된 원본 문단 찾아서 복원
                missing_keywords = important_keywords - preserved_keywords
                for keyword in list(missing_keywords)[:5]:  # 최대 5개 키워드만 복원
                    # 원본에서 해당 키워드가 포함된 문단 찾기
                    keyword_pattern = re.escape(keyword)
                    keyword_matches = list(re.finditer(keyword_pattern, original_answer, re.IGNORECASE))
                    if keyword_matches:
                        for match in keyword_matches[:2]:  # 각 키워드당 최대 2개 문단
                            start_pos = max(0, match.start() - 200)
                            end_pos = min(len(original_answer), match.end() + 300)
                            keyword_context = original_answer[start_pos:end_pos]
                            # 구조화된 답변에 해당 키워드가 없으면 복원 시도
                            if keyword.lower() not in structured_lower:
                                structured_answer = self._restore_missing_content(
                                    structured_answer, keyword_context, keyword
                                )
                                logger.info(f"키워드 복원 시도: {keyword}")
                                break  # 한 번만 복원

            return structured_answer.strip()

        except Exception as e:
            logger.error(f"Error in post-processing LLM response: {e}", exc_info=True)
            return structured_answer

    def _clean_structured_answer(self, structured_answer: str, question_type: QuestionType) -> str:
        """구조화된 답변 최종 정리 (통합 후처리 함수)"""
        try:
            # 1. 제목이 없으면 추가
            if not re.search(r'^##\s+', structured_answer, re.MULTILINE):
                template = self.structure_templates.get(
                    question_type,
                    self.structure_templates[QuestionType.GENERAL_QUESTION]
                )
                title = template.get('title', '법률 질문 답변')
                structured_answer = f"## {title}\n\n{structured_answer}"

            # 2. 불필요한 메타 텍스트 제거
            meta_patterns = [
                r'^위의?\s+.*지침에?\s+따라.*?\n',
                r'^다음과?\s+같이.*?\n',
                r'^구조화된?\s+답변은?\s+다음과?\s+같습니다?.*?\n',
            ]
            for pattern in meta_patterns:
                structured_answer = re.sub(pattern, '', structured_answer, flags=re.MULTILINE | re.IGNORECASE)

            # 3. 대괄호 패턴 제거 (표시 문구 등)
            structured_answer = self._remove_bracket_patterns(structured_answer)

            # 4. 친근한 어투 정리
            structured_answer = self._normalize_tone(structured_answer)

            # 5. 품질 지표 제거
            structured_answer = self._remove_quality_metrics(structured_answer)

            # 6. 빈 섹션 제거
            structured_answer = self._remove_empty_sections(structured_answer)

            # 7. 제목 중복 제거 및 정규화
            structured_answer = self._normalize_titles(structured_answer)

            # 8. 이모지 제거
            structured_answer = self._remove_decorative_emojis(structured_answer)

            # 9. 구조 정규화
            structured_answer = self._normalize_structure(structured_answer)

            # 10. 중복 출처 제거
            structured_answer = self._remove_duplicate_sources(structured_answer)

            # 11. 중복 내용 제거
            structured_answer = self._remove_duplicate_content(structured_answer)

            # 12. 빈 줄 정리 (3개 이상 연속 빈 줄은 2개로)
            structured_answer = re.sub(r'\n{3,}', '\n\n', structured_answer)

            # 13. 중복 헤더 추가 제거 (개선)
            structured_answer = self._remove_duplicate_headers(structured_answer)

            # 14. 빈 섹션 추가 정리 (개선)
            structured_answer = self._remove_empty_sections_enhanced(structured_answer)

            # 15. 챗봇 친화적 구조로 변환 (개선)
            structured_answer = self._make_chatbot_friendly(structured_answer)

            return structured_answer.strip()

        except Exception as e:
            logger.error(f"Error in cleaning structured answer: {e}", exc_info=True)
            return structured_answer

    def _remove_duplicate_headers(self, text: str) -> str:
        """중복 헤더 제거 (개선)"""
        try:
            # 이미 _normalize_titles에서 처리되지만, 추가로 강화
            lines = text.split('\n')
            result_lines = []
            seen_headers = set()

            for line in lines:
                header_match = re.match(r'^(#{2,3})\s+(.+)', line)
                if header_match:
                    level = len(header_match.group(1))
                    header_text = header_match.group(2).strip()
                    # 이모지 제거
                    clean_header = re.sub(r'[📖⚖️💼💡📚📋⭐📌🔍💬🎯📊📝📄⏰🔗⚠️❗✅🚨]+\s*', '', header_text).strip()
                    header_key = f"{level}:{clean_header.lower()}"

                    if header_key in seen_headers:
                        continue  # 중복 헤더 제거
                    seen_headers.add(header_key)

                result_lines.append(line)

            return '\n'.join(result_lines)
        except Exception as e:
            logger.error(f"Error removing duplicate headers: {e}")
            return text

    def _remove_empty_sections_enhanced(self, text: str) -> str:
        """빈 섹션 제거 (강화된 버전)"""
        try:
            # _remove_empty_sections와 유사하지만 더 강화된 로직
            return self._remove_empty_sections(text)
        except Exception as e:
            logger.error(f"Error removing empty sections enhanced: {e}")
            return text

    def _make_chatbot_friendly(self, text: str) -> str:
        """챗봇 친화적 구조로 변환"""
        try:
            # 자연스러운 흐름: 질문 → 핵심 답변 → 상세 설명 → 참고사항
            # 불필요한 마크다운 헤더 최소화
            lines = text.split('\n')
            result_lines = []

            # 연속된 헤더 줄바꿈 최소화 (헤더 바로 다음에 내용이 오도록)
            for i, line in enumerate(lines):
                result_lines.append(line)
                # 헤더 다음에 빈 줄이 두 개 이상이면 하나로 줄임
                if re.match(r'^#{2,3}\s+', line) and i + 1 < len(lines) and lines[i + 1].strip() == '':
                    if i + 2 < len(lines) and lines[i + 2].strip() == '':
                        # 빈 줄 두 개 이상이면 하나만 유지
                        continue

            text = '\n'.join(result_lines)

            # 불필요한 마크다운 헤더 레벨 감소 (### -> 일반 텍스트로 변환할 수도 있지만, 현재는 유지)
            # 단, 너무 많은 헤더는 줄임

            return text
        except Exception as e:
            logger.error(f"Error making chatbot friendly: {e}")
            return text

    def _remove_bracket_patterns(self, text: str) -> str:
        """대괄호 패턴 제거 (예: [질문 내용 분석:], [관련 법령:])"""
        try:
            if not text:
                return text

            lines = text.split('\n')
            result_lines = []
            prev_line_was_section_title = False

            for line in lines:
                # 섹션 제목 확인
                is_section_title = bool(re.match(r'^###\s+', line))

                # 섹션 제목 바로 다음 줄에 대괄호 패턴이 있는 경우 제거
                if prev_line_was_section_title:
                    # 대괄호 패턴 제거 (예: [질문 내용 분석:], [관련 법령:])
                    bracket_pattern = re.match(r'^\s*\[[^\]]*:\]\s*$', line)
                    if bracket_pattern:
                        # 이 줄을 건너뜀 (줄바꿈은 유지하기 위해 빈 줄 추가하지 않음)
                        prev_line_was_section_title = False
                        continue

                # 일반 줄에서 대괄호 패턴 확인
                bracket_match = re.match(r'^\s*\[[^\]]*:\]\s*$', line)
                if bracket_match:
                    # 대괄호 패턴만 있는 줄은 제거
                    continue

                # 대괄호 패턴이 아닌 모든 줄은 추가
                result_lines.append(line)

                prev_line_was_section_title = is_section_title

            return '\n'.join(result_lines)
        except Exception as e:
            logger.error(f"Error removing bracket patterns: {e}", exc_info=True)
            return text

    def _restore_missing_content(self, structured_answer: str, content: str, identifier: str) -> str:
        """
        누락된 내용을 구조화된 답변의 적절한 위치에 복원 (Phase 3.2)

        Args:
            structured_answer: 구조화된 답변
            content: 복원할 내용 (문단 또는 섹션)
            identifier: 내용을 식별하는 키워드 (법조문, 판례, 키워드)

        Returns:
            내용이 복원된 구조화된 답변
        """
        try:
            if not content or not content.strip():
                return structured_answer

            # 법적 근거 관련 섹션 찾기
            legal_section_keywords = ['법적 근거', '관련 법령', '법령', '법조문', '판례', '법적 해설']
            lines = structured_answer.split('\n')
            insertion_point = -1

            # 섹션 제목 라인 찾기
            for i, line in enumerate(lines):
                if re.match(r'^###\s+', line):
                    section_title = re.sub(r'^###\s+', '', line).strip().lower()
                    if any(keyword in section_title for keyword in legal_section_keywords):
                        # 해당 섹션 끝까지 찾기 (다음 ### 또는 끝)
                        insertion_point = i
                        break

            if insertion_point == -1:
                # 법적 근거 섹션이 없으면 끝에 추가
                if structured_answer.strip():
                    structured_answer += f"\n\n### 법적 근거\n\n{content.strip()}"
                else:
                    structured_answer = f"### 법적 근거\n\n{content.strip()}"
            else:
                # 해당 섹션에 내용 추가 (섹션 끝에 추가)
                next_section = -1
                for i in range(insertion_point + 1, len(lines)):
                    if re.match(r'^###\s+', lines[i]):
                        next_section = i
                        break

                if next_section == -1:
                    # 다음 섹션이 없으면 끝에 추가
                    lines.append(f"\n{content.strip()}")
                else:
                    # 다음 섹션 전에 삽입
                    lines.insert(next_section, f"{content.strip()}\n")

                structured_answer = '\n'.join(lines)

            return structured_answer

        except Exception as e:
            logger.warning(f"내용 복원 실패 ({identifier}): {e}")
            return structured_answer

    def _restore_missing_source(self, structured_answer: str, content: str, source_info: str) -> str:
        """
        누락된 출처 정보를 구조화된 답변의 적절한 위치에 복원

        Args:
            structured_answer: 구조화된 답변
            content: 출처 정보가 포함된 원본 문맥
            source_info: 복원할 출처 정보 (예: "**출처**: [문서명]", "[1]")

        Returns:
            출처 정보가 복원된 구조화된 답변
        """
        try:
            if not content or not content.strip() or not source_info:
                return structured_answer

            # 출처 정보가 포함된 문장 추출
            source_sentences = []
            sentences = re.split(r'[.!?]\s+', content)
            
            for sentence in sentences:
                if source_info.lower() in sentence.lower():
                    # 출처 정보 앞뒤 문맥 포함 (최대 200자)
                    source_sentences.append(sentence.strip())
                    if len(source_sentences) >= 2:  # 최대 2개 문장만
                        break

            if not source_sentences:
                return structured_answer

            # 구조화된 답변에서 출처 정보가 있어야 할 위치 찾기
            # 1. 법조문이나 판례가 언급된 위치 찾기
            lines = structured_answer.split('\n')
            insertion_point = -1
            
            # 출처 정보와 관련된 키워드가 있는 문장 찾기
            source_keywords = ['법', '조', '항', '판례', '법원', '판결', '법령']
            for i, line in enumerate(lines):
                line_lower = line.lower()
                # 출처 정보가 없고, 관련 키워드가 있는 라인 찾기
                if source_info.lower() not in line_lower:
                    if any(keyword in line_lower for keyword in source_keywords):
                        # 해당 라인에 출처 정보 추가
                        if line.strip() and not line.strip().startswith('#'):
                            # 출처 정보를 문장 끝에 추가
                            if not re.search(r'출처|\[.*\]', line, re.IGNORECASE):
                                insertion_point = i
                                break

            if insertion_point >= 0:
                # 해당 라인에 출처 정보 추가
                original_line = lines[insertion_point]
                # 출처 정보가 이미 있는지 확인
                if source_info.lower() not in original_line.lower():
                    # 문장 끝에 출처 정보 추가
                    if original_line.strip().endswith(('.', '!', '?')):
                        lines[insertion_point] = f"{original_line.rstrip('.!?')} {source_info}."
                    else:
                        lines[insertion_point] = f"{original_line} {source_info}"
                    structured_answer = '\n'.join(lines)
                    logger.info(f"출처 정보 복원 성공: {source_info[:30]}...")
            else:
                # 적절한 위치를 찾지 못한 경우, 관련 섹션 끝에 추가
                legal_section_keywords = ['법적 근거', '관련 법령', '법령', '법조문', '판례', '법적 해설', '결론']
                for i, line in enumerate(lines):
                    if re.match(r'^###\s+', line):
                        section_title = re.sub(r'^###\s+', '', line).strip().lower()
                        if any(keyword in section_title for keyword in legal_section_keywords):
                            # 해당 섹션 끝에 출처 정보 추가
                            next_section = -1
                            for j in range(i + 1, len(lines)):
                                if re.match(r'^###\s+', lines[j]):
                                    next_section = j
                                    break
                            
                            if next_section == -1:
                                # 다음 섹션이 없으면 끝에 추가
                                lines.append(f"\n{source_sentences[0]} {source_info}")
                            else:
                                # 다음 섹션 전에 삽입
                                lines.insert(next_section, f"{source_sentences[0]} {source_info}\n")
                            
                            structured_answer = '\n'.join(lines)
                            logger.info(f"출처 정보 복원 성공 (섹션 끝): {source_info[:30]}...")
                            break

            return structured_answer

        except Exception as e:
            logger.warning(f"출처 정보 복원 실패 ({source_info[:30]}...): {e}")
            return structured_answer

    def _remove_comparison_table(self, answer: str) -> str:
        """
        문서별 근거 비교 표를 제거하고 내용을 자연스러운 문장으로 변환
        
        Args:
            answer: 답변 텍스트
        
        Returns:
            표가 제거되고 텍스트로 변환된 답변
        """
        if not answer:
            return answer
        
        try:
            result = answer
            lines = result.split('\n')
            new_lines = []
            in_table = False
            table_start_idx = -1
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # 표 제목 감지 ("문서별 근거 비교", "문서별 근거", "근거 비교" 등)
                if re.search(r'문서별\s*근거\s*비교|근거\s*비교|문서\s*근거', line, re.IGNORECASE):
                    table_start_idx = len(new_lines)
                    in_table = True
                    # 표 제목 라인은 제거
                    i += 1
                    continue
                
                # 표 시작 감지 (헤더 라인: | 문서 번호 | ...)
                if in_table and re.match(r'^\s*\|.*문서.*번호.*\|', line, re.IGNORECASE):
                    # 헤더 라인 제거
                    i += 1
                    # 구분선 제거 (|-----------|...)
                    if i < len(lines) and re.match(r'^\s*\|[\s\-:]+\|', lines[i]):
                        i += 1
                    continue
                
                # 표 내용 라인 감지 (| [1] | ... |)
                if in_table and re.match(r'^\s*\|.*\[.*\].*\|', line):
                    # 표 내용을 텍스트로 변환
                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if len(cells) >= 3:
                        doc_num = cells[0]  # 문서 번호
                        source = cells[1] if len(cells) > 1 else ""  # 출처
                        key_basis = cells[2] if len(cells) > 2 else ""  # 핵심 근거
                        related_content = cells[3] if len(cells) > 3 else ""  # 관련 내용
                        
                        # 텍스트로 변환
                        text_parts = []
                        if doc_num:
                            # [1] 형식을 [문서 1] 형식으로 변환
                            doc_num_normalized = re.sub(r'\[(\d+)\]', r'[문서 \1]', doc_num)
                            text_parts.append(f"{doc_num_normalized}에 따르면")
                        if source:
                            text_parts.append(f"({source})")
                        if key_basis:
                            text_parts.append(key_basis)
                        if related_content:
                            text_parts.append(related_content)
                        
                        if text_parts:
                            new_lines.append(" ".join(text_parts) + ".")
                    
                    i += 1
                    continue
                
                # 표 종료 감지 (빈 줄 또는 다른 섹션 시작)
                if in_table:
                    # 빈 줄이거나 다른 제목이면 표 종료
                    if not line.strip() or re.match(r'^#+\s+', line):
                        in_table = False
                        # 표 종료 후 빈 줄 하나 추가
                        if new_lines and new_lines[-1].strip():
                            new_lines.append("")
                    else:
                        # 표가 아닌 일반 텍스트인 경우 표 종료
                        if not re.match(r'^\s*\|', line):
                            in_table = False
                
                # 표가 아닌 경우 일반 라인 추가
                if not in_table:
                    new_lines.append(line)
                
                i += 1
            
            result = '\n'.join(new_lines)
            
            # 연속된 빈 줄 정리
            result = re.sub(r'\n{3,}', '\n\n', result)
            
            if table_start_idx >= 0:
                logger.info("문서별 근거 비교 표 제거 및 텍스트 변환 완료")
            
            return result
            
        except Exception as e:
            logger.warning(f"표 제거 실패: {e}")
            return answer

    def _normalize_source_citations(
        self,
        answer: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        출처 정보를 [문서 N] 형식으로 통일
        
        Args:
            answer: 답변 텍스트
            retrieved_docs: 검색된 문서 목록 (문서명 매핑용)
        
        Returns:
            출처 정보가 [문서 N] 형식으로 통일된 답변
        """
        if not answer:
            return answer
        
        try:
            result = answer
            
            # 문서명 매핑 생성 (retrieved_docs가 있는 경우)
            doc_name_to_index = {}
            if retrieved_docs:
                for idx, doc in enumerate(retrieved_docs, 1):
                    if not isinstance(doc, dict):
                        continue
                    
                    # 문서명 추출 (다양한 필드에서 시도)
                    doc_name = (
                        doc.get("name") or
                        doc.get("title") or
                        doc.get("source") or
                        doc.get("type") or
                        doc.get("metadata", {}).get("name") or
                        doc.get("metadata", {}).get("title") or
                        doc.get("metadata", {}).get("source_type") or
                        ""
                    )
                    
                    if doc_name:
                        # 소문자로 정규화하여 매칭
                        doc_name_to_index[doc_name.lower()] = idx
                        # 타입도 매핑 (statute_article, case_paragraph 등)
                        doc_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "")
                        if doc_type:
                            doc_name_to_index[doc_type.lower()] = idx
            
            # 1. **출처**: [문서명] 형식을 [문서 N] 형식으로 변환
            def replace_source_citation(match):
                source_text = match.group(0)
                source_name_match = re.search(r'\[([^\]]+)\]', source_text)
                if source_name_match:
                    source_name = source_name_match.group(1).strip()
                    # 문서명 매핑에서 찾기
                    doc_index = doc_name_to_index.get(source_name.lower())
                    if doc_index:
                        return f"[문서 {doc_index}]"
                    # 매핑이 없으면 원본 유지 (나중에 처리)
                return source_text
            
            # **출처**: [문서명] 패턴 변환
            result = re.sub(
                r'\*\*출처\*\*:\s*\[[^\]]+\]',
                replace_source_citation,
                result,
                flags=re.IGNORECASE
            )
            
            # [출처: 문서명] 패턴 변환
            result = re.sub(
                r'\[출처:\s*([^\]]+)\]',
                lambda m: f"[문서 {doc_name_to_index.get(m.group(1).strip().lower(), '?')}]" if doc_name_to_index.get(m.group(1).strip().lower()) else m.group(0),
                result,
                flags=re.IGNORECASE
            )
            
            # (출처: 문서명) 패턴 변환
            result = re.sub(
                r'\(출처:\s*([^\)]+)\)',
                lambda m: f"[문서 {doc_name_to_index.get(m.group(1).strip().lower(), '?')}]" if doc_name_to_index.get(m.group(1).strip().lower()) else m.group(0),
                result,
                flags=re.IGNORECASE
            )
            
            # 출처: [문서명] 패턴 변환
            result = re.sub(
                r'출처:\s*\[([^\]]+)\]',
                lambda m: f"[문서 {doc_name_to_index.get(m.group(1).strip().lower(), '?')}]" if doc_name_to_index.get(m.group(1).strip().lower()) else m.group(0),
                result,
                flags=re.IGNORECASE
            )
            
            # 2. [1], [2] 형식을 [문서 1], [문서 2] 형식으로 변환
            result = re.sub(
                r'\[([1-9]\d*)\]',
                r'[문서 \1]',
                result
            )
            
            # 3. 문서[1], 문서[2] 형식을 [문서 1], [문서 2] 형식으로 변환
            result = re.sub(
                r'문서\s*\[\s*([1-9]\d*)\s*\]',
                r'[문서 \1]',
                result
            )
            
            logger.info(f"출처 정보 정규화 완료: {len(doc_name_to_index)}개 문서 매핑")
            return result
            
        except Exception as e:
            logger.warning(f"출처 정보 정규화 실패: {e}")
            return answer

    def _normalize_tone(self, text: str) -> str:
        """친근한 어투를 전문적인 어투로 변환"""
        try:
            if not text:
                return text

            # 친근한 어투 패턴을 전문적 어투로 변환
            replacements = [
                # 어미 변환 (줄바꿈 보존)
                (r'해요\.', '합니다.'),
                (r'이에요\.', '입니다.'),
                (r'예요\.', '입니다.'),
                (r'아요\.', '습니다.'),
                (r'어요\.', '습니다.'),
                (r'해요\s+', '합니다 '),  # 공백만 매칭 (줄바꿈 제외)
                (r'이에요\s+', '입니다 '),
                (r'예요\s+', '입니다 '),
                (r'아요\s+', '습니다 '),
                (r'어요\s+', '습니다 '),

                # 불필요한 어미 변형
                (r'좋아요\.', '좋습니다.'),
                (r'좋아요\s+', '좋습니다 '),
            ]

            result = text
            for pattern, replacement in replacements:
                result = re.sub(pattern, replacement, result)

            # 불필요한 대화형 문구 제거 (줄바꿈 보존)
            # 줄바꿈을 포함하지 않는 패턴 사용
            lines = result.split('\n')
            result_lines = []

            for line in lines:
                # 줄 단위로 처리하여 줄바꿈 보존
                line_processed = line

                # 줄 끝에 있는 불필요한 문구만 제거 (줄바꿈 보존)
                line_processed = re.sub(r'궁금하시군요\.?\s*$', '', line_processed)
                line_processed = re.sub(r'말씀하신\s+', '질문하신 ', line_processed)
                line_processed = re.sub(r'여기서\s+', '여기서 ', line_processed)

                result_lines.append(line_processed)

            result = '\n'.join(result_lines)

            # 문장 시작 부분의 불필요한 대화형 문구 제거 (줄바꿈 보존)
            # 줄바꿈을 유지하면서 앞 문구만 제거
            result = re.sub(r'(^민법\s+제\d+조의\s+내용에\s+대해\s+)궁금하시군요\.?\s*', r'\1', result, flags=re.MULTILINE)

            return result
        except Exception as e:
            logger.error(f"Error normalizing tone: {e}", exc_info=True)
            return text

    def _remove_duplicate_sources(self, text: str) -> str:
        """중복된 출처 표시 제거 및 출처 통합"""
        try:
            if not text:
                return text

            # 출처 패턴 추출: **출처**: [내용]
            source_pattern = r'\*\*출처\*\*:\s*([^\n]+)'
            sources = re.findall(source_pattern, text)

            # 출처가 2개 미만이면 그대로 반환
            if len(sources) < 2:
                return text

            # 동일 출처 확인
            unique_sources = {}

            for match in re.finditer(source_pattern, text):
                source_text = match.group(1).strip()
                source_key = source_text.lower()

                if source_key not in unique_sources:
                    unique_sources[source_key] = {
                        'text': source_text,
                        'positions': []
                    }
                unique_sources[source_key]['positions'].append((match.start(), match.end()))

            # 동일 출처가 2회 이상 나타나는 경우
            result = text
            positions_to_remove = []

            for source_key, source_info in unique_sources.items():
                positions = source_info['positions']
                if len(positions) > 1:
                    # 첫 번째는 유지, 나머지는 제거 대상
                    for start, end in positions[1:]:
                        positions_to_remove.append((start, end))

            # 역순으로 제거 (인덱스 변경 방지)
            for start, end in sorted(positions_to_remove, reverse=True):
                # 출처 줄 전체를 제거
                line_start = result.rfind('\n', 0, start) + 1
                line_end = result.find('\n', end)
                if line_end == -1:
                    line_end = len(result)

                # 빈 줄도 함께 제거
                prev_newline = result.rfind('\n', 0, line_start - 1) + 1 if line_start > 0 else 0
                next_newline = result.find('\n', line_end)

                # 앞뒤 빈 줄 확인
                if line_start > 0 and result[prev_newline:line_start].strip() == '':
                    line_start = prev_newline
                if next_newline != -1 and result[line_end:next_newline].strip() == '':
                    line_end = next_newline

                result = result[:line_start] + result[line_end:]

            return result
        except Exception as e:
            logger.error(f"Error removing duplicate sources: {e}", exc_info=True)
            return text

    def _remove_duplicate_content(self, text: str, similarity_threshold: float = 0.8) -> str:
        """중복 내용 제거 (문단 단위 유사도 검사)"""
        try:
            if not text:
                return text

            # 문단 단위로 분리 (빈 줄로 구분)
            paragraphs = re.split(r'\n\s*\n', text)

            if len(paragraphs) < 2:
                return text

            # 유사도 계산 및 중복 제거
            unique_paragraphs = []
            seen_paragraphs = []

            for para in paragraphs:
                if not para or not para.strip():
                    unique_paragraphs.append(para)
                    continue

                # 제목이나 섹션 마커는 제외
                if re.match(r'^#+\s+', para.strip()):
                    unique_paragraphs.append(para)
                    continue

                # 섹션 제목을 제외한 순수 내용만 추출
                para_lines = para.split('\n')
                para_content_lines = [line for line in para_lines if not re.match(r'^#+\s+', line.strip())]
                para_content = '\n'.join(para_content_lines).strip()

                if not para_content:
                    unique_paragraphs.append(para)
                    continue

                # 기존 문단과 유사도 비교 (섹션 제목 제외한 순수 내용만)
                is_duplicate = False
                para_clean = para_content.lower()

                for seen_para in seen_paragraphs:
                    similarity = SequenceMatcher(None, para_clean, seen_para).ratio()
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_paragraphs.append(para)
                    seen_paragraphs.append(para_clean)

            return '\n\n'.join(unique_paragraphs)
        except Exception as e:
            logger.error(f"Error removing duplicate content: {e}", exc_info=True)
            return text

    def _format_docs_for_prompt(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """법률 문서를 프롬프트용으로 포맷팅"""
        # None 체크
        if retrieved_docs is None:
            return "검색된 문서가 없습니다."

        if not retrieved_docs:
            return "검색된 문서가 없습니다."

        formatted_docs = []
        for i, doc in enumerate(retrieved_docs[:5], 1):  # 최대 5개
            if not isinstance(doc, dict):
                continue

            doc_type = doc.get("type", "문서")
            source = doc.get("source", doc.get("title", "알 수 없음"))
            content = doc.get("content", doc.get("text", ""))
            score = doc.get("relevance_score", doc.get("score", 0.0))

            # 내용 요약 (최대 300자) - None 체크
            if content is None:
                content = ""
            content_preview = content[:300] + "..." if len(content) > 300 else content

            formatted_docs.append(
                f"{i}. **{doc_type}**: {source}\n"
                f"   - 관련도: {score:.2f}\n"
                f"   - 내용: {content_preview}"
            )

        return "\n\n".join(formatted_docs)

    def _build_document_mapping(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """문서명을 문서 번호로 매핑하는 정보 생성"""
        if not retrieved_docs:
            return "문서 정보 없음"
        
        mapping_lines = []
        for idx, doc in enumerate(retrieved_docs, 1):
            if not isinstance(doc, dict):
                continue
            
            # 문서명 추출 (다양한 필드에서 시도)
            doc_name = (
                doc.get("name") or
                doc.get("title") or
                doc.get("source") or
                doc.get("type") or
                doc.get("metadata", {}).get("name") or
                doc.get("metadata", {}).get("title") or
                doc.get("metadata", {}).get("source_type") or
                ""
            )
            
            # 법조문 정보
            law_name = doc.get("law_name", "")
            article_no = doc.get("article_no", "")
            
            # 판례 정보
            case_name = doc.get("case_name", "")
            case_number = doc.get("case_number", "")
            
            # 매핑 정보 구성
            if law_name and article_no:
                mapping_info = f"[문서 {idx}]: {law_name} 제{article_no}조"
            elif case_name or case_number:
                mapping_info = f"[문서 {idx}]: {case_name or case_number}"
            elif doc_name:
                mapping_info = f"[문서 {idx}]: {doc_name}"
            else:
                mapping_info = f"[문서 {idx}]: 문서 {idx}"
            
            mapping_lines.append(mapping_info)
        
        if not mapping_lines:
            return "문서 정보 없음"
        
        return "\n".join(mapping_lines)

    def _enhance_with_template(
        self,
        answer: str,
        mapped_question_type: QuestionType,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        legal_references: List[str],
        legal_citations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """템플릿 기반 구조화 (폴백)"""

        # None 체크 및 타입 안전성 보장
        retrieved_docs = retrieved_docs if retrieved_docs is not None else []
        legal_references = legal_references if legal_references is not None else []
        legal_citations = legal_citations if legal_citations is not None else []

        if not isinstance(retrieved_docs, list):
            retrieved_docs = []
        if not isinstance(legal_references, list):
            legal_references = []
        if not isinstance(legal_citations, list):
            legal_citations = []

        # 구조 템플릿 가져오기
        template = self.structure_templates.get(mapped_question_type,
                                              self.structure_templates[QuestionType.GENERAL_QUESTION])

        if not template:
            return {"error": "Template not found"}

        # 현재 답변 분석
        analysis = self._analyze_current_structure(answer, template)

        # 구조화 개선 제안 (법적 근거 정보 포함)
        improvements = self._generate_structure_improvements(
            analysis, template, retrieved_docs, legal_references, legal_citations
        )

        # 구조화된 답변 생성 (법적 근거 정보 포함)
        structured_answer = self._create_structured_answer(
            answer, template, improvements, retrieved_docs, legal_references, legal_citations
        )

        # 품질 메트릭 계산
        quality_metrics = self._calculate_quality_metrics(structured_answer)

        return {
            "original_answer": answer,
            "structured_answer": structured_answer,
            "question_type": mapped_question_type.value,
            "template_used": template.get("title", "Unknown"),
            "method": "template_based",
            "analysis": analysis,
            "improvements": improvements,
            "quality_metrics": quality_metrics,
            "enhancement_timestamp": datetime.now().isoformat()
        }

    def _map_question_type_fallback(self, question_type: any, question: str) -> QuestionType:
        """폴백 질문 유형 매핑 (기존 방식)"""
        # question_type을 문자열로 변환
        if isinstance(question_type, QuestionType):
            question_type_str = question_type.value if hasattr(question_type, 'value') else str(question_type)
        elif isinstance(question_type, str):
            question_type_str = question_type.lower()
        else:
            question_type_str = str(question_type).lower()

        # 키워드 기반 매핑
        if "판례" in question or "precedent" in question_type_str:
            return QuestionType.PRECEDENT_SEARCH
        elif "계약서" in question or "contract" in question_type_str:
            return QuestionType.CONTRACT_REVIEW
        elif "이혼" in question or "divorce" in question_type_str:
            return QuestionType.DIVORCE_PROCEDURE
        elif "상속" in question or "inheritance" in question_type_str:
            return QuestionType.INHERITANCE_PROCEDURE
        elif "범죄" in question or "criminal" in question_type_str:
            return QuestionType.CRIMINAL_CASE
        elif "노동" in question or "labor" in question_type_str:
            return QuestionType.LABOR_DISPUTE
        elif "절차" in question or "procedure" in question_type_str:
            return QuestionType.PROCEDURE_GUIDE
        elif "용어" in question or "term" in question_type_str:
            return QuestionType.TERM_EXPLANATION
        elif "조언" in question or "advice" in question_type_str:
            return QuestionType.LEGAL_ADVICE
        elif "법률" in question or "law" in question_type_str:
            return QuestionType.LAW_INQUIRY
        else:
            return QuestionType.GENERAL_QUESTION

    def _analyze_question_structure(self, question: str) -> QuestionType:
        """질문 구조 분석을 통한 유형 추정"""
        question_lower = question.lower()

        # 질문 패턴 분석
        if any(word in question_lower for word in ['어떻게', 'how', '방법', '절차']):
            return QuestionType.PROCEDURE_GUIDE

        if any(word in question_lower for word in ['무엇', 'what', '의미', '정의']):
            return QuestionType.TERM_EXPLANATION

        if any(word in question_lower for word in ['도움', 'help', '조언', 'advice']):
            return QuestionType.LEGAL_ADVICE

        if any(word in question_lower for word in ['찾', 'search', '검색', '찾아']):
            return QuestionType.PRECEDENT_SEARCH

        return QuestionType.GENERAL_QUESTION

    def _analyze_current_structure(self, answer: str, template: Dict[str, Any]) -> Dict[str, Any]:
        """현재 답변 구조 분석 (안전한 버전)"""
        analysis = {
            "has_title": False,
            "section_coverage": {},
            "missing_sections": [],
            "structure_score": 0.0,
            "quality_indicators": {}  # 기본값 보장
        }

        try:
            # 제목 존재 여부
            analysis["has_title"] = bool(re.search(r'^#+\s+', answer, re.MULTILINE))

            # 섹션별 포함도 분석
            sections = template.get("sections", [])
            for section in sections:
                try:
                    section_name = section.get("name", "")
                    section_keywords = self._extract_section_keywords(section)

                    # 섹션 키워드가 답변에 포함되어 있는지 확인
                    coverage = self._calculate_section_coverage(answer, section_keywords)
                    analysis["section_coverage"][section_name] = coverage

                    # 누락된 섹션 확인
                    if coverage < 0.3:  # 30% 미만이면 누락으로 간주
                        analysis["missing_sections"].append(section_name)

                except Exception as e:
                    logger.warning(f"Section analysis error for {section.get('name', 'unknown')}: {e}", exc_info=True)
                    continue

            # 구조 점수 계산
            analysis["structure_score"] = self._calculate_structure_score(analysis)

            # 품질 지표 분석
            analysis["quality_indicators"] = self._analyze_quality_indicators(answer)

        except Exception as e:
            logger.error(f"Structure analysis error: {e}", exc_info=True)
            # 기본값 유지

        return analysis

    def _extract_section_keywords(self, section: Dict[str, Any]) -> List[str]:
        """섹션별 키워드 추출"""
        try:
            keywords = []

            # None 체크
            if section is None or not isinstance(section, dict):
                return []

            # 섹션 이름에서 키워드 추출
            section_name = section.get("name", "")
            if section_name:
                keywords.extend(section_name.split())

            # 템플릿에서 키워드 추출
            template_text = section.get("template", "") or ""
            if template_text:
                keywords.extend(re.findall(r'[\w가-힣]+', template_text))

            # 내용 가이드에서 키워드 추출
            content_guide = section.get("content_guide", "") or ""
            if content_guide:
                keywords.extend(re.findall(r'[\w가-힣]+', content_guide))

            return list(set(keywords))  # 중복 제거
        except Exception as e:
            logger.error(f"Error in _extract_section_keywords: {e}", exc_info=True)
            return []

    def _calculate_section_coverage(self, answer: str, keywords: List[str]) -> float:
        """섹션 포함도 계산 (안전한 버전)"""
        try:
            if not keywords or len(keywords) == 0:
                return 0.0

            answer_lower = answer.lower()
            matched_keywords = sum(1 for keyword in keywords if keyword.lower() in answer_lower)

            return matched_keywords / len(keywords)

        except Exception as e:
            logger.error(f"Section coverage calculation error: {e}", exc_info=True)
            return 0.0

    def _calculate_structure_score(self, analysis: Dict[str, Any]) -> float:
        """구조 점수 계산 (안전한 버전)"""
        try:
            score = 0.0

            # 제목 존재 여부 (20점)
            if analysis.get("has_title", False):
                score += 0.2

            # 섹션 포함도 (60점)
            section_coverage = analysis.get("section_coverage", {})
            if section_coverage:
                section_scores = list(section_coverage.values())
                if section_scores:
                    avg_section_coverage = sum(section_scores) / len(section_scores)
                    score += avg_section_coverage * 0.6

            # 품질 지표 (20점)
            quality_indicators = analysis.get("quality_indicators", {})
            if quality_indicators:
                quality_score = sum(quality_indicators.values()) / len(quality_indicators)
                score += quality_score * 0.2

            return min(1.0, score)

        except Exception as e:
            logger.error(f"Structure score calculation error: {e}", exc_info=True)
            return 0.0

    def _analyze_quality_indicators(self, answer: str) -> Dict[str, float]:
        """품질 지표 분석 (안전한 버전)"""
        try:
            answer_lower = answer.lower()
            quality_scores = {}

            for indicator_type, keywords in self.quality_indicators.items():
                if keywords and len(keywords) > 0:
                    matched_keywords = sum(1 for keyword in keywords if keyword.lower() in answer_lower)
                    quality_scores[indicator_type] = matched_keywords / len(keywords)
                else:
                    quality_scores[indicator_type] = 0.0

            return quality_scores

        except Exception as e:
            logger.error(f"Quality indicators analysis error: {e}", exc_info=True)
            # 기본값 반환
            return {indicator_type: 0.0 for indicator_type in self.quality_indicators.keys()}

    def _generate_structure_improvements(self, analysis: Dict[str, Any],
                                       template: Dict[str, Any],
                                       retrieved_docs: Optional[List[Dict[str, Any]]] = None,
                                       legal_references: Optional[List[str]] = None,
                                       legal_citations: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """구조화 개선 제안 생성 (법적 근거 정보 포함)"""
        improvements = []

        # None 체크 및 빈 리스트로 변환
        retrieved_docs = retrieved_docs if retrieved_docs is not None else []
        legal_references = legal_references if legal_references is not None else []
        legal_citations = legal_citations if legal_citations is not None else []

        # 타입 안전성 보장
        if not isinstance(retrieved_docs, list):
            retrieved_docs = []
        if not isinstance(legal_references, list):
            legal_references = []
        if not isinstance(legal_citations, list):
            legal_citations = []

        # 제목 추가 제안
        if not analysis["has_title"]:
            improvements.append({
                "type": "add_title",
                "priority": "high",
                "suggestion": f"답변에 제목을 추가하세요: '{template['title']}'",
                "impact": "높음"
            })

        # 법적 근거 섹션 추가 제안 (근거 정보가 있는 경우)
        if retrieved_docs or legal_references or legal_citations:
            has_legal_basis_section = any(
                section.get("name", "").lower() in ["법적근거", "참고법령", "판례", "legal_basis", "references"]
                for section in analysis.get("found_sections", [])
            )

            if not has_legal_basis_section:
                improvements.append({
                    "type": "add_legal_basis_section",
                    "priority": "high",
                    "suggestion": "법적 근거 및 참고 자료 섹션을 추가하세요",
                    "impact": "높음",
                    "legal_docs_count": len(retrieved_docs),
                    "references_count": len(legal_references),
                    "citations_count": len(legal_citations)
                })

        # 누락된 섹션 추가 제안
        for missing_section in analysis["missing_sections"]:
            section_info = next((s for s in template["sections"] if s["name"] == missing_section), None)
            if section_info:
                improvements.append({
                    "type": "add_section",
                    "priority": section_info["priority"],
                    "section_name": missing_section,
                    "suggestion": f"'{missing_section}' 섹션을 추가하세요",
                    "template": section_info["template"],
                    "content_guide": section_info["content_guide"],
                    "impact": "중간" if section_info["priority"] == "medium" else "높음"
                })

        # 품질 지표 개선 제안
        for indicator_type, score in analysis["quality_indicators"].items():
            if score < 0.5:  # 50% 미만이면 개선 필요
                improvements.append({
                    "type": "improve_quality",
                    "priority": "medium",
                    "indicator_type": indicator_type,
                    "suggestion": f"{indicator_type} 지표를 개선하세요",
                    "current_score": score,
                    "target_score": 0.7,
                    "impact": "중간"
                })

        return improvements

    def _create_structured_answer(self, answer: str, template: Dict[str, Any],
                                improvements: List[Dict[str, Any]],
                                retrieved_docs: Optional[List[Dict[str, Any]]] = None,
                                legal_references: Optional[List[str]] = None,
                                legal_citations: Optional[List[Dict[str, Any]]] = None) -> str:
        """구조화된 답변 생성 (법적 근거 정보 포함)"""
        structured_parts = []

        # None 체크 및 타입 안전성 보장
        retrieved_docs = retrieved_docs if retrieved_docs is not None else []
        legal_references = legal_references if legal_references is not None else []
        legal_citations = legal_citations if legal_citations is not None else []

        if not isinstance(retrieved_docs, list):
            retrieved_docs = []
        if not isinstance(legal_references, list):
            legal_references = []
        if not isinstance(legal_citations, list):
            legal_citations = []

        # 제목 추가
        if not re.search(r'^#+\s+', answer, re.MULTILINE):
            structured_parts.append(f"## {template['title']}")
            structured_parts.append("")

        # 기존 답변을 섹션별로 재구성
        current_answer = answer

        # 섹션별로 내용 재구성
        sections = template.get("sections", [])
        if not isinstance(sections, list):
            sections = []

        for section in sections:
            if not section or not isinstance(section, dict):
                continue

            section_name = section.get("name", "섹션")
            section_template = section.get("template", "")

            # 해당 섹션과 관련된 내용 추출
            section_content = self._extract_section_content(current_answer, section)

            # None 체크
            if section_content is None:
                section_content = ""

            # 법적 근거 섹션인 경우 근거 정보 포함
            is_legal_basis_section = any(
                keyword in section_name.lower()
                for keyword in ["법적근거", "참고법령", "판례", "legal_basis", "references", "출처"]
            )

            if is_legal_basis_section and (retrieved_docs or legal_references or legal_citations):
                # 법적 근거 정보를 섹션 내용에 추가
                legal_basis_content = self._format_legal_basis_content(
                    retrieved_docs, legal_references, legal_citations
                )
                if legal_basis_content:
                    section_content = f"{section_content}\n\n{legal_basis_content}".strip() if section_content else legal_basis_content

            # 빈 섹션 검증 - 내용이 없으면 섹션 생성하지 않음
            if not self._validate_section_content(section_content):
                # 필수 섹션(priority: high)도 내용이 없으면 생성하지 않음
                # (원래는 가이드만 표시했지만, 이제는 완전히 제외)
                continue

            # 유효한 섹션만 추가
            structured_parts.append(f"### {section_name}")
            structured_parts.append(section_template)
            structured_parts.append("")
            structured_parts.append(section_content)
            structured_parts.append("")

            # 개선 제안 적용
        improvements = improvements if improvements is not None else []
        for improvement in improvements:
            if not improvement or not isinstance(improvement, dict):
                continue

            if improvement.get("type") == "add_section":
                section_name = improvement.get("section_name", "섹션")
                section_template = improvement.get("template", "")
                content_guide = improvement.get("content_guide", "")

                # 가이드만 있는 빈 섹션은 생성하지 않음
                if not content_guide or len(content_guide.strip()) < 20:
                    continue

                structured_parts.append(f"### {section_name}")
                structured_parts.append(section_template)
                structured_parts.append("")
                structured_parts.append(f"*{content_guide}*")
                structured_parts.append("")
            elif improvement.get("type") == "add_legal_basis_section":
                # 법적 근거 섹션 추가
                legal_basis_content = self._format_legal_basis_content(
                    retrieved_docs, legal_references, legal_citations
                )
                if legal_basis_content:
                    structured_parts.append("### 참고 법령 및 판례")
                    structured_parts.append("")
                    structured_parts.append(legal_basis_content)
                    structured_parts.append("")

        return "\n".join(structured_parts)

    def _filter_relevant_documents(
        self,
        retrieved_docs: List[Dict[str, Any]],
        min_relevance_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """관련성 검증 및 필터링"""
        filtered_docs = []

        for doc in retrieved_docs:
            if not isinstance(doc, dict):
                continue

            # 관련도 스코어 확인
            score = doc.get("relevance_score", doc.get("score", 0.0))

            # 최소 관련도 미만이면 제외
            if score < min_relevance_score:
                continue

            # 문서 타입 검증 (빈 타입 제외)
            doc_type = doc.get("type", "").strip()
            if not doc_type:
                continue

            filtered_docs.append(doc)

        # 관련도 순으로 정렬
        filtered_docs.sort(
            key=lambda x: x.get("relevance_score", x.get("score", 0.0)),
            reverse=True
        )

        return filtered_docs

    def _format_legal_basis_content(
        self,
        retrieved_docs: List[Dict[str, Any]],
        legal_references: List[str],
        legal_citations: List[Dict[str, Any]]
    ) -> str:
        """법적 근거 정보를 포맷팅 (관련성 검증 포함)"""
        content_parts = []

        # None 체크 및 타입 안전성 보장
        retrieved_docs = retrieved_docs if retrieved_docs is not None else []
        legal_references = legal_references if legal_references is not None else []
        legal_citations = legal_citations if legal_citations is not None else []

        if not isinstance(retrieved_docs, list):
            retrieved_docs = []
        if not isinstance(legal_references, list):
            legal_references = []
        if not isinstance(legal_citations, list):
            legal_citations = []

        # 관련성 검증 및 필터링
        filtered_docs = self._filter_relevant_documents(retrieved_docs, min_relevance_score=0.3)

        # 검색된 문서 정보 (관련성 높은 것만)
        if filtered_docs:
            content_parts.append("#### 검색된 법률 문서")
            for i, doc in enumerate(filtered_docs[:5], 1):  # 최대 5개
                if not isinstance(doc, dict):
                    continue

                doc_type = doc.get("type", "문서")
                source = doc.get("source", doc.get("title", "알 수 없음"))
                content = doc.get("content", doc.get("text", ""))
                score = doc.get("relevance_score", doc.get("score", 0.0))

                # 내용 요약 (최대 200자) - None 체크
                if content is None:
                    content = ""
                content_preview = content[:200] + "..." if len(content) > 200 else content

                content_parts.append(f"{i}. **{doc_type}**: {source}")
                if score > 0:
                    # 관련도 낮은 경우 표시
                    if score < 0.5:
                        content_parts.append(f"   - 관련도: {score:.2f} (참고용)")
                    else:
                        content_parts.append(f"   - 관련도: {score:.2f}")
                content_parts.append(f"   - 내용: {content_preview}")
                content_parts.append("")

        # 법적 참고 자료
        if legal_references:
            content_parts.append("#### 참고 법령")
            for ref in legal_references:
                if ref is not None and ref.strip():
                    content_parts.append(f"- {ref}")
            content_parts.append("")

        # 법적 인용
        if legal_citations:
            content_parts.append("#### 법적 인용")
            for citation in legal_citations:
                if citation is not None:
                    if isinstance(citation, dict):
                        citation_text = citation.get("text", citation.get("citation", str(citation)))
                    else:
                        citation_text = str(citation)

                    if citation_text and citation_text.strip():
                        content_parts.append(f"- {citation_text}")
            content_parts.append("")

        return "\n".join(content_parts).strip()

    def _extract_section_content(self, answer: str, section: Dict[str, Any]) -> str:
        """섹션별 내용 추출"""
        try:
            # None 체크
            if answer is None:
                answer = ""
            if section is None or not isinstance(section, dict):
                return ""

            # 간단한 키워드 매칭으로 관련 내용 추출
            section_keywords = self._extract_section_keywords(section)

            # None 체크
            if section_keywords is None:
                section_keywords = []

            # 문단별로 분리하여 관련 문단 찾기
            paragraphs = answer.split('\n\n')
            relevant_paragraphs = []

            for paragraph in paragraphs:
                if paragraph and any(keyword.lower() in paragraph.lower() for keyword in section_keywords if keyword):
                    relevant_paragraphs.append(paragraph)

            return '\n\n'.join(relevant_paragraphs) if relevant_paragraphs else ""
        except Exception as e:
            logger.error(f"Error in _extract_section_content: {e}", exc_info=True)
            return ""

    def _calculate_quality_metrics(self, structured_answer: str) -> Dict[str, Any]:
        """품질 메트릭 계산 (안전한 버전)"""
        try:
            metrics = {
                "structure_score": 0.0,
                "completeness_score": 0.0,
                "legal_accuracy_score": 0.0,
                "practical_guidance_score": 0.0,
                "overall_score": 0.0
            }

            if not structured_answer or not isinstance(structured_answer, str):
                return metrics

            # 구조 점수 (섹션 수와 제목 존재 여부)
            section_count = len(re.findall(r'^###\s+', structured_answer, re.MULTILINE))
            has_title = bool(re.search(r'^##\s+', structured_answer, re.MULTILINE))
            metrics["structure_score"] = min(1.0, (section_count * 0.2) + (0.2 if has_title else 0))

            # 완성도 점수 (품질 지표 기반)
            quality_indicators = self._analyze_quality_indicators(structured_answer)
            if quality_indicators and len(quality_indicators) > 0:
                metrics["completeness_score"] = sum(quality_indicators.values()) / len(quality_indicators)
            else:
                metrics["completeness_score"] = 0.0

            # 법적 정확성 점수
            metrics["legal_accuracy_score"] = quality_indicators.get("legal_accuracy", 0.0)

            # 실무 조언 점수
            metrics["practical_guidance_score"] = quality_indicators.get("practical_guidance", 0.0)

            # 전체 점수 (가중 평균)
            metrics["overall_score"] = (
                metrics["structure_score"] * 0.3 +
                metrics["completeness_score"] * 0.3 +
                metrics["legal_accuracy_score"] * 0.2 +
                metrics["practical_guidance_score"] * 0.2
            )

            return metrics

        except Exception as e:
            logger.error(f"Quality metrics calculation error: {e}", exc_info=True)
            return {
                "structure_score": 0.0,
                "completeness_score": 0.0,
                "legal_accuracy_score": 0.0,
                "practical_guidance_score": 0.0,
                "overall_score": 0.0
            }

    def enhance_answer_with_legal_basis(self, answer: str, question_type: QuestionType,
                                       query: str = "") -> Dict[str, Any]:
        """법적 근거를 포함한 답변 강화"""
        try:
            # 1. 법적 인용 추출 및 강화
            citation_result = self.citation_enhancer.enhance_text_with_citations(answer)

            # 2. 법적 근거 검증
            validation_result = self.basis_validator.validate_legal_basis(query, answer)

            # 3. 구조화된 답변 생성
            structured_answer = self.create_structured_answer(answer, question_type)

            # 4. 법적 근거 섹션 추가
            enhanced_answer = self._add_legal_basis_section(
                structured_answer, citation_result, validation_result
            )

            return {
                "original_answer": answer,
                "enhanced_answer": enhanced_answer,
                "structured_answer": structured_answer,
                "citations": citation_result,
                "validation": validation_result,
                "legal_basis_summary": citation_result.get("legal_basis_summary", {}),
                "confidence": validation_result.confidence,
                "is_legally_sound": validation_result.is_valid
            }

        except Exception as e:
            logger.error(f"Error enhancing answer with legal basis: {e}", exc_info=True)
            return {
                "original_answer": answer,
                "enhanced_answer": answer,
                "structured_answer": answer,
                "citations": {"citations": [], "citation_count": 0},
                "validation": {"is_valid": False, "confidence": 0.0},
                "legal_basis_summary": {},
                "confidence": 0.0,
                "is_legally_sound": False,
                "error": str(e)
            }

    def _add_legal_basis_section(self, structured_answer: str,
                                citation_result: Dict[str, Any],
                                validation_result: Any) -> str:
        """법적 근거 섹션 추가"""
        try:
            legal_basis_section = "\n\n### 📚 법적 근거\n\n"

            # 인용 통계
            citation_count = citation_result.get("citation_count", 0)
            if citation_count > 0:
                legal_basis_section += f"**총 {citation_count}개의 법적 인용이 발견되었습니다.**\n\n"

                # 법령 인용
                laws_referenced = citation_result.get("legal_basis_summary", {}).get("laws_referenced", [])
                if laws_referenced:
                    legal_basis_section += "**관련 법령:**\n"
                    for law in laws_referenced[:5]:  # 최대 5개
                        legal_basis_section += f"- {law['formatted']} (신뢰도: {law['confidence']:.2f})\n"
                    legal_basis_section += "\n"

                # 판례 인용
                precedents_referenced = citation_result.get("legal_basis_summary", {}).get("precedents_referenced", [])
                if precedents_referenced:
                    legal_basis_section += "**관련 판례:**\n"
                    for precedent in precedents_referenced[:5]:  # 최대 5개
                        legal_basis_section += f"- {precedent['formatted']} (신뢰도: {precedent['confidence']:.2f})\n"
                    legal_basis_section += "\n"

                # 법원 판결
                court_decisions = citation_result.get("legal_basis_summary", {}).get("court_decisions", [])
                if court_decisions:
                    legal_basis_section += "**법원 판결:**\n"
                    for decision in court_decisions[:3]:  # 최대 3개
                        legal_basis_section += f"- {decision['formatted']} (신뢰도: {decision['confidence']:.2f})\n"
                    legal_basis_section += "\n"
            else:
                legal_basis_section += "**법적 인용이 발견되지 않았습니다.**\n\n"

            # 검증 결과
            if hasattr(validation_result, 'confidence'):
                confidence_level = "높음" if validation_result.confidence >= 0.8 else "보통" if validation_result.confidence >= 0.6 else "낮음"
                legal_basis_section += f"**법적 근거 신뢰도:** {confidence_level} ({validation_result.confidence:.2f})\n\n"

                if validation_result.is_valid:
                    legal_basis_section += "✅ **법적 근거가 충분히 검증되었습니다.**\n\n"
                else:
                    legal_basis_section += "⚠️ **법적 근거 검증이 필요합니다.**\n\n"

            # 권장사항
            if hasattr(validation_result, 'recommendations') and validation_result.recommendations:
                legal_basis_section += "**개선 권장사항:**\n"
                for recommendation in validation_result.recommendations[:3]:  # 최대 3개
                    legal_basis_section += f"- {recommendation}\n"
                legal_basis_section += "\n"

            return structured_answer + legal_basis_section

        except Exception as e:
            logger.error(f"Error adding legal basis section: {e}", exc_info=True)
            return structured_answer

    def get_legal_citation_statistics(self, text: str) -> Dict[str, Any]:
        """법적 인용 통계 조회"""
        try:
            citation_result = self.citation_enhancer.enhance_text_with_citations(text)

            return {
                "total_citations": citation_result.get("citation_count", 0),
                "citation_types": citation_result.get("citation_stats", {}),
                "confidence_distribution": citation_result.get("confidence_distribution", {}),
                "legal_basis_summary": citation_result.get("legal_basis_summary", {}),
                "enhanced_text": citation_result.get("enhanced_text", text)
            }

        except Exception as e:
            logger.error(f"Error getting citation statistics: {e}", exc_info=True)
            return {
                "total_citations": 0,
                "citation_types": {},
                "confidence_distribution": {},
                "legal_basis_summary": {},
                "enhanced_text": text
            }

    def validate_answer_legal_basis(self, query: str, answer: str) -> Dict[str, Any]:
        """답변의 법적 근거 검증"""
        try:
            validation_result = self.basis_validator.validate_legal_basis(query, answer)

            return {
                "is_valid": validation_result.is_valid,
                "confidence": validation_result.confidence,
                "validation_details": validation_result.validation_details,
                "legal_sources": validation_result.legal_sources,
                "issues": validation_result.issues,
                "recommendations": validation_result.recommendations
            }

        except Exception as e:
            logger.error(f"Error validating legal basis: {e}", exc_info=True)
            return {
                "is_valid": False,
                "confidence": 0.0,
                "validation_details": [],
                "legal_sources": [],
                "issues": [f"검증 중 오류 발생: {str(e)}"],
                "recommendations": ["시스템 관리자에게 문의하세요"]
            }

    def reload_templates(self):
        """템플릿 동적 리로드"""
        try:
            self.structure_templates = self._load_structure_templates()
            self.quality_indicators = self._load_quality_indicators()
            # 캐시 무효화
            if hasattr(self, '_few_shot_examples_cache'):
                self._few_shot_examples_cache = None
            logger.info("Templates reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload templates: {e}", exc_info=True)

    def get_template_info(self, question_type: str) -> Dict[str, Any]:
        """템플릿 정보 조회"""
        try:
            # 질문 유형 매핑
            try:
                question_type_enum = QuestionType(question_type)
            except ValueError:
                question_type_enum = QuestionType.GENERAL_QUESTION

            template = self.structure_templates.get(question_type_enum)

            if template:
                return {
                    "question_type": question_type,
                    "title": template.get("title", "Unknown"),
                    "section_count": len(template.get("sections", [])),
                    "sections": template.get("sections", []),
                    "source": "hardcoded"
                }
            else:
                return {
                    "question_type": question_type,
                    "title": "Unknown",
                    "section_count": 0,
                    "sections": [],
                    "source": "not_found"
                }
        except Exception as e:
            logger.error(f"Failed to get template info: {e}", exc_info=True)
            return {
                "question_type": question_type,
                "title": "Error",
                "section_count": 0,
                "sections": [],
                "source": "error"
            }

    def create_structured_answer(self, answer: str, question_type: QuestionType) -> str:
        """구조화된 답변 생성"""
        try:
            # 질문 유형에 따른 템플릿 가져오기
            template = self.structure_templates.get(question_type, {})

            if not template:
                return answer

            # 구조화된 답변 생성
            structured_answer = f"## {template.get('title', '답변')}\n\n"

            # 각 섹션별로 내용 구성
            sections = template.get('sections', [])
            for section in sections:
                if section.get('priority') == 'high':
                    structured_answer += f"### {section['name']}\n"
                    structured_answer += f"{section['template']}\n\n"
                    structured_answer += f"{answer}\n\n"
                    break  # 첫 번째 high priority 섹션만 사용

            return structured_answer

        except Exception as e:
            logger.error(f"Error creating structured answer: {e}", exc_info=True)
            return answer


# 전역 인스턴스
answer_structure_enhancer = AnswerStructureEnhancer()
