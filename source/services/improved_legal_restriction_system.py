# -*- coding: utf-8 -*-
"""
Improved Legal Restriction System
개선된 법률 제한 시스템 - 맥락 기반 분석 및 단계적 제한
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class RestrictionLevel(Enum):
    """제한 수준"""
    CRITICAL = "critical"      # 절대 금지 (탈법 행위 등)
    HIGH = "high"              # 높은 위험 (법률 자문)
    MEDIUM = "medium"          # 중간 위험 (민감한 조언)
    LOW = "low"                # 낮은 위험 (주의 필요)
    ALLOWED = "allowed"        # 허용


class LegalArea(Enum):
    """법률 영역"""
    LEGAL_ADVICE = "legal_advice"           # 법률 자문
    MEDICAL_LEGAL = "medical_legal"         # 의료 법률
    CRIMINAL_CASE = "criminal_case"         # 형사 사건
    TAX_EVASION = "tax_evasion"             # 탈세
    CONTRACT_MANIPULATION = "contract_manipulation"  # 계약 조작
    EVIDENCE_TAMPERING = "evidence_tampering"        # 증거 조작
    GENERAL_INFO = "general_info"           # 일반 정보


class ContextType(Enum):
    """맥락 유형"""
    PERSONAL_CASE = "personal_case"          # 개인 사건
    HYPOTHETICAL = "hypothetical"            # 가상의 상황
    ACADEMIC = "academic"                    # 학술적 질문
    PROFESSIONAL = "professional"           # 전문가 질문
    GENERAL_CURIOSITY = "general_curiosity"  # 일반적 호기심


@dataclass
class ContextAnalysis:
    """맥락 분석 결과"""
    context_type: ContextType
    personal_score: float
    general_score: float
    hypothetical_score: float
    confidence: float
    indicators: List[str]


@dataclass
class ImprovedRestrictionRule:
    """개선된 제한 규칙"""
    id: str
    area: LegalArea
    level: RestrictionLevel
    patterns: List[str]
    exceptions: List[str]  # 예외 패턴
    prohibited_phrases: List[str]
    safe_alternatives: List[str]
    description: str
    legal_basis: str


@dataclass
class ImprovedRestrictionResult:
    """개선된 제한 검사 결과"""
    is_restricted: bool
    restriction_level: RestrictionLevel
    context_analysis: ContextAnalysis
    matched_rules: List[ImprovedRestrictionRule]
    matched_patterns: List[str]
    confidence: float
    safe_response: Optional[str]
    warning_message: Optional[str]
    reasoning: str
    timestamp: datetime


class ImprovedLegalRestrictionSystem:
    """개선된 법률 제한 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.restriction_rules = self._initialize_improved_rules()
        self.compiled_patterns = self._compile_patterns()
        self.compiled_exceptions = self._compile_exceptions()
        
    def _initialize_improved_rules(self) -> List[ImprovedRestrictionRule]:
        """개선된 제한 규칙 초기화"""
        rules = []
        
        # 1. 법률 자문 금지 (맥락 기반)
        rules.append(ImprovedRestrictionRule(
            id="legal_advice_improved_001",
            area=LegalArea.LEGAL_ADVICE,
            level=RestrictionLevel.HIGH,
            patterns=[
                # 명확한 개인적 조언 요청
                r"제\s*경우\s*(어떻게|무엇을|해야|해야\s*하나요)",
                r"저는\s*(어떻게|무엇을|해야|해야\s*하나요)",
                r"내\s*사건\s*(어떻게|무엇을|해야|해야\s*하나요)",
                r"이런\s*상황\s*(어떻게|무엇을|해야|해야\s*하나요)",
                
                # 직접적 조언 요청
                r"소송\s*(할까요|해야\s*할까요|하지\s*않을까요)",
                r"변호사\s*(고용|선임|상담)\s*(해야|할까요)",
                r"법적\s*조치\s*(어떻게|무엇을|해야)",
                r"권리\s*(주장|행사)\s*(어떻게|무엇을|해야)",
                
                # 결과 예측 요청
                r"(승소|패소)\s*(할까요|할\s*것인가요|확률)",
                r"(위자료|손해배상|배상금)\s*(얼마|몇\s*만|몇\s*천)",
                
                # 단정적 지시
                r"당신의\s*경우\s*(반드시|꼭|무조건)",
                r"(법적|법률적)\s*판단\s*(을|을\s*내리다)",
                r"(유죄|무죄|과실|책임)\s*(이다|입니다|라고\s*봅니다)"
            ],
            exceptions=[
                # 예외 패턴 (허용)
                r"일반적으로\s*소송",
                r"보통\s*소송",
                r"일반적인\s*소송\s*절차",
                r"소송\s*제기\s*절차",
                r"소송\s*비용",
                r"소송\s*기간",
                r"소송\s*절차\s*(는|은)\s*(어떻게|무엇을)",
                r"소송\s*(신청|제기|진행)\s*절차",
                r"관련\s*법령",
                r"법령\s*(참조|인용|적용)",
                r"판례\s*(참조|인용|적용)",
                r"절차\s*(어떻게|무엇을|해야)",
                r"신청\s*(어떻게|무엇을|해야)",
                r"제출\s*(어떻게|무엇을|해야)",
                r"처리\s*(어떻게|무엇을|해야)",
                r"어디에\s*(신청|제출|문의)",
                r"어떤\s*서류\s*(필요|준비)"
            ],
            prohibited_phrases=[
                "소송하세요", "소송하지 마세요", "승소할 것입니다", "패소할 것입니다",
                "위자료는 얼마 받을 수 있습니다", "손해배상은 몇 만원입니다",
                "당신의 경우 반드시 해야 합니다", "법적 판단을 내리면",
                "유죄입니다", "무죄입니다", "과실이 있습니다", "책임이 있습니다"
            ],
            safe_alternatives=[
                "일반적으로 이런 경우 소송을 고려할 수 있습니다",
                "관련 법령에 따르면 다음과 같은 절차가 있습니다",
                "구체적인 사안은 변호사와 상담하시는 것이 좋습니다",
                "법원의 판례를 참고하시면 도움이 될 것입니다"
            ],
            description="구체적인 법률 자문 및 판단 제시 금지 (맥락 기반)",
            legal_basis="변호사법 제109조 (무자격자의 법률사무 취급 금지)"
        ))
        
        # 2. 의료/건강 법률 조언 (개선됨)
        rules.append(ImprovedRestrictionRule(
            id="medical_legal_improved_001",
            area=LegalArea.MEDICAL_LEGAL,
            level=RestrictionLevel.HIGH,
            patterns=[
                # 의료사고 과실 판단
                r"의료\s*(사고|과실|진\s*과실|행위)\s*(의|에서)\s*(과실|책임)\s*(이|가)\s*(있나요|있나|있어요)",
                r"의료진\s*(이|가)\s*(잘못했나요|잘못했나|잘못했어요)",
                r"의료\s*(과실|사고)\s*(이|가)\s*(인정될까요|인정되나요|인정되나)",
                r"의료진\s*(의|의)\s*(책임|과실)\s*(이|가)\s*(있나요|있나|있어요)",
                
                # 장애등급 예측
                r"장애\s*(등급|인등급|정도|판정)\s*(은|는)\s*(몇\s*급|몇급)",
                r"장애등급\s*(은|는)\s*(몇\s*급|몇급)",
                
                # 의학적 인과관계 판단
                r"의학적\s*인과관계\s*(는|은)\s*(명확|확실|있나요|있나|있어요)",
                r"의학적\s*인과관계\s*(가|이)\s*(인정될까요|인정되나요|인정되나)",
                
                # 의료분쟁 결과 예측
                r"(의료분쟁|의료소송|의료감정)\s*(에서|의)\s*(과실|책임|승소|패소)",
                r"의료\s*(분쟁|소송)\s*(에서|에서)\s*(승소할까요|승소하나요|승소하나)",
                r"의료사고\s*(로|로\s*인한)\s*(손해|배상|위자료)\s*(는|은)\s*(얼마|몇\s*만|몇\s*천)",
                r"의료사고\s*감정\s*(결과|판정)\s*(는|은)\s*(어떻게|무엇으로)\s*(될까요|되나요|되나)",
                r"의료진\s*(이|가)\s*(보상|배상)\s*(해야|해야\s*할까요|해야\s*하나요)"
            ],
            exceptions=[
                r"의료분쟁조정중재원\s*(은|는)\s*(어디에|어디)",
                r"의료사고\s*감정\s*절차",
                r"의료법\s*관련\s*(절차|규정)",
                r"의료소송\s*(절차|비용|기간)",
                r"의료분쟁\s*(조정|중재)\s*절차"
            ],
            prohibited_phrases=[
                "의료사고의 과실이 있습니다", "의료진이 잘못했습니다",
                "장애등급은 몇 급입니다", "의학적 인과관계가 명확합니다",
                "의료과실이 인정됩니다", "의료진의 책임이 있습니다"
            ],
            safe_alternatives=[
                "의료분쟁조정중재원에 문의하시면 도움이 됩니다",
                "의료사고 감정 절차에 대해 안내드릴 수 있습니다",
                "전문 의료소송 변호사와 상담하시는 것을 권합니다",
                "의료법 관련 일반적인 절차를 안내드릴 수 있습니다"
            ],
            description="의료사고 과실 판단 및 의학적 인과관계 판단 금지",
            legal_basis="의료법 제65조 (의료사고 관련 규정)"
        ))
        
        # 3. 형사 사건 관련 민감 조언 (개선됨)
        rules.append(ImprovedRestrictionRule(
            id="criminal_case_improved_001",
            area=LegalArea.CRIMINAL_CASE,
            level=RestrictionLevel.HIGH,
            patterns=[
                # 자백/부인 조언
                r"(자백|부인)\s*(하세요|하지\s*마세요|해야|하지\s*않으면|하시겠습니까|하시겠어요)",
                r"(자백|부인)\s*(할까요|해야\s*할까요|하지\s*않을까요)",
                
                # 증거 조작/은닉 방법
                r"증거\s*(인멸|은닉|조작|숨기기)\s*(방법|수단|기법)",
                r"증거\s*(를|을)\s*(인멸|은닉|조작|숨기기)\s*(방법|수단|기법)",
                r"증거\s*(를|을)\s*(숨기는|인멸하는|조작하는)\s*(방법|수단|기법)",
                
                # 형량 예측
                r"형량\s*(예측|산정|몇\s*년|몇년)",
                r"형량\s*(은|는)\s*(몇\s*년|몇년)",
                r"형량\s*(이|가)\s*(얼마|몇\s*년|몇년)",
                
                # 범죄 수법/전략
                r"범죄\s*(수법|방법|계획|전략)",
                r"수사\s*대응\s*(전략|방법|기법)",
                r"(경찰|검찰)\s*조사\s*(에서|에서)\s*(어떻게|무엇을|해야)",
                r"법망\s*(빠져나가기|회피)\s*(방법|수단|기법)",
                r"법망\s*(을|를)\s*(빠져나가는|회피하는)\s*(방법|수단|기법)"
            ],
            exceptions=[
                r"국선변호인\s*신청\s*(방법|절차)",
                r"변호인\s*조력권\s*(이|가)\s*(무엇|어떤)",
                r"형사절차\s*(의|에서)\s*(일반적|보통)",
                r"형사소송법\s*관련\s*(절차|규정)"
            ],
            prohibited_phrases=[
                "자백하세요", "부인하세요", "증거를 인멸하세요",
                "형량은 몇 년입니다", "이런 범죄 수법이 있습니다"
            ],
            safe_alternatives=[
                "국선변호인 신청 방법을 안내드릴 수 있습니다",
                "변호인 조력권에 대해 설명드릴 수 있습니다",
                "즉시 변호사 상담을 받으시는 것을 권합니다",
                "형사절차의 일반적인 흐름을 안내드릴 수 있습니다"
            ],
            description="수사 대응 전략 및 증거 조작 방법 제시 금지",
            legal_basis="형사소송법 제30조 (변호인 조력권)"
        ))
        
        # 4. 탈법 행위 조력 (절대 금지)
        rules.append(ImprovedRestrictionRule(
            id="tax_evasion_improved_001",
            area=LegalArea.TAX_EVASION,
            level=RestrictionLevel.CRITICAL,
            patterns=[
                # 세금 회피/탈세 방법
                r"세금\s*(회피|탈세|줄이기|피하기)\s*(방법|수단|기법)",
                r"세금\s*(을|를)\s*(회피|탈세|줄이기|피하기)\s*(방법|수단|기법)",
                r"탈세\s*(하는|하는\s*방법|방법|수단|기법)",
                r"세금\s*(을|를)\s*(줄이는|줄이는\s*방법|줄이기)\s*(방법|수단|기법)",
                
                # 위장 행위 방법
                r"위장\s*(이혼|계약|결혼|행위)\s*(방법|수단|기법)",
                r"위장\s*(으로|으로\s*하는|하는)\s*(이혼|계약|결혼)\s*(방법|수단|기법)",
                
                # 가짜/허위 서류 방법
                r"(가짜|허위|거짓)\s*(계약|서류|증명|신고)\s*(방법|수단|기법)",
                r"(가짜|허위|거짓)\s*(계약|서류|증명|신고)\s*(을|를)\s*(만드는|작성하는)\s*(방법|수단|기법)",
                
                # 증거 조작/위조 방법
                r"증거\s*(조작|위조|은닉)\s*(방법|수단|기법)",
                r"서류\s*(위조|조작|변조)\s*(방법|수단|기법)",
                r"증거\s*(를|을)\s*(조작|위조|은닉)\s*(방법|수단|기법)",
                r"서류\s*(를|을)\s*(위조|조작|변조)\s*(방법|수단|기법)",
                
                # 법망 회피 방법
                r"법망\s*(빠져나가기|회피)\s*(방법|수단|기법)",
                r"법망\s*(을|를)\s*(빠져나가는|회피하는)\s*(방법|수단|기법)",
                r"법적\s*구멍\s*(이용|활용)\s*(방법|수단|기법)",
                r"법적\s*구멍\s*(을|를)\s*(이용|활용)\s*(방법|수단|기법)",
                
                # 불법 행위 은닉 방법
                r"불법\s*행위\s*(숨기기|은닉)\s*(방법|수단|기법)",
                r"불법\s*행위\s*(를|을)\s*(숨기는|은닉하는)\s*(방법|수단|기법)",
                r"우회\s*(방법|수단|기법)",
                r"회피\s*(방법|수단|기법)"
            ],
            exceptions=[
                r"세법\s*관련\s*(일반적|보통)\s*(절차|규정)",
                r"국세청\s*(에|에서)\s*(문의|신고)\s*(방법|절차)",
                r"세무\s*전문가\s*(와|과)\s*(상담|문의)"
            ],
            prohibited_phrases=[
                "세금을 회피하는 방법", "탈세하는 방법", "세금을 줄이는 방법",
                "위장 이혼하는 방법", "가짜 계약을 만드는 방법",
                "증거를 조작하는 방법", "서류를 위조하는 방법"
            ],
            safe_alternatives=[
                "세법 관련 일반적인 절차를 안내드릴 수 있습니다",
                "세무 전문가와 상담하시는 것을 권합니다",
                "국세청에 문의하시면 정확한 안내를 받으실 수 있습니다"
            ],
            description="세금 회피 및 탈세 방법 제시 절대 금지",
            legal_basis="국세기본법 제81조 (납세의무)"
        ))
        
        # 5. 일반 정보 질문 허용 (새로 추가)
        rules.append(ImprovedRestrictionRule(
            id="general_info_001",
            area=LegalArea.GENERAL_INFO,
            level=RestrictionLevel.ALLOWED,
            patterns=[
                # 법률 규정 질문
                r"(.+법|.+규정|.+조항)\s*(은|는|이|가)\s*(무엇|어떤|어떠한)",
                r"(.+법|.+규정|.+조항)\s*(에\s*대해|에\s*관해|에\s*대한)",
                r"(.+법|.+규정|.+조항)\s*(규정|내용|조항)",
                
                # 일반적인 절차 질문
                r"(.+절차|.+방법|.+신청|.+제출)\s*(은|는|이|가)\s*(어떻게|무엇|어떠한)",
                r"(.+절차|.+방법|.+신청|.+제출)\s*(에\s*대해|에\s*관해|에\s*대한)",
                
                # 법률 용어 설명 요청
                r"(.+이|.+가|.+은|.+는)\s*(무엇|어떤|어떠한|의미)",
                r"(.+에\s*대해|.+에\s*관해|.+에\s*대한)\s*(설명|해석|의미)",
                
                # 일반적인 정보 요청
                r"(.+에\s*대해|.+에\s*관해|.+에\s*대한)\s*(알고\s*싶|궁금|질문)",
                r"(.+에\s*대해|.+에\s*관해|.+에\s*대한)\s*(정보|자료|내용)"
            ],
            exceptions=[],  # 일반 정보는 예외 없음
            prohibited_phrases=[],  # 금지 구문 없음
            safe_alternatives=[
                "해당 법률의 일반적인 내용을 안내드릴 수 있습니다",
                "관련 절차에 대해 설명드릴 수 있습니다",
                "법률 용어의 의미를 설명드릴 수 있습니다"
            ],
            description="일반적인 법률 정보 및 절차 안내 허용",
            legal_basis="정보공개법 제1조 (목적)"
        ))
        
        return rules
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """패턴 컴파일"""
        compiled = {}
        for rule in self.restriction_rules:
            for pattern in rule.patterns:
                try:
                    compiled[f"{rule.id}_{pattern}"] = re.compile(pattern, re.IGNORECASE)
                except re.error as e:
                    self.logger.warning(f"Pattern compilation failed: {pattern}, error: {e}")
        return compiled
    
    def _compile_exceptions(self) -> Dict[str, re.Pattern]:
        """예외 패턴 컴파일"""
        compiled = {}
        for rule in self.restriction_rules:
            for exception in rule.exceptions:
                try:
                    compiled[f"{rule.id}_exception_{exception}"] = re.compile(exception, re.IGNORECASE)
                except re.error as e:
                    self.logger.warning(f"Exception pattern compilation failed: {exception}, error: {e}")
        return compiled
    
    def analyze_context(self, query: str) -> ContextAnalysis:
        """맥락 분석 (카테고리별 맞춤 정책 포함)"""
        query_lower = query.lower()
        
        # 개인적 표현 지표
        personal_indicators = [
            "제 경우", "저는", "내 사건", "이런 상황", "현재 상황",
            "진행 중인", "당사자", "구체적 사안", "실제 사건", "내 문제"
        ]
        
        # 일반적 표현 지표
        general_indicators = [
            "일반적으로", "보통", "일반적인", "법령", "법률", 
            "관련 법", "절차", "신청", "제출", "처리", "방법"
        ]
        
        # 가상적 표현 지표
        hypothetical_indicators = [
            "만약", "가정", "예를 들어", "가상의", "만일",
            "상상해보면", "가정해보면", "예시로", "만약에"
        ]
        
        # 카테고리별 특별 지표 (더 관대한 처리)
        administrative_indicators = [
            "행정", "행정심판", "행정소송", "행정처분", "행정절차",
            "신청", "제출", "처리", "절차", "방법", "규정"
        ]
        
        corporate_indicators = [
            "법인", "주식회사", "법인세", "설립", "등기", "해산",
            "신고", "납부", "계산", "절차", "방법", "규정"
        ]
        
        real_estate_indicators = [
            "부동산", "매매", "임대차", "등기", "소유권", "이전",
            "절차", "방법", "규정", "신청", "제출", "처리"
        ]
        
        # 점수 계산
        personal_score = sum(1 for indicator in personal_indicators if indicator in query_lower)
        general_score = sum(1 for indicator in general_indicators if indicator in query_lower)
        hypothetical_score = sum(1 for indicator in hypothetical_indicators if indicator in query_lower)
        
        # 카테고리별 점수 계산
        admin_score = sum(1 for indicator in administrative_indicators if indicator in query_lower)
        corporate_score = sum(1 for indicator in corporate_indicators if indicator in query_lower)
        real_estate_score = sum(1 for indicator in real_estate_indicators if indicator in query_lower)
        
        # 맥락 유형 결정 (카테고리별 우선순위 적용)
        if personal_score > 0:
            context_type = ContextType.PERSONAL_CASE
        elif admin_score > 0 or corporate_score > 0 or real_estate_score > 0:
            # 카테고리별 특별 지표가 있으면 일반적 호기심으로 분류 (더 관대한 처리)
            context_type = ContextType.GENERAL_CURIOSITY
        elif general_score > 0:
            context_type = ContextType.GENERAL_CURIOSITY
        elif hypothetical_score > 0:
            context_type = ContextType.HYPOTHETICAL
        else:
            context_type = ContextType.GENERAL_CURIOSITY
        
        # 신뢰도 계산 (카테고리별 지표 고려)
        total_indicators = personal_score + general_score + hypothetical_score + admin_score + corporate_score + real_estate_score
        confidence = min(total_indicators / 3.0, 1.0) if total_indicators > 0 else 0.5
        
        # 매칭된 지표들
        matched_indicators = []
        all_indicators = personal_indicators + general_indicators + hypothetical_indicators + administrative_indicators + corporate_indicators + real_estate_indicators
        for indicator in all_indicators:
            if indicator in query_lower:
                matched_indicators.append(indicator)
        
        return ContextAnalysis(
            context_type=context_type,
            personal_score=personal_score,
            general_score=general_score,
            hypothetical_score=hypothetical_score,
            confidence=confidence,
            indicators=matched_indicators
        )
    
    def check_restrictions(self, query: str, response: str = "") -> ImprovedRestrictionResult:
        """개선된 제한 사항 검사"""
        try:
            # 1. 맥락 분석
            context_analysis = self.analyze_context(query)
            
            # 2. 복합 질문에서 개인적 조언 부분 감지
            has_personal_advice = self._detect_personal_advice_in_complex_query(query)
            
            # 3. 일반 정보 질문 검사 (우선순위 높음)
            general_info_matched = self._check_general_info_patterns(query)
            if general_info_matched:
                return ImprovedRestrictionResult(
                    is_restricted=False,
                    restriction_level=RestrictionLevel.ALLOWED,
                    context_analysis=context_analysis,
                    matched_rules=[general_info_matched],
                    matched_patterns=[],
                    confidence=0.95,
                    safe_response=None,
                    warning_message=None,
                    reasoning=f"일반 정보 질문으로 허용: {general_info_matched.description}",
                    timestamp=datetime.now()
                )
            
            # 4. 예외 패턴 검사 (예외가 있으면 허용)
            exception_matched = self._check_exceptions(query)
            if exception_matched and not has_personal_advice:
                return ImprovedRestrictionResult(
                    is_restricted=False,
                    restriction_level=RestrictionLevel.ALLOWED,
                    context_analysis=context_analysis,
                    matched_rules=[],
                    matched_patterns=[],
                    confidence=0.9,
                    safe_response=None,
                    warning_message=None,
                    reasoning=f"예외 패턴 매칭: {exception_matched}",
                    timestamp=datetime.now()
                )
            
            # 4. 맥락 기반 제한 검사
            if context_analysis.context_type == ContextType.PERSONAL_CASE or has_personal_advice:
                # 개인 사건의 경우 더 엄격한 검사
                restriction_result = self._check_personal_case(query, context_analysis)
            elif context_analysis.context_type == ContextType.GENERAL_CURIOSITY:
                # 일반적 호기심의 경우 관대한 검사
                restriction_result = self._check_general_curiosity(query, context_analysis)
            else:
                # 기타 경우 기본 검사
                restriction_result = self._check_default(query, context_analysis)
            
            return restriction_result
            
        except Exception as e:
            self.logger.error(f"Error checking restrictions: {e}")
            return ImprovedRestrictionResult(
                is_restricted=False,
                restriction_level=RestrictionLevel.LOW,
                context_analysis=ContextAnalysis(
                    context_type=ContextType.GENERAL_CURIOSITY,
                    personal_score=0.0,
                    general_score=0.0,
                    hypothetical_score=0.0,
                    confidence=0.0,
                    indicators=[]
                ),
                matched_rules=[],
                matched_patterns=[],
                confidence=0.0,
                safe_response=None,
                warning_message=None,
                reasoning=f"Error: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _detect_personal_advice_in_complex_query(self, query: str) -> bool:
        """복합 질문에서 개인적 조언 부분 감지"""
        query_lower = query.lower()
        
        # 개인적 조언 지표들
        personal_advice_indicators = [
            "제 경우", "저는", "내 사건", "이런 상황", "현재 상황",
            "진행 중인", "당사자", "구체적 사안", "실제 사건", "내 문제",
            "어떻게 해야", "무엇을 해야", "해야 할까요", "해야 하나요",
            "소송하시겠습니까", "변호사를 고용해야", "법적 조치를 취해야",
            "권리를 주장해야", "승소할까요", "패소할까요", "위자료는 얼마",
            "손해배상은 얼마", "형량은 몇 년", "자백해야 할까요",
            "부인해야 할까요", "의료과실이 있나요", "장애등급은 몇 급"
        ]
        
        # 개인적 조언 패턴들
        personal_advice_patterns = [
            r"제\s*경우\s*(어떻게|무엇을|해야|해야\s*하나요)",
            r"저는\s*(어떻게|무엇을|해야|해야\s*하나요)",
            r"내\s*사건\s*(어떻게|무엇을|해야|해야\s*하나요)",
            r"이런\s*상황\s*(어떻게|무엇을|해야|해야\s*하나요)",
            r"소송\s*(할까요|해야\s*할까요|하지\s*않을까요)",
            r"변호사\s*(고용|선임|상담)\s*(해야|할까요)",
            r"법적\s*조치\s*(어떻게|무엇을|해야)",
            r"권리\s*(주장|행사)\s*(어떻게|무엇을|해야)",
            r"(승소|패소)\s*(할까요|할\s*것인가요|확률)",
            r"(위자료|손해배상|배상금)\s*(얼마|몇\s*만|몇\s*천)",
            r"형량\s*(예측|산정|몇\s*년)",
            r"(자백|부인)\s*(하시겠습니까|해야\s*할까요)",
            r"의료\s*(과실|사고)\s*(이|가)\s*(있나요|있나)",
            r"장애\s*(등급|인등급)\s*(은|는)\s*(몇\s*급|몇급)"
        ]
        
        # 키워드 매칭
        keyword_matches = sum(1 for indicator in personal_advice_indicators if indicator in query_lower)
        
        # 패턴 매칭
        pattern_matches = 0
        for pattern in personal_advice_patterns:
            try:
                if re.search(pattern, query, re.IGNORECASE):
                    pattern_matches += 1
            except re.error:
                continue
        
        # 개인적 조언이 있다고 판단하는 기준
        # 키워드 1개 이상 또는 패턴 1개 이상 매칭
        return keyword_matches > 0 or pattern_matches > 0
    
    def _check_general_info_patterns(self, query: str) -> Optional[ImprovedRestrictionRule]:
        """일반 정보 질문 패턴 검사"""
        try:
            # 일반 정보 규칙 찾기
            general_info_rule = None
            for rule in self.restriction_rules:
                if rule.area == LegalArea.GENERAL_INFO:
                    general_info_rule = rule
                    break
            
            if not general_info_rule:
                return None
            
            # 패턴 매칭 검사
            query_lower = query.lower()
            for pattern in general_info_rule.patterns:
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                    if compiled_pattern.search(query):
                        return general_info_rule
                except re.error:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking general info patterns: {e}")
            return None
    
    def _check_exceptions(self, query: str) -> Optional[str]:
        """예외 패턴 검사"""
        for rule in self.restriction_rules:
            for exception in rule.exceptions:
                pattern_key = f"{rule.id}_exception_{exception}"
                if pattern_key in self.compiled_exceptions:
                    if self.compiled_exceptions[pattern_key].search(query):
                        return exception
        return None
    
    def _check_personal_case(self, query: str, context_analysis: ContextAnalysis) -> ImprovedRestrictionResult:
        """개인 사건 검사 (엄격)"""
        matched_rules = []
        matched_patterns = []
        max_confidence = 0.0
        
        for rule in self.restriction_rules:
            rule_matches = self._check_rule_patterns(query, rule)
            if rule_matches['matched']:
                matched_rules.append(rule)
                matched_patterns.extend(rule_matches['patterns'])
                max_confidence = max(max_confidence, rule_matches['confidence'])
        
        # 개인 사건의 경우 더 낮은 임계값으로 제한 (0.1 → 0.2로 조정)
        is_restricted = max_confidence > 0.2
        
        if is_restricted:
            restriction_level = self._determine_restriction_level(matched_rules)
            safe_response = self._generate_safe_response(matched_rules, query)
            warning_message = self._generate_warning_message(matched_rules)
        else:
            restriction_level = RestrictionLevel.ALLOWED
            safe_response = None
            warning_message = None
        
        return ImprovedRestrictionResult(
            is_restricted=is_restricted,
            restriction_level=restriction_level,
            context_analysis=context_analysis,
            matched_rules=matched_rules,
            matched_patterns=matched_patterns,
            confidence=max_confidence,
            safe_response=safe_response,
            warning_message=warning_message,
            reasoning=f"개인 사건 맥락 분석: {context_analysis.indicators}",
            timestamp=datetime.now()
        )
    
    def _check_general_curiosity(self, query: str, context_analysis: ContextAnalysis) -> ImprovedRestrictionResult:
        """일반적 호기심 검사 (더욱 관대)"""
        matched_rules = []
        matched_patterns = []
        max_confidence = 0.0
        
        # 카테고리별 특별 지표가 있는 경우 더욱 관대한 처리
        has_category_indicators = any(indicator in query.lower() for indicator in [
            "행정", "법인", "부동산", "주식회사", "법인세", "매매", "임대차", 
            "등기", "신청", "제출", "처리", "절차", "방법", "규정"
        ])
        
        for rule in self.restriction_rules:
            rule_matches = self._check_rule_patterns(query, rule)
            if rule_matches['matched']:
                matched_rules.append(rule)
                matched_patterns.extend(rule_matches['patterns'])
                max_confidence = max(max_confidence, rule_matches['confidence'])
        
        # 절차 관련 질문에 대한 특별 처리 (더욱 관대한 처리)
        if any(keyword in query.lower() for keyword in ["절차", "방법", "과정", "규정", "제도"]):
            # 절차 관련 질문은 더욱 관대하게 처리
            threshold = 0.3  # 매우 관대한 임계값
            if max_confidence > threshold * 2:  # 2배 더 관대
                restriction_level = self._determine_restriction_level(matched_rules)
                safe_response = self._generate_safe_response(matched_rules, query)
                warning_message = self._generate_warning_message(matched_rules)
            elif max_confidence < threshold * 2:  # 2배 더 관대
                restriction_level = RestrictionLevel.ALLOWED
                safe_response = None
                warning_message = None
            else:
                # 불확실한 경우도 허용
                restriction_level = RestrictionLevel.ALLOWED
                safe_response = None
                warning_message = None
            
            return ImprovedRestrictionResult(
                is_restricted=restriction_level != RestrictionLevel.ALLOWED,
                restriction_level=restriction_level,
                context_analysis=context_analysis,
                matched_rules=matched_rules,
                matched_patterns=matched_patterns,
                confidence=max_confidence,
                safe_response=safe_response,
                warning_message=warning_message,
                reasoning=f"절차 관련 질문 특별 처리 (관대한 임계값 적용): {context_analysis.indicators}",
                timestamp=datetime.now()
            )
        
        # 일반적 호기심의 경우 더 낮은 임계값으로 제한 (카테고리별 지표가 있으면 더욱 관대하게 처리)
        if has_category_indicators:
            threshold = 0.4  # 관대 (0.3 → 0.4로 조정)
        else:
            threshold = 0.5  # 기본 (유지)
        
        is_restricted = max_confidence > threshold
        
        if is_restricted:
            restriction_level = self._determine_restriction_level(matched_rules)
            safe_response = self._generate_safe_response(matched_rules, query)
            warning_message = self._generate_warning_message(matched_rules)
        else:
            restriction_level = RestrictionLevel.ALLOWED
            safe_response = None
            warning_message = None
        
        return ImprovedRestrictionResult(
            is_restricted=is_restricted,
            restriction_level=restriction_level,
            context_analysis=context_analysis,
            matched_rules=matched_rules,
            matched_patterns=matched_patterns,
            confidence=max_confidence,
            safe_response=safe_response,
            warning_message=warning_message,
            reasoning=f"일반적 호기심 맥락 분석 (카테고리 지표: {has_category_indicators}): {context_analysis.indicators}",
            timestamp=datetime.now()
        )
    
    def _check_default(self, query: str, context_analysis: ContextAnalysis) -> ImprovedRestrictionResult:
        """기본 검사"""
        matched_rules = []
        matched_patterns = []
        max_confidence = 0.0
        
        for rule in self.restriction_rules:
            rule_matches = self._check_rule_patterns(query, rule)
            if rule_matches['matched']:
                matched_rules.append(rule)
                matched_patterns.extend(rule_matches['patterns'])
                max_confidence = max(max_confidence, rule_matches['confidence'])
        
        # 기본 임계값으로 제한
        is_restricted = max_confidence > 0.5
        
        if is_restricted:
            restriction_level = self._determine_restriction_level(matched_rules)
            safe_response = self._generate_safe_response(matched_rules, query)
            warning_message = self._generate_warning_message(matched_rules)
        else:
            restriction_level = RestrictionLevel.ALLOWED
            safe_response = None
            warning_message = None
        
        return ImprovedRestrictionResult(
            is_restricted=is_restricted,
            restriction_level=restriction_level,
            context_analysis=context_analysis,
            matched_rules=matched_rules,
            matched_patterns=matched_patterns,
            confidence=max_confidence,
            safe_response=safe_response,
            warning_message=warning_message,
            reasoning=f"기본 맥락 분석: {context_analysis.indicators}",
            timestamp=datetime.now()
        )
    
    def _check_rule_patterns(self, query: str, rule: ImprovedRestrictionRule) -> Dict[str, Any]:
        """규칙 패턴 검사"""
        matched_patterns = []
        confidence = 0.0
        
        for pattern in rule.patterns:
            pattern_key = f"{rule.id}_{pattern}"
            if pattern_key in self.compiled_patterns:
                if self.compiled_patterns[pattern_key].search(query):
                    matched_patterns.append(pattern)
                    confidence += 0.4
        
        return {
            'matched': confidence > 0.0,
            'patterns': matched_patterns,
            'confidence': min(confidence, 1.0)
        }
    
    def _determine_restriction_level(self, matched_rules: List[ImprovedRestrictionRule]) -> RestrictionLevel:
        """제한 수준 결정"""
        if not matched_rules:
            return RestrictionLevel.LOW
        
        levels = [rule.level for rule in matched_rules]
        
        if RestrictionLevel.CRITICAL in levels:
            return RestrictionLevel.CRITICAL
        elif RestrictionLevel.HIGH in levels:
            return RestrictionLevel.HIGH
        elif RestrictionLevel.MEDIUM in levels:
            return RestrictionLevel.MEDIUM
        else:
            return RestrictionLevel.LOW
    
    def _generate_safe_response(self, matched_rules: List[ImprovedRestrictionRule], query: str) -> str:
        """안전한 응답 생성"""
        if not matched_rules:
            return ""
        
        primary_rule = matched_rules[0]
        
        if primary_rule.area == LegalArea.LEGAL_ADVICE:
            return "구체적인 법률 자문은 변호사와 상담하시는 것이 좋습니다. 일반적인 법률 정보나 절차는 안내드릴 수 있습니다."
        elif primary_rule.area == LegalArea.MEDICAL_LEGAL:
            return "의료사고 관련 구체적인 판단은 의료분쟁조정중재원이나 전문 의료소송 변호사와 상담하시기 바랍니다."
        elif primary_rule.area == LegalArea.CRIMINAL_CASE:
            return "형사사건 관련 구체적인 조언은 변호사와 상담하시는 것이 좋습니다. 국선변호인 신청 방법 등은 안내드릴 수 있습니다."
        elif primary_rule.area in [LegalArea.TAX_EVASION, LegalArea.CONTRACT_MANIPULATION]:
            return "해당 분야는 전문가와 상담하시는 것을 권합니다. 일반적인 법률 절차는 안내드릴 수 있습니다."
        
        return "일반적인 법률 정보를 안내드릴 수 있습니다."
    
    def _generate_warning_message(self, matched_rules: List[ImprovedRestrictionRule]) -> str:
        """경고 메시지 생성"""
        if not matched_rules:
            return ""
        
        primary_rule = matched_rules[0]
        
        warnings = {
            RestrictionLevel.CRITICAL: "⚠️ 이 질문은 법적으로 매우 민감한 내용을 포함하고 있습니다.",
            RestrictionLevel.HIGH: "⚠️ 이 질문은 법률 자문에 해당할 수 있습니다.",
            RestrictionLevel.MEDIUM: "⚠️ 이 질문은 주의가 필요한 내용을 포함하고 있습니다.",
            RestrictionLevel.LOW: "ℹ️ 이 질문은 일반적인 법률 정보 범위 내에서 답변드릴 수 있습니다."
        }
        
        return warnings.get(primary_rule.level, warnings[RestrictionLevel.LOW])
    
    def get_restriction_statistics(self) -> Dict[str, Any]:
        """제한 통계 정보"""
        return {
            "total_rules": len(self.restriction_rules),
            "areas": [area.value for area in LegalArea],
            "levels": [level.value for level in RestrictionLevel],
            "compiled_patterns": len(self.compiled_patterns),
            "compiled_exceptions": len(self.compiled_exceptions)
        }
