# -*- coding: utf-8 -*-
"""
Legal Restriction System
법률 챗봇 답변 제한 시스템
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


class LegalArea(Enum):
    """법률 영역"""
    LEGAL_ADVICE = "legal_advice"           # 법률 자문
    MEDICAL_LEGAL = "medical_legal"         # 의료 법률
    CRIMINAL_CASE = "criminal_case"         # 형사 사건
    TAX_EVASION = "tax_evasion"             # 탈세
    CONTRACT_MANIPULATION = "contract_manipulation"  # 계약 조작
    EVIDENCE_TAMPERING = "evidence_tampering"        # 증거 조작
    GENERAL_INFO = "general_info"           # 일반 정보


@dataclass
class RestrictionRule:
    """제한 규칙"""
    id: str
    area: LegalArea
    level: RestrictionLevel
    keywords: List[str]
    patterns: List[str]
    prohibited_phrases: List[str]
    safe_alternatives: List[str]
    description: str
    legal_basis: str


@dataclass
class RestrictionResult:
    """제한 검사 결과"""
    is_restricted: bool
    restriction_level: RestrictionLevel
    matched_rules: List[RestrictionRule]
    matched_keywords: List[str]
    matched_patterns: List[str]
    confidence: float
    safe_response: Optional[str]
    warning_message: Optional[str]
    timestamp: datetime


class LegalRestrictionSystem:
    """법률 제한 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.restriction_rules = self._initialize_restriction_rules()
        self.compiled_patterns = self._compile_patterns()
        
    def _initialize_restriction_rules(self) -> List[RestrictionRule]:
        """제한 규칙 초기화"""
        rules = []
        
        # 1. 법률 자문 금지 (변호사법 위반 방지)
        rules.append(RestrictionRule(
            id="legal_advice_001",
            area=LegalArea.LEGAL_ADVICE,
            level=RestrictionLevel.HIGH,
            keywords=[
                "소송하세요", "소송하지 마세요", "소송하시면", "소송하지 않으면",
                "승소", "패소", "승소율", "패소율", "승소 확률", "패소 확률",
                "위자료", "손해배상", "배상금", "금액 산정", "얼마 받을 수",
                "당신의 경우", "귀하의 경우", "이런 경우에는", "반드시 해야",
                "법적 판단", "법률적 판단", "유죄", "무죄", "과실", "책임"
            ],
            patterns=[
                r"소송\s*(하세요|하지\s*마세요|하시면|하지\s*않으면)",
                r"(승소|패소)\s*(확률|율|가능성)",
                r"(위자료|손해배상|배상금)\s*(얼마|몇\s*만|몇\s*천)",
                r"당신의\s*경우\s*(반드시|꼭|무조건)",
                r"(법적|법률적)\s*판단\s*(을|을\s*내리다)",
                r"(유죄|무죄|과실|책임)\s*(이다|입니다|라고\s*봅니다)"
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
            description="구체적인 법률 자문 및 판단 제시 금지",
            legal_basis="변호사법 제109조 (무자격자의 법률사무 취급 금지)"
        ))
        
        # 2. 의료/건강 법률 조언
        rules.append(RestrictionRule(
            id="medical_legal_001",
            area=LegalArea.MEDICAL_LEGAL,
            level=RestrictionLevel.HIGH,
            keywords=[
                "의료사고", "의료과실", "의료진 과실", "의사 과실",
                "장애등급", "장애인등급", "장애 정도", "장애 판정",
                "의학적 인과관계", "의료행위", "치료 과실",
                "의료분쟁", "의료소송", "의료감정"
            ],
            patterns=[
                r"의료\s*(사고|과실|진\s*과실|행위)",
                r"장애\s*(등급|인등급|정도|판정)",
                r"의학적\s*인과관계",
                r"(의료분쟁|의료소송|의료감정)\s*(에서|의)"
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
        
        # 3. 형사 사건 관련 민감 조언
        rules.append(RestrictionRule(
            id="criminal_case_001",
            area=LegalArea.CRIMINAL_CASE,
            level=RestrictionLevel.HIGH,
            keywords=[
                "자백하세요", "부인하세요", "자백하지 마세요", "부인하지 마세요",
                "증거 인멸", "증거 은닉", "증거 조작", "증거 숨기기",
                "형량 예측", "형량 산정", "몇 년 형", "징역",
                "범죄 수법", "범행 방법", "범죄 계획", "탈법 방법"
            ],
            patterns=[
                r"(자백|부인)\s*(하세요|하지\s*마세요)",
                r"증거\s*(인멸|은닉|조작|숨기기)",
                r"형량\s*(예측|산정|몇\s*년)",
                r"범죄\s*(수법|방법|계획)",
                r"탈법\s*(방법|수단)"
            ],
            prohibited_phrases=[
                "자백하세요", "부인하세요", "증거를 인멸하세요",
                "형량은 몇 년입니다", "이런 범죄 수법이 있습니다",
                "법망을 빠져나가는 방법", "증거를 숨기는 방법"
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
        rules.append(RestrictionRule(
            id="tax_evasion_001",
            area=LegalArea.TAX_EVASION,
            level=RestrictionLevel.CRITICAL,
            keywords=[
                "세금 회피", "탈세", "세금 줄이기", "세금 피하기",
                "위장 이혼", "위장 계약", "가짜 계약", "허위 계약",
                "증거 조작", "서류 위조", "거짓 증명", "허위 신고"
            ],
            patterns=[
                r"세금\s*(회피|탈세|줄이기|피하기)",
                r"위장\s*(이혼|계약|결혼)",
                r"(가짜|허위|거짓)\s*(계약|서류|증명|신고)",
                r"증거\s*(조작|위조|은닉)"
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
        
        # 5. 계약 조작 및 증거 조작
        rules.append(RestrictionRule(
            id="contract_manipulation_001",
            area=LegalArea.CONTRACT_MANIPULATION,
            level=RestrictionLevel.CRITICAL,
            keywords=[
                "계약 조작", "계약 위조", "계약 날조", "계약 변조",
                "증거 조작", "증거 위조", "증거 날조", "증거 변조",
                "서류 조작", "서류 위조", "서류 날조", "서류 변조",
                "법망 빠져나가기", "법적 구멍", "법률 우회"
            ],
            patterns=[
                r"(계약|증거|서류)\s*(조작|위조|날조|변조)",
                r"법망\s*빠져나가기",
                r"법적\s*구멍",
                r"법률\s*우회"
            ],
            prohibited_phrases=[
                "계약을 조작하는 방법", "증거를 위조하는 방법",
                "서류를 날조하는 방법", "법망을 빠져나가는 방법",
                "법적 구멍을 이용하는 방법", "법률을 우회하는 방법"
            ],
            safe_alternatives=[
                "계약법 관련 일반적인 절차를 안내드릴 수 있습니다",
                "증거법 관련 기본 원칙을 설명드릴 수 있습니다",
                "법무 전문가와 상담하시는 것을 권합니다"
            ],
            description="계약 및 증거 조작 방법 제시 절대 금지",
            legal_basis="형법 제233조 (문서위조)"
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
    
    def check_restrictions(self, query: str, response: str = "") -> RestrictionResult:
        """제한 사항 검사"""
        try:
            matched_rules = []
            matched_keywords = []
            matched_patterns = []
            max_confidence = 0.0
            
            # 질문과 답변 모두 검사
            text_to_check = f"{query} {response}".strip()
            
            for rule in self.restriction_rules:
                rule_matches = self._check_rule(text_to_check, rule)
                if rule_matches['matched']:
                    matched_rules.append(rule)
                    matched_keywords.extend(rule_matches['keywords'])
                    matched_patterns.extend(rule_matches['patterns'])
                    max_confidence = max(max_confidence, rule_matches['confidence'])
            
            # 제한 여부 결정
            is_restricted = len(matched_rules) > 0
            restriction_level = self._determine_restriction_level(matched_rules)
            
            # 안전한 응답 생성
            safe_response = None
            warning_message = None
            
            if is_restricted:
                safe_response = self._generate_safe_response(matched_rules, query)
                warning_message = self._generate_warning_message(matched_rules)
            
            return RestrictionResult(
                is_restricted=is_restricted,
                restriction_level=restriction_level,
                matched_rules=matched_rules,
                matched_keywords=matched_keywords,
                matched_patterns=matched_patterns,
                confidence=max_confidence,
                safe_response=safe_response,
                warning_message=warning_message,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error checking restrictions: {e}")
            return RestrictionResult(
                is_restricted=False,
                restriction_level=RestrictionLevel.LOW,
                matched_rules=[],
                matched_keywords=[],
                matched_patterns=[],
                confidence=0.0,
                safe_response=None,
                warning_message=None,
                timestamp=datetime.now()
            )
    
    def _check_rule(self, text: str, rule: RestrictionRule) -> Dict[str, Any]:
        """개별 규칙 검사"""
        matched_keywords = []
        matched_patterns = []
        confidence = 0.0
        
        # 키워드 검사
        for keyword in rule.keywords:
            if keyword.lower() in text.lower():
                matched_keywords.append(keyword)
                confidence += 0.3
        
        # 패턴 검사
        for pattern in rule.patterns:
            pattern_key = f"{rule.id}_{pattern}"
            if pattern_key in self.compiled_patterns:
                if self.compiled_patterns[pattern_key].search(text):
                    matched_patterns.append(pattern)
                    confidence += 0.4
        
        # 금지 구문 검사
        for phrase in rule.prohibited_phrases:
            if phrase.lower() in text.lower():
                confidence += 0.5
        
        return {
            'matched': confidence > 0.0,
            'keywords': matched_keywords,
            'patterns': matched_patterns,
            'confidence': min(confidence, 1.0)
        }
    
    def _determine_restriction_level(self, matched_rules: List[RestrictionRule]) -> RestrictionLevel:
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
    
    def _generate_safe_response(self, matched_rules: List[RestrictionRule], query: str) -> str:
        """안전한 응답 생성"""
        if not matched_rules:
            return ""
        
        # 가장 높은 우선순위 규칙의 안전한 대안 사용
        primary_rule = matched_rules[0]
        
        safe_responses = [
            "죄송합니다. 구체적인 법률 자문은 변호사와 상담하시는 것이 좋습니다.",
            "일반적인 법률 정보는 제공할 수 있지만, 구체적인 사안에 대한 조언은 전문가와 상담하시기 바랍니다.",
            "관련 법령이나 절차에 대한 일반적인 정보를 안내드릴 수 있습니다.",
            "더 정확한 답변을 위해서는 해당 분야 전문가와 상담하시는 것을 권합니다."
        ]
        
        # 규칙별 맞춤 응답
        if primary_rule.area == LegalArea.LEGAL_ADVICE:
            return "구체적인 법률 자문은 변호사와 상담하시는 것이 좋습니다. 일반적인 법률 정보나 절차는 안내드릴 수 있습니다."
        elif primary_rule.area == LegalArea.MEDICAL_LEGAL:
            return "의료사고 관련 구체적인 판단은 의료분쟁조정중재원이나 전문 의료소송 변호사와 상담하시기 바랍니다."
        elif primary_rule.area == LegalArea.CRIMINAL_CASE:
            return "형사사건 관련 구체적인 조언은 변호사와 상담하시는 것이 좋습니다. 국선변호인 신청 방법 등은 안내드릴 수 있습니다."
        elif primary_rule.area in [LegalArea.TAX_EVASION, LegalArea.CONTRACT_MANIPULATION]:
            return "해당 분야는 전문가와 상담하시는 것을 권합니다. 일반적인 법률 절차는 안내드릴 수 있습니다."
        
        return safe_responses[0]
    
    def _generate_warning_message(self, matched_rules: List[RestrictionRule]) -> str:
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
            "compiled_patterns": len(self.compiled_patterns)
        }
