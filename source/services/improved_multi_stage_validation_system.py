# -*- coding: utf-8 -*-
"""
Edge Cases 처리 개선 시스템
Edge Cases 카테고리의 정확도를 향상시키기 위한 특별 처리 시스템
"""

import re
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from .moderation_gemini import GeminiModerator

logger = logging.getLogger(__name__)

class EdgeCaseType(Enum):
    """Edge Case 유형"""
    INSTITUTION_LOCATION = "institution_location"  # 기관 위치 문의
    GENERAL_PROCEDURE = "general_procedure"         # 일반 절차 문의
    CONCEPT_INQUIRY = "concept_inquiry"            # 개념 문의
    SERVICE_REQUEST = "service_request"            # 서비스 문의
    DOCUMENT_HELP = "document_help"                # 문서 작성 도움
    INFORMATION_REQUEST = "information_request"     # 정보 요청
    INQUIRY_GUIDANCE = "inquiry_guidance"          # 문의처 안내
    DISPUTE_RESOLUTION = "dispute_resolution"      # 분쟁 해결

@dataclass
class EdgeCaseDetection:
    """Edge Case 감지 결과"""
    is_edge_case: bool
    edge_case_type: Optional[EdgeCaseType]
    confidence: float
    matched_patterns: List[str]
    reasoning: str

class EdgeCaseHandler:
    """Edge Cases 특별 처리 핸들러"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Edge Cases 허용 패턴 정의
        self.edge_case_patterns = {
            EdgeCaseType.INSTITUTION_LOCATION: [
                r".*은\s*어디에\s*있나요\?$",
                r".*는\s*어디에\s*있나요\?$",
                r".*의\s*위치는\s*어디인가요\?$",
                r".*주소는\s*어디인가요\?$",
                r".*연락처는\s*어떻게\s*되나요\?$",
                # 변형/대체 표현 추가 (기관/연락정보)
                r"(법원|검찰청|경찰서|국세청|의료분쟁조정중재원).*(위치|주소|어디|찾아가는길)",
                r"(법원|검찰청|경찰서|국세청|의료분쟁조정중재원).*(전화번호|연락처)"
            ],
            EdgeCaseType.GENERAL_PROCEDURE: [
                r".*관련\s*일반적인\s*절차는\s*무엇인가요\?$",
                r".*의\s*일반적인\s*절차는\s*무엇인가요\?$",
                r".*에서\s*일반적인\s*절차는\s*무엇인가요\?$",
                r".*절차는\s*어떻게\s*진행되나요\?$",
                r".*과정은\s*어떻게\s*되나요\?$",
                # 절차/신청 방법 변형 (제도/기관 앵커 포함)
                r"(형사절차|민사절차|국선변호인|의료분쟁|세법|형사소송법|민사소송법).*(절차|과정|신청\s*방법|진행\s*방법)",
                r"(소송|재판|신청|접수).*(절차|방법|순서)"
            ],
            EdgeCaseType.CONCEPT_INQUIRY: [
                r".*에서\s*.*이\s*무엇인가요\?$",
                r".*의\s*.*이\s*무엇인가요\?$",
                r".*는\s*무엇인가요\?$",
                r".*이\s*무엇인가요\?$",
                r".*개념은\s*무엇인가요\?$",
                # 정의/개념 변형
                r".*의\s*정의는\s*무엇인가요\?$",
                r".*의\s*개념을\s*설명해 주세요$"
            ],
            EdgeCaseType.SERVICE_REQUEST: [
                r".*를\s*받고\s*싶은데\s*어떻게\s*해야\s*할까요\?$",
                r".*서비스를\s*받고\s*싶은데\s*어떻게\s*해야\s*할까요\?$",
                r".*를\s*신청하고\s*싶은데\s*어떻게\s*해야\s*할까요\?$",
                r".*를\s*받으려면\s*어떻게\s*해야\s*하나요\?$",
                # 신청/예약/접수 변형 (서비스/기관 앵커)
                r"(상담|법률상담|국선변호인|조정|중재).*(신청|예약|접수)\s*방법"
            ],
            EdgeCaseType.DOCUMENT_HELP: [
                r".*작성에\s*도움이\s*필요해요$",
                r".*작성\s*방법을\s*알려주세요$",
                r".*를\s*어떻게\s*작성하나요\?$",
                r".*작성\s*요령을\s*알려주세요$",
                r".*양식을\s*어디서\s*받을\s*수\s*있나요\?$",
                # 문서 관련 변형
                r"(소장|진단서|계약서|신청서|제출서).*(양식|서식|작성\s*방법)"
            ],
            EdgeCaseType.INFORMATION_REQUEST: [
                r".*를\s*알고\s*싶어요$",
                r".*에\s*대해\s*알고\s*싶어요$",
                r".*정보를\s*알고\s*싶어요$",
                r".*에\s*대한\s*정보를\s*알려주세요$",
                r".*에\s*대해\s*설명해주세요$",
                # 일반 정보 변형
                r".*에\s*관한\s*기본 정보를\s*알려주세요$",
                r".*의\s*개요를\s*알고\s*싶어요$"
            ],
            EdgeCaseType.INQUIRY_GUIDANCE: [
                r".*이\s*생겼는데\s*어디에\s*문의해야\s*할까요\?$",
                r".*문제가\s*생겼는데\s*어디에\s*문의해야\s*할까요\?$",
                r".*상황이\s*생겼는데\s*어디에\s*문의해야\s*할까요\?$",
                r".*에\s*대해\s*어디에\s*문의해야\s*할까요\?$",
                r".*관련해서\s*어디에\s*문의해야\s*할까요\?$",
                # 문의 안내 변형
                r"(기관|부서|창구|콜센터|고객센터).*(문의|연락|전화|상담)"
            ],
            EdgeCaseType.DISPUTE_RESOLUTION: [
                r".*을\s*해결하고\s*싶어요$",
                r".*를\s*해결하고\s*싶어요$",
                r".*분쟁을\s*해결하고\s*싶어요$",
                r".*문제를\s*해결하고\s*싶어요$",
                r".*갈등을\s*해결하고\s*싶어요$",
                # 해결 의사 변형(정보 탐색형)
                r"(분쟁|문제|갈등).*(해결\s*절차|해결\s*방법|대응\s*절차)"
            ]
        }
        
        # Edge Cases 허용 키워드
        self.edge_case_allowed_keywords = [
            # 기관명
            "의료분쟁조정중재원", "법원", "검찰청", "경찰서", "국세청", "법무부",
            "고용노동부", "보건복지부", "여성가족부", "행정안전부",
            
            # 일반적 절차 관련
            "일반적인", "일반적", "보통", "표준", "기본", "정상",
            "절차", "과정", "순서", "단계", "방법",
            
            # 개념 관련
            "개념", "정의", "의미", "내용", "특징", "성격",
            "법적", "법률", "법정", "사법", "민사", "형사",
            
            # 서비스 관련
            "서비스", "신청", "접수", "상담", "도움", "지원",
            "국선변호인", "법률상담", "법률도움", "법률지원",
            
            # 문서 관련
            "작성", "양식", "서식", "문서", "계약서", "소장", "진단서",
            "증명서", "신청서", "제출서", "보고서",
            
            # 정보 요청 관련
            "정보", "자료", "데이터", "내용", "상세", "구체적",
            "알고", "알려", "설명", "안내", "가이드",
            
            # 문의 관련
            "문의", "상담", "안내", "가이드", "도움", "지원",
            "연락", "접촉", "소통", "의사소통",
            
            # 해결 관련
            "해결", "해소", "완화", "개선", "조정", "중재",
            "분쟁", "갈등", "문제", "이슈", "사안"
        ]

        # Edge Case 음성 오버라이드(테스트 기반 상위 오탐 패턴)
        self._edgecase_negative_overrides = [
            r"저는.*얼마", r"제\s*경우.*몇\s*(년|급)", r"받을\s*수\s*있",
            r"(자백|부인).*방법"
        ]
        self._edgecase_negative_rx = [re.compile(p) for p in self._edgecase_negative_overrides]
        
        # Edge Cases 금지 키워드 (개인적 조언 지시어)
        self.edge_case_prohibited_keywords = [
            "제 경우", "저는", "내 사건", "이런 상황", "현재 상황",
            "진행 중인", "당사자", "구체적 사안", "실제 사건", "내 문제",
            "어떻게 해야", "무엇을 해야", "해야 할까요", "해야 하나요",
            "승소할까요", "패소할까요", "위자료는 얼마", "손해배상은 얼마",
            "형량은 몇 년", "자백해야 할까요", "부인해야 할까요",
            "의료과실이 있나요", "장애등급은 몇 급"
        ]
    
    def detect_edge_case(self, query: str) -> EdgeCaseDetection:
        """Edge Case 감지"""
        try:
            query_clean = query.strip()
            
            # 각 Edge Case 유형별로 패턴 매칭
            matched_patterns = []
            detected_type = None
            max_confidence = 0.0
            
            for edge_type, patterns in self.edge_case_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query_clean, re.IGNORECASE):
                        matched_patterns.append(pattern)
                        detected_type = edge_type
                        max_confidence = max(max_confidence, 0.9)  # 패턴 매칭 시 높은 신뢰도
            
            # 허용 키워드 확인
            allowed_keyword_count = sum(1 for keyword in self.edge_case_allowed_keywords if keyword in query_clean)
            prohibited_keyword_count = sum(1 for keyword in self.edge_case_prohibited_keywords if keyword in query_clean)
            
            # 허용 키워드가 있고 금지 키워드가 없으면 Edge Case로 판단
            if allowed_keyword_count > 0 and prohibited_keyword_count == 0:
                if not detected_type:
                    detected_type = EdgeCaseType.INFORMATION_REQUEST  # 기본 유형
                max_confidence = max(max_confidence, 0.7)
                matched_patterns.append(f"허용키워드_{allowed_keyword_count}개")
            
            # 금지 키워드가 있으면 Edge Case가 아님
            if prohibited_keyword_count > 0:
                return EdgeCaseDetection(
                    is_edge_case=False,
                    edge_case_type=None,
                    confidence=0.0,
                    matched_patterns=[],
                    reasoning=f"개인적 조언 지시어 {prohibited_keyword_count}개 감지로 Edge Case 아님"
                )
            
            # Edge Case 판단 보수화: 개인+결과/불법 코어/음성오버라이드 제외
            first_person_rx = re.compile(r"(제경우|제 경우|저는|내사건|내 사건|이런상황|이런 상황|현재상황|현재 상황)")
            result_numeric_rx = re.compile(r"(얼마|[0-9]+\s*(만원|억|원)|몇\s*(년|급|퍼센트)|형량|받을\s*수)")
            illegal_core_rx = re.compile(r"(탈세|세금회피|편법|법망|허점|구멍|빈틈|증거(인멸|은닉|조작|변조)|(문서|서류)(위조|변조|조작)|위장이혼|자금세탁)")
            q_plain = query_clean.replace(" ", "")

            if first_person_rx.search(q_plain) and result_numeric_rx.search(q_plain):
                return EdgeCaseDetection(
                    is_edge_case=False,
                    edge_case_type=None,
                    confidence=0.0,
                    matched_patterns=[],
                    reasoning="1인칭+결과형 수치 → Edge Case 제외"
                )

            if illegal_core_rx.search(q_plain):
                return EdgeCaseDetection(
                    is_edge_case=False,
                    edge_case_type=None,
                    confidence=0.0,
                    matched_patterns=[],
                    reasoning="불법 코어 감지 → Edge Case 제외"
                )

            if any(rx.search(q_plain) for rx in self._edgecase_negative_rx):
                return EdgeCaseDetection(
                    is_edge_case=False,
                    edge_case_type=None,
                    confidence=0.0,
                    matched_patterns=[],
                    reasoning="EdgeCase 음성 오버라이드"
                )

            # Edge Case 판단
            is_edge_case = detected_type is not None and max_confidence > 0.5
            
            reasoning = ""
            if is_edge_case:
                reasoning = f"Edge Case 감지: {detected_type.value}, 패턴: {len(matched_patterns)}개, 신뢰도: {max_confidence:.2f}"
            else:
                reasoning = "Edge Case 패턴 매칭 없음"
            
            return EdgeCaseDetection(
                is_edge_case=is_edge_case,
                edge_case_type=detected_type,
                confidence=max_confidence,
                matched_patterns=matched_patterns,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error in edge case detection: {e}")
            return EdgeCaseDetection(
                is_edge_case=False,
                edge_case_type=None,
                confidence=0.0,
                matched_patterns=[],
                reasoning=f"Edge Case 감지 오류: {str(e)}"
            )
    
    def should_allow_edge_case(self, query: str, edge_detection: EdgeCaseDetection) -> Tuple[bool, str]:
        """Edge Case 허용 여부 결정"""
        try:
            if not edge_detection.is_edge_case:
                return False, "Edge Case가 아님"
            
            # Edge Case 유형별 허용 정책
            if edge_detection.edge_case_type == EdgeCaseType.INSTITUTION_LOCATION:
                return True, "기관 위치 문의는 허용"
            
            if edge_detection.edge_case_type == EdgeCaseType.GENERAL_PROCEDURE:
                return True, "일반 절차 문의는 허용"
            
            if edge_detection.edge_case_type == EdgeCaseType.CONCEPT_INQUIRY:
                return True, "개념 문의는 허용"
            
            if edge_detection.edge_case_type == EdgeCaseType.SERVICE_REQUEST:
                return True, "서비스 문의는 허용"
            
            if edge_detection.edge_case_type == EdgeCaseType.DOCUMENT_HELP:
                return True, "문서 작성 도움은 허용"
            
            if edge_detection.edge_case_type == EdgeCaseType.INFORMATION_REQUEST:
                return True, "정보 요청은 허용"
            
            if edge_detection.edge_case_type == EdgeCaseType.INQUIRY_GUIDANCE:
                return True, "문의처 안내는 허용"
            
            if edge_detection.edge_case_type == EdgeCaseType.DISPUTE_RESOLUTION:
                return True, "분쟁 해결 문의는 허용"
            
            # 기본적으로 Edge Case는 허용
            return True, "Edge Case 기본 허용 정책"
            
        except Exception as e:
            self.logger.error(f"Error in edge case allowance decision: {e}")
            return False, f"Edge Case 허용 결정 오류: {str(e)}"
    
    def get_edge_case_context_type(self, query: str, edge_detection: EdgeCaseDetection) -> str:
        """Edge Case의 컨텍스트 유형 결정"""
        if not edge_detection.is_edge_case:
            return "personal"
        
        # Edge Case는 항상 일반적 맥락
        return "general"
    
    def get_edge_case_intent_type(self, query: str, edge_detection: EdgeCaseDetection) -> str:
        """Edge Case의 의도 유형 결정"""
        if not edge_detection.is_edge_case:
            return "advice"
        
        # Edge Case는 항상 정보 요청 의도
        return "information"

class ImprovedMultiStageValidationSystem:
    """개선된 다단계 검증 시스템 (Edge Cases 특별 처리 포함)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.edge_case_handler = EdgeCaseHandler()
        self.gemini_moderator = GeminiModerator(model_name="gemini-2.0-flash-lite")
        # 환경 플래그: 개인 법률 자문 탐지 비활성화
        self.disable_personal = os.getenv("DISABLE_PERSONAL_LEGAL_ADVICE", "0") == "1"
        
        # 기존 시스템의 설정들 (간소화)
        self.stage_weights = {
            "keyword_check": 0.25,
            "pattern_check": 0.25,
            "context_check": 0.20,
            "intent_check": 0.20,
            "final_decision": 0.10
        }
        
        # 금지 키워드 (기존 시스템과 동일)
        self.prohibited_keywords = {
            "의료사고", "의료과실", "의료진", "장애등급", "의학적인과관계",
            "자백", "부인", "증거인멸", "형량", "범죄수법", "수사대응",
            "세금회피", "탈세", "위장이혼", "가짜계약", "서류위조",
            "제경우", "저는", "내사건", "이런상황", "현재상황",
            # 추가된 개인적 조언 키워드
            "제 경우", "내 사건", "이런 상황", "현재 상황", "진행 중인",
            "당사자", "구체적 사안", "실제 사건", "내 문제",
            "어떻게 해야", "무엇을 해야", "해야 할까요", "해야 하나요",
            "승소할까요", "패소할까요", "위자료는 얼마", "손해배상은 얼마",
            "형량은 몇 년", "자백해야 할까요", "부인해야 할까요",
            "의료과실이 있나요", "장애등급은 몇 급",
            # 의료법 관련 강화
            "의료사고의 과실", "의료진이 잘못", "장애등급은 몇 급",
            "의학적 인과관계", "의료과실이 인정", "의료진의 책임",
            "의료분쟁에서 승소", "의료진이 보상", "의료사고 감정",
            "의료진의 진료과실", "의료사고로 인한 손해배상",
            "의료진이 진단을 잘못", "의료사고로 인한 치료비",
            "의료진의 치료 방법"
        }
        # 개인 조언 관련 금지 키워드 집합(플래그용)
        self._personal_prohibited_keywords = {
            "제경우", "제 경우", "저는", "내사건", "내 사건", "이런상황", "이런 상황", "현재상황", "현재 상황",
            "당사자", "구체적 사안", "실제 사건", "내 문제",
            "어떻게 해야", "무엇을 해야", "해야 할까요", "해야 하나요",
            "승소할까요", "패소할까요", "위자료는 얼마", "손해배상은 얼마",
            "형량은 몇 년", "자백해야 할까요", "부인해야 할까요",
            "의료과실이 있나요", "장애등급은 몇 급"
        }
        if self.disable_personal:
            # 개인 조언 관련 키워드를 금지 목록에서 제거
            self.prohibited_keywords = set(k for k in self.prohibited_keywords if k not in self._personal_prohibited_keywords)
        
        # 허용 키워드 (기존 시스템 + Edge Cases 키워드)
        self.allowed_keywords = {
            "일반적으로", "보통", "절차", "방법", "관련법령", "판례찾기",
            "의료분쟁조정중재원", "국선변호인신청", "변호인조력권",
            "형사절차", "형사소송법", "세법", "국세청", "세무전문가"
        }
        # Edge Cases 허용 키워드 추가
        self.allowed_keywords.update(self.edge_case_handler.edge_case_allowed_keywords)

        # ===== Illegal activity assistance precompiled resources =====
        self._allowlist_patterns = [
            r"(신청|문의|제출)\s*방법",
            r"일반적인\s*(절차|방법)",
            r"(기관|국세청|법원|검찰청|경찰서).*(문의|방문|위치|어디)",
            r"(절차|과정)\s*(은|는)\s*(무엇|어떻게)",
            r"(정의|개념)\s*(은|는)\s*무엇"
        ]
        self._illegal_core_patterns = [
            r"탈세", r"세금회피", r"편법", r"법망", r"허점", r"구멍", r"빈틈",
            r"증거(인멸|은닉|조작|변조)", r"(문서|서류)(위조|변조|조작)", r"위장이혼", r"자금세탁",
            r"세\s*금\s*(회\s*피|탈\s*세|탈\s*루|기\s*피)",
            r"증거\s*(인\s*멸|은\s*닉|조\s*작|변\s*조)"
        ]
        # 의도어를 강/약으로 분리
        self._illegal_intent_strong = [r"알려줘", r"알려주", r"가르쳐", r"하는법", r"하는방법", r"구체적으로알려"]
        self._illegal_intent_weak = [r"방법", r"요령", r"수법"]
        self._allowlist_rx = [re.compile(p) for p in self._allowlist_patterns]
        self._illegal_core_rx = [re.compile(p) for p in self._illegal_core_patterns]
        self._illegal_intent_strong_rx = [re.compile(p) for p in self._illegal_intent_strong]
        self._illegal_intent_weak_rx = [re.compile(p) for p in self._illegal_intent_weak]

        # ===== Personal advice detection helpers =====
        self._personal_pronouns_rx = re.compile(r"(제경우|제 경우|저는|내사건|내 사건|제가|저의|저한테|본인|본인의)")
        self._advice_verbs_rx = re.compile(r"(해야\s*할까요|해야\s*하나요|해야\s*합니까|어떻게\s*해야|무엇을\s*해야|가능할까요|받을\s*수\s*있나요|해줄\s*수\s*있나요)")
        self._legal_objects_rx = re.compile(r"(계약서|소송|소장|위자료|손해배상|형량|장애등급|고소|고발|합의|중재|조정)")

    def _normalize_text(self, text: str) -> str:
        try:
            t = text.lower()
            t = "".join(ch for ch in t if ch.isprintable())
            t = " ".join(t.split())
            t = t.replace(" ", "")
            return t
        except Exception:
            return text
    
    def validate(self, query: str) -> Dict[str, Any]:
        """개선된 다단계 검증 수행"""
        try:
            self.logger.info(f"Starting improved multi-stage validation for query: {query[:50]}...")
            # 0단계: Gemini 사전 모더레이션(옵션)
            mod = self.gemini_moderator.moderate(query)
            gemini_stage = {
                "stage": "gemini_moderation",
                "result": "allowed" if mod.get("allowed", True) else "restricted",
                "score": mod.get("score", 0.0),
                "reasoning": mod.get("reasoning", "Gemini moderation disabled"),
                "details": mod.get("details", {})
            }
            
            # Edge Case 감지
            edge_detection = self.edge_case_handler.detect_edge_case(query)
            
            stages = [gemini_stage]
            total_score = 0.0
            # 위험도가 높을 때(>=0.75) 즉시 제한으로 빠르게 차단 (보수화)
            if gemini_stage["score"] >= 0.75 and gemini_stage["result"] == "restricted":
                final_decision = "restricted"
                confidence = 0.85
                return {
                    "query": query,
                    "stages": stages,
                    "final_decision": final_decision,
                    "confidence": confidence,
                    "reasoning": [gemini_stage["reasoning"], "High-risk pre-moderation"],
                    "total_score": gemini_stage["score"],
                    "timestamp": datetime.now().isoformat(),
                    "edge_case_info": {
                        "is_edge_case": False,
                        "edge_case_type": None,
                        "edge_confidence": 0.0,
                        "matched_patterns": [],
                        "edge_reasoning": "pre-moderation gate"
                    }
                }
            
            # 1단계: 키워드 검사 (Edge Cases 특별 처리)
            stage1_result = self._stage1_keyword_check(query, edge_detection)
            stages.append(stage1_result)
            total_score += stage1_result["score"] * self.stage_weights["keyword_check"]
            
            # 2단계: 패턴 검사
            stage2_result = self._stage2_pattern_check(query, edge_detection)
            stages.append(stage2_result)
            total_score += stage2_result["score"] * self.stage_weights["pattern_check"]
            
            # 3단계: 맥락 검사 (Edge Cases 특별 처리)
            stage3_result = self._stage3_context_check(query, edge_detection)
            stages.append(stage3_result)
            total_score += stage3_result["score"] * self.stage_weights["context_check"]
            
            # 4단계: 의도 검사 (Edge Cases 특별 처리)
            stage4_result = self._stage4_intent_check(query, edge_detection)
            stages.append(stage4_result)
            total_score += stage4_result["score"] * self.stage_weights["intent_check"]
            
            # 5단계: 최종 결정 (Edge Cases 특별 처리)
            stage5_result = self._stage5_final_decision(query, stages, edge_detection)
            stages.append(stage5_result)
            total_score += stage5_result["score"] * self.stage_weights["final_decision"]
            
            # 최종 결정
            final_decision = self._make_final_decision(total_score, stages, edge_detection)
            # 신뢰도 간단 정규화: 단계 점수 가중합을 0.2~0.9 구간으로 리스케일
            raw_conf = min(max(total_score, 0.0), 1.0)
            confidence = 0.2 + 0.7 * raw_conf
            
            reasoning = [stage["reasoning"] for stage in stages]
            
            result = {
                "query": query,
                "stages": stages,
                "final_decision": final_decision,
                "confidence": confidence,
                "reasoning": reasoning,
                "total_score": total_score,
                "timestamp": datetime.now().isoformat(),
                "edge_case_info": {
                    "is_edge_case": edge_detection.is_edge_case,
                    "edge_case_type": edge_detection.edge_case_type.value if edge_detection.edge_case_type else None,
                    "edge_confidence": edge_detection.confidence,
                    "matched_patterns": edge_detection.matched_patterns,
                    "edge_reasoning": edge_detection.reasoning
                }
            }
            
            self.logger.info(f"Improved multi-stage validation completed. Final decision: {final_decision}, Confidence: {confidence:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in improved multi-stage validation: {e}")
            # 오류 시 안전한 기본값 반환
            return {
                "query": query,
                "stages": [],
                "final_decision": "restricted",
                "confidence": 0.9,
                "reasoning": ["검증 오류로 인한 제한"],
                "total_score": 0.9,
                "timestamp": datetime.now().isoformat(),
                "edge_case_info": {
                    "is_edge_case": False,
                    "edge_case_type": None,
                    "edge_confidence": 0.0,
                    "matched_patterns": [],
                    "edge_reasoning": f"검증 오류: {str(e)}"
                }
            }
    
    def _stage1_keyword_check(self, query: str, edge_detection: EdgeCaseDetection) -> Dict[str, Any]:
        """1단계: 키워드 검사 (Edge Cases 특별 처리)"""
        try:
            query_clean = query.replace(" ", "").replace("?", "").replace("요", "")
            
            # Edge Case인 경우 특별 처리
            if edge_detection.is_edge_case:
                should_allow, reason = self.edge_case_handler.should_allow_edge_case(query, edge_detection)
                if should_allow:
                    return {
                        "stage": "keyword_check",
                        "result": "allowed",
                        "score": 0.0,
                        "reasoning": f"Edge Case 특별 허용: {reason}",
                        "details": {"edge_case": True, "edge_type": edge_detection.edge_case_type.value if edge_detection.edge_case_type else None}
                    }
            
            # 기존 키워드 검사 로직
            allowed_matches = sum(1 for keyword in self.allowed_keywords if keyword in query_clean)
            prohibited_matches = sum(1 for keyword in self.prohibited_keywords if keyword in query_clean)
            
            # 허용 키워드가 있으면 허용
            if allowed_matches > 0:
                return {
                    "stage": "keyword_check",
                    "result": "allowed",
                    "score": 0.0,
                    "reasoning": f"허용 키워드 {allowed_matches}개 매칭으로 허용 (금지 키워드 {prohibited_matches}개도 있음)",
                    "details": {"allowed_matches": allowed_matches, "prohibited_matches": prohibited_matches}
                }
            
            # 금지 키워드가 있으면 제한
            if prohibited_matches > 0:
                score = prohibited_matches / len(self.prohibited_keywords)
                return {
                    "stage": "keyword_check",
                    "result": "restricted",
                    "score": score,
                    "reasoning": f"금지 키워드 {prohibited_matches}개 매칭으로 제한",
                    "details": {"prohibited_matches": prohibited_matches, "allowed_matches": 0}
                }
            
            # 둘 다 없으면 허용
            return {
                "stage": "keyword_check",
                "result": "allowed",
                "score": 0.0,
                "reasoning": "허용/금지 키워드 없음",
                "details": {"allowed_matches": 0, "prohibited_matches": 0}
            }
            
        except Exception as e:
            self.logger.error(f"Error in stage 1 keyword check: {e}")
            return {
                "stage": "keyword_check",
                "result": "restricted",
                "score": 0.9,
                "reasoning": f"키워드 검사 오류: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def _stage2_pattern_check(self, query: str, edge_detection: EdgeCaseDetection) -> Dict[str, Any]:
        """2단계: 패턴 검사"""
        try:
            # Edge Case인 경우 특별 처리
            if edge_detection.is_edge_case:
                return {
                    "stage": "pattern_check",
                    "result": "allowed",
                    "score": 0.0,
                    "reasoning": f"Edge Case 패턴 허용: {edge_detection.reasoning}",
                    "details": {"edge_case": True, "matched_patterns": edge_detection.matched_patterns}
                }
            
            # 강화된 패턴 검사 로직 (정규화 + 허용 우선 + 결합 규칙)
            q_norm = self._normalize_text(query)

            # 허용 패턴 우선(보수화): 허용 패턴+정보형 앵커이고 불법 코어가 없을 때만 즉시 허용
            allow_hit = any(rx.search(q_norm) for rx in self._allowlist_rx)
            illegal_core_hit_prefilter = any(rx.search(q_norm) for rx in self._illegal_core_rx)
            info_anchor_rx = re.compile(r"(무엇|무엇인가요|정의|개념|절차|과정|위치|주소|연락처|전화|어디)")
            if allow_hit and not illegal_core_hit_prefilter and info_anchor_rx.search(q_norm):
                return {
                    "stage": "pattern_check",
                    "result": "allowed",
                    "score": 0.0,
                    "reasoning": "허용 패턴+정보형 앵커",
                    "details": {"allowlist": True, "anchor": True}
                }

            prohibited_patterns = [
                # 개인적 조언 패턴 (강화)
                r"제\s*경우\s*(어떻게|무엇을|해야|해야\s*하나요|문제가|상황이)",
                r"저는\s*(어떻게|무엇을|해야|해야\s*하나요|현재|진행)",
                r"내\s*사건\s*(어떻게|무엇을|해야|해야\s*하나요|에서)",
                r"이런\s*상황\s*(어떻게|무엇을|해야|해야\s*하나요|에서)",
                r"현재\s*상황\s*(어떻게|무엇을|해야|해야\s*하나요|에서)",
                
                # 의료법 관련 패턴 (강화)
                r"의료\s*(사고|과실)\s*(의|에서)\s*(과실|책임)\s*(이|가)\s*(있나요|있나|있어요)",
                r"의료진\s*(이|가)\s*(잘못했나요|잘못했나|잘못했어요|책임)",
                r"장애\s*(등급|인등급|정도|판정)\s*(은|는)\s*(몇\s*급|몇급)",
                r"의학적\s*인과관계\s*(는|은)\s*(명확|확실|있나요|있나|있어요)",
                r"의료과실\s*(이|가)\s*(인정|될까요|있나요)",
                
                # 형사법 관련 패턴
                r"(자백|부인)\s*(하세요|하지\s*마세요|해야|하지\s*않으면|하시겠습니까)",
                # 불법/탈법 조력(강화: 공백/변형 허용)
                r"세\s*금\s*(회\s*피|탈\s*세|탈\s*루|기\s*피|줄이기|우회|빠져나가|편법)\s*(방법|수단|기법|요령)?",
                r"(법\s*망|허점|구멍|빈틈)\s*(을|을\s*이용|을\s*노려)?\s*(빠져나|피하|우회)",
                r"(증거)\s*(인\s*멸|은\s*닉|조\s*작|변\s*조)\s*(방법|요령)?",
                r"(위장)\s*(이혼|계약|전입|회사)\s*(방법|수단|요령)?",
                r"(문서|서류)\s*(위조|변조|조작)\s*(방법|수단|요령)?",
                
                # 추가 개인적 조언 패턴
                r"(승소|패소)할까요",
                r"(위자료|손해배상).*얼마",
                r"얼마나\s*받을\s*수\s*있",
                r"형량은\s*몇\s*년",
                r"(자백|부인)해야\s*할까요",
                r"의료과실이\s*있나요",
                r"장애등급은\s*몇\s*급"
            ]
            # 결합 규칙: 불법 코어 + 의도어 동시 충족 시 강한 제한
            illegal_hit = any(rx.search(q_norm) for rx in self._illegal_core_rx)
            strong_intent_hit = any(rx.search(q_norm) for rx in self._illegal_intent_strong_rx)
            weak_intent_hit = any(rx.search(q_norm) for rx in self._illegal_intent_weak_rx)
            # 방법/요령/수법 + 요청형(알려줘/자세히/구체적으로/가르쳐) 결합은 강 의도 취급
            method_request_rx = re.compile(r"(방법|요령|수법).*(알려줘|알려주|자세히|구체적으로|가르쳐)")
            if illegal_hit and method_request_rx.search(q_norm):
                return {
                    "stage": "pattern_check",
                    "result": "restricted",
                    "score": 0.9,
                    "reasoning": "불법 키워드+방법 요청형 결합(강)",
                    "details": {"illegal_core": True, "illegal_intent": "method_request"}
                }
            if illegal_hit and strong_intent_hit:
                return {
                    "stage": "pattern_check",
                    "result": "restricted",
                    "score": 0.9,
                    "reasoning": "불법 키워드+행위 유도어 결합 매칭",
                    "details": {"illegal_core": True, "illegal_intent": "strong"}
                }
            if illegal_hit and weak_intent_hit:
                return {
                    "stage": "pattern_check",
                    "result": "restricted",
                    "score": 0.75,
                    "reasoning": "불법 키워드+행위 유도어(약) 결합 매칭",
                    "details": {"illegal_core": True, "illegal_intent": "weak"}
                }

            prohibited_pattern_matches = sum(1 for pattern in prohibited_patterns if re.search(pattern, q_norm, re.IGNORECASE))
            
            if prohibited_pattern_matches > 0:
                score = prohibited_pattern_matches / len(prohibited_patterns)
                return {
                    "stage": "pattern_check",
                    "result": "restricted",
                    "score": score,
                    "reasoning": f"금지 패턴 {prohibited_pattern_matches}개 매칭으로 제한",
                    "details": {"prohibited_pattern_matches": prohibited_pattern_matches}
                }
            else:
                return {
                    "stage": "pattern_check",
                    "result": "allowed",
                    "score": 0.0,
                    "reasoning": f"금지 패턴 없음",
                    "details": {"prohibited_pattern_matches": prohibited_pattern_matches}
                }
                
        except Exception as e:
            self.logger.error(f"Error in stage 2 pattern check: {e}")
            return {
                "stage": "pattern_check",
                "result": "restricted",
                "score": 0.9,
                "reasoning": f"패턴 검사 오류: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def _stage3_context_check(self, query: str, edge_detection: EdgeCaseDetection) -> Dict[str, Any]:
        """3단계: 맥락 검사 (Edge Cases 특별 처리)"""
        try:
            # Edge Case인 경우 특별 처리
            if edge_detection.is_edge_case:
                context_type = self.edge_case_handler.get_edge_case_context_type(query, edge_detection)
                return {
                    "stage": "context_check",
                    "result": "allowed",
                    "score": 0.0,
                    "reasoning": f"Edge Case 일반적 맥락: {context_type}",
                    "details": {"edge_case": True, "context_type": context_type}
                }
            
            # 강화된 맥락 검사 로직
            if self.disable_personal:
                return {
                    "stage": "context_check",
                    "result": "allowed",
                    "score": 0.0,
                    "reasoning": "개인 법률 자문 탐지 비활성화 플래그로 맥락 검사 우회",
                    "details": {"disable_personal": True}
                }
            personal_indicators = [
                "제 경우", "저는", "내 사건", "이런 상황", "현재 상황", "진행 중인",
                "당사자", "구체적 사안", "실제 사건", "내 문제",
                "어떻게 해야", "무엇을 해야", "해야 할까요", "해야 하나요",
                # 결과형·수치형 결합(강화)
                "승소할까요", "패소할까요", "위자료는 얼마", "손해배상은 얼마",
                "얼마나 받을 수", "형량은 몇 년", "몇 년", "몇 급", "몇 퍼센트",
                "자백해야 할까요", "부인해야 할까요",
                "의료과실이 있나요", "장애등급은 몇 급"
            ]
            general_indicators = [
                "일반적으로", "보통", "절차", "방법", "관련법령", "판례",
                "의료분쟁조정중재원", "국선변호인신청", "변호인조력권",
                "형사절차", "형사소송법", "세법", "국세청", "세무전문가",
                "일반적인", "관련 일반적인", "어디에 있나요", "무엇인가요",
                "알고 싶어요", "도움이 필요해요", "해결하고 싶어요"
            ]
            
            q_lower_ctx = query.lower()
            # 개인 조언 강 신호(1인칭 + 조언동사 + 법률 객체) 결합 시 즉시 제한
            q_plain = query.replace(" ", "")
            if self._personal_pronouns_rx.search(q_plain) and self._advice_verbs_rx.search(q_lower_ctx) and self._legal_objects_rx.search(q_lower_ctx):
                return {
                    "stage": "context_check",
                    "result": "restricted",
                    "score": 0.9,
                    "reasoning": "개인 1인칭+조언동사+법률 객체 결합 감지",
                    "details": {"personal_pronoun": True, "advice_verbs": True, "legal_object": True}
                }
            personal_score = sum(1.2 for indicator in personal_indicators if indicator in q_lower_ctx)
            general_score = sum(1.0 for indicator in general_indicators if indicator in q_lower_ctx)
            
            if personal_score > general_score and personal_score > 0:
                return {
                    "stage": "context_check",
                    "result": "restricted",
                    "score": personal_score / (personal_score + general_score),
                    "reasoning": f"개인적 맥락 감지 (개인: {personal_score}, 일반: {general_score})",
                    "details": {"personal_score": personal_score, "general_score": general_score}
                }
            else:
                return {
                    "stage": "context_check",
                    "result": "allowed",
                    "score": general_score / (personal_score + general_score) if (personal_score + general_score) > 0 else 0,
                    "reasoning": f"일반적 맥락 감지 (개인: {personal_score}, 일반: {general_score})",
                    "details": {"personal_score": personal_score, "general_score": general_score}
                }
                
        except Exception as e:
            self.logger.error(f"Error in stage 3 context check: {e}")
            return {
                "stage": "context_check",
                "result": "restricted",
                "score": 0.9,
                "reasoning": f"맥락 검사 오류: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def _stage4_intent_check(self, query: str, edge_detection: EdgeCaseDetection) -> Dict[str, Any]:
        """4단계: 의도 검사 (Edge Cases 특별 처리)"""
        try:
            # Edge Case인 경우 특별 처리
            if edge_detection.is_edge_case:
                intent_type = self.edge_case_handler.get_edge_case_intent_type(query, edge_detection)
                return {
                    "stage": "intent_check",
                    "result": "allowed",
                    "score": 0.0,
                    "reasoning": f"Edge Case 정보 요청 의도: {intent_type}",
                    "details": {"edge_case": True, "intent_type": intent_type}
                }
            
            # 강화된 의도 검사 로직
            if self.disable_personal:
                return {
                    "stage": "intent_check",
                    "result": "allowed",
                    "score": 0.0,
                    "reasoning": "개인 법률 자문 탐지 비활성화 플래그로 의도 검사 우회",
                    "details": {"disable_personal": True}
                }
            advice_indicators = [
                "어떻게", "해야", "할까요", "하시겠습니까", "해야 할까요", "해야 하나요",
                "어떻게 해야", "무엇을 해야", "해야 할까요", "해야 하나요",
                "승소할까요", "패소할까요", "위자료는 얼마", "손해배상은 얼마",
                "형량은 몇 년", "자백해야 할까요", "부인해야 할까요",
                "의료과실이 있나요", "장애등급은 몇 급", "과실이 있나요",
                "잘못했나요", "책임이 있나요", "인정될까요"
            ]
            info_indicators = [
                "무엇", "어디", "언제", "왜", "어떤", "무엇인가", "어디에", "언제부터",
                "왜 그런", "어떤 경우", "어떻게 되는", "무엇인가요", "어디에 있나요",
                "일반적인", "관련 일반적인", "절차는", "방법은", "개념은",
                "알고 싶어요", "알려주세요", "설명해주세요", "안내해주세요"
            ]
            
            q_lower_int = query.lower()
            advice_score = sum(1.2 for indicator in advice_indicators if indicator in q_lower_int)
            info_score = sum(1.0 for indicator in info_indicators if indicator in q_lower_int)
            # 불법 의도어가 있는 경우 조언 점수 가산
            illegal_intent_words = ["방법", "요령", "수법", "알려줘", "가르쳐", "하는법", "하는방법"]
            if any(w in q_lower_int for w in illegal_intent_words):
                advice_score += 0.5
            
            # 1인칭 + 조언동사 + 법률 객체 조합 시 의도 점수 가산(개인 조언 강화)
            if self._personal_pronouns_rx.search(query.replace(" ","")) and self._advice_verbs_rx.search(q_lower_int) and self._legal_objects_rx.search(q_lower_int):
                advice_score += 0.8

            if advice_score > info_score and advice_score > 0:
                return {
                    "stage": "intent_check",
                    "result": "restricted",
                    "score": advice_score / (advice_score + info_score),
                    "reasoning": f"조언 요청 의도 감지 (조언: {advice_score}, 정보: {info_score})",
                    "details": {"advice_score": advice_score, "info_score": info_score}
                }
            else:
                return {
                    "stage": "intent_check",
                    "result": "allowed",
                    "score": info_score / (advice_score + info_score) if (advice_score + info_score) > 0 else 0,
                    "reasoning": f"정보 요청 의도 감지 (조언: {advice_score}, 정보: {info_score})",
                    "details": {"advice_score": advice_score, "info_score": info_score}
                }
                
        except Exception as e:
            self.logger.error(f"Error in stage 4 intent check: {e}")
            return {
                "stage": "intent_check",
                "result": "restricted",
                "score": 0.9,
                "reasoning": f"의도 검사 오류: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def _stage5_final_decision(self, query: str, previous_stages: List[Dict], edge_detection: EdgeCaseDetection) -> Dict[str, Any]:
        """5단계: 최종 결정 (Edge Cases 특별 처리)"""
        try:
            # Edge Case인 경우 특별 처리
            if edge_detection.is_edge_case:
                should_allow, reason = self.edge_case_handler.should_allow_edge_case(query, edge_detection)
                result = "allowed" if should_allow else "restricted"
                return {
                    "stage": "final_decision",
                    "result": result,
                    "score": 0.9 if should_allow else 0.1,
                    "reasoning": f"Edge Case 최종 결정: {reason}",
                    "details": {"edge_case": True, "should_allow": should_allow, "reason": reason}
                }
            
            # 기존 최종 결정 로직
            restricted_stages = sum(1 for stage in previous_stages if stage["result"] == "restricted")
            allowed_stages = sum(1 for stage in previous_stages if stage["result"] == "allowed")
            
            total_stages = len(previous_stages)
            restriction_ratio = restricted_stages / total_stages if total_stages > 0 else 0
            
            # 간단한 캘리브레이션: 경계(0.47~0.53) 구간에서 개인/의도 신호 가중 반영(완만)
            if 0.47 <= restriction_ratio <= 0.53:
                intent_stage = next((s for s in previous_stages if s["stage"] == "intent_check"), None)
                context_stage = next((s for s in previous_stages if s["stage"] == "context_check"), None)
                calibrated_ratio = restriction_ratio
                if intent_stage and intent_stage["result"] == "restricted":
                    calibrated_ratio += 0.03  # 과보정 방지
                if context_stage and context_stage["result"] == "restricted":
                    calibrated_ratio += 0.03
                restriction_ratio = max(0.0, min(1.0, calibrated_ratio))

            if restriction_ratio > 0.5:
                return {
                    "stage": "final_decision",
                    "result": "restricted",
                    "score": restriction_ratio,
                    "reasoning": f"이전 단계들에서 제한 비율 {restriction_ratio:.2f}로 제한 결정",
                    "details": {"restricted_stages": restricted_stages, "allowed_stages": allowed_stages, "restriction_ratio": restriction_ratio}
                }
            else:
                return {
                    "stage": "final_decision",
                    "result": "allowed",
                    "score": 1.0 - restriction_ratio,
                    "reasoning": f"이전 단계들에서 허용 비율 {1.0 - restriction_ratio:.2f}로 허용 결정",
                    "details": {"restricted_stages": restricted_stages, "allowed_stages": allowed_stages, "restriction_ratio": restriction_ratio}
                }
                
        except Exception as e:
            self.logger.error(f"Error in stage 5 final decision: {e}")
            return {
                "stage": "final_decision",
                "result": "restricted",
                "score": 0.9,
                "reasoning": f"최종 결정 오류: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def _make_final_decision(self, total_score: float, stages: List[Dict], edge_detection: EdgeCaseDetection) -> str:
        """최종 결정"""
        try:
            # Edge Case인 경우 특별 처리
            if edge_detection.is_edge_case:
                should_allow, _ = self.edge_case_handler.should_allow_edge_case("", edge_detection)
                return "allowed" if should_allow else "restricted"
            
            # 기존 최종 결정 로직
            restricted_stages = sum(1 for stage in stages if stage["result"] == "restricted")
            total_stages = len(stages)
            
            if total_stages == 0:
                return "restricted"  # 안전한 기본값
            
            restriction_ratio = restricted_stages / total_stages
            
            # 제한 비율이 0.5 이상이면 제한, 아니면 허용
            return "restricted" if restriction_ratio >= 0.5 else "allowed"
            
        except Exception as e:
            self.logger.error(f"Error in final decision: {e}")
            return "restricted"  # 안전한 기본값
