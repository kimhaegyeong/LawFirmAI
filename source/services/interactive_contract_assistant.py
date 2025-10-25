# -*- coding: utf-8 -*-
"""
Interactive Contract Assistant
대화형 계약서 작성 어시스턴트
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ContractType(Enum):
    """계약 유형"""
    SERVICE = "용역계약"
    EMPLOYMENT = "근로계약"
    REAL_ESTATE = "부동산계약"
    INTELLECTUAL_PROPERTY = "지적재산권계약"
    PARTNERSHIP = "제휴계약"
    OTHER = "기타"


class ConversationState(Enum):
    """대화 상태"""
    INITIAL = "initial"
    COLLECTING_BASIC_INFO = "collecting_basic_info"
    COLLECTING_DETAILED_INFO = "collecting_detailed_info"
    GENERATING_CONTRACT = "generating_contract"
    COMPLETED = "completed"


@dataclass
class ContractInformation:
    """계약 정보"""
    contract_type: Optional[ContractType] = None
    parties: Dict[str, str] = None  # {"client": "갑", "contractor": "을"}
    purpose: Optional[str] = None
    scope: Optional[str] = None
    payment_amount: Optional[str] = None
    payment_method: Optional[str] = None
    timeline: Optional[str] = None
    special_terms: List[str] = None
    risk_factors: List[str] = None
    
    def __post_init__(self):
        if self.parties is None:
            self.parties = {}
        if self.special_terms is None:
            self.special_terms = []
        if self.risk_factors is None:
            self.risk_factors = []


@dataclass
class ClarificationQuestion:
    """명확화 질문"""
    question: str
    question_type: str  # "multiple_choice", "text_input", "detailed_text"
    options: List[str] = None
    required: bool = True
    example: str = None
    guidance: str = None
    field_name: str = None


@dataclass
class ContractSession:
    """계약서 작성 세션"""
    session_id: str
    user_id: str
    contract_info: ContractInformation
    conversation_state: ConversationState
    collected_fields: List[str]
    missing_fields: List[str]
    created_at: datetime
    updated_at: datetime
    
    def __post_init__(self):
        if self.collected_fields is None:
            self.collected_fields = []
        if self.missing_fields is None:
            self.missing_fields = []


class InteractiveContractAssistant:
    """대화형 계약서 작성 어시스턴트"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 계약 유형별 필수 정보 정의
        self.required_fields = {
            ContractType.SERVICE: [
                "contract_type", "parties", "purpose", "scope", 
                "payment_amount", "timeline"
            ],
            ContractType.EMPLOYMENT: [
                "contract_type", "parties", "purpose", "payment_amount",
                "timeline", "special_terms"
            ],
            ContractType.REAL_ESTATE: [
                "contract_type", "parties", "purpose", "payment_amount",
                "timeline", "special_terms"
            ],
            ContractType.INTELLECTUAL_PROPERTY: [
                "contract_type", "parties", "purpose", "scope",
                "payment_amount", "special_terms"
            ],
            ContractType.PARTNERSHIP: [
                "contract_type", "parties", "purpose", "scope",
                "payment_method", "special_terms"
            ]
        }
        
        # 질문 템플릿 정의
        self.question_templates = {
            "contract_type": {
                "question": "어떤 종류의 계약서를 작성하시나요?",
                "type": "multiple_choice",
                "options": [
                    "용역계약 (디자인, 개발, 컨설팅 등)",
                    "근로계약 (직원 채용)",
                    "부동산계약 (매매, 임대차)",
                    "지적재산권계약 (저작권, 특허 등)",
                    "제휴계약 (업무 협력)",
                    "기타"
                ]
            },
            "parties": {
                "question": "계약 당사자 정보를 알려주세요.",
                "type": "detailed_text",
                "guidance": "의뢰인(갑)과 수급인(을)의 정보를 구체적으로 입력해주세요.",
                "example": "갑: (주)ABC회사, 대표: 홍길동, 주소: 서울시 강남구...\n을: 프리랜서 디자이너 김철수, 주소: 경기도 성남시..."
            },
            "purpose": {
                "question": "계약의 목적을 설명해주세요.",
                "type": "text_input",
                "guidance": "이 계약을 통해 달성하고자 하는 목표를 간단히 설명해주세요."
            },
            "scope": {
                "question": "구체적인 작업 범위나 서비스 내용을 설명해주세요.",
                "type": "detailed_text",
                "guidance": "정확한 작업 범위를 명시하여 나중에 분쟁이 생기지 않도록 해주세요.",
                "example": "웹사이트 디자인: 메인페이지 1개, 상품페이지 5개, 관리자 페이지 1개"
            },
            "payment_amount": {
                "question": "계약 금액은 얼마인가요?",
                "type": "multiple_choice",
                "options": [
                    "100만원 미만",
                    "100만원 ~ 1,000만원",
                    "1,000만원 ~ 1억원",
                    "1억원 이상"
                ]
            },
            "payment_method": {
                "question": "대금 지급 방법을 선택해주세요.",
                "type": "multiple_choice",
                "options": [
                    "일시불",
                    "계약금 + 잔금",
                    "계약금 + 중도금 + 잔금",
                    "월 단위 분할 지급",
                    "기타"
                ]
            },
            "timeline": {
                "question": "작업 기간은 어떻게 되나요?",
                "type": "text_input",
                "guidance": "시작일과 완료일, 또는 작업 기간을 명시해주세요.",
                "example": "2024년 1월 1일 ~ 2024년 3월 31일 (3개월)"
            },
            "special_terms": {
                "question": "특별히 포함하고 싶은 조항이 있나요?",
                "type": "multiple_choice",
                "options": [
                    "위약금 조항",
                    "비밀유지 조항",
                    "지적재산권 귀속 조항",
                    "계약 해지 조항",
                    "분쟁 해결 조항",
                    "없음"
                ],
                "required": False
            }
        }
        
        # 활성 세션 저장소
        self.active_sessions: Dict[str, ContractSession] = {}
        
        self.logger.info("InteractiveContractAssistant initialized")
    
    async def process_contract_query(self, message: str, session_id: str, user_id: str) -> Dict[str, Any]:
        """계약서 관련 질문 처리"""
        try:
            self.logger.info(f"Processing contract query: {message} for session {session_id}")
            
            # 1. 세션 가져오기 또는 생성
            session = await self._get_or_create_session(session_id, user_id)
            
            # 2. 메시지에서 정보 추출 시도
            extracted_info = await self._extract_information_from_message(message, session)
            
            # 3. 추출된 정보로 세션 업데이트
            if extracted_info:
                session = await self._update_session_with_info(session, extracted_info)
            
            # 4. 다음 필요한 정보 결정
            next_questions = await self._determine_next_questions(session)
            
            # 5. 응답 생성
            if next_questions:
                return await self._generate_clarification_response(session, next_questions)
            else:
                return await self._generate_contract_document(session)
                
        except Exception as e:
            self.logger.error(f"Error processing contract query: {e}")
            return {
                "response": "계약서 작성 중 오류가 발생했습니다. 다시 시도해주세요.",
                "error": str(e),
                "session_id": session_id
            }
    
    async def _get_or_create_session(self, session_id: str, user_id: str) -> ContractSession:
        """세션 가져오기 또는 생성"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.updated_at = datetime.now()
            return session
        
        # 새 세션 생성
        session = ContractSession(
            session_id=session_id,
            user_id=user_id,
            contract_info=ContractInformation(),
            conversation_state=ConversationState.INITIAL,
            collected_fields=[],
            missing_fields=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        return session
    
    async def _extract_information_from_message(self, message: str, session: ContractSession) -> Dict[str, Any]:
        """메시지에서 정보 추출"""
        extracted = {}
        message_lower = message.lower()
        
        # 계약 유형 추출
        if not session.contract_info.contract_type:
            if "용역" in message or "디자인" in message or "개발" in message or "컨설팅" in message:
                extracted["contract_type"] = ContractType.SERVICE
            elif "근로" in message or "채용" in message or "직원" in message:
                extracted["contract_type"] = ContractType.EMPLOYMENT
            elif "부동산" in message or "매매" in message or "임대" in message:
                extracted["contract_type"] = ContractType.REAL_ESTATE
            elif "저작권" in message or "특허" in message or "지적재산" in message:
                extracted["contract_type"] = ContractType.INTELLECTUAL_PROPERTY
            elif "제휴" in message or "협력" in message:
                extracted["contract_type"] = ContractType.PARTNERSHIP
        
        # 금액 정보 추출
        if not session.contract_info.payment_amount:
            import re
            amount_patterns = [
                r'(\d+)만원',
                r'(\d+)원',
                r'(\d+)천원',
                r'(\d+)억원'
            ]
            
            for pattern in amount_patterns:
                match = re.search(pattern, message)
                if match:
                    extracted["payment_amount"] = match.group(0)
                    break
        
        # 기간 정보 추출
        if not session.contract_info.timeline:
            timeline_patterns = [
                r'(\d+)개월',
                r'(\d+)일',
                r'(\d+)주',
                r'(\d+)년'
            ]
            
            for pattern in timeline_patterns:
                match = re.search(pattern, message)
                if match:
                    extracted["timeline"] = match.group(0)
                    break
        
        return extracted
    
    async def _update_session_with_info(self, session: ContractSession, extracted_info: Dict[str, Any]) -> ContractSession:
        """추출된 정보로 세션 업데이트"""
        for field, value in extracted_info.items():
            if field == "contract_type":
                session.contract_info.contract_type = value
                session.collected_fields.append("contract_type")
            elif field == "payment_amount":
                session.contract_info.payment_amount = value
                session.collected_fields.append("payment_amount")
            elif field == "timeline":
                session.contract_info.timeline = value
                session.collected_fields.append("timeline")
        
        session.updated_at = datetime.now()
        return session
    
    async def _determine_next_questions(self, session: ContractSession) -> List[ClarificationQuestion]:
        """다음에 필요한 질문들 결정"""
        questions = []
        
        # 계약 유형이 결정되지 않은 경우
        if not session.contract_info.contract_type:
            questions.append(ClarificationQuestion(
                question=self.question_templates["contract_type"]["question"],
                question_type=self.question_templates["contract_type"]["type"],
                options=self.question_templates["contract_type"]["options"],
                field_name="contract_type"
            ))
            return questions
        
        # 계약 유형별 필수 필드 확인
        required_fields = self.required_fields.get(session.contract_info.contract_type, [])
        
        for field in required_fields:
            if field not in session.collected_fields:
                template = self.question_templates.get(field, {})
                if template:
                    question = ClarificationQuestion(
                        question=template["question"],
                        question_type=template["type"],
                        options=template.get("options"),
                        required=template.get("required", True),
                        example=template.get("example"),
                        guidance=template.get("guidance"),
                        field_name=field
                    )
                    questions.append(question)
        
        return questions
    
    async def _generate_clarification_response(self, session: ContractSession, questions: List[ClarificationQuestion]) -> Dict[str, Any]:
        """명확화 질문 응답 생성"""
        if len(questions) == 1:
            question = questions[0]
            response_text = f"📋 **계약서 작성을 도와드리겠습니다!**\n\n{question.question}"
            
            if question.question_type == "multiple_choice" and question.options:
                response_text += "\n\n"
                for i, option in enumerate(question.options, 1):
                    response_text += f"○ {option}\n"
            
            if question.guidance:
                response_text += f"\n💡 **참고사항:** {question.guidance}"
            
            if question.example:
                response_text += f"\n\n📝 **예시:**\n{question.example}"
        
        else:
            response_text = f"📋 **계약서 작성을 위해 몇 가지 추가 정보가 필요합니다.**\n\n"
            for i, question in enumerate(questions[:3], 1):  # 최대 3개 질문
                response_text += f"**{i}. {question.question}**\n"
                if question.guidance:
                    response_text += f"   💡 {question.guidance}\n"
                response_text += "\n"
        
        return {
            "response": response_text,
            "questions": [asdict(q) for q in questions],
            "conversation_state": "collecting_info",
            "session_id": session.session_id,
            "contract_type": session.contract_info.contract_type.value if session.contract_info.contract_type else None,
            "collected_fields": session.collected_fields,
            "confidence": 0.9
        }
    
    async def _generate_contract_document(self, session: ContractSession) -> Dict[str, Any]:
        """계약서 문서 생성"""
        contract_info = session.contract_info
        
        # 계약서 템플릿 생성
        contract_template = await self._create_contract_template(contract_info)
        
        # 법적 리스크 분석
        risk_analysis = await self._analyze_legal_risks(contract_info)
        
        # 권장 조항 생성
        recommended_clauses = await self._generate_recommended_clauses(contract_info)
        
        response_text = f"""📄 **맞춤형 계약서가 생성되었습니다!**

## 📋 계약서 요약
- **계약 유형:** {contract_info.contract_type.value if contract_info.contract_type else '미정'}
- **계약 금액:** {contract_info.payment_amount or '미정'}
- **작업 기간:** {contract_info.timeline or '미정'}

## 📝 계약서 템플릿
{contract_template}

## ⚖️ 법적 리스크 분석
{risk_analysis}

## 💡 권장 조항
{recommended_clauses}

## ⚠️ 중요 안내
- 이 계약서는 참고용 템플릿입니다
- 실제 계약 체결 전 변호사 검토를 권장합니다
- 계약 금액이 큰 경우 전문가 상담이 필요합니다
"""
        
        # 세션 완료로 표시
        session.conversation_state = ConversationState.COMPLETED
        
        return {
            "response": response_text,
            "contract_generated": True,
            "contract_template": contract_template,
            "risk_analysis": risk_analysis,
            "recommended_clauses": recommended_clauses,
            "session_id": session.session_id,
            "confidence": 0.95
        }
    
    async def _create_contract_template(self, contract_info: ContractInformation) -> str:
        """계약서 템플릿 생성"""
        template = f"""
## 계약서

**제1조 (계약의 목적)**
본 계약은 {contract_info.purpose or '[계약 목적을 명시하세요]'}에 관한 사항을 정함을 목적으로 한다.

**제2조 (계약 당사자)**
- 갑: {contract_info.parties.get('client', '[의뢰인 정보]')}
- 을: {contract_info.parties.get('contractor', '[수급인 정보]')}

**제3조 (작업 범위)**
{contract_info.scope or '[구체적인 작업 범위를 명시하세요]'}

**제4조 (계약 기간)**
본 계약의 기간은 {contract_info.timeline or '[계약 기간을 명시하세요]'}로 한다.

**제5조 (대금 및 지급)**
1. 총 계약 금액: {contract_info.payment_amount or '[계약 금액을 명시하세요]'}
2. 지급 방법: {contract_info.payment_method or '[지급 방법을 명시하세요]'}

**제6조 (계약의 해지)**
양 당사자는 상대방이 본 계약을 위반한 경우 계약을 해지할 수 있다.

**제7조 (분쟁 해결)**
본 계약과 관련하여 분쟁이 발생할 경우 서울중앙지방법원을 관할 법원으로 한다.

**제8조 (기타)**
본 계약에 명시되지 않은 사항은 관련 법령에 따른다.

본 계약의 성립을 증명하기 위하여 계약서 2통을 작성하여 각자 1통씩 보관한다.

{datetime.now().strftime('%Y년 %m월 %d일')}

갑: _________________ (인)
을: _________________ (인)
"""
        return template.strip()
    
    async def _analyze_legal_risks(self, contract_info: ContractInformation) -> str:
        """법적 리스크 분석"""
        risks = []
        
        if contract_info.contract_type == ContractType.SERVICE:
            risks.extend([
                "• 작업 범위 불명확으로 인한 분쟁 (35%)",
                "• 대금 지급 조건 불일치 (28%)",
                "• 저작권 귀속 문제 (22%)"
            ])
        elif contract_info.contract_type == ContractType.EMPLOYMENT:
            risks.extend([
                "• 근로시간 및 임금 관련 분쟁 (40%)",
                "• 해고 관련 분쟁 (30%)",
                "• 부당해고 및 손해배상 (25%)"
            ])
        elif contract_info.contract_type == ContractType.REAL_ESTATE:
            risks.extend([
                "• 매물 하자 관련 분쟁 (45%)",
                "• 계약금 및 중도금 반환 (30%)",
                "• 등기 이전 관련 분쟁 (20%)"
            ])
        
        if not risks:
            risks = ["• 계약 조건 불명확", "• 대금 지급 조건 미명시", "• 계약 해지 조건 부재"]
        
        return "\n".join(risks)
    
    async def _generate_recommended_clauses(self, contract_info: ContractInformation) -> str:
        """권장 조항 생성"""
        clauses = []
        
        if contract_info.contract_type == ContractType.SERVICE:
            clauses.extend([
                "• 작업 범위를 구체적 수치로 명시",
                "• 대금 지급 조건을 단계별로 세분화",
                "• 저작권 귀속 시점을 명확히 규정",
                "• 수정 횟수 제한 조항 추가"
            ])
        elif contract_info.contract_type == ContractType.EMPLOYMENT:
            clauses.extend([
                "• 근로시간 및 휴게시간 명시",
                "• 임금 지급일 및 방법 명시",
                "• 해고 사유 및 절차 명시",
                "• 비밀유지 의무 조항 추가"
            ])
        elif contract_info.contract_type == ContractType.REAL_ESTATE:
            clauses.extend([
                "• 매물 하자 관련 조항 명시",
                "• 계약금 및 중도금 반환 조건",
                "• 등기 이전 의무 및 기간 명시",
                "• 인수인계 절차 명시"
            ])
        
        if not clauses:
            clauses = [
                "• 계약 목적을 구체적으로 명시",
                "• 대금 지급 조건을 명확히 규정",
                "• 계약 해지 조건 및 절차 명시",
                "• 분쟁 해결 방법 명시"
            ]
        
        return "\n".join(clauses)
    
    def get_session_info(self, session_id: str) -> Optional[ContractSession]:
        """세션 정보 조회"""
        return self.active_sessions.get(session_id)
    
    def clear_session(self, session_id: str) -> bool:
        """세션 삭제"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
