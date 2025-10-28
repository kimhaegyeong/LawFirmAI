# -*- coding: utf-8 -*-
"""
답변 구조화 향상 시스템
질문 유형별 맞춤형 답변 구조 템플릿 적용
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from .legal_basis_validator import LegalBasisValidator
from .legal_citation_enhancer import LegalCitationEnhancer


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

    def __init__(self):
        """초기화"""
        # 하드코딩된 템플릿 로드
        self.structure_templates = self._load_structure_templates()
        self.quality_indicators = self._load_quality_indicators()

        # 법적 근거 강화 시스템 초기화
        self.citation_enhancer = LegalCitationEnhancer()
        self.basis_validator = LegalBasisValidator()

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
            print(f"Error in classify_question_type: {e}")
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
            print(f"Failed to load templates: {e}")
            import traceback
            traceback.print_exc()
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
                               question: str = "", domain: str = "general") -> Dict[str, Any]:
        """답변 구조화 향상 (안전한 버전)"""
        try:
            # 입력 검증
            if not answer or not isinstance(answer, str):
                return {"error": "Invalid answer input"}

            # 질문 유형 매핑
            mapped_question_type = self._map_question_type(question_type, question)

            # 구조 템플릿 가져오기
            template = self.structure_templates.get(mapped_question_type,
                                                  self.structure_templates[QuestionType.GENERAL_QUESTION])

            if not template:
                return {"error": "Template not found"}

            # 현재 답변 분석
            analysis = self._analyze_current_structure(answer, template)

            # 구조화 개선 제안
            improvements = self._generate_structure_improvements(analysis, template)

            # 구조화된 답변 생성
            structured_answer = self._create_structured_answer(answer, template, improvements)

            # 품질 메트릭 계산
            quality_metrics = self._calculate_quality_metrics(structured_answer)

            return {
                "original_answer": answer,
                "structured_answer": structured_answer,
                "question_type": mapped_question_type.value,
                "template_used": template.get("title", "Unknown"),
                "analysis": analysis,
                "improvements": improvements,
                "quality_metrics": quality_metrics,
                "enhancement_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"답변 구조화 향상 실패: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"Question type mapping failed: {e}")
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

    # 사용되지 않는 데이터베이스 기반 메서드들 제거됨
    # 실제 구현에서는 하드코딩된 키워드 매칭 방식 사용

    def _map_question_type_fallback(self, question_type: any, question: str) -> QuestionType:
        """폴백 질문 유형 매핑 (기존 방식)"""
        question_lower = question.lower()

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
                    print(f"Section analysis error for {section.get('name', 'unknown')}: {e}")
                    continue

            # 구조 점수 계산
            analysis["structure_score"] = self._calculate_structure_score(analysis)

            # 품질 지표 분석
            analysis["quality_indicators"] = self._analyze_quality_indicators(answer)

        except Exception as e:
            print(f"Structure analysis error: {e}")
            # 기본값 유지

        return analysis

    def _extract_section_keywords(self, section: Dict[str, Any]) -> List[str]:
        """섹션별 키워드 추출"""
        keywords = []

        # 섹션 이름에서 키워드 추출
        keywords.extend(section["name"].split())

        # 템플릿에서 키워드 추출
        template_text = section.get("template", "")
        keywords.extend(re.findall(r'[\w가-힣]+', template_text))

        # 내용 가이드에서 키워드 추출
        content_guide = section.get("content_guide", "")
        keywords.extend(re.findall(r'[\w가-힣]+', content_guide))

        return list(set(keywords))  # 중복 제거

    def _calculate_section_coverage(self, answer: str, keywords: List[str]) -> float:
        """섹션 포함도 계산 (안전한 버전)"""
        try:
            if not keywords or len(keywords) == 0:
                return 0.0

            answer_lower = answer.lower()
            matched_keywords = sum(1 for keyword in keywords if keyword.lower() in answer_lower)

            return matched_keywords / len(keywords)

        except Exception as e:
            print(f"Section coverage calculation error: {e}")
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
            print(f"Structure score calculation error: {e}")
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
            print(f"Quality indicators analysis error: {e}")
            # 기본값 반환
            return {indicator_type: 0.0 for indicator_type in self.quality_indicators.keys()}

    def _generate_structure_improvements(self, analysis: Dict[str, Any],
                                       template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """구조화 개선 제안 생성"""
        improvements = []

        # 제목 추가 제안
        if not analysis["has_title"]:
            improvements.append({
                "type": "add_title",
                "priority": "high",
                "suggestion": f"답변에 제목을 추가하세요: '{template['title']}'",
                "impact": "높음"
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
                                improvements: List[Dict[str, Any]]) -> str:
        """구조화된 답변 생성"""
        structured_parts = []

        # 제목 추가
        if not re.search(r'^#+\s+', answer, re.MULTILINE):
            structured_parts.append(f"## {template['title']}")
            structured_parts.append("")

        # 기존 답변을 섹션별로 재구성
        current_answer = answer

        # 섹션별로 내용 재구성
        for section in template["sections"]:
            section_name = section["name"]
            section_template = section["template"]

            # 해당 섹션과 관련된 내용 추출
            section_content = self._extract_section_content(current_answer, section)

            if section_content:
                structured_parts.append(f"### {section_name}")
                structured_parts.append(section_template)
                structured_parts.append("")
                structured_parts.append(section_content)
                structured_parts.append("")

        # 개선 제안 적용
        for improvement in improvements:
            if improvement["type"] == "add_section":
                section_name = improvement["section_name"]
                section_template = improvement["template"]
                content_guide = improvement["content_guide"]

                structured_parts.append(f"### {section_name}")
                structured_parts.append(section_template)
                structured_parts.append("")
                structured_parts.append(f"*{content_guide}*")
                structured_parts.append("")

        return "\n".join(structured_parts)

    def _extract_section_content(self, answer: str, section: Dict[str, Any]) -> str:
        """섹션별 내용 추출"""
        # 간단한 키워드 매칭으로 관련 내용 추출
        section_keywords = self._extract_section_keywords(section)

        # 문단별로 분리하여 관련 문단 찾기
        paragraphs = answer.split('\n\n')
        relevant_paragraphs = []

        for paragraph in paragraphs:
            if any(keyword.lower() in paragraph.lower() for keyword in section_keywords):
                relevant_paragraphs.append(paragraph)

        return '\n\n'.join(relevant_paragraphs) if relevant_paragraphs else ""

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
            print(f"Quality metrics calculation error: {e}")
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
            print(f"Error enhancing answer with legal basis: {e}")
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

            # 면책 조항
            legal_basis_section += "> **면책 조항:** 본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다. 구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다.\n"

            return structured_answer + legal_basis_section

        except Exception as e:
            print(f"Error adding legal basis section: {e}")
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
            print(f"Error getting citation statistics: {e}")
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
            print(f"Error validating legal basis: {e}")
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
            print("Templates reloaded successfully")
        except Exception as e:
            print(f"Failed to reload templates: {e}")

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
            print(f"Failed to get template info: {e}")
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
            print(f"Error creating structured answer: {e}")
            return answer


# 전역 인스턴스
answer_structure_enhancer = AnswerStructureEnhancer()
