# -*- coding: utf-8 -*-
"""
답변 구조화 향상 시스템
질문 유형별 맞춤형 답변 구조 템플릿 적용
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .legal_citation_enhancer import LegalCitationEnhancer
from .validation.legal_basis_validator import LegalBasisValidator


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
    """답변 구조화 향상 시스템 (데이터베이스 기반)"""

    def __init__(self):
        """초기화"""
        try:
            # 데이터베이스 기반 템플릿 매니저 초기화 (안전한 방식)
            try:
                from .template_database_manager import template_db_manager
                self.template_db_manager = template_db_manager
            except ImportError:
                print("Template database manager not available, using fallback")
                self.template_db_manager = None

            # 동적 템플릿 로드
            self.structure_templates = self._load_structure_templates_from_db()
            self.quality_indicators = self._load_quality_indicators_from_db()

            # 법적 근거 강화 시스템 초기화 (안전한 방식)
            try:
                self.citation_enhancer = LegalCitationEnhancer()
                self.basis_validator = LegalBasisValidator()
            except Exception as e:
                print(f"Legal citation systems not available: {e}")
                self.citation_enhancer = None
                self.basis_validator = None

            print("AnswerStructureEnhancer initialized successfully")

        except Exception as e:
            print(f"AnswerStructureEnhancer initialization failed: {e}")
            # 폴백 초기화
            self.template_db_manager = None
            self.structure_templates = self._get_fallback_templates()
            self.quality_indicators = self._get_fallback_quality_indicators()
            self.citation_enhancer = None
            self.basis_validator = None

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

    def _load_structure_templates_from_db(self) -> Dict[QuestionType, Dict[str, Any]]:
        """데이터베이스에서 구조 템플릿 로드 (안전한 방식)"""
        try:
            if not self.template_db_manager:
                return self._get_fallback_templates()

            templates = {}

            # 데이터베이스에서 모든 템플릿 조회
            db_templates = self.template_db_manager.get_all_templates()

            for question_type_str, template_data in db_templates.items():
                try:
                    question_type = QuestionType(question_type_str)
                    templates[question_type] = template_data
                except ValueError:
                    # 잘못된 질문 유형은 무시
                    continue

            # 기본 템플릿이 없으면 폴백 템플릿 생성
            if not templates:
                templates = self._get_fallback_templates()

            return templates

        except Exception as e:
            print(f"Failed to load templates from database: {e}")
            return self._get_fallback_templates()

    def _get_fallback_templates(self) -> Dict[QuestionType, Dict[str, Any]]:
        """폴백 템플릿 생성 - 템플릿 완전 제거"""
        return {
            QuestionType.GENERAL_QUESTION: {
                "title": "",
                "sections": []
            },
            QuestionType.DIVORCE_PROCEDURE: {
                "title": "",
                "sections": []
            },
            QuestionType.CONTRACT_REVIEW: {
                "title": "",
                "sections": []
            },
            QuestionType.PROCEDURE_GUIDE: {
                "title": "",
                "sections": []
            }
        }

    def _load_quality_indicators_from_db(self) -> Dict[str, List[str]]:
        """데이터베이스에서 품질 지표 로드 (안전한 방식)"""
        try:
            if not self.template_db_manager:
                return self._get_fallback_quality_indicators()

            return self.template_db_manager.get_quality_indicators()
        except Exception as e:
            print(f"Failed to load quality indicators from database: {e}")
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

    def _map_question_type(self, question_type: str, question: str) -> QuestionType:
        """데이터베이스 기반 질문 유형 매핑 (하이브리드 방식)"""
        try:
            # 데이터베이스 매니저 임포트
            from .database_keyword_manager import db_keyword_manager

            # 1단계: 명시적 질문 유형 처리
            explicit_result = self._handle_explicit_question_type(question_type)
            if explicit_result != QuestionType.GENERAL_QUESTION:
                return explicit_result

            # 2단계: 데이터베이스 기반 가중치 매핑
            db_result = self._map_with_database_keywords(question, db_keyword_manager)
            if db_result != QuestionType.GENERAL_QUESTION:
                return db_result

            # 3단계: 데이터베이스 기반 패턴 매칭
            pattern_result = self._map_with_database_patterns(question, db_keyword_manager)
            if pattern_result != QuestionType.GENERAL_QUESTION:
                return pattern_result

            # 4단계: 질문 구조 분석
            structure_result = self._analyze_question_structure(question)
            if structure_result != QuestionType.GENERAL_QUESTION:
                return structure_result

            # 최종 폴백
            return QuestionType.GENERAL_QUESTION

        except Exception as e:
            print(f"Database-based question type mapping failed: {e}")
            # 폴백: 기존 방식 사용
            return self._map_question_type_fallback(question_type, question)

    def _handle_explicit_question_type(self, question_type: str) -> QuestionType:
        """명시적 질문 유형 처리 (데이터베이스 기반)"""
        if not question_type or question_type.lower() == "general":
            return QuestionType.GENERAL_QUESTION

        # 데이터베이스에서 질문 유형 설정 조회
        try:
            config = self.template_db_manager.get_question_type_config(question_type.lower())
            if config:
                try:
                    return QuestionType(question_type.lower())
                except ValueError:
                    pass
        except Exception as e:
            print(f"Failed to get question type config: {e}")

        # 폴백: 기존 명시적 매핑
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

        return explicit_mapping.get(question_type.lower(), QuestionType.GENERAL_QUESTION)

    def _map_with_database_keywords(self, question: str, db_manager) -> QuestionType:
        """개선된 데이터베이스 기반 키워드 매핑"""
        try:
            question_lower = question.lower()
            question_words = set(question_lower.split())

            # 법조문 패턴 우선 체크
            if re.search(r'제\d+조|제\d+항|제\d+호', question):
                # 법령명과 함께 나타나는지 확인
                law_names = ['민법', '형법', '근로기준법', '상법', '행정법']
                if any(law_name in question_lower for law_name in law_names):
                    return QuestionType.LAW_INQUIRY

            # 모든 질문 유형에 대해 점수 계산
            scores = {}
            question_types = db_manager.get_all_question_types()

            for qt_info in question_types:
                question_type = qt_info['type_name']

                # 법률 문의에 특별 가중치 적용
                if question_type == "law_inquiry":
                    score = self._calculate_law_inquiry_score(question_lower, question_words, db_manager)
                else:
                    score = self._calculate_normal_score(question_type, question_lower, question_words, db_manager)

                scores[question_type] = score

            # 키워드 충돌 해결
            resolved_type = self._resolve_keyword_conflicts(question_lower, scores)

            # 최고 점수 질문 유형 반환
            if scores:
                best_match = max(scores.items(), key=lambda x: x[1])
                if best_match[1] >= 2.0:  # 최소 임계값
                    try:
                        return QuestionType(best_match[0])
                    except ValueError:
                        pass

            return QuestionType.GENERAL_QUESTION

        except Exception as e:
            print(f"Database keyword mapping error: {e}")
            return QuestionType.GENERAL_QUESTION

    def _calculate_law_inquiry_score(self, question_lower: str, question_words: set, db_manager) -> float:
        """법률 문의 전용 점수 계산 (데이터베이스 기반)"""
        score = 0.0

        # 데이터베이스에서 법률 문의 설정 조회
        try:
            config = self.template_db_manager.get_question_type_config("law_inquiry")
            if config:
                # 동적 법령명 조회
                law_names = config.get('law_names', [])
                for law_name in law_names:
                    if law_name in question_lower:
                        score += 4.0

                # 동적 질문어 조회
                question_words_special = config.get('question_words', [])
                for word in question_words_special:
                    if word in question_lower:
                        score += 3.0

                # 특별 키워드 조회
                special_keywords = config.get('special_keywords', [])
                for keyword in special_keywords:
                    if keyword in question_lower:
                        score += 2.0

                # 보너스 점수 적용
                bonus_score = config.get('bonus_score', 0.0)
                if bonus_score > 0:
                    score += bonus_score
        except Exception as e:
            print(f"Failed to get law_inquiry config: {e}")
            # 폴백: 하드코딩된 값 사용
            law_names = ['민법', '형법', '근로기준법', '상법', '행정법']
            for law_name in law_names:
                if law_name in question_lower:
                    score += 4.0

            question_words_special = ['내용', '규정', '기준', '처벌', '얼마', '몇', '언제']
            for word in question_words_special:
                if word in question_lower:
                    score += 3.0

        # 법조문 패턴 보너스
        if re.search(r'제\d+조|제\d+항|제\d+호', question_lower):
            score += 5.0

        # 일반 키워드 점수
        keywords = db_manager.get_keywords_for_type("law_inquiry", limit=50)
        for kw in keywords:
            if kw['keyword'].lower() in question_words:
                score += kw['weight_value']
            elif kw['keyword'].lower() in question_lower:
                score += kw['weight_value'] * 0.7

        return score

    def _calculate_normal_score(self, question_type: str, question_lower: str, question_words: set, db_manager) -> float:
        """일반 질문 유형 점수 계산"""
        score = 0.0

        # 각 가중치 레벨별로 키워드 조회
        high_keywords = db_manager.get_keywords_for_type(question_type, 'high', 50)
        medium_keywords = db_manager.get_keywords_for_type(question_type, 'medium', 50)
        low_keywords = db_manager.get_keywords_for_type(question_type, 'low', 50)

        # 고가중치 키워드
        for kw in high_keywords:
            if kw['keyword'].lower() in question_words:
                score += kw['weight_value']
            elif kw['keyword'].lower() in question_lower:
                score += kw['weight_value'] * 0.7  # 부분 매칭

        # 중가중치 키워드
        for kw in medium_keywords:
            if kw['keyword'].lower() in question_words:
                score += kw['weight_value']
            elif kw['keyword'].lower() in question_lower:
                score += kw['weight_value'] * 0.7  # 부분 매칭

        # 저가중치 키워드
        for kw in low_keywords:
            if kw['keyword'].lower() in question_words:
                score += kw['weight_value']
            elif kw['keyword'].lower() in question_lower:
                score += kw['weight_value'] * 0.7  # 부분 매칭

        return score

    def _resolve_keyword_conflicts(self, question_lower: str, scores: Dict[str, float]) -> str:
        """동적 키워드 충돌 해결 (데이터베이스 기반)"""
        try:
            # 데이터베이스에서 충돌 해결 규칙 조회
            conflict_rules = self.template_db_manager.get_conflict_resolution_rules()

            # 각 충돌 규칙 적용
            for conflict_type, rule in conflict_rules.items():
                target_type = rule['target_type']
                keywords = rule['keywords']
                bonus_score = rule['bonus_score']

                # 충돌하는 질문 유형이 모두 점수가 있는지 확인
                if self._should_apply_conflict_rule(conflict_type, scores):
                    if any(word in question_lower for word in keywords):
                        scores[target_type] += bonus_score

            # 기존 하드코딩된 충돌 해결 로직 (폴백)
            self._apply_fallback_conflict_rules(question_lower, scores)

        except Exception as e:
            print(f"Failed to resolve conflicts with database rules: {e}")
            # 폴백: 기존 하드코딩된 규칙 사용
            self._apply_fallback_conflict_rules(question_lower, scores)

        return max(scores.items(), key=lambda x: x[1])[0]

    def _should_apply_conflict_rule(self, conflict_type: str, scores: Dict[str, float]) -> bool:
        """충돌 규칙 적용 여부 확인"""
        # 충돌 유형에서 관련 질문 유형들 추출
        if "law_inquiry_vs_" in conflict_type:
            target_type = conflict_type.replace("law_inquiry_vs_", "")
            return "law_inquiry" in scores and target_type in scores

        return False

    def _apply_fallback_conflict_rules(self, question_lower: str, scores: Dict[str, float]):
        """폴백 충돌 해결 규칙 적용"""
        # 법률 문의 vs 계약서 검토 충돌 해결
        if "law_inquiry" in scores and "contract_review" in scores:
            if any(word in question_lower for word in ['계약서', '계약', '조항', '검토', '수정', '불리한']):
                scores["contract_review"] += 3.0

        # 법률 문의 vs 노동 분쟁 충돌 해결
        if "law_inquiry" in scores and "labor_dispute" in scores:
            if any(word in question_lower for word in ['노동', '근로', '임금', '해고', '부당해고', '임금체불', '근로시간']):
                scores["labor_dispute"] += 3.0
            elif re.search(r'제\d+조|제\d+항|제\d+호', question_lower):
                scores["law_inquiry"] += 2.0

        # 법률 문의 vs 상속 절차 충돌 해결
        if "law_inquiry" in scores and "inheritance_procedure" in scores:
            if any(word in question_lower for word in ['상속', '유산', '상속인', '상속세', '유언']):
                scores["inheritance_procedure"] += 3.0

        # 법률 문의 vs 절차 안내 충돌 해결
        if "law_inquiry" in scores and "procedure_guide" in scores:
            if any(word in question_lower for word in ['절차', '신청', '방법', '어떻게', '소액사건', '민사조정', '이혼조정']):
                scores["procedure_guide"] += 3.0

        # 법률 문의 vs 일반 질문 충돌 해결
        if "law_inquiry" in scores and "general_question" in scores:
            if any(word in question_lower for word in ['어디서', '얼마나', '비용', '상담', '변호사', '소송', '제기']):
                scores["general_question"] += 3.0

    def _map_with_database_patterns(self, question: str, db_manager) -> QuestionType:
        """데이터베이스 기반 패턴 매칭"""
        try:
            import re

            question_types = db_manager.get_all_question_types()

            for qt_info in question_types:
                question_type = qt_info['type_name']
                patterns = db_manager.get_patterns_for_type(question_type)

                for pattern_info in patterns:
                    try:
                        if pattern_info['pattern_type'] == 'regex':
                            if re.search(pattern_info['pattern'], question, re.IGNORECASE):
                                return QuestionType(question_type)
                        elif pattern_info['pattern_type'] == 'keyword':
                            if pattern_info['pattern'].lower() in question.lower():
                                return QuestionType(question_type)
                        elif pattern_info['pattern_type'] == 'phrase':
                            if pattern_info['pattern'].lower() in question.lower():
                                return QuestionType(question_type)
                    except ValueError:
                        continue  # 잘못된 QuestionType 무시

            return QuestionType.GENERAL_QUESTION

        except Exception as e:
            print(f"Database pattern mapping error: {e}")
            return QuestionType.GENERAL_QUESTION

    def _map_question_type_fallback(self, question_type: str, question: str) -> QuestionType:
        """폴백 질문 유형 매핑 (기존 방식)"""
        question_lower = question.lower()

        # 키워드 기반 매핑
        if "판례" in question or "precedent" in question_type:
            return QuestionType.PRECEDENT_SEARCH
        elif "계약서" in question or "contract" in question_type:
            return QuestionType.CONTRACT_REVIEW
        elif "이혼" in question or "divorce" in question_type:
            return QuestionType.DIVORCE_PROCEDURE
        elif "상속" in question or "inheritance" in question_type:
            return QuestionType.INHERITANCE_PROCEDURE
        elif "범죄" in question or "criminal" in question_type:
            return QuestionType.CRIMINAL_CASE
        elif "노동" in question or "labor" in question_type:
            return QuestionType.LABOR_DISPUTE
        elif "절차" in question or "procedure" in question_type:
            return QuestionType.PROCEDURE_GUIDE
        elif "용어" in question or "term" in question_type:
            return QuestionType.TERM_EXPLANATION
        elif "조언" in question or "advice" in question_type:
            return QuestionType.LEGAL_ADVICE
        elif "법률" in question or "law" in question_type:
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
            legal_basis_section = "\n\n"

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

            # legal_basis_section += "> **면책 조항:** 본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다. 구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다.\n"

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
            self.structure_templates = self._load_structure_templates_from_db()
            self.quality_indicators = self._load_quality_indicators_from_db()
            print("Templates reloaded successfully from database")
        except Exception as e:
            print(f"Failed to reload templates: {e}")

    def get_template_info(self, question_type: str) -> Dict[str, Any]:
        """템플릿 정보 조회"""
        try:
            template = self.template_db_manager.get_template(question_type)
            if template:
                return {
                    "question_type": question_type,
                    "title": template["title"],
                    "section_count": len(template["sections"]),
                    "sections": template["sections"],
                    "source": "database"
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
        """자연스러운 답변 생성 - 템플릿 완전 제거"""
        try:
            # 템플릿 패턴이 있는지 확인
            template_patterns = [
                "## 법률 문의 답변",
                "### 관련 법령",
                "### 법령 해설",
                "### 적용 사례",
                "### 주의사항",
                "*정확한 조문 번호와 내용*",
                "*쉬운 말로 풀어서 설명*",
                "*구체적 예시와 설명*"
            ]

            for pattern in template_patterns:
                if pattern in answer:
                    # 템플릿이 감지되면 자연스러운 답변으로 변환
                    return self._convert_to_natural_answer(answer, question_type)

            # 이미 자연스러운 답변이면 그대로 반환
            return answer

        except Exception as e:
            print(f"Error creating structured answer: {e}")
            return answer

    def _convert_to_natural_answer(self, template_answer: str, question_type: QuestionType) -> str:
        """템플릿 답변을 자연스러운 답변으로 변환 - Gemini API 사용"""
        try:
            # Gemini API를 사용하여 강력한 변환
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()

            prompt = f"""다음 템플릿 기반 답변을 자연스러운 대화체로 변환해주세요:

원본: {template_answer}

변환 규칙:
1. 모든 섹션 제목(##, ###)을 제거하세요
2. 모든 플레이스홀더(*정확한 조문 번호와 내용* 등)를 제거하세요
3. 하나의 연속된 자연스러운 답변으로 작성하세요
4. 친근하고 대화체로 변환하세요
5. 불필요한 면책 조항은 제거하세요
6. 마치 친한 변호사와 대화하는 것처럼 자연스럽게 작성하세요

변환된 답변:"""

            response = gemini_client.generate(prompt)
            return response.response

        except Exception as e:
            print(f"Error converting to natural answer with Gemini: {e}")
            # 폴백: 간단한 정리
            import re

            # 템플릿 패턴 제거
            cleaned = template_answer
            cleaned = re.sub(r'##\s*법률\s*문의\s*답변\s*', '', cleaned)
            cleaned = re.sub(r'###\s*관련\s*법령\s*', '', cleaned)
            cleaned = re.sub(r'###\s*법령\s*해설\s*', '', cleaned)
            cleaned = re.sub(r'###\s*적용\s*사례\s*', '', cleaned)
            cleaned = re.sub(r'###\s*주의사항\s*', '', cleaned)
            cleaned = re.sub(r'\*정확한\s*조문\s*번호와\s*내용\*', '', cleaned)
            cleaned = re.sub(r'\*쉬운\s*말로\s*풀어서\s*설명\*', '', cleaned)
            cleaned = re.sub(r'\*구체적\s*예시와\s*설명\*', '', cleaned)

            # 빈 줄 정리
            cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
            cleaned = cleaned.strip()

            # 의미있는 내용이 있으면 반환, 없으면 기본 답변
            if len(cleaned) > 50:
                return cleaned
            else:
                return f"질문에 대한 답변을 준비 중입니다. 더 구체적인 정보가 필요하시면 추가 질문을 해주세요."

    def _is_well_structured(self, answer: str) -> bool:
        """답변이 잘 구조화되어 있는지 확인"""
        # 불필요한 섹션 제목이 있는지 확인
        unwanted_patterns = [
            r'###\s*관련\s*법령\s*\n+\s*관련\s*법령\s*:',
            r'###\s*법령\s*해설\s*\n+\s*법령\s*해설\s*:',
            r'###\s*적용\s*사례\s*\n+\s*실제\s*적용\s*사례\s*:',
            r'\*쉬운\s*말로\s*풀어서\s*설명\*',
            r'\*구체적\s*예시와\s*설명\*',
            r'\*법적\s*리스크와\s*제한사항\*'
        ]

        for pattern in unwanted_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return False

        return True


# 전역 인스턴스
answer_structure_enhancer = AnswerStructureEnhancer()
