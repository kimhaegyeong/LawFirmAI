#!/usr/bin/env python3
"""
법적 근거 제시 시스템 종합 테스트
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# 서비스 임포트
from source.services.answer_structure_enhancer import (
    AnswerStructureEnhancer,
    QuestionType,
)
from source.services.legal_basis_integration_service import LegalBasisIntegrationService
from source.services.legal_basis_validator import LegalBasisValidator
from source.services.legal_citation_enhancer import LegalCitationEnhancer


class LegalBasisSystemTester:
    """법적 근거 제시 시스템 테스터"""

    def __init__(self):
        self.citation_enhancer = LegalCitationEnhancer()
        self.basis_validator = LegalBasisValidator()
        self.structure_enhancer = AnswerStructureEnhancer()
        self.integration_service = LegalBasisIntegrationService()

        # 테스트 데이터
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self) -> Dict[str, Any]:
        """테스트 케이스 로드 - 확장된 버전"""
        return {
            "law_inquiry": {
                "query": "계약 해지 시 손해배상 범위는 어떻게 되나요?",
                "answer": """
                계약 해지 시 손해배상 범위는 민법 제543조와 제544조에 따라 결정됩니다.

                제543조에 따르면, 계약의 해지로 인한 손해배상은 계약 이행으로 얻을 수 있었던 이익의 상실을 의미합니다.
                제544조에서는 해지로 인한 손해배상의 범위를 제한하고 있습니다.

                대법원 2023다12345 판례에서는 계약 해지 시 손해배상 범위를 구체적으로 명시하고 있습니다.
                """,
                "expected_citations": ["제543조", "제544조", "2023다12345"]
            },
            "precedent_search": {
                "query": "부동산 매매계약에서 하자담보책임 관련 판례를 찾아주세요",
                "answer": """
                부동산 매매계약에서 하자담보책임과 관련된 주요 판례는 다음과 같습니다:

                1. 대법원 2022다56789 판례: 부동산의 하자로 인한 손해배상 범위
                2. 서울고등법원 2023나12345 판례: 하자 발견 시 계약 해지 요건

                민법 제580조와 제581조가 하자담보책임의 법적 근거가 됩니다.
                """,
                "expected_citations": ["2022다56789", "2023나12345", "제580조", "제581조"]
            },
            "legal_advice": {
                "query": "임금 체불 시 어떻게 대응해야 하나요?",
                "answer": """
                임금 체불 시 다음과 같은 방법으로 대응할 수 있습니다:

                1. 노동위원회에 진정 제기 (근로기준법 제28조)
                2. 법원에 임금 지급 명령 신청 (민사집행법 제226조)
                3. 형사고발 (근로기준법 제109조)

                대법원 2021다98765 판례에서는 임금 체불의 법적 효과를 명확히 하고 있습니다.
                """,
                "expected_citations": ["제28조", "제226조", "제109조", "2021다98765"]
            },
            "term_explanation": {
                "query": "손해배상의 의미를 설명해주세요",
                "answer": """
                손해배상은 민법 제750조에 정의된 개념으로, 불법행위로 인한 손해를 배상하는 것을 의미합니다.

                손해배상의 요건:
                1. 불법행위의 존재
                2. 손해의 발생
                3. 인과관계의 존재
                4. 고의 또는 과실

                대법원 2020다54321 판례에서는 손해배상의 범위를 구체적으로 설명하고 있습니다.
                """,
                "expected_citations": ["제750조", "2020다54321"]
            },
            "contract_review": {
                "query": "이 계약서의 불리한 조항들을 검토해주세요",
                "answer": """
                계약서 검토 결과 다음과 같은 불리한 조항들이 발견되었습니다:

                1. 민법 제104조에 따른 불공정한 계약 조항
                2. 약관의 규제에 관한 법률 제6조 위반 가능성
                3. 대법원 2019다23456 판례에서 유사한 조항이 무효로 판단된 사례

                특히 제3조의 손해배상 제한 조항은 민법 제398조에 위배될 수 있습니다.
                """,
                "expected_citations": ["제104조", "제6조", "2019다23456", "제398조"]
            },
            "divorce_procedure": {
                "query": "협의이혼 절차를 안내해주세요",
                "answer": """
                협의이혼 절차는 다음과 같습니다:

                1. 협의이혼서 작성 (가족관계의 등록 등에 관한 법률 제80조)
                2. 가정법원에 협의이혼 확인신청 (가사소송법 제2조)
                3. 대법원 2018다78901 판례에 따른 협의이혼 요건 확인

                민법 제836조에서 협의이혼의 요건을 명시하고 있습니다.
                """,
                "expected_citations": ["제80조", "제2조", "2018다78901", "제836조"]
            },
            "inheritance_procedure": {
                "query": "상속 포기 절차는 어떻게 되나요?",
                "answer": """
                상속 포기 절차는 다음과 같습니다:

                1. 상속 개시일로부터 3개월 내 신청 (민법 제1019조)
                2. 가정법원에 상속포기신고서 제출 (가사소송법 제52조)
                3. 대법원 2017다45678 판례에서 상속포기 요건을 명확히 함

                상속세법 제65조에 따른 상속세 신고도 필요합니다.
                """,
                "expected_citations": ["제1019조", "제52조", "2017다45678", "제65조"]
            },
            "criminal_case": {
                "query": "사기죄의 구성요건을 설명해주세요",
                "answer": """
                사기죄의 구성요건은 형법 제347조에 명시되어 있습니다:

                1. 기망행위 (사람을 기망하는 행위)
                2. 착오유발 (상대방으로 하여금 착오에 빠뜨리는 것)
                3. 재산상 이익 (재산상 이익을 취득하거나 제3자로 하여금 취득하게 하는 것)

                대법원 2016다34567 판례에서 사기죄의 기망행위를 구체적으로 해석했습니다.
                """,
                "expected_citations": ["제347조", "2016다34567"]
            },
            "labor_dispute": {
                "query": "부당해고 구제절차를 안내해주세요",
                "answer": """
                부당해고 구제절차는 다음과 같습니다:

                1. 노동위원회에 부당해고 구제신청 (근로기준법 제28조)
                2. 구제신청서 제출 후 조정절차 진행
                3. 대법원 2015다23456 판례에서 부당해고의 요건을 명확히 함

                근로기준법 제23조에서 부당해고를 금지하고 있습니다.
                """,
                "expected_citations": ["제28조", "2015다23456", "제23조"]
            },
            "procedure_guide": {
                "query": "소액사건심판절차는 어떻게 진행되나요?",
                "answer": """
                소액사건심판절차는 다음과 같이 진행됩니다:

                1. 소액사건심판법 제1조에 따른 소액사건의 범위 확인
                2. 관할법원에 소장 제출 (민사소송법 제248조)
                3. 대법원 2014다12345 판례에서 소액사건심판의 특례를 인정

                소액사건심판법 제15조에서 간이절차를 규정하고 있습니다.
                """,
                "expected_citations": ["제1조", "제248조", "2014다12345", "제15조"]
            },
            "constitutional_law": {
                "query": "헌법 제10조의 인간의 존엄과 가치에 대해 설명해주세요",
                "answer": """
                헌법 제10조는 인간의 존엄과 가치를 명시하고 있습니다:

                1. 모든 국민은 인간으로서의 존엄과 가치를 가진다
                2. 헌법재판소 2013헌마456 판결에서 인간의 존엄성을 구체적으로 해석
                3. 국가는 개인이 가지는 불가침의 기본적 인권을 확인하고 이를 보장할 의무를 진다

                헌법 제37조에서 기본권의 제한과 그 한계를 규정하고 있습니다.
                """,
                "expected_citations": ["제10조", "2013헌마456", "제37조"]
            },
            "administrative_law": {
                "query": "행정처분의 취소요건을 설명해주세요",
                "answer": """
                행정처분의 취소요건은 다음과 같습니다:

                1. 행정절차법 제1조에 따른 행정절차 준수
                2. 행정소송법 제1조에 따른 행정소송 제기 가능
                3. 대법원 2012다78901 판례에서 행정처분 취소의 요건을 명확히 함

                행정절차법 제20조에서 행정처분의 근거와 이유를 제시할 의무를 규정하고 있습니다.
                """,
                "expected_citations": ["제1조", "제1조", "2012다78901", "제20조"]
            },
            "commercial_law": {
                "query": "주식회사의 이사 책임에 대해 설명해주세요",
                "answer": """
                주식회사의 이사 책임은 상법에 명시되어 있습니다:

                1. 상법 제382조에서 이사의 선임과 임기 규정
                2. 상법 제399조에서 이사의 의무와 책임 규정
                3. 대법원 2011다45678 판례에서 이사의 주의의무를 구체적으로 해석

                상법 제401조에서 이사의 손해배상책임을 규정하고 있습니다.
                """,
                "expected_citations": ["제382조", "제399조", "2011다45678", "제401조"]
            },
            "tax_law": {
                "query": "소득세 신고 절차를 안내해주세요",
                "answer": """
                소득세 신고 절차는 다음과 같습니다:

                1. 소득세법 제70조에 따른 신고의무자 확인
                2. 소득세법 제71조에 따른 신고기한 준수
                3. 대법원 2010다34567 판례에서 소득세 신고의무를 명확히 함

                국세기본법 제25조에서 납세의무의 성립과 확정을 규정하고 있습니다.
                """,
                "expected_citations": ["제70조", "제71조", "2010다34567", "제25조"]
            },
            "intellectual_property": {
                "query": "특허권 침해 시 구제방법을 설명해주세요",
                "answer": """
                특허권 침해 시 구제방법은 다음과 같습니다:

                1. 특허법 제126조에 따른 침해금지청구권
                2. 특허법 제128조에 따른 손해배상청구권
                3. 대법원 2009다23456 판례에서 특허권 침해의 요건을 명확히 함

                특허법 제130조에서 침해행위의 추정을 규정하고 있습니다.
                """,
                "expected_citations": ["제126조", "제128조", "2009다23456", "제130조"]
            },
            "family_law": {
                "query": "친생자 추정의 원칙에 대해 설명해주세요",
                "answer": """
                친생자 추정의 원칙은 민법에 명시되어 있습니다:

                1. 민법 제844조에서 혼인 중의 잉태로 출생한 자는 부의 자로 추정
                2. 민법 제845조에서 친생자 부인소송 규정
                3. 대법원 2008다12345 판례에서 친생자 추정의 요건을 구체적으로 해석

                민법 제846조에서 친생자 인지의 효력을 규정하고 있습니다.
                """,
                "expected_citations": ["제844조", "제845조", "2008다12345", "제846조"]
            },
            "real_estate_law": {
                "query": "부동산 등기 절차를 안내해주세요",
                "answer": """
                부동산 등기 절차는 다음과 같습니다:

                1. 부동산등기법 제1조에 따른 등기의 효력 규정
                2. 부동산등기법 제40조에 따른 등기신청 절차
                3. 대법원 2007다78901 판례에서 등기의 효력을 구체적으로 해석

                부동산등기법 제55조에서 등기원인을 규정하고 있습니다.
                """,
                "expected_citations": ["제1조", "제40조", "2007다78901", "제55조"]
            },
            "complex_legal_question": {
                "query": "회사가 직원을 부당해고한 후, 해당 직원이 회사에 대해 손해배상을 청구할 수 있는지, 그리고 이 경우 민법 제750조와 근로기준법 제23조가 어떻게 적용되는지 설명해주세요",
                "answer": """
                이는 복합적인 법적 문제로 여러 법령이 적용됩니다:

                1. 근로기준법 제23조에서 부당해고를 금지하고 있습니다.
                2. 민법 제750조에서 불법행위로 인한 손해배상을 규정하고 있습니다.
                3. 대법원 2006다45678 판례에서 부당해고와 손해배상의 관계를 명확히 했습니다.
                4. 근로기준법 제28조에서 부당해고 구제절차를 규정하고 있습니다.

                따라서 부당해고된 직원은 근로기준법상 구제절차와 민법상 손해배상을 모두 청구할 수 있습니다.
                """,
                "expected_citations": ["제23조", "제750조", "2006다45678", "제28조"]
            },
            "no_citations": {
                "query": "법률 상담은 어디서 받을 수 있나요?",
                "answer": """
                법률 상담은 다음과 같은 곳에서 받을 수 있습니다:

                1. 대한변호사협회 법률구조재단
                2. 각 지방법원의 법률상담소
                3. 시군구청의 법률상담실
                4. 온라인 법률상담 서비스

                대부분 무료 또는 저렴한 비용으로 상담을 받을 수 있습니다.
                """,
                "expected_citations": []
            },
            "mixed_citations": {
                "query": "계약서에 명시된 위약금 조항이 민법 제398조와 약관의 규제에 관한 법률 제6조에 위배되는지, 그리고 대법원 2005다12345 판례와 서울고등법원 2004나56789 판례는 어떻게 판단했는지 알려주세요",
                "answer": """
                위약금 조항의 유효성은 여러 법령과 판례를 종합적으로 검토해야 합니다:

                1. 민법 제398조에서 위약금의 성질과 효력을 규정하고 있습니다.
                2. 약관의 규제에 관한 법률 제6조에서 불공정한 약관을 규제하고 있습니다.
                3. 대법원 2005다12345 판례에서 위약금 조항의 유효성 요건을 명확히 했습니다.
                4. 서울고등법원 2004나56789 판례에서 유사한 사안을 다루었습니다.
                5. 근로기준법 제15조에서 근로계약의 위약금을 제한하고 있습니다.

                각 사안의 구체적 내용에 따라 판단이 달라질 수 있습니다.
                """,
                "expected_citations": ["제398조", "제6조", "2005다12345", "2004나56789", "제15조"]
            },

            # === 추가 법률 분야 테스트 ===
            "environmental_law": {
                "query": "환경오염으로 인한 손해배상 책임에 대해 설명해주세요.",
                "answer": """
                환경오염으로 인한 손해배상 책임은 환경정책기본법 제31조와 환경오염피해 배상책임법에 규정되어 있습니다:

                1. 무과실 책임주의 원칙 적용
                2. 원인자 불명 시 공동책임
                3. 환경오염피해 배상책임보험 가입 의무

                대법원 2022다12345 판례에서 환경오염 손해배상의 구체적인 범위를 판시하고 있습니다.
                """,
                "expected_citations": ["제31조", "2022다12345"]
            },
            "medical_law": {
                "query": "의료사고 발생 시 의사의 책임에 대해 설명해주세요.",
                "answer": """
                의료사고 발생 시 의사의 책임은 의료법 제27조와 민법 제750조에 따라 다음과 같이 규정됩니다:

                1. 의료진의 주의의무 위반
                2. 환자에 대한 설명의무 위반
                3. 의료기록 작성 및 보관의무

                대법원 2021다12345 판례에서 의료사고의 책임 인정 기준을 구체적으로 제시하고 있습니다.
                """,
                "expected_citations": ["제27조", "제750조", "2021다12345"]
            },
            "cyber_law": {
                "query": "개인정보보호법 위반 시 처벌에 대해 설명해주세요.",
                "answer": """
                개인정보보호법 위반 시 처벌은 개인정보보호법 제71조부터 제76조까지에서 정하고 있습니다:

                1. 개인정보 처리방침 위반: 3년 이하의 징역 또는 3천만원 이하의 벌금
                2. 개인정보 유출 신고 의무 위반: 1년 이하의 징역 또는 1천만원 이하의 벌금
                3. 개인정보보호위원회의 시정명령 위반

                대법원 2022도12345 판례에서 개인정보보호법 위반의 구체적인 처벌 기준을 제시하고 있습니다.
                """,
                "expected_citations": ["제71조", "제76조", "2022도12345"]
            },
            "international_law": {
                "query": "국제계약에서 준거법 선택에 대해 설명해주세요.",
                "answer": """
                국제계약에서 준거법 선택은 국제사법 제25조부터 제28조까지에서 규정하고 있습니다:

                1. 당사자의 자유로운 준거법 선택
                2. 준거법 선택이 없는 경우의 결정 기준
                3. 강행법규의 적용 범위

                대법원 2021다12345 판례에서 국제계약의 준거법 결정 기준을 구체적으로 제시하고 있습니다.
                """,
                "expected_citations": ["제25조", "제28조", "2021다12345"]
            },
            "banking_law": {
                "query": "은행의 대출계약에서 이자율 제한에 대해 설명해주세요.",
                "answer": """
                은행의 대출계약에서 이자율 제한은 금융소비자보호법 제19조와 약관의 규제에 관한 법률 제6조에 규정되어 있습니다:

                1. 연 20%를 초과하는 이자율의 무효
                2. 복리 계산의 금지
                3. 선이자 공제의 금지

                대법원 2020다12345 판례에서 대출 이자율 제한의 구체적인 적용을 판시하고 있습니다.
                """,
                "expected_citations": ["제19조", "제6조", "2020다12345"]
            },
            "insurance_law": {
                "query": "보험계약에서 보험사기 방지를 위한 조치에 대해 설명해주세요.",
                "answer": """
                보험계약에서 보험사기 방지를 위한 조치는 보험업법 제108조와 상법 제644조에 규정되어 있습니다:

                1. 보험사기 신고 의무
                2. 보험사기 조사권
                3. 보험금 지급 거부 사유

                대법원 2021다12345 판례에서 보험사기 방지 조치의 구체적인 범위를 판시하고 있습니다.
                """,
                "expected_citations": ["제108조", "제644조", "2021다12345"]
            },
            "competition_law": {
                "query": "공정거래법 위반 시 제재에 대해 설명해주세요.",
                "answer": """
                공정거래법 위반 시 제재는 공정거래법 제22조부터 제25조까지에서 정하고 있습니다:

                1. 시정명령 및 과징금 부과
                2. 과징금의 계산 기준
                3. 시정명령 위반 시 형사처벌

                대법원 2022다12345 판례에서 공정거래법 위반 제재의 구체적인 기준을 제시하고 있습니다.
                """,
                "expected_citations": ["제22조", "제25조", "2022다12345"]
            },
            "consumer_law": {
                "query": "소비자기본법에 따른 소비자 권리에 대해 설명해주세요.",
                "answer": """
                소비자기본법에 따른 소비자 권리는 소비자기본법 제5조부터 제8조까지에서 규정하고 있습니다:

                1. 안전한 상품과 서비스를 받을 권리
                2. 정확한 정보를 받을 권리
                3. 선택할 권리
                4. 피해를 받았을 때 보상받을 권리

                대법원 2021다12345 판례에서 소비자 권리의 구체적인 보호 범위를 판시하고 있습니다.
                """,
                "expected_citations": ["제5조", "제8조", "2021다12345"]
            },
            "education_law": {
                "query": "교육기본법에 따른 교육의 권리에 대해 설명해주세요.",
                "answer": """
                교육기본법에 따른 교육의 권리는 교육기본법 제3조와 헌법 제31조에서 규정하고 있습니다:

                1. 모든 국민의 교육받을 권리
                2. 교육의 기회 균등
                3. 교육의 자주성과 전문성 보장

                헌법재판소 2019헌마123 결정에서 교육권의 구체적인 보호 범위를 제시하고 있습니다.
                """,
                "expected_citations": ["제3조", "제31조", "2019헌마123"]
            },

            # === 엣지 케이스 및 특수 상황 테스트 ===
            "ambiguous_citations": {
                "query": "제123조에 따라 처벌한다고 되어 있는데, 이게 형법인지 민법인지 불분명합니다.",
                "answer": """
                제123조는 여러 법률에 존재할 수 있으므로 정확한 법률명을 확인해야 합니다:

                1. 형법 제123조: 공문서 위조
                2. 민법 제123조: 법률행위의 해석
                3. 상법 제123조: 주식회사의 설립

                대법원 2020다12345 판례에서는 법조문 해석 시 법률명의 중요성을 강조하고 있습니다.
                """,
                "expected_citations": ["제123조", "제123조", "제123조", "2020다12345"]
            },
            "outdated_citations": {
                "query": "구 민법 제123조에 따르면 어떻게 되나요?",
                "answer": """
                구 민법 제123조는 현재 개정되어 새로운 조문으로 변경되었습니다:

                1. 구 민법 제123조: 구체적인 내용
                2. 현행 민법 제124조: 개정된 내용
                3. 법률 개정의 소급효 문제

                대법원 2019다12345 판례에서는 법률 개정의 소급효 적용 범위를 구체적으로 판시하고 있습니다.
                """,
                "expected_citations": ["제123조", "제124조", "2019다12345"]
            },
            "foreign_law_citations": {
                "query": "미국 법률과 한국 법률의 차이점은 무엇인가요?",
                "answer": """
                미국 법률과 한국 법률의 주요 차이점은 다음과 같습니다:

                1. 법체계: 판례법 vs 성문법
                2. 법원구조: 연방법원 vs 단일법원
                3. 배심제도: 배심제 vs 전문법관제

                국제사법 제1조에서 외국법의 적용 범위를 규정하고 있으며,
                대법원 2020다12345 판례에서 외국법 적용의 구체적인 기준을 제시하고 있습니다.
                """,
                "expected_citations": ["제1조", "2020다12345"]
            },
            "multiple_law_conflicts": {
                "query": "민법과 상법이 충돌할 때 어떤 법이 우선되나요?",
                "answer": """
                민법과 상법의 충돌 시 적용 원칙은 다음과 같습니다:

                1. 특별법 우선의 원칙: 상법이 민법에 우선
                2. 상법 제1조: 상사에 관하여 상법에 특별한 규정이 없으면 민법의 규정을 준용
                3. 법률의 해석 원칙

                대법원 2021다12345 판례에서 민법과 상법의 충돌 해결 기준을 구체적으로 제시하고 있습니다.
                """,
                "expected_citations": ["제1조", "2021다12345"]
            },
            "constitutional_challenge": {
                "query": "헌법재판소에서 위헌 결정을 받은 법률은 어떻게 되나요?",
                "answer": """
                헌법재판소의 위헌 결정은 헌법재판소법 제47조에 따라 다음과 같은 효력을 가집니다:

                1. 법률의 효력 상실
                2. 소급효의 문제
                3. 관련 판례의 변경

                헌법재판소 2019헌마123 결정에서 위헌 결정의 구체적인 효력 범위를 제시하고 있습니다.
                """,
                "expected_citations": ["제47조", "2019헌마123"]
            },
            "emergency_law": {
                "query": "긴급상황에서 발령된 행정명령의 효력은 어떻게 되나요?",
                "answer": """
                긴급상황에서 발령된 행정명령의 효력은 행정절차법 제3조와 긴급명령권에 관한 법률에서 규정하고 있습니다:

                1. 긴급명령의 발령 요건
                2. 국회의 사후 승인
                3. 명령의 효력 기간

                대법원 2020다12345 판례에서 긴급명령의 구체적인 효력 범위를 판시하고 있습니다.
                """,
                "expected_citations": ["제3조", "2020다12345"]
            },
            "international_treaty": {
                "query": "국제조약이 국내법과 충돌할 때 어떻게 해결되나요?",
                "answer": """
                국제조약과 국내법의 충돌 시 해결 원칙은 다음과 같습니다:

                1. 조약의 국내법적 효력
                2. 국제법 우선의 원칙
                3. 헌법 제6조의 조약 승인

                대법원 2021다12345 판례에서 국제조약과 국내법의 충돌 해결 기준을 구체적으로 제시하고 있습니다.
                """,
                "expected_citations": ["제6조", "2021다12345"]
            },
            "customary_law": {
                "query": "관습법의 효력과 적용 범위에 대해 설명해주세요.",
                "answer": """
                관습법의 효력과 적용 범위는 민법 제1조와 대법원 판례에서 규정하고 있습니다:

                1. 법령에 특별한 규정이 없는 경우 관습법 적용
                2. 관습법의 성립 요건
                3. 관습법과 성문법의 관계

                대법원 2020다12345 판례에서 관습법의 구체적인 적용 기준을 제시하고 있습니다.
                """,
                "expected_citations": ["제1조", "2020다12345"]
            },
            "legal_fiction": {
                "query": "법률상 추정과 법률상 가정의 차이점은 무엇인가요?",
                "answer": """
                법률상 추정과 법률상 가정의 차이점은 다음과 같습니다:

                1. 법률상 추정: 반증이 가능한 추정
                2. 법률상 가정: 반증이 불가능한 가정
                3. 입증책임의 전환

                대법원 2021다12345 판례에서 법률상 추정과 가정의 구체적인 적용을 판시하고 있습니다.
                """,
                "expected_citations": ["2021다12345"]
            },

            # === 성능 및 부하 테스트 ===
            "long_text": {
                "query": "매우 긴 법률 문서에서 여러 조문을 인용하는 경우",
                "answer": """
                민법 제1조는 법원의 법률 해석에 관한 기본 원칙을 규정하고 있으며, 제2조는 법률의 효력에 관하여 규정하고 있습니다.
                제3조는 법률행위의 해석에 관하여 규정하고 있으며, 제4조는 법률행위의 성립과 효력에 관하여 규정하고 있습니다.
                제5조는 법률행위의 무효와 취소에 관하여 규정하고 있으며, 제6조는 법률행위의 조건과 기한에 관하여 규정하고 있습니다.
                제7조는 법률행위의 대리에 관하여 규정하고 있으며, 제8조는 법률행위의 대표에 관하여 규정하고 있습니다.
                제9조는 법률행위의 공시에 관하여 규정하고 있으며, 제10조는 법률행위의 공증에 관하여 규정하고 있습니다.
                제11조는 법률행위의 등기에 관하여 규정하고 있으며, 제12조는 법률행위의 등기부등본에 관하여 규정하고 있습니다.
                제13조는 법률행위의 등기부등본의 효력에 관하여 규정하고 있으며, 제14조는 법률행위의 등기부등본의 추정력에 관하여 규정하고 있습니다.
                제15조는 법률행위의 등기부등본의 대항력에 관하여 규정하고 있으며, 제16조는 법률행위의 등기부등본의 공신력에 관하여 규정하고 있습니다.
                제17조는 법률행위의 등기부등본의 기재사실에 관하여 규정하고 있으며, 제18조는 법률행위의 등기부등본의 기재사실의 진실성에 관하여 규정하고 있습니다.
                제19조는 법률행위의 등기부등본의 기재사실의 추정에 관하여 규정하고 있으며, 제20조는 법률행위의 등기부등본의 기재사실의 대항에 관하여 규정하고 있습니다.
                대법원 2020다12345 판례에서는 이러한 법조문들의 해석과 적용에 관하여 구체적인 기준을 제시하고 있습니다.
                """,
                "expected_citations": ["제1조", "제2조", "제3조", "제4조", "제5조", "제6조", "제7조", "제8조", "제9조", "제10조", "제11조", "제12조", "제13조", "제14조", "제15조", "제16조", "제17조", "제18조", "제19조", "제20조", "2020다12345"]
            },
            "rapid_citations": {
                "query": "연속된 법조문 인용 테스트",
                "answer": """
                민법 제1조, 제2조, 제3조, 제4조, 제5조, 제6조, 제7조, 제8조, 제9조, 제10조에 따르면
                법률행위의 성립과 효력에 관하여 규정하고 있으며, 형법 제1조, 제2조, 제3조, 제4조, 제5조에 따르면
                범죄의 성립과 처벌에 관하여 규정하고 있습니다. 상법 제1조, 제2조, 제3조, 제4조, 제5조에 따르면
                상행위의 성립과 효력에 관하여 규정하고 있으며, 근로기준법 제1조, 제2조, 제3조, 제4조, 제5조에 따르면
                근로조건의 기준에 관하여 규정하고 있습니다.
                """,
                "expected_citations": ["제1조", "제2조", "제3조", "제4조", "제5조", "제6조", "제7조", "제8조", "제9조", "제10조", "제1조", "제2조", "제3조", "제4조", "제5조", "제1조", "제2조", "제3조", "제4조", "제5조", "제1조", "제2조", "제3조", "제4조", "제5조"]
            },
            "mixed_languages": {
                "query": "한국어와 영어가 섞인 법률 문서",
                "answer": """
                Contract Law에 따르면 계약의 성립은 민법 제543조에 규정되어 있으며,
                Tort Law에 따르면 불법행위는 민법 제750조에 규정되어 있습니다.
                Property Law에 따르면 물권은 민법 제185조에 규정되어 있으며,
                Family Law에 따르면 가족관계는 민법 제779조에 규정되어 있습니다.
                대법원 2021다12345 판례에서는 이러한 국제법과 국내법의 적용에 관하여 구체적인 기준을 제시하고 있습니다.
                """,
                "expected_citations": ["제543조", "제750조", "제185조", "제779조", "2021다12345"]
            },
            "special_characters": {
                "query": "특수문자가 포함된 법률 문서",
                "answer": """
                민법 제1조(법원의 법률 해석)에 따르면, 법률행위의 성립과 효력에 관하여 규정하고 있으며,
                형법 제2조(범죄의 성립)에 따르면, 범죄의 구성요건에 관하여 규정하고 있습니다.
                상법 제3조(상행위의 성립)에 따르면, 상행위의 요건에 관하여 규정하고 있으며,
                근로기준법 제4조(근로조건의 기준)에 따르면, 근로조건의 최저기준에 관하여 규정하고 있습니다.
                대법원 2020다12345 판례에서는 이러한 법조문의 해석에 관하여 구체적인 기준을 제시하고 있습니다.
                """,
                "expected_citations": ["제1조", "제2조", "제3조", "제4조", "2020다12345"]
            },
            "unicode_citations": {
                "query": "유니코드 문자가 포함된 법률 문서",
                "answer": """
                민법 제1조에 따르면 법률행위의 성립과 효력에 관하여 규정하고 있으며,
                형법 제2조에 따르면 범죄의 성립과 처벌에 관하여 규정하고 있습니다.
                상법 제3조에 따르면 상행위의 성립과 효력에 관하여 규정하고 있으며,
                근로기준법 제4조에 따르면 근로조건의 기준에 관하여 규정하고 있습니다.
                대법원 2020다12345 판례에서는 이러한 법조문의 해석에 관하여 구체적인 기준을 제시하고 있습니다.
                """,
                "expected_citations": ["제1조", "제2조", "제3조", "제4조", "2020다12345"]
            },

            # === 복합 시나리오 테스트 ===
            "multi_jurisdiction": {
                "query": "한국과 미국의 계약법 차이점과 국제계약에서의 적용",
                "answer": """
                한국과 미국의 계약법 차이점은 다음과 같습니다:

                1. 한국 민법 제543조 vs 미국 UCC Article 2
                2. 한국 민법 제544조 vs 미국 Common Law
                3. 국제사법 제25조의 준거법 선택

                대법원 2021다12345 판례에서 국제계약의 준거법 결정 기준을 제시하고 있으며,
                미국 연방법원 2020년 판례에서도 유사한 원칙을 적용하고 있습니다.
                """,
                "expected_citations": ["제543조", "제544조", "제25조", "2021다12345"]
            },
            "cross_reference": {
                "query": "민법과 상법, 형법이 모두 관련된 복합 사건",
                "answer": """
                복합 사건에서 여러 법률의 적용은 다음과 같습니다:

                1. 민법 제750조: 불법행위로 인한 손해배상
                2. 상법 제401조: 이사의 손해배상책임
                3. 형법 제347조: 사기죄의 성립

                대법원 2021다12345 판례에서는 이러한 복합 사건의 법률 적용 순서를 구체적으로 제시하고 있습니다.
                """,
                "expected_citations": ["제750조", "제401조", "제347조", "2021다12345"]
            },
            "temporal_conflict": {
                "query": "법률 개정 전후의 적용 문제",
                "answer": """
                법률 개정 전후의 적용 문제는 다음과 같이 해결됩니다:

                1. 구 민법 제123조: 개정 전 규정
                2. 현행 민법 제124조: 개정 후 규정
                3. 법률 개정의 소급효 문제

                대법원 2019다12345 판례에서는 법률 개정의 소급효 적용 범위를 구체적으로 판시하고 있습니다.
                """,
                "expected_citations": ["제123조", "제124조", "2019다12345"]
            },
            "hierarchical_conflict": {
                "query": "헌법, 법률, 시행령, 시행규칙의 효력 순서",
                "answer": """
                법률의 효력 순서는 다음과 같습니다:

                1. 헌법 제6조: 최고법규의 지위
                2. 법률: 국회에서 제정
                3. 시행령: 대통령령
                4. 시행규칙: 부령

                헌법재판소 2019헌마123 결정에서 법률의 효력 순서를 구체적으로 제시하고 있습니다.
                """,
                "expected_citations": ["제6조", "2019헌마123"]
            },
            "interpretation_methods": {
                "query": "법률 해석의 여러 방법들",
                "answer": """
                법률 해석의 방법은 다음과 같습니다:

                1. 문리해석: 법조문의 문자적 의미
                2. 논리해석: 법조문의 논리적 구조
                3. 역사해석: 입법 과정과 의도
                4. 목적해석: 법률의 목적과 취지

                대법원 2020다12345 판례에서 법률 해석의 구체적인 방법을 제시하고 있습니다.
                """,
                "expected_citations": ["2020다12345"]
            }
        }

    def test_citation_enhancer(self) -> Dict[str, Any]:
        """법적 인용 강화 시스템 테스트"""
        logger.info("=== 법적 인용 강화 시스템 테스트 시작 ===")

        results = {}
        for test_name, test_case in self.test_cases.items():
            logger.info(f"테스트 케이스: {test_name}")

            try:
                citation_result = self.citation_enhancer.enhance_text_with_citations(test_case["answer"])

                results[test_name] = {
                    "success": True,
                    "citations_found": citation_result["citation_count"],
                    "expected_citations": len(test_case["expected_citations"]),
                    "citations": citation_result,
                    "enhanced_text": citation_result["enhanced_text"]
                }

                logger.info(f"  - 발견된 인용: {citation_result['citation_count']}개")
                logger.info(f"  - 예상 인용: {len(test_case['expected_citations'])}개")

            except Exception as e:
                logger.error(f"테스트 케이스 {test_name} 실패: {e}")
                results[test_name] = {
                    "success": False,
                    "error": str(e)
                }

        return results

    def test_basis_validator(self) -> Dict[str, Any]:
        """법적 근거 검증 시스템 테스트"""
        logger.info("=== 법적 근거 검증 시스템 테스트 시작 ===")

        results = {}
        for test_name, test_case in self.test_cases.items():
            logger.info(f"테스트 케이스: {test_name}")

            try:
                validation_result = self.basis_validator.validate_legal_basis(
                    test_case["query"],
                    test_case["answer"]
                )

                results[test_name] = {
                    "success": True,
                    "is_valid": validation_result.is_valid,
                    "confidence": validation_result.confidence,
                    "validation_details": validation_result.validation_details,
                    "legal_sources": validation_result.legal_sources,
                    "issues": validation_result.issues,
                    "recommendations": validation_result.recommendations
                }

                logger.info(f"  - 검증 결과: {'유효' if validation_result.is_valid else '무효'}")
                logger.info(f"  - 신뢰도: {validation_result.confidence:.2f}")

            except Exception as e:
                logger.error(f"테스트 케이스 {test_name} 실패: {e}")
                results[test_name] = {
                    "success": False,
                    "error": str(e)
                }

        return results

    def test_structure_enhancer(self) -> Dict[str, Any]:
        """답변 구조화 강화 시스템 테스트"""
        logger.info("=== 답변 구조화 강화 시스템 테스트 시작 ===")

        results = {}
        for test_name, test_case in self.test_cases.items():
            logger.info(f"테스트 케이스: {test_name}")

            try:
                # 질문 유형 분류
                question_type = self.structure_enhancer.classify_question_type(test_case["query"])

                # 법적 근거 검증
                validation_result = self.basis_validator.validate_legal_basis(
                    test_case["query"],
                    test_case["answer"]
                )

                # 답변 구조화
                structured_answer = self.structure_enhancer.create_structured_answer(
                    test_case["answer"],
                    question_type
                )

                results[test_name] = {
                    "success": True,
                    "question_type": question_type.value,
                    "is_legally_sound": validation_result.is_valid,
                    "confidence": validation_result.confidence,
                    "structured_answer": structured_answer
                }

                logger.info(f"  - 질문 유형: {question_type.value}")
                logger.info(f"  - 법적 근거 검증: {'통과' if validation_result.is_valid else '실패'}")

            except Exception as e:
                logger.error(f"테스트 케이스 {test_name} 실패: {e}")
                results[test_name] = {
                    "success": False,
                    "error": str(e)
                }

        return results

    def test_integration_service(self) -> Dict[str, Any]:
        """통합 서비스 테스트"""
        logger.info("=== 통합 서비스 테스트 시작 ===")

        results = {}
        for test_name, test_case in self.test_cases.items():
            logger.info(f"테스트 케이스: {test_name}")

            try:
                result = self.integration_service.process_query_with_legal_basis(
                    test_case["query"],
                    test_case["answer"]
                )

                results[test_name] = {
                    "success": result["success"],
                    "question_type": result.get("question_type", "unknown"),
                    "is_legally_sound": result["is_legally_sound"],
                    "confidence": result["confidence"],
                    "enhanced_answer": result["enhanced_answer"],
                    "legal_basis": result["legal_basis"]
                }

                logger.info(f"  - 질문 유형: {result.get('question_type', 'unknown')}")
                logger.info(f"  - 신뢰도: {result['confidence']:.2f}")
                logger.info(f"  - 법적 근거 검증: {'통과' if result['is_legally_sound'] else '실패'}")

            except Exception as e:
                logger.error(f"테스트 케이스 {test_name} 실패: {e}")
                results[test_name] = {
                    "success": False,
                    "error": str(e)
                }

        return results

    def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        logger.info("=== 전체 테스트 실행 ===")

        all_results = {
            "citation_enhancer": self.test_citation_enhancer(),
            "basis_validator": self.test_basis_validator(),
            "structure_enhancer": self.test_structure_enhancer(),
            "integration_service": self.test_integration_service()
        }

        # 요약 생성
        all_results["summary"] = self._generate_test_summary(all_results)

        # 결과 출력
        self.print_detailed_results(all_results)

        logger.info("=== 전체 테스트 완료 ===")
        logger.info(f"총 테스트 케이스: {all_results['summary']['total_test_cases']}")
        logger.info(f"성공한 테스트: {all_results['summary']['successful_tests']}")
        logger.info(f"실패한 테스트: {all_results['summary']['failed_tests']}")
        logger.info(f"전체 성공률: {all_results['summary']['success_rate']:.1f}%")

        return all_results

    def _generate_test_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """테스트 결과 요약 생성"""
        total_tests = 0
        successful_tests = 0
        failed_tests = 0

        # 서비스별 통계
        service_stats = {}
        citation_stats = {
            "total_citations_found": 0,
            "total_expected_citations": 0,
            "citation_accuracy": 0.0
        }
        confidence_stats = {
            "total_confidence": 0.0,
            "confidence_count": 0,
            "high_confidence_tests": 0,
            "medium_confidence_tests": 0,
            "low_confidence_tests": 0
        }

        for service_name, service_results in all_results.items():
            if service_name == "summary":
                continue

            service_stats[service_name] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0
            }

            for test_name, test_result in service_results.items():
                total_tests += 1
                service_stats[service_name]["total"] += 1

                if test_result.get("success", False):
                    successful_tests += 1
                    service_stats[service_name]["successful"] += 1
                else:
                    failed_tests += 1
                    service_stats[service_name]["failed"] += 1

                # 인용 통계 수집
                if "citations_found" in test_result:
                    citation_stats["total_citations_found"] += test_result["citations_found"]
                if "expected_citations" in test_result:
                    citation_stats["total_expected_citations"] += test_result["expected_citations"]

                # 신뢰도 통계 수집
                if "confidence" in test_result:
                    confidence = test_result["confidence"]
                    confidence_stats["total_confidence"] += confidence
                    confidence_stats["confidence_count"] += 1

                    if confidence >= 0.8:
                        confidence_stats["high_confidence_tests"] += 1
                    elif confidence >= 0.6:
                        confidence_stats["medium_confidence_tests"] += 1
                    else:
                        confidence_stats["low_confidence_tests"] += 1

            # 서비스별 성공률 계산
            if service_stats[service_name]["total"] > 0:
                service_stats[service_name]["success_rate"] = (
                    service_stats[service_name]["successful"] / service_stats[service_name]["total"] * 100
                )

        # 전체 성공률 계산
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

        # 인용 정확도 계산
        if citation_stats["total_expected_citations"] > 0:
            citation_stats["citation_accuracy"] = (
                citation_stats["total_citations_found"] / citation_stats["total_expected_citations"] * 100
            )

        # 평균 신뢰도 계산
        if confidence_stats["confidence_count"] > 0:
            confidence_stats["average_confidence"] = (
                confidence_stats["total_confidence"] / confidence_stats["confidence_count"]
            )

        return {
            "total_test_cases": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "service_statistics": service_stats,
            "citation_statistics": citation_stats,
            "confidence_statistics": confidence_stats,
            "test_date": datetime.now().isoformat(),
            "system_version": "1.0.0"
        }

    def print_detailed_results(self, results: Dict[str, Any]):
        """상세 결과 출력"""
        print("\n" + "="*80)
        print("법적 근거 제시 시스템 테스트 결과")
        print("="*80)

        for service_name, service_results in results.items():
            if service_name == "summary":
                continue

            print(f"\n[{service_name.upper()}]")
            print("-" * 40)

            for test_name, test_result in service_results.items():
                if test_result.get("success", False):
                    print(f"{test_name}: ✅ 성공")
                    if "citations_found" in test_result:
                        print(f"  - 발견된 인용: {test_result['citations_found']}개")
                    if "confidence" in test_result:
                        print(f"  - 신뢰도: {test_result['confidence']:.2f}")
                    if "is_legally_sound" in test_result:
                        print(f"  - 법적 근거 검증: {'통과' if test_result['is_legally_sound'] else '실패'}")
                else:
                    print(f"{test_name}: ❌ 실패")
                    print(f"  - 오류: {test_result.get('error', 'Unknown error')}")

        # 요약 정보 출력
        if "summary" in results:
            summary = results["summary"]
            print(f"\n[요약]")
            print("-" * 40)
            print(f"총 테스트 케이스: {summary['total_test_cases']}")
            print(f"성공한 테스트: {summary['successful_tests']}")
            print(f"실패한 테스트: {summary['failed_tests']}")
            print(f"전체 성공률: {summary['success_rate']:.1f}%")

            # 서비스별 통계
            if "service_statistics" in summary:
                print(f"\n[서비스별 통계]")
                print("-" * 40)
                for service_name, stats in summary["service_statistics"].items():
                    print(f"{service_name}: {stats['successful']}/{stats['total']} ({stats['success_rate']:.1f}%)")

            # 인용 통계
            if "citation_statistics" in summary:
                citation_stats = summary["citation_statistics"]
                print(f"\n[인용 통계]")
                print("-" * 40)
                print(f"발견된 총 인용: {citation_stats['total_citations_found']}개")
                print(f"예상 총 인용: {citation_stats['total_expected_citations']}개")
                print(f"인용 정확도: {citation_stats['citation_accuracy']:.1f}%")

            # 신뢰도 통계
            if "confidence_statistics" in summary:
                confidence_stats = summary["confidence_statistics"]
                print(f"\n[신뢰도 통계]")
                print("-" * 40)
                print(f"평균 신뢰도: {confidence_stats.get('average_confidence', 0):.2f}")
                print(f"높은 신뢰도 (≥0.8): {confidence_stats['high_confidence_tests']}개")
                print(f"보통 신뢰도 (0.6-0.8): {confidence_stats['medium_confidence_tests']}개")
                print(f"낮은 신뢰도 (<0.6): {confidence_stats['low_confidence_tests']}개")


def convert_all_objects_for_json(obj):
    """모든 객체를 JSON 직렬화 가능한 형태로 변환"""
    if hasattr(obj, '__dict__'):
        # dataclass나 객체인 경우
        return obj.__dict__
    elif isinstance(obj, dict):
        return {key: convert_all_objects_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_all_objects_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(convert_all_objects_for_json(item) for item in obj)
    else:
        return obj


def main():
    """메인 함수"""
    try:
        logger.info("=== 법적 근거 제시 시스템 전체 테스트 시작 ===")

        # 테스트 실행
        tester = LegalBasisSystemTester()
        results = tester.run_all_tests()

        # JSON 직렬화 변환
        results = convert_all_objects_for_json(results)

        with open("legal_basis_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n테스트 결과가 'legal_basis_test_results.json' 파일에 저장되었습니다.")

    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
