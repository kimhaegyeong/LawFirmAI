#!/usr/bin/env python3
"""
통합된 하이브리드 분류기를 위한 훈련 데이터 생성
"""

import sys
import os
from typing import List, Tuple
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.services.unified_question_types import UnifiedQuestionType

def create_integrated_training_data() -> List[Tuple[str, UnifiedQuestionType]]:
    """통합된 훈련 데이터 생성"""
    training_data = []
    
    # 법률 문의
    law_inquiry_examples = [
        "민법 제123조의 내용이 무엇인가요?",
        "형법 제250조 처벌 기준은?",
        "근로기준법 제15조 의미는?",
        "상법 제123조 해석해주세요",
        "헌법 제10조 내용은?",
        "특허법 제25조 규정은?",
        "부동산등기법 제123조는?",
        "민법 제456항의 의미는?",
        "형법 제789호 규정은?",
        "근로기준법 제12조 제3항은?"
    ]
    
    # 판례 검색
    precedent_examples = [
        "대법원 판례를 찾아주세요",
        "관련 판례가 있나요?",
        "고등법원 판결을 알려주세요",
        "지방법원 판례 검색",
        "최근 판례를 찾아주세요",
        "유사한 판례가 있나요?",
        "대법원 판례 검색",
        "고등법원 판례 찾기",
        "지방법원 판례를 찾아주세요",
        "판례를 찾아주세요"
    ]
    
    # 계약서 검토
    contract_examples = [
        "계약서를 검토해주세요",
        "이 계약 조항이 불리한가요?",
        "계약서 수정이 필요한가요?",
        "계약서를 작성해주세요",
        "계약서를 체결하고 싶어요",
        "계약 조항을 확인해주세요",
        "계약 조건이 적절한가요?",
        "계약 내용을 검토해주세요",
        "불리한 조항이 있나요?",
        "계약서의 문제점은?"
    ]
    
    # 이혼 절차
    divorce_examples = [
        "이혼 절차를 알려주세요",
        "협의이혼 방법은?",
        "재판이혼 절차는?",
        "이혼절차 신청 방법",
        "이혼 어떻게 해야 하나요?",
        "이혼 어디서 신청하나요?",
        "이혼 비용은 얼마인가요?",
        "협의이혼 절차",
        "재판이혼 절차",
        "이혼 방법"
    ]
    
    # 상속 절차
    inheritance_examples = [
        "상속 절차를 알려주세요",
        "유산 분할 방법은?",
        "상속인 확인 방법",
        "상속세 신고 절차",
        "유언 검인 절차",
        "상속포기 방법",
        "상속 신청 방법",
        "유산 분할 절차",
        "상속인 확인 절차",
        "상속세 신고 방법"
    ]
    
    # 형사 사건
    criminal_examples = [
        "사기죄 구성요건은?",
        "절도 범죄 처벌은?",
        "강도 사건 대응 방법",
        "살인죄 형량은?",
        "형사 사건 절차",
        "절도죄 구성요건",
        "강도죄 처벌",
        "살인죄 형량",
        "사기범죄 처벌",
        "절도범죄 처벌"
    ]
    
    # 노동 분쟁
    labor_examples = [
        "노동 분쟁 해결 방법",
        "근로 시간 규정은?",
        "임금 체불 대응",
        "부당해고 구제 방법",
        "해고 통보 대응",
        "노동위원회 신청",
        "근로 분쟁 해결",
        "임금 지급 문제",
        "부당해고 구제",
        "해고 대응"
    ]
    
    # 절차 안내
    procedure_examples = [
        "소송 절차를 알려주세요",
        "민사조정 신청 방법",
        "소액사건 절차는?",
        "어떻게 신청하나요?",
        "어디서 신청하나요?",
        "신청 방법을 알려주세요",
        "신청 절차를 알려주세요",
        "처리 절차를 알려주세요",
        "진행 절차를 알려주세요",
        "소송 제기 방법"
    ]
    
    # 용어 설명
    term_examples = [
        "법인격의 의미는?",
        "소멸시효 정의는?",
        "무효와 취소의 개념",
        "무엇이 계약인가요?",
        "뜻을 설명해주세요",
        "계약의 의미는?",
        "계약의 정의는?",
        "계약의 개념은?",
        "계약이 무엇인가요?",
        "계약이란 무엇인가요?"
    ]
    
    # 법률 자문
    advice_examples = [
        "어떻게 대응해야 하나요?",
        "권리 구제 방법은?",
        "의무 이행 방법",
        "해야 할 일은?",
        "법률 상담을 받고 싶어요",
        "변호사 상담이 필요해요",
        "법적 대응 방법은?",
        "법적 조언을 구하고 싶어요",
        "어떤 조치를 취해야 하나요?",
        "권리 보호 방법은?"
    ]
    
    # 일반 질문
    general_examples = [
        "안녕하세요",
        "도움이 필요합니다",
        "질문이 있습니다",
        "고마워요",
        "감사합니다",
        "좋은 하루 되세요",
        "수고하세요",
        "잘 부탁드립니다",
        "도와주세요",
        "궁금한 것이 있어요"
    ]
    
    # 훈련 데이터 구성
    all_examples = [
        (law_inquiry_examples, UnifiedQuestionType.LAW_INQUIRY),
        (precedent_examples, UnifiedQuestionType.PRECEDENT_SEARCH),
        (contract_examples, UnifiedQuestionType.CONTRACT_REVIEW),
        (divorce_examples, UnifiedQuestionType.DIVORCE_PROCEDURE),
        (inheritance_examples, UnifiedQuestionType.INHERITANCE_PROCEDURE),
        (criminal_examples, UnifiedQuestionType.CRIMINAL_CASE),
        (labor_examples, UnifiedQuestionType.LABOR_DISPUTE),
        (procedure_examples, UnifiedQuestionType.PROCEDURE_GUIDE),
        (term_examples, UnifiedQuestionType.TERM_EXPLANATION),
        (advice_examples, UnifiedQuestionType.LEGAL_ADVICE),
        (general_examples, UnifiedQuestionType.GENERAL_QUESTION)
    ]
    
    for examples, question_type in all_examples:
        for example in examples:
            training_data.append((example, question_type))
    
    return training_data

if __name__ == "__main__":
    training_data = create_integrated_training_data()
    print(f"총 {len(training_data)}개의 훈련 데이터 생성")
    
    # 데이터 저장
    import json
    from pathlib import Path
    
    output_file = Path("data/ml_training/integrated_training_data.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # JSON 직렬화 가능한 형태로 변환
    json_data = []
    for question, question_type in training_data:
        json_data.append({
            "question": question,
            "question_type": question_type.value
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"훈련 데이터 저장 완료: {output_file}")
    
    # 각 질문 유형별 개수 출력
    type_counts = {}
    for _, question_type in training_data:
        type_counts[question_type.value] = type_counts.get(question_type.value, 0) + 1
    
    print("\n질문 유형별 데이터 개수:")
    for question_type, count in type_counts.items():
        print(f"- {question_type}: {count}개")
