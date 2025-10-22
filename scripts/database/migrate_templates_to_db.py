#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 하드코딩된 템플릿을 데이터베이스로 마이그레이션하는 스크립트
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from source.services.template_database_manager import TemplateDatabaseManager


def migrate_hardcoded_templates():
    """하드코딩된 템플릿을 데이터베이스로 마이그레이션"""
    print("=" * 60)
    print("하드코딩된 템플릿 데이터베이스 마이그레이션")
    print("=" * 60)
    
    db_manager = TemplateDatabaseManager()
    
    # 기존 하드코딩된 템플릿 데이터
    templates_data = {
        "precedent_search": {
            "title": "판례 검색 결과",
            "sections": [
                {
                    "name": "관련 판례",
                    "priority": "high",
                    "template": "다음과 같은 관련 판례를 찾았습니다:",
                    "content_guide": "판례 번호, 사건명, 핵심 판결요지 포함",
                    "legal_citations": True
                },
                {
                    "name": "판례 분석",
                    "priority": "high",
                    "template": "해당 판례의 주요 쟁점과 법원의 판단:",
                    "content_guide": "법리적 분석과 실무적 시사점"
                },
                {
                    "name": "적용 가능성",
                    "priority": "medium",
                    "template": "귀하의 사안에의 적용 가능성:",
                    "content_guide": "유사점과 차이점 분석"
                },
                {
                    "name": "실무 조언",
                    "priority": "medium",
                    "template": "실무적 권장사항:",
                    "content_guide": "구체적 행동 방안"
                }
            ]
        },
        "law_inquiry": {
            "title": "법률 문의 답변",
            "sections": [
                {
                    "name": "관련 법령",
                    "priority": "high",
                    "template": "관련 법령:",
                    "content_guide": "정확한 조문 번호와 내용",
                    "legal_citations": True
                },
                {
                    "name": "법령 해설",
                    "priority": "high",
                    "template": "법령 해설:",
                    "content_guide": "쉬운 말로 풀어서 설명"
                },
                {
                    "name": "적용 사례",
                    "priority": "medium",
                    "template": "실제 적용 사례:",
                    "content_guide": "구체적 예시와 설명"
                },
                {
                    "name": "주의사항",
                    "priority": "medium",
                    "template": "주의사항:",
                    "content_guide": "법적 리스크와 제한사항"
                }
            ]
        },
        "legal_advice": {
            "title": "법률 상담 답변",
            "sections": [
                {
                    "name": "상황 정리",
                    "priority": "high",
                    "template": "말씀하신 상황을 정리하면:",
                    "content_guide": "핵심 사실 관계 정리"
                },
                {
                    "name": "법적 분석",
                    "priority": "high",
                    "template": "법적 분석:",
                    "content_guide": "적용 법률과 법리 분석",
                    "legal_citations": True
                },
                {
                    "name": "권리 구제 방법",
                    "priority": "high",
                    "template": "권리 구제 방법:",
                    "content_guide": "단계별 구체적 방안"
                },
                {
                    "name": "필요 증거",
                    "priority": "medium",
                    "template": "필요한 증거 자료:",
                    "content_guide": "구체적 증거 목록"
                },
                {
                    "name": "전문가 상담",
                    "priority": "low",
                    "template": "전문가 상담 권유:",
                    "content_guide": "변호사 상담 필요성"
                }
            ]
        },
        "procedure_guide": {
            "title": "절차 안내",
            "sections": [
                {
                    "name": "절차 개요",
                    "priority": "high",
                    "template": "전체 절차 개요:",
                    "content_guide": "절차의 전체적인 흐름"
                },
                {
                    "name": "단계별 절차",
                    "priority": "high",
                    "template": "단계별 절차:",
                    "content_guide": "구체적 단계별 설명"
                },
                {
                    "name": "필요 서류",
                    "priority": "high",
                    "template": "필요한 서류:",
                    "content_guide": "구체적 서류 목록"
                },
                {
                    "name": "처리 기간",
                    "priority": "medium",
                    "template": "처리 기간 및 비용:",
                    "content_guide": "예상 소요시간과 비용"
                },
                {
                    "name": "주의사항",
                    "priority": "medium",
                    "template": "주의사항:",
                    "content_guide": "절차 진행 시 주의할 점"
                }
            ]
        },
        "term_explanation": {
            "title": "법률 용어 해설",
            "sections": [
                {
                    "name": "용어 정의",
                    "priority": "high",
                    "template": "용어 정의:",
                    "content_guide": "정확한 법률적 정의"
                },
                {
                    "name": "법적 근거",
                    "priority": "high",
                    "template": "법적 근거:",
                    "content_guide": "관련 법조문과 판례",
                    "legal_citations": True
                },
                {
                    "name": "실제 적용",
                    "priority": "medium",
                    "template": "실제 적용 사례:",
                    "content_guide": "구체적 적용 예시"
                },
                {
                    "name": "관련 용어",
                    "priority": "low",
                    "template": "관련 용어:",
                    "content_guide": "비슷하거나 관련된 용어들"
                }
            ]
        },
        "contract_review": {
            "title": "계약서 검토 결과",
            "sections": [
                {
                    "name": "계약서 분석",
                    "priority": "high",
                    "template": "계약서 주요 내용 분석:",
                    "content_guide": "계약의 핵심 조항 분석"
                },
                {
                    "name": "법적 검토",
                    "priority": "high",
                    "template": "법적 검토 결과:",
                    "content_guide": "법적 유효성과 문제점"
                },
                {
                    "name": "주의사항",
                    "priority": "high",
                    "template": "주의해야 할 사항:",
                    "content_guide": "불리한 조항과 리스크"
                },
                {
                    "name": "개선 제안",
                    "priority": "medium",
                    "template": "개선 제안:",
                    "content_guide": "구체적 수정 권장사항"
                }
            ]
        },
        "divorce_procedure": {
            "title": "이혼 절차 안내",
            "sections": [
                {
                    "name": "이혼 방법",
                    "priority": "high",
                    "template": "이혼 방법 선택:",
                    "content_guide": "협의이혼, 조정이혼, 재판이혼 비교"
                },
                {
                    "name": "절차 단계",
                    "priority": "high",
                    "template": "구체적 절차:",
                    "content_guide": "단계별 상세 절차"
                },
                {
                    "name": "필요 서류",
                    "priority": "high",
                    "template": "필요한 서류:",
                    "content_guide": "구체적 서류 목록"
                },
                {
                    "name": "재산분할",
                    "priority": "medium",
                    "template": "재산분할 및 위자료:",
                    "content_guide": "재산분할 기준과 위자료 산정"
                },
                {
                    "name": "양육권",
                    "priority": "medium",
                    "template": "양육권 및 면접교섭권:",
                    "content_guide": "자녀 양육 관련 사항"
                }
            ]
        },
        "inheritance_procedure": {
            "title": "상속 절차 안내",
            "sections": [
                {
                    "name": "상속인 확인",
                    "priority": "high",
                    "template": "상속인 및 상속분:",
                    "content_guide": "법정상속인과 상속분 계산"
                },
                {
                    "name": "상속 절차",
                    "priority": "high",
                    "template": "상속 절차:",
                    "content_guide": "단계별 상속 절차"
                },
                {
                    "name": "필요 서류",
                    "priority": "high",
                    "template": "필요한 서류:",
                    "content_guide": "상속 관련 서류 목록"
                },
                {
                    "name": "세금 문제",
                    "priority": "medium",
                    "template": "상속세 및 증여세:",
                    "content_guide": "세금 관련 주의사항"
                },
                {
                    "name": "유언 검인",
                    "priority": "low",
                    "template": "유언 검인 절차:",
                    "content_guide": "유언이 있는 경우 절차"
                }
            ]
        },
        "criminal_case": {
            "title": "형사 사건 안내",
            "sections": [
                {
                    "name": "범죄 분석",
                    "priority": "high",
                    "template": "해당 범죄의 구성요건:",
                    "content_guide": "범죄 성립요건 분석"
                },
                {
                    "name": "법정형",
                    "priority": "high",
                    "template": "법정형 및 형량:",
                    "content_guide": "처벌 기준과 형량"
                },
                {
                    "name": "수사 절차",
                    "priority": "medium",
                    "template": "수사 및 재판 절차:",
                    "content_guide": "수사부터 재판까지 절차"
                },
                {
                    "name": "변호인 조력",
                    "priority": "high",
                    "template": "변호인 조력권:",
                    "content_guide": "변호인 선임과 조력권"
                },
                {
                    "name": "구제 방법",
                    "priority": "medium",
                    "template": "권리 구제 방법:",
                    "content_guide": "항소, 상고 등 구제 절차"
                }
            ]
        },
        "labor_dispute": {
            "title": "노동 분쟁 안내",
            "sections": [
                {
                    "name": "분쟁 분석",
                    "priority": "high",
                    "template": "노동 분쟁 분석:",
                    "content_guide": "분쟁의 성격과 쟁점"
                },
                {
                    "name": "적용 법령",
                    "priority": "high",
                    "template": "적용 법령:",
                    "content_guide": "근로기준법 등 관련 법령"
                },
                {
                    "name": "구제 절차",
                    "priority": "high",
                    "template": "구제 절차:",
                    "content_guide": "노동위원회, 법원 절차"
                },
                {
                    "name": "필요 증거",
                    "priority": "medium",
                    "template": "필요한 증거:",
                    "content_guide": "임금대장, 근로계약서 등"
                },
                {
                    "name": "시효 문제",
                    "priority": "medium",
                    "template": "시효 및 제한:",
                    "content_guide": "신청 기한과 제한사항"
                }
            ]
        },
        "general_question": {
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
    
    # 템플릿 마이그레이션
    print("\n1. 템플릿 마이그레이션 중...")
    migrated_count = 0
    
    for question_type, template_data in templates_data.items():
        print(f"\n   {question_type} 템플릿 마이그레이션 중...")
        
        # 템플릿 추가
        template_id = db_manager.add_template(
            question_type=question_type,
            template_name=f"{question_type}_template",
            title=template_data["title"],
            description=f"{question_type} 질문 유형용 답변 템플릿",
            priority=1
        )
        
        if template_id > 0:
            # 섹션 추가
            section_count = 0
            for i, section in enumerate(template_data["sections"]):
                success = db_manager.add_template_section(
                    template_id=template_id,
                    section_name=section["name"],
                    priority=section["priority"],
                    template_text=section["template"],
                    content_guide=section["content_guide"],
                    legal_citations=section.get("legal_citations", False),
                    section_order=i
                )
                if success:
                    section_count += 1
            
            print(f"     ✅ 템플릿 추가 완료 (섹션 {section_count}개)")
            migrated_count += 1
        else:
            print(f"     ❌ 템플릿 추가 실패")
    
    # 품질 지표 마이그레이션
    print(f"\n2. 품질 지표 마이그레이션 중...")
    quality_indicators = {
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
    
    indicator_count = 0
    for indicator_type, keywords in quality_indicators.items():
        for keyword in keywords:
            success = db_manager.add_quality_indicator(
                indicator_type=indicator_type,
                keyword=keyword,
                weight=1.0,
                description=f"{indicator_type} 지표 키워드"
            )
            if success:
                indicator_count += 1
    
    print(f"   ✅ 품질 지표 {indicator_count}개 추가 완료")
    
    # 질문 유형 설정 마이그레이션
    print(f"\n3. 질문 유형 설정 마이그레이션 중...")
    question_type_configs = {
        "law_inquiry": {
            "display_name": "법률 문의",
            "law_names": ["민법", "형법", "근로기준법", "상법", "행정법"],
            "question_words": ["내용", "규정", "기준", "처벌", "얼마", "몇", "언제"],
            "special_keywords": ["제", "조", "항", "호"],
            "bonus_score": 2.0
        },
        "precedent_search": {
            "display_name": "판례 검색",
            "law_names": [],
            "question_words": ["찾아", "검색", "유사", "관련", "최근"],
            "special_keywords": ["판례", "사건", "판결"],
            "bonus_score": 1.5
        },
        "contract_review": {
            "display_name": "계약서 검토",
            "law_names": [],
            "question_words": ["검토", "분석", "확인", "수정"],
            "special_keywords": ["계약서", "계약", "조항"],
            "bonus_score": 1.5
        }
    }
    
    config_count = 0
    for question_type, config in question_type_configs.items():
        success = db_manager.add_question_type_config(
            question_type=question_type,
            display_name=config["display_name"],
            law_names=config["law_names"],
            question_words=config["question_words"],
            special_keywords=config["special_keywords"],
            bonus_score=config["bonus_score"]
        )
        if success:
            config_count += 1
    
    print(f"   ✅ 질문 유형 설정 {config_count}개 추가 완료")
    
    # 충돌 해결 규칙 마이그레이션
    print(f"\n4. 충돌 해결 규칙 마이그레이션 중...")
    conflict_rules = [
        {
            "conflict_type": "law_inquiry_vs_contract_review",
            "target_type": "contract_review",
            "keywords": ["계약서", "계약", "조항", "검토", "수정", "불리한"],
            "bonus_score": 3.0,
            "priority": 1
        },
        {
            "conflict_type": "law_inquiry_vs_labor_dispute",
            "target_type": "labor_dispute",
            "keywords": ["노동", "근로", "임금", "해고", "부당해고", "임금체불", "근로시간"],
            "bonus_score": 3.0,
            "priority": 1
        },
        {
            "conflict_type": "law_inquiry_vs_inheritance_procedure",
            "target_type": "inheritance_procedure",
            "keywords": ["상속", "유산", "상속인", "상속세", "유언"],
            "bonus_score": 3.0,
            "priority": 1
        },
        {
            "conflict_type": "law_inquiry_vs_procedure_guide",
            "target_type": "procedure_guide",
            "keywords": ["절차", "신청", "방법", "어떻게", "소액사건", "민사조정", "이혼조정"],
            "bonus_score": 3.0,
            "priority": 1
        },
        {
            "conflict_type": "law_inquiry_vs_general_question",
            "target_type": "general_question",
            "keywords": ["어디서", "얼마나", "비용", "상담", "변호사", "소송", "제기"],
            "bonus_score": 3.0,
            "priority": 1
        }
    ]
    
    rule_count = 0
    for rule in conflict_rules:
        success = db_manager.add_conflict_resolution_rule(
            conflict_type=rule["conflict_type"],
            target_type=rule["target_type"],
            keywords=rule["keywords"],
            bonus_score=rule["bonus_score"],
            priority=rule["priority"]
        )
        if success:
            rule_count += 1
    
    print(f"   ✅ 충돌 해결 규칙 {rule_count}개 추가 완료")
    
    # 마이그레이션 결과 요약
    print(f"\n" + "=" * 60)
    print("마이그레이션 완료!")
    print("=" * 60)
    print(f"마이그레이션된 템플릿: {migrated_count}개")
    print(f"추가된 품질 지표: {indicator_count}개")
    print(f"추가된 질문 유형 설정: {config_count}개")
    print(f"추가된 충돌 해결 규칙: {rule_count}개")
    
    # 통계 출력
    stats = db_manager.get_template_statistics()
    print(f"\n데이터베이스 통계:")
    print(f"  전체 템플릿: {stats.get('total_templates', 0)}")
    print(f"  활성 템플릿: {stats.get('active_templates', 0)}")
    print(f"  전체 섹션: {stats.get('total_sections', 0)}")
    print(f"  활성 섹션: {stats.get('active_sections', 0)}")
    
    print(f"\n질문 유형별 템플릿 수:")
    by_type = stats.get('by_question_type', {})
    for question_type, count in by_type.items():
        print(f"  {question_type}: {count}개")
    
    print(f"\n데이터베이스 파일 위치: {db_manager.db_path}")
    print("=" * 60)


def test_migrated_templates():
    """마이그레이션된 템플릿 테스트"""
    print("\n" + "=" * 60)
    print("마이그레이션된 템플릿 테스트")
    print("=" * 60)
    
    db_manager = TemplateDatabaseManager()
    
    # 테스트할 질문 유형들
    test_types = [
        "precedent_search",
        "law_inquiry", 
        "legal_advice",
        "contract_review",
        "divorce_procedure"
    ]
    
    for question_type in test_types:
        print(f"\n테스트: {question_type}")
        
        # 템플릿 조회
        template = db_manager.get_template(question_type)
        if template:
            print(f"  ✅ 템플릿 조회 성공")
            print(f"     제목: {template['title']}")
            print(f"     섹션 수: {len(template['sections'])}")
            
            for section in template['sections'][:2]:  # 상위 2개만 표시
                print(f"     - {section['name']} ({section['priority']})")
        else:
            print(f"  ❌ 템플릿 조회 실패")
    
    # 품질 지표 테스트
    print(f"\n품질 지표 테스트:")
    indicators = db_manager.get_quality_indicators()
    print(f"  지표 유형 수: {len(indicators)}")
    for indicator_type, keywords in indicators.items():
        print(f"  {indicator_type}: {len(keywords)}개 키워드")
    
    # 충돌 해결 규칙 테스트
    print(f"\n충돌 해결 규칙 테스트:")
    rules = db_manager.get_conflict_resolution_rules()
    print(f"  규칙 수: {len(rules)}")
    for conflict_type, rule in rules.items():
        print(f"  {conflict_type}: {len(rule['keywords'])}개 키워드")


def main():
    """메인 함수"""
    print("하드코딩된 템플릿 데이터베이스 마이그레이션")
    
    try:
        # 1. 템플릿 마이그레이션
        migrate_hardcoded_templates()
        
        # 2. 마이그레이션 테스트
        test_migrated_templates()
        
        print(f"\n🎉 마이그레이션이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 마이그레이션 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
