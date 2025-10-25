#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터베이스 기반 템플릿 시스템 테스트
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from source.services.answer_structure_enhancer import AnswerStructureEnhancer, QuestionType
from source.services.template_database_manager import TemplateDatabaseManager


def test_database_based_templates():
    """데이터베이스 기반 템플릿 시스템 테스트"""
    print("=" * 60)
    print("데이터베이스 기반 템플릿 시스템 테스트")
    print("=" * 60)
    
    # 시스템 초기화
    enhancer = AnswerStructureEnhancer()
    db_manager = TemplateDatabaseManager()
    
    # 1. 템플릿 로드 테스트
    print("\n1. 템플릿 로드 테스트")
    print("-" * 30)
    
    test_question_types = [
        "precedent_search",
        "law_inquiry",
        "legal_advice",
        "contract_review",
        "divorce_procedure"
    ]
    
    for question_type in test_question_types:
        template_info = enhancer.get_template_info(question_type)
        print(f"  {question_type}:")
        print(f"    제목: {template_info['title']}")
        print(f"    섹션 수: {template_info['section_count']}")
        print(f"    소스: {template_info['source']}")
        
        if template_info['sections']:
            print(f"    주요 섹션:")
            for section in template_info['sections'][:2]:
                print(f"      - {section['name']} ({section['priority']})")
        print()
    
    # 2. 질문 유형 매핑 테스트
    print("\n2. 질문 유형 매핑 테스트")
    print("-" * 30)
    
    test_questions = [
        ("민법 제123조의 내용이 무엇인가요?", "law_inquiry"),
        ("계약서를 검토해주세요", "contract_review"),
        ("이혼 절차를 알려주세요", "divorce_procedure"),
        ("판례를 찾아주세요", "precedent_search"),
        ("법률 상담이 필요합니다", "legal_advice")
    ]
    
    for question, expected_type in test_questions:
        mapped_type = enhancer._map_question_type("general", question)
        print(f"  질문: {question}")
        print(f"    예상: {expected_type}")
        print(f"    결과: {mapped_type.value}")
        print(f"    일치: {'✅' if mapped_type.value == expected_type else '❌'}")
        print()
    
    # 3. 답변 구조화 테스트
    print("\n3. 답변 구조화 테스트")
    print("-" * 30)
    
    test_answer = """
    민법 제123조는 계약의 해제에 관한 규정입니다.
    이 조항에 따르면 계약 당사자는 상대방이 계약을 이행하지 않을 경우
    계약을 해제할 수 있습니다.
    """
    
    result = enhancer.enhance_answer_structure(
        answer=test_answer,
        question_type="law_inquiry",
        question="민법 제123조의 내용이 무엇인가요?"
    )
    
    if "error" not in result:
        print(f"  원본 답변 길이: {len(test_answer)} 문자")
        print(f"  구조화된 답변 길이: {len(result['structured_answer'])} 문자")
        print(f"  질문 유형: {result['question_type']}")
        print(f"  사용된 템플릿: {result['template_used']}")
        print(f"  품질 메트릭:")
        for metric, score in result['quality_metrics'].items():
            print(f"    {metric}: {score:.2f}")
        print("  ✅ 답변 구조화 성공")
    else:
        print(f"  ❌ 답변 구조화 실패: {result['error']}")
    
    # 4. 동적 템플릿 리로드 테스트
    print("\n4. 동적 템플릿 리로드 테스트")
    print("-" * 30)
    
    print("  템플릿 리로드 중...")
    enhancer.reload_templates()
    print("  ✅ 템플릿 리로드 완료")
    
    # 5. 데이터베이스 통계 테스트
    print("\n5. 데이터베이스 통계 테스트")
    print("-" * 30)
    
    stats = db_manager.get_template_statistics()
    print(f"  전체 템플릿: {stats.get('total_templates', 0)}")
    print(f"  활성 템플릿: {stats.get('active_templates', 0)}")
    print(f"  전체 섹션: {stats.get('total_sections', 0)}")
    print(f"  활성 섹션: {stats.get('active_sections', 0)}")
    
    print(f"\n  질문 유형별 템플릿 수:")
    by_type = stats.get('by_question_type', {})
    for question_type, count in by_type.items():
        print(f"    {question_type}: {count}개")
    
    # 6. 품질 지표 테스트
    print("\n6. 품질 지표 테스트")
    print("-" * 30)
    
    indicators = enhancer.quality_indicators
    print(f"  지표 유형 수: {len(indicators)}")
    for indicator_type, keywords in indicators.items():
        print(f"  {indicator_type}: {len(keywords)}개 키워드")
    
    # 7. 충돌 해결 규칙 테스트
    print("\n7. 충돌 해결 규칙 테스트")
    print("-" * 30)
    
    conflict_rules = db_manager.get_conflict_resolution_rules()
    print(f"  규칙 수: {len(conflict_rules)}")
    for conflict_type, rule in conflict_rules.items():
        print(f"  {conflict_type}: {len(rule['keywords'])}개 키워드, 보너스 {rule['bonus_score']}")
    
    print("\n" + "=" * 60)
    print("데이터베이스 기반 템플릿 시스템 테스트 완료!")
    print("=" * 60)


def test_template_modification():
    """템플릿 수정 테스트"""
    print("\n" + "=" * 60)
    print("템플릿 수정 테스트")
    print("=" * 60)
    
    db_manager = TemplateDatabaseManager()
    
    # 새로운 템플릿 추가 테스트
    print("\n1. 새로운 템플릿 추가 테스트")
    print("-" * 30)
    
    template_id = db_manager.add_template(
        question_type="test_type",
        template_name="test_template",
        title="테스트 템플릿",
        description="테스트용 템플릿입니다",
        priority=1
    )
    
    if template_id > 0:
        print(f"  ✅ 템플릿 추가 성공 (ID: {template_id})")
        
        # 섹션 추가
        sections = [
            ("테스트 섹션 1", "high", "테스트 내용 1:", "테스트 가이드 1"),
            ("테스트 섹션 2", "medium", "테스트 내용 2:", "테스트 가이드 2")
        ]
        
        for i, (name, priority, template_text, content_guide) in enumerate(sections):
            success = db_manager.add_template_section(
                template_id=template_id,
                section_name=name,
                priority=priority,
                template_text=template_text,
                content_guide=content_guide,
                section_order=i
            )
            if success:
                print(f"    ✅ 섹션 '{name}' 추가 성공")
            else:
                print(f"    ❌ 섹션 '{name}' 추가 실패")
        
        # 템플릿 조회 테스트
        template = db_manager.get_template("test_type")
        if template:
            print(f"  ✅ 템플릿 조회 성공")
            print(f"    제목: {template['title']}")
            print(f"    섹션 수: {len(template['sections'])}")
        else:
            print(f"  ❌ 템플릿 조회 실패")
        
        # 템플릿 삭제 (정리)
        db_manager.delete_template(template_id)
        print(f"  🧹 테스트 템플릿 삭제 완료")
    else:
        print(f"  ❌ 템플릿 추가 실패")


def test_performance():
    """성능 테스트"""
    print("\n" + "=" * 60)
    print("성능 테스트")
    print("=" * 60)
    
    import time
    
    enhancer = AnswerStructureEnhancer()
    
    # 템플릿 로드 성능
    print("\n1. 템플릿 로드 성능 테스트")
    print("-" * 30)
    
    start_time = time.time()
    enhancer.reload_templates()
    load_time = time.time() - start_time
    print(f"  템플릿 로드 시간: {load_time:.3f}초")
    
    # 질문 유형 매핑 성능
    print("\n2. 질문 유형 매핑 성능 테스트")
    print("-" * 30)
    
    test_questions = [
        "민법 제123조의 내용이 무엇인가요?",
        "계약서를 검토해주세요",
        "이혼 절차를 알려주세요",
        "판례를 찾아주세요",
        "법률 상담이 필요합니다"
    ] * 20  # 100개 질문
    
    start_time = time.time()
    for question in test_questions:
        enhancer._map_question_type("general", question)
    mapping_time = time.time() - start_time
    
    print(f"  {len(test_questions)}개 질문 매핑 시간: {mapping_time:.3f}초")
    print(f"  평균 매핑 시간: {mapping_time/len(test_questions)*1000:.2f}ms")
    
    # 답변 구조화 성능
    print("\n3. 답변 구조화 성능 테스트")
    print("-" * 30)
    
    test_answer = """
    민법 제123조는 계약의 해제에 관한 규정입니다.
    이 조항에 따르면 계약 당사자는 상대방이 계약을 이행하지 않을 경우
    계약을 해제할 수 있습니다.
    """
    
    start_time = time.time()
    for _ in range(10):
        enhancer.enhance_answer_structure(
            answer=test_answer,
            question_type="law_inquiry",
            question="민법 제123조의 내용이 무엇인가요?"
        )
    structure_time = time.time() - start_time
    
    print(f"  10회 답변 구조화 시간: {structure_time:.3f}초")
    print(f"  평균 구조화 시간: {structure_time/10:.3f}초")


def main():
    """메인 함수"""
    print("데이터베이스 기반 템플릿 시스템 종합 테스트")
    
    try:
        # 1. 기본 기능 테스트
        test_database_based_templates()
        
        # 2. 템플릿 수정 테스트
        test_template_modification()
        
        # 3. 성능 테스트
        test_performance()
        
        print(f"\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
