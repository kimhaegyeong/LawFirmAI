#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatService 검증 시스템 통합 테스트
"""

import sys
import os
import logging

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_chat_service_validation():
    """ChatService 검증 시스템 통합 테스트"""
    print("=" * 60)
    print("ChatService 검증 시스템 통합 테스트")
    print("=" * 60)
    
    try:
        from source.services.chat_service import ChatService
        from source.utils.config import Config
        
        # 테스트 질문들
        test_questions = [
            "계약서 작성에 대해 알려주세요.",
            "이혼 절차는 어떻게 진행되나요?",
            "형사 사건에서 변호사 선임은 필수인가요?",
        ]
        
        config = Config()
        chat_service = ChatService(config)
        
        print("\n" + "-" * 40)
        print("질문별 검증 결과 분석")
        print("-" * 40)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[질문 {i}] {question}")
            print("-" * 30)
            
            # 각 검증 시스템 개별 테스트
            print("개별 검증 시스템 테스트:")
            
            # 1. MultiStageValidationSystem
            if chat_service.multi_stage_validation_system:
                multi_result = chat_service.multi_stage_validation_system.validate(question)
                print(f"  MultiStage: {'제한' if multi_result.final_decision.value == 'restricted' else '허용'}")
            
            # 2. ImprovedLegalRestrictionSystem
            if chat_service.improved_legal_restriction_system:
                improved_result = chat_service.improved_legal_restriction_system.check_restrictions(question)
                print(f"  ImprovedRestriction: {'제한' if improved_result.is_restricted else '허용'}")
            
            # 3. ContentFilterEngine
            if chat_service.content_filter_engine:
                filter_result = chat_service.content_filter_engine.filter_content(question)
                print(f"  ContentFilter: {'차단' if filter_result.is_blocked else '허용'}")
                print(f"    의도: {filter_result.intent_analysis.intent_type}")
                print(f"    신뢰도: {filter_result.intent_analysis.confidence:.3f}")
            
            # 4. IntentBasedProcessor
            if chat_service.intent_based_processor and chat_service.improved_legal_restriction_system:
                improved_result = chat_service.improved_legal_restriction_system.check_restrictions(question)
                processing_result = chat_service.intent_based_processor.process_by_intent(question, improved_result)
                print(f"  IntentProcessor: {'허용' if processing_result.allowed else '제한'}")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("ChatService 검증 시스템 통합 테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    # Windows 콘솔 인코딩 설정
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    
    test_chat_service_validation()
