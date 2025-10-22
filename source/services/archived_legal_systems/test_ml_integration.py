#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 통합 검증 시스템 테스트
민법 제750조 설명 요청 문제 해결 테스트
"""

import sys
import os
import asyncio
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from source.services.ml_integrated_validation_system import MLIntegratedValidationSystem
from source.services.chat_service import ChatService
from source.utils.config import Config

def test_ml_integrated_system():
    """ML 통합 시스템 직접 테스트"""
    print("=" * 60)
    print("ML 통합 검증 시스템 직접 테스트")
    print("=" * 60)
    
    try:
        # ML 통합 시스템 초기화
        ml_system = MLIntegratedValidationSystem()
        print("✓ ML 통합 시스템 초기화 성공")
        
        # 테스트 쿼리들
        test_queries = [
            "민법 제 750조에 대해서 설명해줘",
            "민법 제750조의 내용이 무엇인가요?",
            "불법행위로 인한 손해배상에 대해 알려주세요",
            "제 경우 어떻게 해야 할까요?",
            "소송을 해야 할까요?",
            "변호사를 고용해야 할까요?",
            "의료사고 과실이 있나요?",
            "형량은 몇 년일까요?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. 테스트 쿼리: {query}")
            print("-" * 50)
            
            try:
                result = ml_system.validate(
                    query=query,
                    user_id="test_user",
                    session_id="test_session",
                    collect_feedback=True
                )
                
                print(f"결과: {result.get('final_decision', 'unknown')}")
                print(f"신뢰도: {result.get('confidence', 0.0):.2f}")
                print(f"Edge Case: {result.get('edge_case_info', {}).get('is_edge_case', False)}")
                print(f"이유: {result.get('reasoning', [])}")
                
                if 'ml_prediction' in result:
                    print(f"ML 예측: {result['ml_prediction']}")
                
            except Exception as e:
                print(f"✗ 오류: {e}")
        
        print("\n" + "=" * 60)
        print("ML 통합 시스템 상태 조회")
        print("=" * 60)
        
        status = ml_system.get_system_status()
        for key, value in status.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"✗ ML 통합 시스템 초기화 실패: {e}")

async def test_chat_service():
    """ChatService를 통한 통합 테스트"""
    print("\n" + "=" * 60)
    print("ChatService 통합 테스트")
    print("=" * 60)
    
    try:
        # ChatService 초기화
        config = Config()
        chat_service = ChatService(config)
        print("✓ ChatService 초기화 성공")
        
        # 테스트 쿼리들
        test_queries = [
            "민법 제 750조에 대해서 설명해줘",
            "민법 제750조의 내용이 무엇인가요?",
            "불법행위로 인한 손해배상에 대해 알려주세요"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. 테스트 쿼리: {query}")
            print("-" * 50)
            
            try:
                result = await chat_service.process_message(
                    message=query,
                    user_id="test_user",
                    session_id="test_session"
                )
                
                print(f"응답: {result.get('response', 'No response')[:100]}...")
                print(f"신뢰도: {result.get('confidence', 0.0):.2f}")
                print(f"처리 시간: {result.get('processing_time', 0.0):.2f}초")
                
                if 'validation_info' in result:
                    validation_info = result['validation_info']
                    print(f"검증 시스템: {validation_info.get('blocked_by', 'Not blocked')}")
                    print(f"Edge Case: {validation_info.get('edge_case', {}).get('is_edge_case', False)}")
                
            except Exception as e:
                print(f"✗ 오류: {e}")
                
    except Exception as e:
        print(f"✗ ChatService 초기화 실패: {e}")

def main():
    """메인 테스트 함수"""
    print("ML 통합 검증 시스템 테스트 시작")
    print(f"테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. ML 통합 시스템 직접 테스트
    test_ml_integrated_system()
    
    # 2. ChatService 통합 테스트
    asyncio.run(test_chat_service())
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)

if __name__ == "__main__":
    main()
