#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatService 테스트 스크립트
"""

import sys
import os
import asyncio
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from source.utils.config import Config
from source.services.chat_service import ChatService

async def test_chat_service():
    """ChatService 테스트"""
    print("=" * 60)
    print("ChatService 테스트 시작")
    print("=" * 60)
    
    try:
        # 설정 로드
        config = Config()
        
        # ChatService 초기화
        print("ChatService 초기화 중...")
        chat_service = ChatService(config)
        print("ChatService 초기화 완료!")
        
        # 서비스 상태 확인
        print("\n" + "-" * 40)
        print("서비스 상태 확인")
        print("-" * 40)
        status = chat_service.get_service_status()
        print(f"서비스 이름: {status.get('service_name', 'Unknown')}")
        print(f"LangGraph 활성화: {status.get('langgraph_enabled', False)}")
        print(f"전체 상태: {status.get('overall_status', 'Unknown')}")
        
        # RAG 컴포넌트 상태
        rag_components = status.get('rag_components', {})
        print(f"RAG 서비스: {'✓' if rag_components.get('rag_service') else '✗'}")
        print(f"하이브리드 검색 엔진: {'✓' if rag_components.get('hybrid_search_engine') else '✗'}")
        print(f"질문 분류기: {'✓' if rag_components.get('question_classifier') else '✗'}")
        print(f"답변 생성기: {'✓' if rag_components.get('improved_answer_generator') else '✗'}")
        
        # Phase 컴포넌트 상태
        phase1_components = status.get('phase1_components', {})
        print(f"세션 관리자: {'✓' if phase1_components.get('session_manager') else '✗'}")
        print(f"다중 턴 핸들러: {'✓' if phase1_components.get('multi_turn_handler') else '✗'}")
        print(f"컨텍스트 압축기: {'✓' if phase1_components.get('context_compressor') else '✗'}")
        
        # 테스트 질문들
        test_questions = [
            "안녕하세요! 법률 상담을 받고 싶습니다.",
            "계약서 작성에 대해 알려주세요.",
            "부동산 매매 계약 시 주의사항이 있나요?",
            "이혼 절차는 어떻게 진행되나요?",
            "형사 사건에서 변호사 선임은 필수인가요?"
        ]
        
        print("\n" + "-" * 40)
        print("질문 테스트 시작")
        print("-" * 40)
        
        session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        user_id = "test_user"
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[테스트 {i}] 질문: {question}")
            print("-" * 30)
            
            try:
                # 메시지 처리
                start_time = datetime.now()
                result = await chat_service.process_message(
                    message=question,
                    session_id=session_id,
                    user_id=user_id
                )
                end_time = datetime.now()
                
                # 결과 출력
                print(f"응답: {result.get('response', '응답 없음')[:200]}...")
                print(f"신뢰도: {result.get('confidence', 0.0):.2f}")
                print(f"처리 시간: {result.get('processing_time', 0.0):.2f}초")
                print(f"소스 수: {len(result.get('sources', []))}")
                
                # 제한 정보 확인
                restriction_info = result.get('restriction_info')
                if restriction_info:
                    print(f"제한 여부: {'제한됨' if restriction_info.get('is_restricted') else '허용됨'}")
                    if restriction_info.get('is_restricted'):
                        print(f"제한 수준: {restriction_info.get('restriction_level', 'unknown')}")
                
                # Phase 정보 확인
                phase_info = result.get('phase_info', {})
                print(f"Phase 1 활성화: {'✓' if phase_info.get('phase1', {}).get('enabled') else '✗'}")
                print(f"Phase 2 활성화: {'✓' if phase_info.get('phase2', {}).get('enabled') else '✗'}")
                print(f"Phase 3 활성화: {'✓' if phase_info.get('phase3', {}).get('enabled') else '✗'}")
                
                # 오류 확인
                errors = result.get('errors', [])
                if errors:
                    print(f"오류: {errors}")
                
            except Exception as e:
                print(f"오류 발생: {str(e)}")
        
        # 성능 메트릭 확인
        print("\n" + "-" * 40)
        print("성능 메트릭")
        print("-" * 40)
        
        try:
            metrics = chat_service.get_performance_metrics()
            print(f"메트릭 수집 시간: {metrics.get('timestamp', 'Unknown')}")
            
            performance_monitor = metrics.get('performance_monitor', {})
            if performance_monitor:
                summary = performance_monitor.get('summary', {})
                print(f"평균 응답 시간: {summary.get('avg_response_time', 0.0):.2f}초")
                print(f"총 요청 수: {summary.get('total_requests', 0)}")
            
            memory_optimizer = metrics.get('memory_optimizer', {})
            if memory_optimizer:
                memory_usage = memory_optimizer.get('memory_usage', {})
                print(f"메모리 사용량: {memory_usage.get('used_mb', 0.0):.2f}MB")
                print(f"메모리 비율: {memory_usage.get('percentage', 0.0):.1f}%")
        
        except Exception as e:
            print(f"성능 메트릭 조회 오류: {str(e)}")
        
        # Phase 통계 확인
        print("\n" + "-" * 40)
        print("Phase 통계")
        print("-" * 40)
        
        try:
            stats = chat_service.get_phase_statistics()
            for phase, stat in stats.items():
                print(f"{phase}: {'활성화' if stat.get('enabled') else '비활성화'}")
                if stat.get('enabled'):
                    for key, value in stat.items():
                        if key != 'enabled':
                            print(f"  {key}: {value}")
        
        except Exception as e:
            print(f"Phase 통계 조회 오류: {str(e)}")
        
        print("\n" + "=" * 60)
        print("ChatService 테스트 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Windows 콘솔 인코딩 설정
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    
    # 비동기 테스트 실행
    asyncio.run(test_chat_service())
