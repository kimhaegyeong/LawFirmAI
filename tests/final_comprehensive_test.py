# -*- coding: utf-8 -*-
"""
Final Comprehensive Answer Quality Test
최종 종합 답변 품질 테스트
"""

import asyncio
import sys
import os
import time
from typing import List, Dict, Any

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, '.')

print("🚀 최종 종합 답변 품질 테스트")
print("=" * 70)

try:
    from source.utils.config import Config
    from source.services.enhanced_chat_service import EnhancedChatService
    print("✅ 모든 모듈 import 성공")
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    sys.exit(1)


def generate_comprehensive_test_questions() -> List[Dict[str, Any]]:
    """종합 테스트 질문 생성 (5개 질문)"""
    questions = [
        # 법률 조문 질문
        {"question": "민법 제 750조에 대해서 설명해줘", "category": "법률조문", "expected_type": "statute", "priority": "high"},
        
        # 계약서 관련 질문
        {"question": "계약서 작성 방법을 알려주세요", "category": "계약서", "expected_type": "contract", "priority": "high"},
        
        # 부동산 관련 질문
        {"question": "부동산 매매 절차를 알려주세요", "category": "부동산", "expected_type": "real_estate", "priority": "high"},
        
        # 가족법 관련 질문
        {"question": "이혼 소송 절차가 궁금합니다", "category": "가족법", "expected_type": "family_law", "priority": "high"},
        
        # 민사법 관련 질문
        {"question": "손해배상 청구 방법", "category": "민사법", "expected_type": "civil_law", "priority": "medium"},
    ]
    
    return questions


async def test_comprehensive_answer_quality():
    """종합 답변 품질 테스트"""
    print("\n🚀 종합 답변 품질 테스트 시작")
    print("=" * 50)
    
    try:
        # 설정 로드
        config = Config()
        print("✅ Config 로드 성공")
        
        # Enhanced Chat Service 초기화
        chat_service = EnhancedChatService(config)
        print("✅ Enhanced Chat Service 초기화 성공")
        print(f"Chat service type: {type(chat_service)}")
        print(f"Chat service has process_message: {hasattr(chat_service, 'process_message')}")
        
        # 테스트 질문 생성
        test_questions = generate_comprehensive_test_questions()
        print(f"📝 총 {len(test_questions)}개의 종합 테스트 질문 생성")
        
        # 우선순위별 분류
        high_priority = [q for q in test_questions if q["priority"] == "high"]
        medium_priority = [q for q in test_questions if q["priority"] == "medium"]
        low_priority = [q for q in test_questions if q["priority"] == "low"]
        
        print(f"📊 우선순위별 질문 수: High({len(high_priority)}), Medium({len(medium_priority)}), Low({len(low_priority)})")
        
        # 테스트 실행
        results = []
        start_time = time.time()
        
        print(f"\n🔄 종합 답변 품질 테스트 실행 중...")
        print("-" * 50)
        
        for i, test_case in enumerate(test_questions, 1):
            question = test_case["question"]
            category = test_case["category"]
            expected_type = test_case["expected_type"]
            priority = test_case["priority"]
            
            print(f"\n질문 {i}: {question}")
            print(f"카테고리: {category} | 예상유형: {expected_type} | 우선순위: {priority}")
            
            try:
                # 메시지 처리
                result = await chat_service.process_message(
                    message=question,
                    user_id=f"comprehensive_test_user_{i}",
                    session_id=f"comprehensive_test_session_{i}"
                )
                
                # 결과 분석
                response = result.get('response', 'N/A')
                confidence = result.get('confidence', 0.0)
                processing_time = result.get('processing_time', 0.0)
                is_restricted = result.get('restricted', False)
                generation_method = result.get('generation_method', 'unknown')
                sources = result.get('sources', [])
                
                print(f"응답: {response}")
                print(f"신뢰도: {confidence:.2f}")
                print(f"처리 시간: {processing_time:.3f}초")
                print(f"제한 여부: {is_restricted}")
                print(f"생성 방법: {generation_method}")
                print(f"검색 결과 수: {len(sources)}")
                if sources:
                    print(f"검색 소스: {sources}")
                print("-" * 80)
                
                # 결과 저장
                results.append({
                    'test_case': test_case,
                    'result': result,
                    'success': True,
                    'processing_time': processing_time,
                    'confidence': confidence,
                    'is_restricted': is_restricted,
                    'generation_method': generation_method,
                    'sources_count': len(sources)
                })
                
            except Exception as e:
                print(f"❌ 질문 {i} 처리 실패: {e}")
                results.append({
                    'test_case': test_case,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        
        # 테스트 결과 분석
        print(f"\n📊 종합 답변 품질 테스트 결과")
        print("=" * 50)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        failed_tests = total_tests - successful_tests
        restricted_tests = sum(1 for r in results if r.get('is_restricted', False))
        
        print(f"총 테스트: {total_tests}")
        print(f"성공한 테스트: {successful_tests}")
        print(f"실패한 테스트: {failed_tests}")
        print(f"제한된 테스트: {restricted_tests}")
        print(f"총 실행 시간: {total_time:.2f}초")
        
        if successful_tests > 0:
            avg_confidence = sum(r.get('confidence', 0) for r in results if r['success']) / successful_tests
            avg_processing_time = sum(r.get('processing_time', 0) for r in results if r['success']) / successful_tests
            
            print(f"평균 신뢰도: {avg_confidence:.2f}")
            print(f"평균 처리 시간: {avg_processing_time:.3f}초")
        
        # 생성 방법별 분석
        print(f"\n🔧 생성 방법별 분석")
        print("-" * 30)
        
        generation_methods = {}
        for result in results:
            if result['success']:
                method = result.get('generation_method', 'unknown')
                if method not in generation_methods:
                    generation_methods[method] = {'count': 0, 'total_confidence': 0, 'avg_confidence': 0, 'avg_time': 0}
                generation_methods[method]['count'] += 1
                generation_methods[method]['total_confidence'] += result.get('confidence', 0)
                generation_methods[method]['avg_time'] += result.get('processing_time', 0)
        
        for method, stats in generation_methods.items():
            stats['avg_confidence'] = stats['total_confidence'] / stats['count']
            stats['avg_time'] = stats['avg_time'] / stats['count']
            print(f"{method}: {stats['count']}개, 평균 신뢰도: {stats['avg_confidence']:.2f}, 평균 시간: {stats['avg_time']:.3f}초")
        
        # 우선순위별 분석
        print(f"\n📈 우선순위별 분석")
        print("-" * 30)
        
        priority_stats = {}
        for result in results:
            if result['success']:
                priority = result['test_case']['priority']
                if priority not in priority_stats:
                    priority_stats[priority] = {'total': 0, 'success': 0, 'avg_conf': 0, 'avg_time': 0}
                
                priority_stats[priority]['total'] += 1
                priority_stats[priority]['success'] += 1
                priority_stats[priority]['avg_conf'] += result.get('confidence', 0)
                priority_stats[priority]['avg_time'] += result.get('processing_time', 0)
        
        for priority, stats in priority_stats.items():
            success_rate = (stats['success'] / stats['total']) * 100
            avg_conf = stats['avg_conf'] / stats['success'] if stats['success'] > 0 else 0
            avg_time = stats['avg_time'] / stats['success'] if stats['success'] > 0 else 0
            
            print(f"{priority.upper()}: {stats['success']}/{stats['total']} 성공 ({success_rate:.1f}%), 평균신뢰도 {avg_conf:.2f}, 평균시간 {avg_time:.3f}초")
        
        # 카테고리별 분석
        print(f"\n📊 카테고리별 분석")
        print("-" * 30)
        
        categories = {}
        for result in results:
            if result['success']:
                category = result['test_case']['category']
                if category not in categories:
                    categories[category] = {'total': 0, 'success': 0, 'restricted': 0, 'avg_time': 0, 'avg_conf': 0}
                
                categories[category]['total'] += 1
                categories[category]['success'] += 1
                categories[category]['avg_time'] += result.get('processing_time', 0)
                categories[category]['avg_conf'] += result.get('confidence', 0)
                if result.get('is_restricted', False):
                    categories[category]['restricted'] += 1
        
        for category, stats in categories.items():
            success_rate = (stats['success'] / stats['total']) * 100
            restriction_rate = (stats['restricted'] / stats['total']) * 100
            avg_time = stats['avg_time'] / stats['success'] if stats['success'] > 0 else 0
            avg_conf = stats['avg_conf'] / stats['success'] if stats['success'] > 0 else 0
            
            print(f"{category}: {stats['success']}/{stats['total']} 성공 ({success_rate:.1f}%), 제한 {restriction_rate:.1f}%, 평균시간 {avg_time:.3f}초, 평균신뢰도 {avg_conf:.2f}")
        
        # 품질 개선 효과 분석
        print(f"\n🎯 품질 개선 효과 분석")
        print("-" * 30)
        
        statute_results = [r for r in results if r['success'] and r['test_case']['category'] == '법률조문']
        template_results = [r for r in results if r['success'] and 'template' in r.get('generation_method', '')]
        
        if statute_results:
            statute_avg_conf = sum(r.get('confidence', 0) for r in statute_results) / len(statute_results)
            print(f"법률 조문 질문 평균 신뢰도: {statute_avg_conf:.2f}")
        
        if template_results:
            template_avg_conf = sum(r.get('confidence', 0) for r in template_results) / len(template_results)
            print(f"템플릿 기반 답변 평균 신뢰도: {template_avg_conf:.2f}")
        
        print(f"\n✅ 종합 답변 품질 테스트 완료!")
        
        return results
        
    except Exception as e:
        print(f"❌ 종합 답변 품질 테스트 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return []


if __name__ == "__main__":
    print("🚀 Final Comprehensive Answer Quality Test")
    print("=" * 80)
    
    # 종합 테스트 실행
    results = asyncio.run(test_comprehensive_answer_quality())
    
    print("\n🎉 최종 종합 답변 품질 테스트 완료!")
