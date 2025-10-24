# -*- coding: utf-8 -*-
"""
Enhanced Law Search Test
향상된 조문 검색 시스템 테스트
"""

import asyncio
import sys
import os
import time
from typing import List, Dict, Any

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, '.')

print("🚀 향상된 조문 검색 시스템 테스트")
print("=" * 70)

try:
    from source.utils.config import Config
    from source.services.enhanced_chat_service import EnhancedChatService
    print("✅ 모든 모듈 import 성공")
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    sys.exit(1)


def generate_law_search_test_questions() -> List[Dict[str, Any]]:
    """조문 검색 테스트 질문 생성"""
    questions = [
        # 정확한 조문 질문
        {"question": "민법 제750조에 대해서 설명해줘", "category": "정확조문", "expected_method": "exact_match", "priority": "high"},
        
        # 항번호 포함 질문
        {"question": "민법 제750조 제1항의 내용을 알려주세요", "category": "항번호포함", "expected_method": "exact_match", "priority": "high"},
        
        # 조문번호만 질문
        {"question": "제750조에 대해 설명해줘", "category": "조문번호만", "expected_method": "fuzzy_match", "priority": "high"},
        
        # 다른 법령 질문
        {"question": "형법 제250조 살인죄에 대해 알려주세요", "category": "다른법령", "expected_method": "exact_match", "priority": "high"},
        
        # 일반적인 법률 질문 (조문 검색이 아닌 경우)
        {"question": "계약서 작성 방법을 알려주세요", "category": "일반질문", "expected_method": "general", "priority": "medium"},
    ]
    
    return questions


async def test_enhanced_law_search():
    """향상된 조문 검색 테스트"""
    print("\n🚀 향상된 조문 검색 테스트 시작")
    print("=" * 50)
    
    try:
        # 설정 로드
        config = Config()
        print("✅ Config 로드 성공")
        
        # Enhanced Chat Service 초기화
        chat_service = EnhancedChatService(config)
        print("✅ Enhanced Chat Service 초기화 성공")
        
        # 테스트 질문 생성
        test_questions = generate_law_search_test_questions()
        print(f"📝 총 {len(test_questions)}개의 조문 검색 테스트 질문 생성")
        
        # 테스트 실행
        results = []
        start_time = time.time()
        
        print(f"\n🔄 향상된 조문 검색 테스트 실행 중...")
        print("-" * 50)
        
        for i, test_case in enumerate(test_questions, 1):
            question = test_case["question"]
            category = test_case["category"]
            expected_method = test_case["expected_method"]
            priority = test_case["priority"]
            
            print(f"\n질문 {i}: {question}")
            print(f"카테고리: {category} | 예상방법: {expected_method} | 우선순위: {priority}")
            
            try:
                # 메시지 처리
                result = await chat_service.process_message(
                    message=question,
                    user_id=f"law_search_test_user_{i}",
                    session_id=f"law_search_test_session_{i}"
                )
                
                # 결과 분석
                response = result.get('response', 'N/A')
                confidence = result.get('confidence', 0.0)
                processing_time = result.get('processing_time', 0.0)
                generation_method = result.get('generation_method', 'unknown')
                sources = result.get('sources', [])
                additional_options = result.get('additional_options', [])
                has_more_detail = result.get('has_more_detail', False)
                
                print(f"응답 길이: {len(response)}자")
                print(f"신뢰도: {confidence:.2f}")
                print(f"처리 시간: {processing_time:.3f}초")
                print(f"생성 방법: {generation_method}")
                print(f"검색 결과 수: {len(sources)}")
                print(f"추가 옵션 수: {len(additional_options)}")
                print(f"더 자세한 정보: {has_more_detail}")
                
                if sources:
                    print(f"검색 소스: {sources[0].get('law_name', 'N/A')} 제{sources[0].get('article_number', 'N/A')}조")
                
                if additional_options:
                    print(f"추가 옵션: {[opt.title if hasattr(opt, 'title') else str(opt) for opt in additional_options]}")
                
                print("-" * 80)
                
                # 결과 저장
                results.append({
                    'test_case': test_case,
                    'result': result,
                    'success': True,
                    'processing_time': processing_time,
                    'confidence': confidence,
                    'generation_method': generation_method,
                    'sources_count': len(sources),
                    'additional_options_count': len(additional_options),
                    'has_more_detail': has_more_detail
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
        print(f"\n📊 향상된 조문 검색 테스트 결과")
        print("=" * 50)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        failed_tests = total_tests - successful_tests
        
        print(f"총 테스트: {total_tests}")
        print(f"성공한 테스트: {successful_tests}")
        print(f"실패한 테스트: {failed_tests}")
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
        
        # 카테고리별 분석
        print(f"\n📊 카테고리별 분석")
        print("-" * 30)
        
        categories = {}
        for result in results:
            if result['success']:
                category = result['test_case']['category']
                if category not in categories:
                    categories[category] = {'total': 0, 'success': 0, 'avg_time': 0, 'avg_conf': 0, 'methods': []}
                
                categories[category]['total'] += 1
                categories[category]['success'] += 1
                categories[category]['avg_time'] += result.get('processing_time', 0)
                categories[category]['avg_conf'] += result.get('confidence', 0)
                categories[category]['methods'].append(result.get('generation_method', 'unknown'))
        
        for category, stats in categories.items():
            success_rate = (stats['success'] / stats['total']) * 100
            avg_time = stats['avg_time'] / stats['success'] if stats['success'] > 0 else 0
            avg_conf = stats['avg_conf'] / stats['success'] if stats['success'] > 0 else 0
            
            print(f"{category}: {stats['success']}/{stats['total']} 성공 ({success_rate:.1f}%), 평균시간 {avg_time:.3f}초, 평균신뢰도 {avg_conf:.2f}")
            print(f"  사용된 방법: {set(stats['methods'])}")
        
        # 개선 효과 분석
        print(f"\n🎯 개선 효과 분석")
        print("-" * 30)
        
        exact_match_results = [r for r in results if r['success'] and r.get('generation_method') == 'integrated_law_search']
        general_results = [r for r in results if r['success'] and r.get('generation_method') != 'integrated_law_search']
        
        if exact_match_results:
            exact_avg_conf = sum(r.get('confidence', 0) for r in exact_match_results) / len(exact_match_results)
            print(f"통합 조문 검색 평균 신뢰도: {exact_avg_conf:.2f}")
        
        if general_results:
            general_avg_conf = sum(r.get('confidence', 0) for r in general_results) / len(general_results)
            print(f"일반 검색 평균 신뢰도: {general_avg_conf:.2f}")
        
        # 추가 기능 활용도
        total_additional_options = sum(r.get('additional_options_count', 0) for r in results if r['success'])
        total_more_detail = sum(1 for r in results if r.get('has_more_detail', False))
        
        print(f"총 추가 옵션 제공: {total_additional_options}개")
        print(f"더 자세한 정보 제공 가능: {total_more_detail}개")
        
        print(f"\n✅ 향상된 조문 검색 테스트 완료!")
        
        return results
        
    except Exception as e:
        print(f"❌ 향상된 조문 검색 테스트 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return []


if __name__ == "__main__":
    print("🚀 Enhanced Law Search Test")
    print("=" * 80)
    
    # 향상된 조문 검색 테스트 실행
    results = asyncio.run(test_enhanced_law_search())
    
    print("\n🎉 향상된 조문 검색 테스트 완료!")
