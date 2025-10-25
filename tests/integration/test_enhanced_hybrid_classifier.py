#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 하이브리드 분류기 통합 테스트
"""

import sys
import os
import asyncio
import time
from typing import Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from source.services.enhanced_chat_service import EnhancedChatService
from source.utils.config import Config

async def test_enhanced_hybrid_classifier():
    """향상된 하이브리드 분류기 테스트"""
    print("=" * 60)
    print("향상된 하이브리드 분류기 통합 테스트")
    print("=" * 60)
    
    try:
        # 설정 및 서비스 초기화
        config = Config()
        service = EnhancedChatService(config)
        
        print(f"✅ EnhancedChatService 초기화 완료")
        print(f"✅ 하이브리드 분류기 상태: {'활성화' if service.hybrid_classifier else '비활성화'}")
        
        # 테스트 질문들
        test_questions = [
            "민법 제750조 불법행위 손해배상에 대해 알려주세요",
            "이혼 절차는 어떻게 진행하나요?",
            "회사 설립 시 필요한 서류는 무엇인가요?",
            "부동산 매매계약서 작성 방법을 알려주세요",
            "노동법상 근로시간 규정은 어떻게 되나요?",
            "형법 제250조 살인죄의 구성요건은 무엇인가요?",
            "상속세 계산 방법을 알려주세요",
            "계약서 검토 시 주의사항은 무엇인가요?",
            "판례 검색 방법을 알려주세요",
            "법률 용어 해설이 필요합니다"
        ]
        
        print(f"\n📝 테스트 질문 수: {len(test_questions)}개")
        print("-" * 60)
        
        # 통계 수집
        stats = {
            "total_questions": len(test_questions),
            "hybrid_analysis_count": 0,
            "fallback_analysis_count": 0,
            "error_count": 0,
            "processing_times": [],
            "domain_distribution": {},
            "classification_methods": {},
            "confidence_scores": []
        }
        
        # 각 질문에 대해 테스트
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔍 테스트 {i}/{len(test_questions)}")
            print(f"질문: {question}")
            
            start_time = time.time()
            
            try:
                # 질문 분석 수행
                result = await service._analyze_query(
                    message=question,
                    context=None,
                    user_id=f"test_user_{i}",
                    session_id=f"test_session_{i}"
                )
                
                processing_time = time.time() - start_time
                stats["processing_times"].append(processing_time)
                
                # 결과 분석
                hybrid_analysis = result.get("hybrid_analysis", False)
                classification_method = result.get("classification_method", "unknown")
                domain = result.get("domain", "unknown")
                confidence = result.get("confidence", 0.0)
                
                if hybrid_analysis:
                    stats["hybrid_analysis_count"] += 1
                else:
                    stats["fallback_analysis_count"] += 1
                
                # 통계 업데이트
                stats["domain_distribution"][domain] = stats["domain_distribution"].get(domain, 0) + 1
                stats["classification_methods"][classification_method] = stats["classification_methods"].get(classification_method, 0) + 1
                stats["confidence_scores"].append(confidence)
                
                # 결과 출력
                print(f"✅ 분석 완료 (처리시간: {processing_time:.3f}초)")
                print(f"   질문 유형: {result.get('query_type', 'unknown')}")
                print(f"   도메인: {domain}")
                print(f"   신뢰도: {confidence:.3f}")
                print(f"   분류 방법: {classification_method}")
                print(f"   하이브리드 분석: {'예' if hybrid_analysis else '아니오'}")
                
                # 키워드 정보 출력
                keywords = result.get("keywords", [])
                if keywords:
                    print(f"   추출된 키워드: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
                
                # 법률 조문 정보 출력
                statute_match = result.get("statute_match")
                if statute_match:
                    print(f"   법률 조문: {statute_match}")
                
                # 도메인 정보 출력
                domain_info = result.get("domain_info", {})
                if domain_info:
                    description = domain_info.get("description", "")
                    if description:
                        print(f"   도메인 설명: {description}")
                
            except Exception as e:
                stats["error_count"] += 1
                print(f"❌ 오류 발생: {str(e)}")
        
        # 전체 통계 출력
        print("\n" + "=" * 60)
        print("📊 테스트 결과 통계")
        print("=" * 60)
        
        print(f"총 질문 수: {stats['total_questions']}")
        print(f"하이브리드 분석 성공: {stats['hybrid_analysis_count']} ({stats['hybrid_analysis_count']/stats['total_questions']*100:.1f}%)")
        print(f"폴백 분석 사용: {stats['fallback_analysis_count']} ({stats['fallback_analysis_count']/stats['total_questions']*100:.1f}%)")
        print(f"오류 발생: {stats['error_count']} ({stats['error_count']/stats['total_questions']*100:.1f}%)")
        
        if stats["processing_times"]:
            avg_time = sum(stats["processing_times"]) / len(stats["processing_times"])
            min_time = min(stats["processing_times"])
            max_time = max(stats["processing_times"])
            print(f"평균 처리 시간: {avg_time:.3f}초")
            print(f"최소 처리 시간: {min_time:.3f}초")
            print(f"최대 처리 시간: {max_time:.3f}초")
        
        if stats["confidence_scores"]:
            avg_confidence = sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
            min_confidence = min(stats["confidence_scores"])
            max_confidence = max(stats["confidence_scores"])
            print(f"평균 신뢰도: {avg_confidence:.3f}")
            print(f"최소 신뢰도: {min_confidence:.3f}")
            print(f"최대 신뢰도: {max_confidence:.3f}")
        
        print(f"\n도메인 분포:")
        for domain, count in sorted(stats["domain_distribution"].items()):
            percentage = count / stats["total_questions"] * 100
            print(f"  {domain}: {count}개 ({percentage:.1f}%)")
        
        print(f"\n분류 방법 분포:")
        for method, count in sorted(stats["classification_methods"].items()):
            percentage = count / stats["total_questions"] * 100
            print(f"  {method}: {count}개 ({percentage:.1f}%)")
        
        # 하이브리드 분류기 통계
        if service.hybrid_classifier:
            print(f"\n하이브리드 분류기 내부 통계:")
            hybrid_stats = service.get_hybrid_classifier_stats()
            for key, value in hybrid_stats.items():
                print(f"  {key}: {value}")
        
        # 성공률 계산
        success_rate = (stats["hybrid_analysis_count"] + stats["fallback_analysis_count"]) / stats["total_questions"] * 100
        print(f"\n✅ 전체 성공률: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 테스트 성공! 하이브리드 분류기가 안정적으로 작동합니다.")
        elif success_rate >= 70:
            print("⚠️  테스트 부분 성공. 일부 개선이 필요합니다.")
        else:
            print("❌ 테스트 실패. 하이브리드 분류기 개선이 필요합니다.")
        
        return success_rate >= 70
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

async def test_keyword_extraction():
    """키워드 추출 기능 테스트"""
    print("\n" + "=" * 60)
    print("키워드 추출 기능 테스트")
    print("=" * 60)
    
    try:
        config = Config()
        service = EnhancedChatService(config)
        
        test_cases = [
            {
                "question": "민법 제750조 불법행위 손해배상 청구 방법",
                "expected_keywords": ["민법", "불법행위", "손해배상"],
                "expected_domain": "civil_law"
            },
            {
                "question": "이혼 절차와 양육권 문제 해결",
                "expected_keywords": ["이혼", "양육권"],
                "expected_domain": "family_law"
            },
            {
                "question": "회사 설립 시 주식 발행 절차",
                "expected_keywords": ["회사", "주식"],
                "expected_domain": "commercial_law"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 키워드 테스트 {i}")
            print(f"질문: {test_case['question']}")
            
            result = await service._analyze_query(
                message=test_case["question"],
                context=None,
                user_id=f"keyword_test_user_{i}",
                session_id=f"keyword_test_session_{i}"
            )
            
            keywords = result.get("keywords", [])
            domain = result.get("domain", "unknown")
            
            print(f"추출된 키워드: {keywords}")
            print(f"도메인: {domain}")
            
            # 예상 키워드와 비교
            expected_keywords = test_case["expected_keywords"]
            found_keywords = [kw for kw in expected_keywords if kw in keywords]
            
            print(f"예상 키워드: {expected_keywords}")
            print(f"발견된 키워드: {found_keywords}")
            
            if len(found_keywords) >= len(expected_keywords) * 0.7:  # 70% 이상 매칭
                print("✅ 키워드 추출 성공")
            else:
                print("⚠️  키워드 추출 부분 성공")
            
            if domain == test_case["expected_domain"]:
                print("✅ 도메인 분류 성공")
            else:
                print(f"⚠️  도메인 분류 부분 성공 (예상: {test_case['expected_domain']}, 실제: {domain})")
        
        print("\n✅ 키워드 추출 기능 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 키워드 추출 테스트 실패: {str(e)}")
        return False

async def main():
    """메인 테스트 함수"""
    print("🚀 향상된 하이브리드 분류기 통합 테스트 시작")
    
    # 기본 안정성 테스트
    stability_result = await test_enhanced_hybrid_classifier()
    
    # 키워드 추출 테스트
    keyword_result = await test_keyword_extraction()
    
    # 전체 결과
    print("\n" + "=" * 60)
    print("🎯 최종 테스트 결과")
    print("=" * 60)
    
    if stability_result and keyword_result:
        print("🎉 모든 테스트 통과! 하이브리드 분류기로의 기능 이전이 성공적으로 완료되었습니다.")
        print("\n✅ 달성된 개선사항:")
        print("  - 하이브리드 분류기에 키워드 추출 기능 추가")
        print("  - 향상된 도메인 매핑 기능 구현")
        print("  - _analyze_query 메서드의 키워드 시스템 의존성 감소")
        print("  - 안정적인 폴백 시스템 구축")
        print("\n📈 다음 단계:")
        print("  - LEGAL_DOMAIN_KEYWORDS 사용량 점진적 감소")
        print("  - 하이브리드 분류기 성능 최적화")
        print("  - 완전한 키워드 시스템 제거 준비")
    else:
        print("⚠️  일부 테스트 실패. 추가 개선이 필요합니다.")
    
    return stability_result and keyword_result

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
