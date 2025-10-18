#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 성능 최적화 테스트
더 작은 모델과 최적화된 설정으로 성능 개선 테스트
"""

import os
import sys
import asyncio
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 환경 변수 설정
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"

# 테스트 질문들 (간소화)
TEST_QUESTIONS = [
    "계약서 작성 시 주의사항은?",
    "이혼 절차는 어떻게 되나요?",
    "절도죄의 구성요건은?",
    "손해배상 청구 방법은?",
    "부당해고 구제 절차는?",
    "부동산 매매계약서 필수 조항은?",
    "특허권 침해 시 구제 방법은?",
    "소득세 신고 누락 시 가산세는?",
    "법정대리인의 권한은?",
    "소송 제기 시 관할 법원은?"
]

async def test_optimized_performance():
    """최적화된 성능 테스트"""
    print("=" * 80)
    print("LangGraph 성능 최적화 테스트")
    print("=" * 80)
    print("변경사항:")
    print("- 모델: qwen2.5:7b → qwen2.5:3b (더 작은 모델)")
    print("- 응답 길이: 200 → 100 토큰")
    print("- 타임아웃: 30초 → 15초")
    print("- Temperature: 0.3 (일관성 향상)")
    print("=" * 80)
    
    try:
        from source.services.langgraph.workflow_service import LangGraphWorkflowService
        from source.utils.langgraph_config import LangGraphConfig
        
        # 임시 데이터베이스 파일 사용
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        # 설정 생성
        config = LangGraphConfig.from_env()
        config.checkpoint_db_path = db_path
        
        # 워크플로우 서비스 초기화
        service = LangGraphWorkflowService(config)
        print("✅ 최적화된 워크플로우 서비스 초기화 완료")
        
        # 테스트 실행
        results = []
        total_start_time = time.time()
        
        print(f"\n🚀 {len(TEST_QUESTIONS)}개 질문 테스트 시작...")
        
        for i, question in enumerate(TEST_QUESTIONS, 1):
            print(f"\n[{i:2d}/{len(TEST_QUESTIONS)}] {question}")
            
            start_time = time.time()
            result = await service.process_query(question)
            processing_time = time.time() - start_time
            
            results.append({
                "question_id": i,
                "question": question,
                "processing_time": processing_time,
                "response_length": len(result.get("answer", "")),
                "confidence": result.get("confidence", 0),
                "query_type": result.get("query_type", "unknown"),
                "errors": result.get("errors", [])
            })
            
            print(f"    ⚡ 처리시간: {processing_time:.2f}초")
            print(f"    📝 답변 길이: {len(result.get('answer', ''))}자")
            print(f"    🎯 신뢰도: {result.get('confidence', 0):.2f}")
            print(f"    📋 질문 유형: {result.get('query_type', 'unknown')}")
            
            if result.get("errors"):
                print(f"    ❌ 오류: {result['errors']}")
            
            # 진행률 표시
            if i % 3 == 0:
                avg_time = sum(r["processing_time"] for r in results) / len(results)
                print(f"\n📊 진행률: {i}/{len(TEST_QUESTIONS)} - 평균 처리시간: {avg_time:.2f}초")
        
        total_time = time.time() - total_start_time
        
        # 결과 분석
        analyze_performance_results(results, total_time)
        
        # 임시 파일 정리
        os.unlink(db_path)
        
        return results
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return []

def analyze_performance_results(results: List[Dict[str, Any]], total_time: float):
    """성능 결과 분석"""
    print("\n" + "=" * 80)
    print("성능 최적화 결과 분석")
    print("=" * 80)
    
    if not results:
        print("❌ 분석할 결과가 없습니다.")
        return
    
    # 기본 통계
    total_questions = len(results)
    avg_processing_time = sum(r["processing_time"] for r in results) / total_questions
    min_time = min(r["processing_time"] for r in results)
    max_time = max(r["processing_time"] for r in results)
    avg_response_length = sum(r["response_length"] for r in results) / total_questions
    avg_confidence = sum(r["confidence"] for r in results) / total_questions
    
    print(f"📊 성능 통계:")
    print(f"   - 총 질문 수: {total_questions}")
    print(f"   - 총 소요 시간: {total_time:.2f}초")
    print(f"   - 평균 처리 시간: {avg_processing_time:.2f}초")
    print(f"   - 최단 처리 시간: {min_time:.2f}초")
    print(f"   - 최장 처리 시간: {max_time:.2f}초")
    print(f"   - 평균 답변 길이: {avg_response_length:.0f}자")
    print(f"   - 평균 신뢰도: {avg_confidence:.3f}")
    print(f"   - 초당 처리 질문 수: {total_questions/total_time:.2f}개")
    
    # 처리 시간 분포
    fast_tests = [r for r in results if r["processing_time"] < 5]
    medium_tests = [r for r in results if 5 <= r["processing_time"] < 15]
    slow_tests = [r for r in results if r["processing_time"] >= 15]
    
    print(f"\n⏱️ 처리 시간 분포:")
    print(f"   - 매우 빠름 (<5초): {len(fast_tests)}개 ({len(fast_tests)/total_questions*100:.1f}%)")
    print(f"   - 빠름 (5-15초): {len(medium_tests)}개 ({len(medium_tests)/total_questions*100:.1f}%)")
    print(f"   - 보통 (≥15초): {len(slow_tests)}개 ({len(slow_tests)/total_questions*100:.1f}%)")
    
    # 질문 유형별 성능
    query_types = {}
    for r in results:
        qtype = r["query_type"]
        if qtype not in query_types:
            query_types[qtype] = []
        query_types[qtype].append(r["processing_time"])
    
    print(f"\n📋 질문 유형별 평균 처리 시간:")
    for qtype, times in query_types.items():
        avg_time = sum(times) / len(times)
        print(f"   - {qtype}: {avg_time:.2f}초 ({len(times)}개)")
    
    # 성능 개선 평가
    print(f"\n🏆 성능 개선 평가:")
    if avg_processing_time < 5:
        print("   ✅ 우수 - 평균 5초 미만")
    elif avg_processing_time < 10:
        print("   ✅ 양호 - 평균 10초 미만")
    elif avg_processing_time < 20:
        print("   ⚠️ 보통 - 평균 20초 미만")
    else:
        print("   ❌ 개선 필요 - 평균 20초 이상")
    
    # 이전 성능과 비교
    previous_avg = 28.04  # 이전 테스트 결과
    improvement = ((previous_avg - avg_processing_time) / previous_avg) * 100
    
    print(f"\n📈 이전 성능 대비 개선:")
    print(f"   - 이전 평균: {previous_avg:.2f}초")
    print(f"   - 현재 평균: {avg_processing_time:.2f}초")
    print(f"   - 개선율: {improvement:.1f}%")
    
    if improvement > 0:
        print(f"   ✅ 성능이 {improvement:.1f}% 향상되었습니다!")
    else:
        print(f"   ⚠️ 성능이 {abs(improvement):.1f}% 저하되었습니다.")
    
    # 추가 최적화 권장사항
    print(f"\n💡 추가 최적화 권장사항:")
    if avg_processing_time > 10:
        print("   1. 더 작은 모델 사용 (qwen2.5:1.5b)")
        print("   2. 응답 길이 더 제한 (num_predict=50)")
        print("   3. 캐싱 시스템 도입")
        print("   4. 프롬프트 템플릿 최적화")
    
    if len(slow_tests) > total_questions * 0.3:
        print("   5. 느린 질문들에 대한 특별 처리 로직 필요")
    
    print("   6. 비동기 처리 및 병렬 처리 도입")
    print("   7. 응답 스트리밍 구현")

async def main():
    """메인 함수"""
    results = await test_optimized_performance()
    
    if results:
        print(f"\n🎉 성능 최적화 테스트 완료!")
        avg_time = sum(r["processing_time"] for r in results) / len(results)
        print(f"평균 처리 시간: {avg_time:.2f}초")
        
        if avg_time < 15:
            print("🚀 성능이 크게 개선되었습니다!")
        else:
            print("⚠️ 추가 최적화가 필요합니다.")
    else:
        print("❌ 테스트 실행 실패")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # 로그 레벨 낮춤
    
    asyncio.run(main())
