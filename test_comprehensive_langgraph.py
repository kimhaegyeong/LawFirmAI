#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 대규모 테스트 - 20개 이상 법률 질문 테스트
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

# 테스트 질문 목록
TEST_QUESTIONS = [
    # 계약법 관련
    "계약서 작성 시 주의사항은 무엇인가요?",
    "계약 해지 조건이 무엇인가요?",
    "불공정한 계약 조항은 어떻게 대응해야 하나요?",
    "계약 위반 시 손해배상 범위는 어떻게 정해지나요?",
    
    # 가족법 관련
    "이혼 절차에 대해 알려주세요",
    "위자료 산정 기준은 무엇인가요?",
    "양육비 지급 기준과 방법은?",
    "상속 포기 절차는 어떻게 되나요?",
    
    # 형사법 관련
    "절도죄의 구성요건은 무엇인가요?",
    "사기죄와 횡령죄의 차이점은?",
    "형사합의는 어떻게 진행되나요?",
    "보석 조건과 절차는?",
    
    # 민사법 관련
    "손해배상 청구 시 입증책임은 누구에게 있나요?",
    "소멸시효 중단 사유는 무엇인가요?",
    "채권자 대위권 행사 조건은?",
    "담보물권의 우선순위는 어떻게 정해지나요?",
    
    # 노동법 관련
    "부당해고 구제 절차는 어떻게 되나요?",
    "임금 체불 시 대응 방법은?",
    "근로시간 제한 규정은 무엇인가요?",
    "산업재해 인정 기준은?",
    
    # 부동산법 관련
    "부동산 매매계약서 필수 조항은?",
    "전세권 설정 절차와 효력은?",
    "임대차보호법 적용 범위는?",
    "건축허가 취소 사유는 무엇인가요?",
    
    # 지적재산권법 관련
    "특허권 침해 시 구제 방법은?",
    "상표권 등록 절차는 어떻게 되나요?",
    "저작권 침해 금지청구권 행사 방법은?",
    "영업비밀 보호 요건은 무엇인가요?",
    
    # 세법 관련
    "소득세 신고 누락 시 가산세는?",
    "법인세 계산 방법과 절차는?",
    "부가가치세 환급 신청 조건은?",
    "상속세 계산 시 공제 항목은?",
    
    # 기타 법률 질문
    "법정대리인의 권한과 책임은?",
    "소송 제기 시 관할 법원은 어떻게 정해지나요?",
    "중재 절차와 법원 소송의 차이점은?",
    "법률 자문 비용은 어떻게 산정되나요?"
]

async def test_single_question(service, question: str, question_id: int) -> Dict[str, Any]:
    """단일 질문 테스트"""
    try:
        start_time = time.time()
        
        print(f"\n[{question_id:2d}/40] 질문: {question}")
        
        # 워크플로우 실행
        result = await service.process_query(question)
        
        processing_time = time.time() - start_time
        
        # 결과 분석
        test_result = {
            "question_id": question_id,
            "question": question,
            "success": True,
            "processing_time": processing_time,
            "response_length": len(result.get("answer", "")),
            "confidence": result.get("confidence", 0),
            "sources_count": len(result.get("sources", [])),
            "processing_steps": len(result.get("processing_steps", [])),
            "query_type": result.get("query_type", "unknown"),
            "session_id": result.get("session_id", ""),
            "errors": result.get("errors", [])
        }
        
        print(f"    ✅ 성공 - 처리시간: {processing_time:.2f}초, 신뢰도: {test_result['confidence']:.2f}")
        print(f"    📝 답변 길이: {test_result['response_length']}자, 소스: {test_result['sources_count']}개")
        
        return test_result
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"    ❌ 실패 - 오류: {str(e)}")
        
        return {
            "question_id": question_id,
            "question": question,
            "success": False,
            "processing_time": processing_time,
            "error": str(e),
            "response_length": 0,
            "confidence": 0,
            "sources_count": 0,
            "processing_steps": 0,
            "query_type": "error",
            "session_id": "",
            "errors": [str(e)]
        }

async def run_comprehensive_test():
    """종합 테스트 실행"""
    print("=" * 80)
    print("LangGraph 대규모 테스트 - 40개 법률 질문 테스트")
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
        print("✅ 워크플로우 서비스 초기화 완료")
        
        # 테스트 실행
        results = []
        total_start_time = time.time()
        
        for i, question in enumerate(TEST_QUESTIONS, 1):
            result = await test_single_question(service, question, i)
            results.append(result)
            
            # 진행률 표시
            if i % 5 == 0:
                success_count = sum(1 for r in results if r["success"])
                print(f"\n📊 진행률: {i}/40 ({i*2.5:.1f}%) - 성공: {success_count}/{i}")
        
        total_time = time.time() - total_start_time
        
        # 결과 분석
        analyze_results(results, total_time)
        
        # 임시 파일 정리
        os.unlink(db_path)
        
        return results
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return []

def analyze_results(results: List[Dict[str, Any]], total_time: float):
    """테스트 결과 분석"""
    print("\n" + "=" * 80)
    print("테스트 결과 분석")
    print("=" * 80)
    
    # 기본 통계
    total_questions = len(results)
    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]
    
    success_rate = len(successful_tests) / total_questions * 100
    
    print(f"📊 전체 통계:")
    print(f"   - 총 질문 수: {total_questions}")
    print(f"   - 성공: {len(successful_tests)} ({success_rate:.1f}%)")
    print(f"   - 실패: {len(failed_tests)} ({100-success_rate:.1f}%)")
    print(f"   - 총 소요 시간: {total_time:.2f}초")
    print(f"   - 평균 처리 시간: {total_time/total_questions:.2f}초")
    
    if successful_tests:
        # 성공한 테스트들의 상세 통계
        avg_processing_time = sum(r["processing_time"] for r in successful_tests) / len(successful_tests)
        avg_confidence = sum(r["confidence"] for r in successful_tests) / len(successful_tests)
        avg_response_length = sum(r["response_length"] for r in successful_tests) / len(successful_tests)
        avg_sources = sum(r["sources_count"] for r in successful_tests) / len(successful_tests)
        
        print(f"\n📈 성공한 테스트 상세 통계:")
        print(f"   - 평균 처리 시간: {avg_processing_time:.2f}초")
        print(f"   - 평균 신뢰도: {avg_confidence:.3f}")
        print(f"   - 평균 답변 길이: {avg_response_length:.0f}자")
        print(f"   - 평균 소스 수: {avg_sources:.1f}개")
        
        # 처리 시간 분포
        fast_tests = [r for r in successful_tests if r["processing_time"] < 30]
        medium_tests = [r for r in successful_tests if 30 <= r["processing_time"] < 60]
        slow_tests = [r for r in successful_tests if r["processing_time"] >= 60]
        
        print(f"\n⏱️ 처리 시간 분포:")
        print(f"   - 빠름 (<30초): {len(fast_tests)}개")
        print(f"   - 보통 (30-60초): {len(medium_tests)}개")
        print(f"   - 느림 (≥60초): {len(slow_tests)}개")
        
        # 신뢰도 분포
        high_confidence = [r for r in successful_tests if r["confidence"] >= 0.8]
        medium_confidence = [r for r in successful_tests if 0.6 <= r["confidence"] < 0.8]
        low_confidence = [r for r in successful_tests if r["confidence"] < 0.6]
        
        print(f"\n🎯 신뢰도 분포:")
        print(f"   - 높음 (≥0.8): {len(high_confidence)}개")
        print(f"   - 보통 (0.6-0.8): {len(medium_confidence)}개")
        print(f"   - 낮음 (<0.6): {len(low_confidence)}개")
        
        # 질문 유형별 분석
        query_types = {}
        for r in successful_tests:
            qtype = r["query_type"]
            query_types[qtype] = query_types.get(qtype, 0) + 1
        
        print(f"\n📋 질문 유형별 분포:")
        for qtype, count in sorted(query_types.items()):
            print(f"   - {qtype}: {count}개")
    
    # 실패한 테스트 분석
    if failed_tests:
        print(f"\n❌ 실패한 테스트:")
        for test in failed_tests:
            print(f"   - [{test['question_id']:2d}] {test['question'][:50]}...")
            print(f"     오류: {test.get('error', 'Unknown error')}")
    
    # 성능 평가
    print(f"\n🏆 성능 평가:")
    if success_rate >= 95:
        print("   ✅ 우수 - 95% 이상 성공률")
    elif success_rate >= 90:
        print("   ✅ 양호 - 90% 이상 성공률")
    elif success_rate >= 80:
        print("   ⚠️ 보통 - 80% 이상 성공률")
    else:
        print("   ❌ 개선 필요 - 80% 미만 성공률")
    
    if avg_processing_time < 30:
        print("   ✅ 빠른 응답 - 평균 30초 미만")
    elif avg_processing_time < 60:
        print("   ✅ 적절한 응답 - 평균 60초 미만")
    else:
        print("   ⚠️ 느린 응답 - 평균 60초 이상")

async def main():
    """메인 함수"""
    results = await run_comprehensive_test()
    
    if results:
        print(f"\n🎉 대규모 테스트 완료!")
        print(f"총 {len(results)}개 질문 테스트 완료")
        
        success_count = sum(1 for r in results if r["success"])
        print(f"성공률: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    else:
        print("❌ 테스트 실행 실패")

if __name__ == "__main__":
    asyncio.run(main())
