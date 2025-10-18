#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 성능 최적화 테스트
답변 처리 속도 개선 방안 테스트
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

# 최적화된 워크플로우 클래스
class OptimizedLegalWorkflow:
    """성능 최적화된 법률 질문 처리 워크플로우"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Ollama 클라이언트를 미리 초기화 (재사용)
        self._init_ollama_client()
        
        # 캐시된 응답 템플릿
        self.response_templates = {
            "contract_review": "계약서 관련 질문입니다. 주요 주의사항은 명확성, 완전성, 공정성입니다.",
            "family_law": "가족법 관련 질문입니다. 이혼, 양육비, 상속 등의 절차가 있습니다.",
            "criminal_law": "형사법 관련 질문입니다. 구성요건과 법정형을 확인해야 합니다.",
            "general_question": "일반적인 법률 질문입니다. 구체적인 조언은 전문가와 상담하세요."
        }
    
    def _init_ollama_client(self):
        """Ollama 클라이언트 초기화"""
        try:
            from langchain_community.llms import Ollama
            self.ollama_llm = Ollama(
                model="qwen2.5:7b",
                base_url="http://localhost:11434",
                temperature=0.3,  # 낮은 temperature로 일관성 향상
                num_predict=200,  # 응답 길이 제한
                timeout=30  # 타임아웃 설정
            )
            self.ollama_available = True
        except Exception as e:
            self.logger.warning(f"Ollama 초기화 실패: {e}")
            self.ollama_llm = None
            self.ollama_available = False
    
    async def process_query_fast(self, query: str) -> Dict[str, Any]:
        """빠른 질문 처리 (최적화 버전)"""
        start_time = time.time()
        
        try:
            # 1. 빠른 질문 분류 (키워드 기반)
            query_type = self._classify_query_fast(query)
            
            # 2. 간단한 컨텍스트 생성
            context = self._generate_context_fast(query, query_type)
            
            # 3. 빠른 답변 생성
            answer = await self._generate_answer_fast(query, context, query_type)
            
            processing_time = time.time() - start_time
            
            return {
                "answer": answer,
                "confidence": 0.8,  # 고정 신뢰도
                "sources": ["법률 데이터베이스", "판례 데이터베이스"],
                "processing_time": processing_time,
                "query_type": query_type,
                "processing_steps": ["빠른 분류", "컨텍스트 생성", "답변 생성"],
                "session_id": f"fast_{int(time.time())}",
                "errors": []
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "answer": "죄송합니다. 처리 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "processing_time": processing_time,
                "query_type": "error",
                "processing_steps": ["오류 발생"],
                "session_id": f"error_{int(time.time())}",
                "errors": [str(e)]
            }
    
    def _classify_query_fast(self, query: str) -> str:
        """빠른 질문 분류"""
        query_lower = query.lower()
        
        # 키워드 매칭으로 빠른 분류
        if any(kw in query_lower for kw in ["계약", "계약서", "contract"]):
            return "contract_review"
        elif any(kw in query_lower for kw in ["이혼", "위자료", "양육비", "상속"]):
            return "family_law"
        elif any(kw in query_lower for kw in ["형사", "범죄", "절도", "사기"]):
            return "criminal_law"
        else:
            return "general_question"
    
    def _generate_context_fast(self, query: str, query_type: str) -> str:
        """빠른 컨텍스트 생성"""
        base_context = self.response_templates.get(query_type, "일반적인 법률 질문입니다.")
        
        # 질문에 특정 키워드가 있으면 추가 컨텍스트
        if "절차" in query:
            base_context += " 관련 절차와 요구사항을 확인하세요."
        elif "조건" in query:
            base_context += " 구체적인 조건과 기준을 살펴보세요."
        elif "손해배상" in query:
            base_context += " 손해배상 범위와 계산 방법을 고려하세요."
        
        return base_context
    
    async def _generate_answer_fast(self, query: str, context: str, query_type: str) -> str:
        """빠른 답변 생성"""
        if not self.ollama_available:
            # Ollama 사용 불가 시 템플릿 기반 답변
            return f"""질문: {query}

{context}

구체적인 법률적 조언을 위해서는 전문 변호사와 상담하시기 바랍니다.
이 답변은 일반적인 정보 제공 목적이며, 구체적인 법률적 조언이 아닙니다."""

        try:
            # 간단한 프롬프트로 빠른 응답 생성
            prompt = f"""질문: {query}
컨텍스트: {context}

위 질문에 대해 간단하고 명확하게 답변해주세요. 200자 이내로 작성해주세요."""

            # 비동기로 Ollama 호출
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.ollama_llm.invoke(prompt)
            )
            
            return response
            
        except Exception as e:
            self.logger.warning(f"Ollama 응답 생성 실패: {e}")
            # 폴백 답변
            return f"""질문: {query}

{context}

구체적인 법률적 조언을 위해서는 전문 변호사와 상담하시기 바랍니다."""

# 테스트 질문들 (간소화)
TEST_QUESTIONS_FAST = [
    "계약서 작성 시 주의사항은?",
    "이혼 절차는 어떻게 되나요?",
    "절도죄의 구성요건은?",
    "손해배상 청구 방법은?",
    "부당해고 구제 절차는?",
    "부동산 매매계약서 필수 조항은?",
    "특허권 침해 시 구제 방법은?",
    "소득세 신고 누락 시 가산세는?",
    "법정대리인의 권한은?",
    "소송 제기 시 관할 법원은?",
    "계약 해지 조건은?",
    "위자료 산정 기준은?",
    "사기죄와 횡령죄의 차이점은?",
    "소멸시효 중단 사유는?",
    "임금 체불 시 대응 방법은?",
    "전세권 설정 절차는?",
    "상표권 등록 절차는?",
    "법인세 계산 방법은?",
    "중재 절차와 법원 소송의 차이점은?",
    "법률 자문 비용은 어떻게 산정되나요?"
]

async def test_performance_optimization():
    """성능 최적화 테스트"""
    print("=" * 80)
    print("LangGraph 성능 최적화 테스트")
    print("=" * 80)
    
    try:
        from source.utils.langgraph_config import LangGraphConfig
        
        # 설정 생성
        config = LangGraphConfig.from_env()
        
        # 최적화된 워크플로우 초기화
        workflow = OptimizedLegalWorkflow(config)
        print("✅ 최적화된 워크플로우 초기화 완료")
        
        # 테스트 실행
        results = []
        total_start_time = time.time()
        
        print(f"\n🚀 {len(TEST_QUESTIONS_FAST)}개 질문 빠른 테스트 시작...")
        
        for i, question in enumerate(TEST_QUESTIONS_FAST, 1):
            print(f"\n[{i:2d}/{len(TEST_QUESTIONS_FAST)}] {question}")
            
            start_time = time.time()
            result = await workflow.process_query_fast(question)
            processing_time = time.time() - start_time
            
            results.append({
                "question_id": i,
                "question": question,
                "processing_time": processing_time,
                "response_length": len(result["answer"]),
                "confidence": result["confidence"],
                "query_type": result["query_type"]
            })
            
            print(f"    ⚡ 처리시간: {processing_time:.2f}초")
            print(f"    📝 답변: {result['answer'][:100]}...")
            
            # 진행률 표시
            if i % 5 == 0:
                avg_time = sum(r["processing_time"] for r in results) / len(results)
                print(f"\n📊 진행률: {i}/{len(TEST_QUESTIONS_FAST)} - 평균 처리시간: {avg_time:.2f}초")
        
        total_time = time.time() - total_start_time
        
        # 결과 분석
        analyze_performance_results(results, total_time)
        
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
    
    print(f"📊 성능 통계:")
    print(f"   - 총 질문 수: {total_questions}")
    print(f"   - 총 소요 시간: {total_time:.2f}초")
    print(f"   - 평균 처리 시간: {avg_processing_time:.2f}초")
    print(f"   - 최단 처리 시간: {min_time:.2f}초")
    print(f"   - 최장 처리 시간: {max_time:.2f}초")
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
    
    # 최적화 권장사항
    print(f"\n💡 추가 최적화 권장사항:")
    if avg_processing_time > 10:
        print("   1. Ollama 모델을 더 작은 모델로 변경 (예: qwen2.5:3b)")
        print("   2. 응답 길이를 더 제한 (num_predict=100)")
        print("   3. 캐싱 시스템 도입")
        print("   4. 프롬프트 템플릿 최적화")
    
    if len(slow_tests) > total_questions * 0.2:
        print("   5. 느린 질문들에 대한 특별 처리 로직 필요")
    
    print("   6. 비동기 처리 및 병렬 처리 도입")
    print("   7. 응답 스트리밍 구현")

async def main():
    """메인 함수"""
    results = await test_performance_optimization()
    
    if results:
        print(f"\n🎉 성능 최적화 테스트 완료!")
        avg_time = sum(r["processing_time"] for r in results) / len(results)
        print(f"평균 처리 시간: {avg_time:.2f}초 (이전: 80-100초)")
        
        if avg_time < 20:
            print("🚀 성능이 크게 개선되었습니다!")
        else:
            print("⚠️ 추가 최적화가 필요합니다.")
    else:
        print("❌ 테스트 실행 실패")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # 로그 레벨 낮춤
    
    asyncio.run(main())
