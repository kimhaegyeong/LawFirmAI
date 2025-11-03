# -*- coding: utf-8 -*-
"""
LangSmith 통합 테스트
개선된 LangGraph 워크플로우를 LangSmith로 모니터링하는 테스트
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_path))

# 환경변수 파일 로드 확인 (langgraph_config.py에서 이미 처리됨)
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ 환경변수 파일 로드 완료: {env_path}")
    else:
        print(f"⚠ .env 파일을 찾을 수 없습니다: {env_path}")
except ImportError:
    print("⚠ python-dotenv가 설치되지 않았습니다. pip install python-dotenv로 설치하세요.")
except Exception as e:
    print(f"⚠ 환경변수 파일 로드 중 오류 발생: {e}")

# 필수 패키지 설치 확인
missing_packages = []

def check_package(package_name, import_name=None, install_name=None):
    """패키지 설치 확인"""
    if import_name is None:
        import_name = package_name
    if install_name is None:
        install_name = package_name

    try:
        __import__(import_name)
        return True
    except ImportError:
        missing_packages.append(install_name)
        print(f"⚠ {package_name}가 설치되지 않았습니다.")
        print(f"  설치 명령: pip install {install_name}")
        return False

# 필수 패키지 확인
print("\n환경 설정 확인 중...")
check_package("pydantic-settings", "pydantic_settings", "pydantic-settings")
check_package("numpy", "numpy", "numpy")
check_package("faiss", "faiss", "faiss-cpu")
check_package("structlog", "structlog", "structlog")
check_package("google-generativeai", "google.generativeai", "google-generativeai")

if missing_packages:
    print(f"\n⚠ 총 {len(missing_packages)}개 패키지가 누락되었습니다:")
    print(f"  pip install {' '.join(missing_packages)}")
    print("\n이 패키지들을 설치한 후 테스트를 다시 실행하세요.\n")
    # sys.exit(1)  # 주석 처리: 설치 안해도 테스트 진행 가능하게
else:
    print("✓ 모든 필수 패키지가 설치되어 있습니다.\n")

# LangGraph 관련 import (환경변수 로드 후)
from lawfirm_langgraph.langgraph_core.services.workflow_service import (
    LangGraphWorkflowService,  # noqa: E402
)
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig  # noqa: E402

# 테스트 환경 설정
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"
os.environ["LANGGRAPH_CHECKPOINT_STORAGE"] = "memory"  # 빠른 테스트를 위해 메모리 사용

# LangSmith 활성화
# .env 파일에서 설정을 가져와서 LangChain 환경변수로 변환
langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "") or os.getenv("LANGCHAIN_API_KEY", "")
langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false") or os.getenv("LANGCHAIN_TRACING_V2", "false")
langsmith_project = os.getenv("LANGSMITH_PROJECT", "LawFirmAI-Test") or os.getenv("LANGCHAIN_PROJECT", "LawFirmAI-Test")

# LangChain 환경변수로 설정 (LangChain SDK가 사용함)
os.environ["LANGCHAIN_TRACING_V2"] = "true" if langsmith_tracing.lower() in ["true", "1", "yes"] else "false"
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
os.environ["LANGCHAIN_PROJECT"] = langsmith_project

# LangSmith 환경변수도 설정 (하위 호환성)
if langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
if langsmith_project:
    os.environ["LANGSMITH_PROJECT"] = langsmith_project


class LangSmithIntegrationTest:
    """LangSmith 통합 테스트 클래스"""

    def __init__(self):
        self.config = LangGraphConfig()
        self.service = LangGraphWorkflowService(self.config)
        self.test_start_time = None

    def calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """답변 품질 점수 계산"""
        if not result:
            return 0.0

        score = 0.0
        max_score = 100

        # 답변 존재 (20점)
        answer = result.get('answer', '')
        if answer:
            score += 10
            if len(answer) >= 50:
                score += 10

        # 신뢰도 (30점)
        confidence = result.get('confidence', 0.0)
        score += confidence * 30

        # 소스 제공 (25점)
        sources_count = len(result.get('sources', []))
        if sources_count > 0:
            score += min(25, sources_count * 5)

        # 법률 참조 (15점)
        legal_refs_count = len(result.get('legal_references', []))
        if legal_refs_count > 0:
            score += min(15, legal_refs_count * 5)

        # 에러 없음 (10점)
        errors_count = len(result.get('errors', []))
        if errors_count == 0:
            score += 10

        return round(score / max_score, 2)

    def validate_answer_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """답변 품질 검증"""
        if not result:
            return {
                'valid': False,
                'has_answer': False,
                'overall_score': 0.0
            }

        quality = {
            'valid': True,
            'has_answer': bool(result.get('answer')),
            'answer_length_sufficient': len(result.get('answer', '')) >= 50,
            'has_sources': len(result.get('sources', [])) > 0,
            'has_legal_references': len(result.get('legal_references', [])) > 0,
            'confidence_threshold': result.get('confidence', 0) >= 0.5,
            'no_errors': len(result.get('errors', [])) == 0,
            'processing_time_reasonable': result.get('processing_time', 0) < 60
        }

        quality['overall_score'] = self.calculate_quality_score(result)
        quality['valid'] = quality['overall_score'] >= 0.5

        return quality

    async def save_results(self, results: List[Dict[str, Any]]) -> str:
        """테스트 결과를 JSON 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = project_root / 'tests' / 'results'
        result_dir.mkdir(parents=True, exist_ok=True)
        filename = result_dir / f'langsmith_test_{timestamp}.json'

        # 결과 요약 생성
        summary = {
            'timestamp': timestamp,
            'total_queries': len(results),
            'successful_queries': len([r for r in results if r.get('result')]),
            'failed_queries': len([r for r in results if not r.get('result')]),
            'langsmith_enabled': os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
            'test_duration': time.time() - self.test_start_time if self.test_start_time else 0,
            'results': results
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return str(filename)

    async def run_single_query(self, query: str, session_id: str, query_number: int) -> Optional[Dict[str, Any]]:
        """단일 질의 실행"""
        print(f"\n{'='*80}")
        print(f"질의 #{query_number}: {query}")
        print(f"{'='*80}")

        try:
            # LangGraph 워크플로우 실행
            result = await self.service.process_query(
                query=query,
                session_id=session_id
            )

            # 품질 검증
            quality = self.validate_answer_quality(result)

            # 결과 출력
            print("\n✓ 처리 완료")
            print(f"  답변 길이: {len(result.get('answer', ''))}자")
            print(f"  신뢰도: {result.get('confidence', 0):.2%}")
            print(f"  소스 수: {len(result.get('sources', []))}개")
            print(f"  법률 참조 수: {len(result.get('legal_references', []))}개")
            print(f"  처리 단계 수: {len(result.get('processing_steps', []))}개")
            print(f"  처리 시간: {result.get('processing_time', 0):.2f}초")
            print(f"  품질 점수: {quality.get('overall_score', 0):.2%}")

            # 키워드 확장 정보 (metadata에서)
            metadata = result.get('metadata', {})
            if 'ai_keyword_expansion' in metadata:
                expansion = metadata['ai_keyword_expansion']
                print(f"  AI 키워드 확장: {expansion.get('method', 'N/A')}")
                print(f"    - 원본 키워드: {len(expansion.get('original_keywords', []))}개")
                print(f"    - 확장 키워드: {len(expansion.get('expanded_keywords', []))}개")
                print(f"    - 신뢰도: {expansion.get('confidence', 0):.2%}")

            # 처리 단계 출력
            steps = result.get('processing_steps', [])
            if steps:
                print("\n  처리 단계:")
                for i, step in enumerate(steps[-5:], 1):  # 마지막 5개만
                    print(f"    {i}. {step}")

            return result

        except Exception as e:
            print(f"\n✗ 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def run_all_tests(self):
        """전체 테스트 실행"""
        self.test_start_time = time.time()

        print("=" * 80)
        print("LangSmith 통합 테스트 시작")
        print("개선된 LangGraph 워크플로우 테스트")
        print("=" * 80)

        # 설정 검증
        validation_errors = self.config.validate()
        if validation_errors:
            print("\n⚠ 설정 검증 오류:")
            for error in validation_errors:
                print(f"  - {error}")
            print("테스트를 계속 진행하지만 일부 기능이 제한될 수 있습니다.")

        print("=" * 80)

        # 테스트 케이스 정의 (확대됨)
        test_cases = [
            {
                "query": "손해배상 청구 방법을 알려주세요",
                "description": "기본 법률 조언 질문"
            },
            {
                "query": "계약 위반 시 법적 조치 방법",
                "description": "계약 관련 질문"
            },
            {
                "query": "민사소송에서 승소하기 위한 증거 수집 방법",
                "description": "민사소송 절차 질문"
            },
            {
                "query": "계약서에 따르면 배송 지연 시 어떻게 대응해야 하나요?",
                "description": "구체적 사안 질문"
            },
            {
                "query": "이전에 소개해주신 손해배상 청구에서 과실비율은 어떻게 결정되나요?",
                "description": "멀티턴 질문 (이전 질문 참조)"
            },
            {
                "query": "민법 제750조 손해배상의 범위는?",
                "description": "특정 법조문 해석 질문"
            },
            {
                "query": "이행불능과 이행불가능의 차이",
                "description": "법률 용어 비교 질문"
            }
        ]

        results = []

        for i, test_case in enumerate(test_cases, 1):
            # 타임스탬프를 포함한 세션 ID로 격리 강화
            session_id = f"test_{int(time.time())}_{i:03d}"

            print(f"\n\n{'#'*80}")
            print(f"테스트 케이스 #{i}/{len(test_cases)}")
            print(f"설명: {test_case['description']}")
            print(f"세션: {session_id}")
            print(f"{'#'*80}")

            try:
                result = await self.run_single_query(
                    query=test_case['query'],
                    session_id=session_id,
                    query_number=i
                )

                # 품질 검증 추가
                if result:
                    quality = self.validate_answer_quality(result)
                    results.append({
                        'case': i,
                        'query': test_case['query'],
                        'description': test_case['description'],
                        'session_id': session_id,
                        'result': result,
                        'quality': quality,
                        'success': True
                    })
                else:
                    results.append({
                        'case': i,
                        'query': test_case['query'],
                        'description': test_case['description'],
                        'session_id': session_id,
                        'result': None,
                        'success': False,
                        'error': 'Result is None'
                    })
            except Exception as e:
                print(f"\n✗ 테스트 케이스 #{i} 실행 실패: {e}")
                results.append({
                    'case': i,
                    'query': test_case['query'],
                    'description': test_case['description'],
                    'session_id': session_id,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })

        # 통계 출력
        print("\n" + "=" * 80)
        print("테스트 결과 통계")
        print("=" * 80)

        total_queries = len(test_cases)
        successful_results = [r for r in results if r.get('success')]
        failed_results = [r for r in results if not r.get('success')]

        print(f"\n총 질의 수: {total_queries}")
        print(f"성공한 질의: {len(successful_results)}")
        print(f"실패한 질의: {len(failed_results)}")

        if successful_results:
            # 평균 값 계산
            total_confidence = sum(r['result'].get('confidence', 0) for r in successful_results if r.get('result'))
            total_docs = sum(len(r['result'].get('sources', [])) for r in successful_results if r.get('result'))
            total_legal_refs = sum(len(r['result'].get('legal_references', [])) for r in successful_results if r.get('result'))
            total_steps = sum(len(r['result'].get('processing_steps', [])) for r in successful_results if r.get('result'))
            total_time = sum(r['result'].get('processing_time', 0) for r in successful_results if r.get('result'))
            total_quality = sum(r.get('quality', {}).get('overall_score', 0) for r in successful_results)

            success_count = len(successful_results)

            print(f"\n평균 신뢰도: {total_confidence/success_count:.2%}")
            print(f"평균 품질 점수: {total_quality/success_count:.2%}")
            print(f"평균 소스 수: {total_docs/success_count:.1f}개")
            print(f"평균 법률 참조 수: {total_legal_refs/success_count:.1f}개")
            print(f"평균 처리 단계 수: {total_steps/success_count:.1f}개")
            print(f"평균 처리 시간: {total_time/success_count:.2f}초")

            # 품질 통계
            valid_qualities = [r.get('quality', {}).get('overall_score', 0) for r in successful_results if r.get('quality')]
            if valid_qualities:
                min_quality = min(valid_qualities)
                max_quality = max(valid_qualities)
                print(f"\n품질 점수 범위: {min_quality:.2%} ~ {max_quality:.2%}")
                high_quality_count = len([q for q in valid_qualities if q >= 0.7])
                print(f"고품질 답변 (≥70%): {high_quality_count}개")

            # AI 키워드 확장 통계
            ai_expansions = []
            for r in successful_results:
                if r.get('result') and 'metadata' in r['result']:
                    metadata = r['result']['metadata']
                    if 'ai_keyword_expansion' in metadata:
                        ai_expansions.append(metadata['ai_keyword_expansion'])

            if ai_expansions:
                print(f"\nAI 키워드 확장 실행: {len(ai_expansions)}회")
                gemini_count = len([e for e in ai_expansions if e.get('method') == 'gemini_ai'])
                fallback_count = len([e for e in ai_expansions if e.get('method') == 'fallback'])
                print(f"  - Gemini AI: {gemini_count}회")
                print(f"  - Fallback: {fallback_count}회")

        # 결과 저장
        try:
            filename = await self.save_results(results)
            print(f"\n✓ 결과 저장됨: {filename}")
        except Exception as e:
            print(f"\n⚠ 결과 저장 실패: {e}")

        print("\n" + "=" * 80)
        print("LangSmith 모니터링 확인")
        print("=" * 80)
        print("\nLangSmith 대시보드에서 다음 정보를 확인할 수 있습니다:")
        print("  - 각 노드의 실행 시간")
        print("  - 노드 간 데이터 흐름")
        print("  - AI 키워드 확장 과정")
        print("  - 에러 및 경고 메시지")
        print("  - 토큰 사용량")
        print("  - 비용 추적")
        print("\nLangSmith URL: https://smith.langchain.com (클라우드 설정인 경우)")

        return results


async def main():
    """메인 함수"""
    test_runner = LangSmithIntegrationTest()
    results = await test_runner.run_all_tests()

    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    # LangSmith 설정 확인 및 출력
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false")
    langchain_project = os.getenv("LANGCHAIN_PROJECT", "LawFirmAI-Test")

    print("\n" + "=" * 80)
    print("LangSmith 설정 확인")
    print("=" * 80)

    if langchain_api_key and langchain_tracing.lower() == "true":
        print("✓ LangSmith 활성화됨")
        print(f"  API Key: {langchain_api_key[:20]}...{langchain_api_key[-10:]} (부분 표시)")
        print(f"  Project: {langchain_project}")
        print(f"  Tracing: {langchain_tracing}")
    else:
        print("⚠ LangSmith 설정 경고:")
        print(f"  API Key: {'설정됨' if langchain_api_key else '❌ 설정되지 않음'}")
        print(f"  Tracing: {langchain_tracing}")
        print(f"  Project: {langchain_project}")
        print("\n설정 방법:")
        print("  .env 파일에 추가:")
        print("    LANGSMITH_API_KEY=your-api-key")
        print("    LANGSMITH_TRACING=true")
        print("    LANGSMITH_PROJECT=your-project-name")
        print("\nLangSmith 없이도 테스트는 진행됩니다.")

    print("=" * 80 + "\n")

    # 테스트 실행
    asyncio.run(main())
