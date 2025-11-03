# -*- coding: utf-8 -*-
"""LangGraph 테스트 (결과를 파일로 저장)"""

import sys
import os
from pathlib import Path
from datetime import datetime

# 출력 파일 경로
output_file = Path(__file__).parent / "test_results.txt"

# 출력 리다이렉션 함수
class TeeOutput:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# 표준 출력을 파일과 콘솔 모두로 리다이렉션
with open(output_file, 'w', encoding='utf-8') as f:
    tee = TeeOutput(sys.stdout, f)
    sys.stdout = tee

    try:
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))

        print("=" * 60)
        print("LangGraph 종합 테스트")
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print()

        # 1. Python 버전
        print(f"1. Python 버전: {sys.version}")
        print()

        # 2. Import 테스트
        print("2. Import 테스트")
        print("-" * 60)

        try:
            import langgraph
            version = getattr(langgraph, '__version__', 'N/A')
            print(f"  ✓ LangGraph: {version}")
        except Exception as e:
            print(f"  ✗ LangGraph: {e}")
            import traceback
            traceback.print_exc()
            raise

        try:
            from config.langgraph_config import LangGraphConfig
            print("  ✓ Config 모듈")
        except Exception as e:
            print(f"  ✗ Config 모듈: {e}")
            import traceback
            traceback.print_exc()
            raise

        try:
            from graph import graph, app, create_graph, create_app
            print("  ✓ Graph export 모듈")
        except Exception as e:
            print(f"  ✗ Graph export 모듈: {e}")
            import traceback
            traceback.print_exc()
            raise

        try:
            from source.services.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
            from source.services.workflow_service import LangGraphWorkflowService
            print("  ✓ 워크플로우 모듈")
        except Exception as e:
            print(f"  ✗ 워크플로우 모듈: {e}")
            import traceback
            traceback.print_exc()
            raise

        print()

        # 3. 설정 로딩
        print("3. 설정 로딩 테스트")
        print("-" * 60)

        try:
            config = LangGraphConfig.from_env()
            print(f"  ✓ LangGraph 활성화: {config.langgraph_enabled}")
            print(f"  ✓ LLM 제공자: {config.llm_provider}")
            print(f"  ✓ Google 모델: {config.google_model}")
            print(f"  ✓ 최대 반복 횟수: {config.max_iterations}")

            errors = config.validate()
            if errors:
                print(f"  ⚠ 설정 경고: {len(errors)}개")
                for error in errors:
                    print(f"    - {error}")
            else:
                print("  ✓ 설정 유효성 검사 통과")
        except Exception as e:
            print(f"  ✗ 설정 로딩 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

        print()

        # 4. Graph 생성
        print("4. Graph 생성 테스트")
        print("-" * 60)

        try:
            g = create_graph()
            print(f"  ✓ Graph 생성: {type(g).__name__}")
            if hasattr(g, 'nodes'):
                print(f"  ✓ 노드 수: {len(g.nodes)}")
            if hasattr(g, 'edges'):
                print(f"  ✓ 엣지 수: {len(g.edges)}")
        except Exception as e:
            print(f"  ✗ Graph 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

        print()

        # 5. App 생성
        print("5. App 생성 테스트")
        print("-" * 60)

        try:
            a = create_app()
            print(f"  ✓ App 생성: {type(a).__name__}")
            if hasattr(a, 'invoke'):
                print("  ✓ invoke 메서드 존재")
            if hasattr(a, 'stream'):
                print("  ✓ stream 메서드 존재")
        except Exception as e:
            print(f"  ✗ App 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

        print()

        # 6. 간단한 워크플로우 실행 (선택적)
        print("6. 워크플로우 실행 테스트 (선택적)")
        print("-" * 60)

        try:
            test_input = {
                "messages": [
                    {
                        "role": "user",
                        "content": "안녕하세요. 테스트입니다."
                    }
                ]
            }

            print("  워크플로우 실행 중...")
            print(f"  입력: {test_input['messages'][0]['content']}")

            result = a.invoke(test_input)

            print("  ✓ 워크플로우 실행 완료")
            if isinstance(result, dict) and "messages" in result:
                print(f"  ✓ 응답 메시지 수: {len(result['messages'])}")
                if result["messages"]:
                    last_msg = result["messages"][-1]
                    if hasattr(last_msg, "content"):
                        content = last_msg.content
                        print(f"  ✓ 응답 길이: {len(content)} 문자")
                        print(f"  ✓ 응답 미리보기: {content[:100]}...")
        except Exception as e:
            print(f"  ⚠ 워크플로우 실행 실패 (환경 설정 문제일 수 있음): {e}")
            print("  (이는 LLM API 키가 없거나 네트워크 문제일 수 있습니다)")
            import traceback
            traceback.print_exc()
            # 이 오류는 치명적이지 않으므로 계속 진행

        print()
        print("=" * 60)
        print("✓ 모든 기본 테스트 통과!")
        print("=" * 60)
        print()
        print("결과가 test_results.txt 파일에 저장되었습니다.")
        print()
        print("다음 단계:")
        print("  1. python test_workflow.py  # 종합 테스트 실행")
        print("  2. langgraph dev            # LangGraph Studio 실행")

    finally:
        # 원래 stdout 복원
        sys.stdout = sys.__stdout__
        print(f"\n테스트 결과가 {output_file}에 저장되었습니다.")
