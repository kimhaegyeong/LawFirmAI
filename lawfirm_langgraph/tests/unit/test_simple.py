# -*- coding: utf-8 -*-
"""간단한 테스트 - Python 실행 확인"""

print("=" * 60)
print("Python 실행 확인")
print("=" * 60)
print()

print(f"Python 버전: {__import__('sys').version}")
print()

try:
    import langgraph
    print(f"✓ LangGraph 설치됨: {getattr(langgraph, '__version__', 'N/A')}")
except Exception as e:
    print(f"✗ LangGraph import 실패: {e}")

print()
print("테스트 완료!")
