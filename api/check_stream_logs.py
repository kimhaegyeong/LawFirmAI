"""
스트리밍 로그 확인 스크립트
백엔드 서버의 최근 로그를 확인합니다.
"""
import subprocess
import sys
import os

def check_recent_logs():
    """최근 로그 확인"""
    print("=" * 80)
    print("스트리밍 로그 확인")
    print("=" * 80)
    print()
    print("백엔드 서버 콘솔에서 다음 로그를 확인하세요:")
    print()
    print("1. 스트리밍 시작 로그:")
    print("   - '스트리밍 시작: astream_events(version='v2') 사용'")
    print("   또는")
    print("   - '스트리밍 시작: astream_events() 사용 (기본 버전)'")
    print()
    print("2. 스트리밍 이벤트 로그:")
    print("   - '스트리밍 이벤트 #X: type=on_llm_stream, name=...'")
    print("   - 'LLM 스트리밍 이벤트 감지: ... (총 X개)'")
    print()
    print("3. 스트리밍 완료 로그:")
    print("   - '스트리밍 이벤트 처리 완료: 총 X개 이벤트, LLM 스트리밍 이벤트 Y개, 토큰 수신 Z개'")
    print()
    print("4. 폴백 로그 (만약 스트리밍이 실패한 경우):")
    print("   - '스트리밍 이벤트 없음, on_chain_end에서 폴백 전송'")
    print("   또는")
    print("   - 'LLM 스트리밍 이벤트에서 답변을 찾지 못했습니다. 일반 처리로 폴백합니다.'")
    print()
    print("=" * 80)
    print()
    print("중요한 확인 사항:")
    print("- LLM 스트리밍 이벤트가 0개면: 실제 토큰 스트리밍이 작동하지 않음")
    print("- 토큰 수신이 0개면: 이벤트 구조가 예상과 다름")
    print("- 폴백 메시지가 나오면: 전체 답변을 받아서 나눠서 보냄 (의사 스트리밍)")
    print()
    print("=" * 80)

if __name__ == "__main__":
    check_recent_logs()

