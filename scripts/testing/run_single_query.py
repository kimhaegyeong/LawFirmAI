# -*- coding: utf-8 -*-
"""
Single-query runner for evaluating current workflow answer quality.

Usage:
    python scripts/testing/run_single_query.py "질의"
If no argument is provided, a default legal query will be used.
"""

import asyncio
import sys
import os
from pathlib import Path


def project_bootstrap():
    # Ensure project root on sys.path
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def normalize_query(query: str) -> str:
    """
    쿼리 문자열의 인코딩을 정규화합니다.
    
    Args:
        query: 입력 쿼리 문자열
        
    Returns:
        정규화된 UTF-8 쿼리 문자열
    """
    if not query:
        return query
    
    # 바이트 문자열인 경우 먼저 처리
    if isinstance(query, bytes):
        # 여러 인코딩 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1', 'utf-16']
        for encoding in encodings:
            try:
                decoded = query.decode(encoding)
                # 성공적으로 디코딩된 경우 UTF-8로 재인코딩
                return decoded.encode('utf-8', errors='replace').decode('utf-8')
            except (UnicodeDecodeError, LookupError):
                continue
        # 모든 인코딩 실패 시 UTF-8로 강제 디코딩 (손실 가능)
        return query.decode('utf-8', errors='replace')
    
    # 문자열인 경우
    if isinstance(query, str):
        # 이미 올바른 유니코드 문자열인지 확인
        try:
            # UTF-8로 인코딩 후 다시 디코딩하여 정규화
            normalized = query.encode('utf-8', errors='replace').decode('utf-8')
            # 깨진 문자가 있는지 확인 (대체 문자 포함 여부)
            if '\ufffd' in normalized:
                # 깨진 문자가 있으면 다른 인코딩 시도
                # 원본이 바이트였다고 가정하고 재시도
                if hasattr(query, 'encode'):
                    # 원본을 다시 인코딩 시도
                    for encoding in ['cp949', 'euc-kr']:
                        try:
                            # 원본을 해당 인코딩으로 인코딩 후 UTF-8로 디코딩
                            temp_bytes = query.encode(encoding, errors='replace')
                            normalized = temp_bytes.decode(encoding, errors='replace')
                            # UTF-8로 재인코딩
                            normalized = normalized.encode('utf-8', errors='replace').decode('utf-8')
                            if '\ufffd' not in normalized:
                                return normalized
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            continue
            return normalized
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            # 인코딩 오류가 발생하면 원본 반환
            print(f"[WARNING] Query encoding error: {e}, using original query", flush=True)
            return query
    
    return str(query)


async def run(query: str):
    from lawfirm_langgraph.core.utils.langgraph_config import LangGraphConfig
    from lawfirm_langgraph.core.agents.workflow_service import LangGraphWorkflowService

    # 쿼리 인코딩 정규화
    normalized_query = normalize_query(query)
    
    # 인코딩 검증
    if normalized_query != query:
        print(f"[INFO] Query normalized: '{query[:50]}...' -> '{normalized_query[:50]}...'", flush=True)
    
    config = LangGraphConfig.from_env()
    service = LangGraphWorkflowService(config)

    result = await service.process_query(normalized_query, session_id="single_query_eval", enable_checkpoint=False)
    # Normalize fields for readability
    answer = result.get("answer")
    answer_text = answer
    if isinstance(answer_text, dict):
        # try nested extraction
        for key in ("answer", "content", "text"):
            if isinstance(answer_text, dict) and key in answer_text:
                answer_text = answer_text[key]
        if isinstance(answer_text, dict):
            answer_text = str(answer_text)

    print("\n===== Single Query Evaluation =====", flush=True)
    print(f"Query: {normalized_query}", flush=True)
    print(f"Answer length: {len(str(answer_text)) if answer_text else 0}", flush=True)
    print("--- Answer Preview (first 600 chars) ---", flush=True)
    print(str(answer_text)[:600] if answer_text else "<no answer>", flush=True)
    print("\n--- Sources ---", flush=True)
    print(result.get("sources", []), flush=True)
    print("\n--- Legal References ---", flush=True)
    print(result.get("legal_references", []), flush=True)
    print("\n--- Metadata ---", flush=True)
    print(result.get("metadata", {}), flush=True)
    print("\n--- Confidence ---", flush=True)
    print(result.get("confidence", 0.0), flush=True)


def main():
    project_bootstrap()
    
    # Windows에서 UTF-8 인코딩 강제 설정
    if sys.platform == 'win32':
        # 표준 출력/입력 인코딩을 UTF-8로 설정
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
        if hasattr(sys.stderr, 'reconfigure'):
            try:
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
        # 환경 변수 설정
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 기본 쿼리 (인코딩 문제 수정)
    query = "손해배상의 범위는 어떻게 결정되나요?"
    
    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        # 첫 번째 인자가 파일 경로인지 확인
        first_arg = sys.argv[1]
        if first_arg.startswith('--file=') or first_arg == '--file':
            # 파일에서 쿼리 읽기
            file_path = first_arg.split('=', 1)[1] if '=' in first_arg else sys.argv[2] if len(sys.argv) > 2 else None
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        query = f.read().strip()
                    print(f"[INFO] Query loaded from file: {file_path}", flush=True)
                except Exception as e:
                    print(f"[ERROR] Failed to read query from file: {e}", flush=True)
                    return
            else:
                print(f"[ERROR] File not found: {file_path}", flush=True)
                return
        elif first_arg == '--input' or first_arg == '-i':
            # 표준 입력에서 쿼리 읽기
            try:
                query = input("Enter query: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("[ERROR] Failed to read query from input", flush=True)
                return
        else:
            # 명령줄 인자에서 쿼리 추출
            query = " ".join(sys.argv[1:])
            
            # Windows PowerShell에서 전달된 인코딩 문제 해결
            if sys.platform == 'win32':
                # PowerShell에서 한글 인자 전달 시 인코딩 문제가 발생할 수 있음
                # 쿼리가 깨진 문자를 포함하는지 확인
                try:
                    # 쿼리를 UTF-8로 재인코딩 시도
                    # 먼저 쿼리를 바이트로 변환 후 UTF-8로 디코딩
                    query_bytes = query.encode('utf-8', errors='replace')
                    test_query = query_bytes.decode('utf-8', errors='replace')
                    
                    # 깨진 문자가 있는지 확인 (대체 문자 포함 여부)
                    if '\ufffd' in test_query or any(ord(c) > 0x10FFFF for c in query if c):
                        # 깨진 문자가 있으면 다른 인코딩 시도
                        import locale
                        system_encoding = locale.getpreferredencoding()
                        
                        # 여러 인코딩 시도
                        for encoding in ['cp949', 'euc-kr', 'utf-8']:
                            try:
                                # 원본을 해당 인코딩으로 인코딩 후 UTF-8로 디코딩
                                query_bytes = query.encode(encoding, errors='replace')
                                decoded = query_bytes.decode(encoding, errors='replace')
                                # UTF-8로 재인코딩
                                query = decoded.encode('utf-8', errors='replace').decode('utf-8')
                                if '\ufffd' not in query:
                                    break
                            except Exception:
                                continue
                        
                        # 여전히 깨진 문자가 있으면 경고 출력
                        if '\ufffd' in query:
                            print(f"[WARNING] Query may contain corrupted characters. Using fallback query.", flush=True)
                            # 대체 쿼리 사용
                            query = "손해배상의 범위는 어떻게 결정되나요?"
                except Exception as e:
                    print(f"[WARNING] Encoding normalization failed: {e}", flush=True)
    
    # 쿼리 정규화
    normalized_query = normalize_query(query)
    
    # 인코딩 검증 출력
    print(f"[INFO] Original query: {repr(query[:50])}", flush=True)
    print(f"[INFO] Normalized query: {repr(normalized_query[:50])}", flush=True)
    
    # 쿼리가 여전히 깨진 문자를 포함하는지 확인
    if '\ufffd' in normalized_query or any(ord(c) > 0x10FFFF for c in normalized_query if c):
        print(f"[WARNING] Query still contains corrupted characters. Using fallback query.", flush=True)
        normalized_query = "손해배상의 범위는 어떻게 결정되나요?"
    
    asyncio.run(run(normalized_query))


if __name__ == "__main__":
    main()
