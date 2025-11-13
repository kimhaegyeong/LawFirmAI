# -*- coding: utf-8 -*-
"""
LangGraph 단일 질의 테스트 스크립트

Usage:
    python lawfirm_langgraph/tests/run_single_query_test.py "질의 내용"
    질의 내용이 없으면 기본 법률 질문을 사용합니다.
"""

import asyncio
import sys
import os
from pathlib import Path

# UTF-8 인코딩 설정 (Windows PowerShell 호환)
if sys.platform == 'win32':
    # Windows에서 UTF-8 출력 설정
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # 환경 변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 디렉토리를 sys.path에 추가
lawfirm_langgraph_path = Path(__file__).parent.parent
sys.path.insert(0, str(lawfirm_langgraph_path))


async def run_single_query_test(query: str):
    """단일 질의 테스트 실행"""
    print("\n" + "="*80)
    print("LangGraph 단일 질의 테스트")
    print("="*80)
    
    try:
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService
        
        print(f"\n📋 질의: {query}")
        print("-" * 80)
        
        # 설정 로드
        print("\n1️⃣  설정 로드 중...")
        config = LangGraphConfig.from_env()
        # 테스트를 위해 체크포인트 비활성화
        config.enable_checkpoint = False
        print(f"   ✅ LangGraph 활성화: {config.langgraph_enabled}")
        print(f"   ✅ 체크포인트 사용: {config.enable_checkpoint} (테스트 모드: 비활성화)")
        
        # 서비스 초기화
        print("\n2️⃣  LangGraphWorkflowService 초기화 중...")
        service = LangGraphWorkflowService(config)
        print("   ✅ 서비스 초기화 완료")
        
        # 질의 처리
        print("\n3️⃣  질의 처리 중...")
        print("   (이 작업은 몇 초에서 몇 분이 걸릴 수 있습니다)")
        
        result = await service.process_query(
            query=query,
            session_id="single_query_test",
            enable_checkpoint=False  # 테스트이므로 체크포인트 비활성화
        )
        
        print("\n4️⃣  결과:")
        print("="*80)
        
        # 답변 추출
        answer = result.get("answer", "")
        answer_text = answer
        if isinstance(answer_text, dict):
            # 중첩된 딕셔너리에서 답변 추출 시도
            for key in ("answer", "content", "text"):
                if isinstance(answer_text, dict) and key in answer_text:
                    answer_text = answer_text[key]
            if isinstance(answer_text, dict):
                answer_text = str(answer_text)
        
        # 답변 출력
        print(f"\n📝 답변 (길이: {len(str(answer_text)) if answer_text else 0}자):")
        print("-" * 80)
        if answer_text:
            print(str(answer_text)[:1000])  # 처음 1000자만 출력
            if len(str(answer_text)) > 1000:
                print(f"\n... (총 {len(str(answer_text))}자, 나머지 생략)")
        else:
            print("<답변 없음>")
        
        # 소스 정보
        sources = result.get("sources", [])
        if sources:
            print(f"\n📚 소스 ({len(sources)}개):")
            print("-" * 80)
            for i, source in enumerate(sources[:5], 1):  # 최대 5개만 출력
                print(f"   {i}. {source}")
            if len(sources) > 5:
                print(f"   ... (총 {len(sources)}개)")
        
        # 법률 참조
        legal_references = result.get("legal_references", [])
        if legal_references:
            print(f"\n⚖️  법률 참조 ({len(legal_references)}개):")
            print("-" * 80)
            for i, ref in enumerate(legal_references[:5], 1):
                print(f"   {i}. {ref}")
            if len(legal_references) > 5:
                print(f"   ... (총 {len(legal_references)}개)")
        
        # 메타데이터
        metadata = result.get("metadata", {})
        if metadata:
            print(f"\n📊 메타데이터:")
            print("-" * 80)
            for key, value in list(metadata.items())[:10]:  # 최대 10개만 출력
                print(f"   {key}: {value}")
        
        # 신뢰도
        confidence = result.get("confidence", 0.0)
        if confidence:
            print(f"\n🎯 신뢰도: {confidence:.2f}")
        
        # 처리 시간
        processing_time = result.get("processing_time", 0.0)
        if processing_time:
            print(f"\n⏱️  처리 시간: {processing_time:.2f}초")
        
        print("\n" + "="*80)
        print("✅ 테스트 완료!")
        print("="*80)
        
        return result
        
    except ImportError as e:
        print(f"\n❌ Import 오류: {e}")
        print("\n필요한 패키지가 설치되어 있는지 확인하세요:")
        print("  - lawfirm_langgraph.config.langgraph_config")
        print("  - lawfirm_langgraph.langgraph_core.workflow.workflow_service")
        raise
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {type(e).__name__}: {e}")
        import traceback
        print("\n상세 오류:")
        traceback.print_exc()
        raise


def main():
    """메인 실행 함수"""
    # 기본 질의 목록
    default_queries = [
        "계약서 작성 시 주의할 사항은 무엇인가요?",
        "민법 제750조 손해배상에 대해 설명해주세요",
        "임대차 계약 해지 시 주의사항은 무엇인가요?",
    ]
    default_query = default_queries[0]
    
    # 질의 선택 방법:
    # 1. 명령줄 인자로 숫자 (0, 1, 2 등) - 기본 질의 목록에서 선택
    # 2. 명령줄 인자로 직접 질의 텍스트
    # 3. 인자가 없으면 첫 번째 기본 질의 사용
    
    query = None
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        
        # 숫자로 시작하면 기본 질의 목록에서 선택
        if arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(default_queries):
                query = default_queries[idx]
                print(f"\n💡 기본 질의 목록에서 선택: [{idx}]")
            else:
                print(f"\n⚠️  인덱스 {idx}가 범위를 벗어났습니다. 기본 질의를 사용합니다.")
                query = default_query
        else:
            # 직접 질의 텍스트로 간주
            # PowerShell 인코딩 문제 해결을 위해 여러 인자를 합침
            query_parts = sys.argv[1:]
            query = " ".join(query_parts)
            # 가능하면 UTF-8로 디코딩 시도
            try:
                if isinstance(query, bytes):
                    query = query.decode('utf-8')
            except (UnicodeDecodeError, AttributeError):
                pass  # 이미 문자열이면 그대로 사용
            
            print(f"\n💡 명령줄에서 질의를 받았습니다.")
    
    if query is None:
        query = default_query
        # 이미 출력됨
        print(f"   사용 가능한 기본 질의: 0='{default_queries[0]}', 1='{default_queries[1]}', 2='{default_queries[2]}'")
        print(f"   사용법: python run_single_query_test.py 0  (또는 직접 질의 입력)")
    
    # 비동기 실행
    try:
        result = asyncio.run(run_single_query_test(query))
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"\n\n❌ 테스트 실패: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

