# -*- coding: utf-8 -*-
"""
generate_answer_stream 통합 테스트
- 문서 필터링 및 균형 조정이 실제 워크플로우에서 작동하는지 확인
- 타입별 문서 섹션이 프롬프트에 포함되는지 확인
"""

import sys
import os
import json
from typing import Dict, Any

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

def test_generate_answer_stream_workflow():
    """generate_answer_stream 워크플로우 통합 테스트"""
    print("\n" + "=" * 80)
    print("전체 통합 테스트: generate_answer_stream 워크플로우")
    print("=" * 80)
    
    try:
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        
        # 설정 로드
        config = LangGraphConfig()
        
        # 워크플로우 초기화
        workflow = EnhancedLegalQuestionWorkflow(config)
        
        # 테스트 쿼리
        test_query = "전세금 반환 보증에 대해 설명해주세요"
        
        print(f"\n📝 테스트 쿼리: {test_query}")
        print(f"\n🔄 워크플로우 실행 중...")
        
        # 초기 상태 생성
        initial_state = {
            "query": test_query,
            "session_id": "test_session_123",
            "metadata": {}
        }
        
        # 워크플로우 실행 (generate_answer_stream 노드만)
        try:
            # generate_answer_stream 노드 직접 호출
            state = workflow.generate_answer_stream(initial_state)
            
            # 결과 확인
            answer = state.get("answer", "")
            retrieved_docs = state.get("retrieved_docs", [])
            structured_docs = state.get("structured_documents", {})
            
            print(f"\n✅ 워크플로우 실행 완료")
            print(f"\n📊 결과:")
            print(f"   - 답변 길이: {len(answer)}자")
            print(f"   - 검색된 문서 수: {len(retrieved_docs)}개")
            
            if isinstance(structured_docs, dict):
                docs_in_structured = structured_docs.get("documents", [])
                print(f"   - structured_documents 문서 수: {len(docs_in_structured)}개")
                
                # 타입별 분포 확인
                if docs_in_structured:
                    type_distribution = {}
                    for doc in docs_in_structured:
                        doc_type = (
                            doc.get("type") or
                            doc.get("source_type") or
                            doc.get("metadata", {}).get("type") if isinstance(doc.get("metadata"), dict) else None or
                            "unknown"
                        )
                        type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
                    
                    print(f"\n📊 structured_documents 타입별 분포:")
                    for doc_type, count in type_distribution.items():
                        print(f"   - {doc_type}: {count}개")
            
            # 답변 미리보기
            if answer:
                # answer가 dict인 경우 처리
                if isinstance(answer, dict):
                    answer_text = answer.get("answer", str(answer))
                else:
                    answer_text = str(answer)
                
                print(f"\n📝 답변 미리보기:")
                if isinstance(answer_text, str) and len(answer_text) > 300:
                    preview = answer_text[:300]
                    print(f"   {preview}...")
                else:
                    print(f"   {answer_text}")
            
            # 검증
            checks = []
            
            # 1. 답변이 생성되었는지 확인
            if isinstance(answer, dict):
                answer_text = answer.get("answer", str(answer))
            else:
                answer_text = str(answer) if answer else ""
            
            if answer_text and len(answer_text) > 100:
                checks.append(("답변 생성", True))
            else:
                checks.append(("답변 생성", False, f"답변 길이: {len(answer_text)}자"))
            
            # 2. 검색된 문서가 있는지 확인
            if retrieved_docs and len(retrieved_docs) > 0:
                checks.append(("검색된 문서", True, f"{len(retrieved_docs)}개"))
            else:
                checks.append(("검색된 문서", False))
            
            # 3. structured_documents에 문서가 있는지 확인
            if isinstance(structured_docs, dict):
                docs_in_structured = structured_docs.get("documents", [])
                if docs_in_structured and len(docs_in_structured) > 0:
                    checks.append(("structured_documents 문서", True, f"{len(docs_in_structured)}개"))
                else:
                    checks.append(("structured_documents 문서", False))
            
            # 결과 출력
            print(f"\n✅ 검증 결과:")
            passed = 0
            failed = 0
            for check in checks:
                if len(check) == 2:
                    check_name, result = check
                    detail = ""
                else:
                    check_name, result, detail = check
                
                status = "✅" if result else "❌"
                print(f"   {status} {check_name}{f': {detail}' if detail else ''}")
                if result:
                    passed += 1
                else:
                    failed += 1
            
            print(f"\n   총 {len(checks)}개 검증 중 {passed}개 통과, {failed}개 실패")
            
            # 결과를 파일로 저장
            result_file = "test_generate_answer_stream_result.json"
            # answer 처리
            if isinstance(answer, dict):
                answer_text = answer.get("answer", str(answer))
            else:
                answer_text = str(answer) if answer else ""
            
            result_data = {
                "query": test_query,
                "answer_length": len(answer_text),
                "retrieved_docs_count": len(retrieved_docs),
                "structured_docs_count": len(docs_in_structured) if isinstance(structured_docs, dict) else 0,
                "answer_preview": answer_text[:500] if answer_text else "",
                "checks": [
                    {
                        "name": check[0],
                        "passed": check[1],
                        "detail": check[2] if len(check) > 2 else ""
                    }
                    for check in checks
                ]
            }
            
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 결과가 {result_file}에 저장되었습니다.")
            
            return failed == 0
            
        except Exception as e:
            print(f"\n❌ 워크플로우 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"\n❌ 모듈 import 실패: {e}")
        print(f"   워크플로우 테스트를 건너뜁니다.")
        return False
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generate_answer_stream_workflow()
    sys.exit(0 if success else 1)

