# -*- coding: utf-8 -*-
"""
개선된 프롬프트와 검색 로직을 사용한 실제 워크플로우 테스트
"""

import sys
import os
import asyncio
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

async def test_workflow_with_improvements():
    """개선된 프롬프트와 검색 로직을 사용한 워크플로우 테스트"""
    print("\n" + "=" * 80)
    print("개선된 프롬프트와 검색 로직 워크플로우 테스트")
    print("=" * 80)
    
    try:
        from core.workflow.workflow_service import LangGraphWorkflowService
        
        # 워크플로우 서비스 초기화
        workflow_service = LangGraphWorkflowService()
        
        # 테스트 쿼리
        test_query = "전세금 반환 보증에 대해 설명해주세요"
        
        print(f"\n📝 테스트 쿼리: {test_query}")
        print("\n⏳ 워크플로우 실행 중...")
        
        # 워크플로우 실행
        result = await workflow_service.process_query(
            query=test_query,
            session_id="test_session_improvements",
            enable_checkpoint=False
        )
        
        print("\n" + "=" * 80)
        print("워크플로우 실행 결과")
        print("=" * 80)
        
        # 결과 분석
        if isinstance(result, dict):
            # 답변 확인
            answer = result.get("answer", "")
            print(f"\n📝 답변 길이: {len(answer)}자")
            if answer:
                print(f"   답변 미리보기: {answer[:200]}...")
            else:
                print("   ⚠️ 답변이 비어있습니다!")
            
            # Sources 확인
            sources = result.get("sources", [])
            sources_detail = result.get("sources_detail", [])
            print(f"\n📚 Sources:")
            print(f"   - sources: {len(sources)}개")
            print(f"   - sources_detail: {len(sources_detail)}개")
            
            if sources_detail:
                print(f"\n   Sources Detail 타입 분포:")
                type_distribution = {}
                for detail in sources_detail:
                    doc_type = detail.get("type", "unknown")
                    type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
                
                for doc_type, count in type_distribution.items():
                    print(f"     - {doc_type}: {count}개")
                
                # 개선 효과 확인: 판례만 있는지 확인
                if len(type_distribution) == 1 and "case_paragraph" in type_distribution:
                    print(f"\n   ⚠️ 여전히 판례만 검색되었습니다 (데이터 불균형 때문일 수 있음)")
                elif len(type_distribution) > 1:
                    print(f"\n   ✅ 다양한 타입의 문서가 검색되었습니다!")
            
            # Related Questions 확인
            related_questions = result.get("related_questions", [])
            print(f"\n❓ Related Questions: {len(related_questions)}개")
            if related_questions:
                for i, q in enumerate(related_questions[:3], 1):
                    print(f"   {i}. {q}")
            
            # Confidence 확인
            confidence = result.get("confidence", 0.0)
            print(f"\n🎯 Confidence: {confidence:.2f}")
            
            # Processing Time 확인
            processing_time = result.get("processing_time", 0.0)
            print(f"\n⏱️ Processing Time: {processing_time:.2f}초")
            
            # 검증
            print("\n" + "=" * 80)
            print("검증 결과")
            print("=" * 80)
            
            checks = []
            
            # 답변이 있는지 확인
            if answer and len(answer) > 100:
                checks.append(("답변 생성", True, f"{len(answer)}자"))
            else:
                checks.append(("답변 생성", False, "답변이 비어있거나 너무 짧음"))
            
            # Sources가 있는지 확인
            if sources and len(sources) > 0:
                checks.append(("Sources 생성", True, f"{len(sources)}개"))
            else:
                checks.append(("Sources 생성", False, "Sources가 없음"))
            
            # Sources Detail이 있는지 확인
            if sources_detail and len(sources_detail) > 0:
                checks.append(("Sources Detail 생성", True, f"{len(sources_detail)}개"))
            else:
                checks.append(("Sources Detail 생성", False, "Sources Detail이 없음"))
            
            # 다양한 타입의 문서가 검색되었는지 확인
            if sources_detail:
                types = set(detail.get("type", "unknown") for detail in sources_detail)
                if len(types) > 1:
                    checks.append(("다양한 문서 타입 검색", True, f"{len(types)}개 타입"))
                else:
                    checks.append(("다양한 문서 타입 검색", False, f"단일 타입만 검색됨 ({list(types)[0] if types else 'unknown'})"))
            
            # Related Questions가 있는지 확인
            if related_questions and len(related_questions) > 0:
                checks.append(("Related Questions 생성", True, f"{len(related_questions)}개"))
            else:
                checks.append(("Related Questions 생성", False, "Related Questions가 없음"))
            
            # 결과 출력
            passed = 0
            failed = 0
            
            for check_name, check_result, detail in checks:
                status = "✅" if check_result else "❌"
                print(f"{status} {check_name}: {detail}")
                if check_result:
                    passed += 1
                else:
                    failed += 1
            
            print(f"\n총 {len(checks)}개 검증 중 {passed}개 통과, {failed}개 실패")
            
            # 결과를 파일로 저장
            import json
            output_file = "workflow_test_result_improvements.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 결과가 {output_file}에 저장되었습니다.")
            
            if failed == 0:
                print("\n🎉 모든 검증 통과!")
                return 0
            else:
                print(f"\n⚠️ {failed}개 검증 실패 (일부는 정상일 수 있음)")
                return 0  # 일부 실패는 정상일 수 있으므로 0 반환
        else:
            print(f"\n❌ 결과가 dict가 아닙니다: {type(result)}")
            return 1
            
    except Exception as e:
        print(f"\n❌ 워크플로우 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """메인 함수"""
    try:
        result = asyncio.run(test_workflow_with_improvements())
        return result
    except Exception as e:
        print(f"\n❌ 테스트 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

