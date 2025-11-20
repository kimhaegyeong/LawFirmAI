# -*- coding: utf-8 -*-
"""
실제 LLM 프롬프트 분석 테스트
- 생성된 프롬프트 확인
- 검색된 문서와 프롬프트에 포함된 문서 비교
- 누락된 문서 확인
- 프롬프트 개선 사항 도출
"""

import sys
import os
import json
import re
from typing import Dict, Any, List

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

def extract_documents_from_prompt(prompt: str) -> Dict[str, List[Dict[str, Any]]]:
    """프롬프트에서 문서 정보 추출"""
    documents = {
        "statute_article": [],
        "case_paragraph": [],
        "decision_paragraph": [],
        "interpretation_paragraph": []
    }
    
    # 타입별 섹션 찾기
    statute_section = re.search(r'### 📜 법령 조문\n\n(.*?)(?=###|$)', prompt, re.DOTALL)
    case_section = re.search(r'### ⚖️ 판례\n\n(.*?)(?=###|$)', prompt, re.DOTALL)
    decision_section = re.search(r'### 📋 결정례\n\n(.*?)(?=###|$)', prompt, re.DOTALL)
    interpretation_section = re.search(r'### 📖 해석례\n\n(.*?)(?=###|$)', prompt, re.DOTALL)
    
    # 각 섹션에서 문서 추출
    if statute_section:
        doc_matches = re.finditer(r'\*\*문서 (\d+)\*\*: (.*?) \(관련도: ([\d.]+)\)', statute_section.group(1))
        for match in doc_matches:
            documents["statute_article"].append({
                "number": int(match.group(1)),
                "title": match.group(2),
                "relevance": float(match.group(3))
            })
    
    if case_section:
        doc_matches = re.finditer(r'\*\*문서 (\d+)\*\*: (.*?) \(관련도: ([\d.]+)\)', case_section.group(1))
        for match in doc_matches:
            documents["case_paragraph"].append({
                "number": int(match.group(1)),
                "title": match.group(2),
                "relevance": float(match.group(3))
            })
    
    if decision_section:
        doc_matches = re.finditer(r'\*\*문서 (\d+)\*\*: (.*?) \(관련도: ([\d.]+)\)', decision_section.group(1))
        for match in doc_matches:
            documents["decision_paragraph"].append({
                "number": int(match.group(1)),
                "title": match.group(2),
                "relevance": float(match.group(3))
            })
    
    if interpretation_section:
        doc_matches = re.finditer(r'\*\*문서 (\d+)\*\*: (.*?) \(관련도: ([\d.]+)\)', interpretation_section.group(1))
        for match in doc_matches:
            documents["interpretation_paragraph"].append({
                "number": int(match.group(1)),
                "title": match.group(2),
                "relevance": float(match.group(3))
            })
    
    return documents

def analyze_prompt_improvements(prompt: str, retrieved_docs: List[Dict], structured_docs: List[Dict]) -> Dict[str, Any]:
    """프롬프트 개선 사항 분석"""
    improvements = {
        "missing_documents": [],
        "document_count_issues": [],
        "prompt_structure_issues": [],
        "data_quality_issues": []
    }
    
    # 1. 누락된 문서 확인
    retrieved_doc_ids = {doc.get("document_id") or doc.get("doc_id") for doc in retrieved_docs if doc}
    
    # 프롬프트에서 문서 ID 추출 (간단한 방법)
    prompt_doc_ids = set()
    for doc_type in ["statute_article", "case_paragraph", "decision_paragraph", "interpretation_paragraph"]:
        # 프롬프트에서 해당 타입의 문서 찾기
        pattern = rf'### .*?{doc_type.replace("_", " ")}\n\n(.*?)(?=###|$)'
        section = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)
        if section:
            # 문서 ID나 제목 추출 시도
            pass
    
    # 2. 문서 수 확인
    prompt_docs = extract_documents_from_prompt(prompt)
    total_prompt_docs = sum(len(docs) for docs in prompt_docs.values())
    total_retrieved = len(retrieved_docs)
    total_structured = len(structured_docs)
    
    if total_prompt_docs < total_retrieved * 0.8:
        improvements["document_count_issues"].append(
            f"프롬프트에 포함된 문서가 검색된 문서의 80% 미만입니다 "
            f"(프롬프트: {total_prompt_docs}개, 검색: {total_retrieved}개)"
        )
    
    # 3. 프롬프트 구조 확인
    if "## 🔍 검색된 법률 문서" not in prompt:
        improvements["prompt_structure_issues"].append("검색된 법률 문서 섹션이 없습니다")
    
    # 타입별 섹션 확인
    type_sections = {
        "법령 조문": "📜 법령 조문" in prompt,
        "판례": "⚖️ 판례" in prompt,
        "결정례": "📋 결정례" in prompt,
        "해석례": "📖 해석례" in prompt
    }
    
    missing_types = [t for t, exists in type_sections.items() if not exists]
    if missing_types:
        improvements["prompt_structure_issues"].append(
            f"다음 타입의 문서 섹션이 없습니다: {', '.join(missing_types)}"
        )
    
    # 4. 데이터 품질 확인
    if total_prompt_docs == 0:
        improvements["data_quality_issues"].append("프롬프트에 문서가 전혀 포함되지 않았습니다")
    
    # 관련도 분포 확인
    relevance_scores = []
    for doc_type, docs in prompt_docs.items():
        for doc in docs:
            relevance_scores.append(doc.get("relevance", 0.0))
    
    if relevance_scores:
        min_relevance = min(relevance_scores)
        if min_relevance < 0.2:
            improvements["data_quality_issues"].append(
                f"관련도가 0.2 미만인 문서가 포함되어 있습니다 (최소: {min_relevance:.3f})"
            )
    
    return improvements

def test_prompt_analysis():
    """프롬프트 분석 테스트"""
    print("\n" + "=" * 80)
    print("실제 LLM 프롬프트 분석 테스트")
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
            "session_id": "test_session_prompt_analysis",
            "metadata": {}
        }
        
        # 프롬프트를 캡처하기 위해 UnifiedPromptManager의 _build_final_prompt를 모니터링
        # 또는 generate_answer_enhanced 실행 후 프롬프트 확인
        
        # 워크플로우 실행
        state = workflow.generate_answer_enhanced(initial_state)
        
        # 결과 확인
        retrieved_docs = state.get("retrieved_docs", [])
        structured_docs_dict = state.get("structured_documents", {})
        structured_docs = structured_docs_dict.get("documents", []) if isinstance(structured_docs_dict, dict) else []
        
        print(f"\n📊 검색 결과:")
        print(f"   - 검색된 문서 수: {len(retrieved_docs)}개")
        print(f"   - structured_documents 문서 수: {len(structured_docs)}개")
        
        # 타입별 분포 확인
        if retrieved_docs:
            type_distribution = {}
            for doc in retrieved_docs:
                doc_type = (
                    doc.get("type") or
                    doc.get("source_type") or
                    doc.get("metadata", {}).get("type") if isinstance(doc.get("metadata"), dict) else None or
                    "unknown"
                )
                type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
            
            print(f"\n📊 검색된 문서 타입별 분포:")
            for doc_type, count in type_distribution.items():
                print(f"   - {doc_type}: {count}개")
        
        # 프롬프트 확인을 위해 answer_generator에서 프롬프트 가져오기
        # 또는 로그에서 프롬프트 확인
        
        # 실제 프롬프트를 확인하기 위해 UnifiedPromptManager를 직접 호출
        from lawfirm_langgraph.core.services.unified_prompt_manager import UnifiedPromptManager
        from lawfirm_langgraph.core.classification.classifiers.question_classifier import QuestionType
        
        prompt_manager = UnifiedPromptManager()
        
        # context 구성
        context = {
            "structured_documents": {
                "documents": structured_docs if structured_docs else retrieved_docs,
                "total_count": len(structured_docs) if structured_docs else len(retrieved_docs)
            },
            "document_count": len(structured_docs) if structured_docs else len(retrieved_docs)
        }
        
        # 프롬프트 생성
        base_prompt = "테스트"
        final_prompt = prompt_manager._build_final_prompt(
            base_prompt=base_prompt,
            query=test_query,
            context=context,
            question_type=QuestionType.TERM_EXPLANATION
        )
        
        # 프롬프트 분석
        prompt_docs = extract_documents_from_prompt(final_prompt)
        improvements = analyze_prompt_improvements(final_prompt, retrieved_docs, structured_docs)
        
        print(f"\n📋 프롬프트에 포함된 문서:")
        total_prompt_docs = 0
        for doc_type, docs in prompt_docs.items():
            if docs:
                print(f"   - {doc_type}: {len(docs)}개")
                total_prompt_docs += len(docs)
                for doc in docs[:3]:  # 상위 3개만 표시
                    print(f"     * 문서 {doc['number']}: {doc['title'][:50]}... (관련도: {doc['relevance']:.3f})")
        
        print(f"\n   총 포함된 문서: {total_prompt_docs}개")
        
        # 개선 사항 출력
        print(f"\n🔍 프롬프트 개선 사항:")
        
        if improvements["missing_documents"]:
            print(f"\n   ❌ 누락된 문서:")
            for issue in improvements["missing_documents"]:
                print(f"      - {issue}")
        
        if improvements["document_count_issues"]:
            print(f"\n   ⚠️ 문서 수 문제:")
            for issue in improvements["document_count_issues"]:
                print(f"      - {issue}")
        
        if improvements["prompt_structure_issues"]:
            print(f"\n   ⚠️ 프롬프트 구조 문제:")
            for issue in improvements["prompt_structure_issues"]:
                print(f"      - {issue}")
        
        if improvements["data_quality_issues"]:
            print(f"\n   ⚠️ 데이터 품질 문제:")
            for issue in improvements["data_quality_issues"]:
                print(f"      - {issue}")
        
        if not any(improvements.values()):
            print(f"\n   ✅ 특별한 문제가 발견되지 않았습니다.")
        
        # 프롬프트 저장
        prompt_file = "test_prompt_analysis_prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(final_prompt)
        print(f"\n💾 프롬프트가 {prompt_file}에 저장되었습니다.")
        
        # 결과 저장
        result_file = "test_prompt_analysis_result.json"
        result_data = {
            "query": test_query,
            "retrieved_docs_count": len(retrieved_docs),
            "structured_docs_count": len(structured_docs),
            "prompt_docs_count": total_prompt_docs,
            "prompt_docs_by_type": {k: len(v) for k, v in prompt_docs.items()},
            "improvements": improvements
        }
        
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"💾 분석 결과가 {result_file}에 저장되었습니다.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prompt_analysis()
    sys.exit(0 if success else 1)

