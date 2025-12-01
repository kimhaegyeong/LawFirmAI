# -*- coding: utf-8 -*-
"""
DocumentType 및 DocumentTypeConfig 테스트 실행 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 경로 설정 (run_query_test.py와 동일한 방식)
script_dir = Path(__file__).parent
unit_dir = script_dir.parent
tests_dir = unit_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

from lawfirm_langgraph.core.workflow.constants.document_types import (
    DocumentType,
    DocumentTypeConfig,
)


def test_document_type_from_string():
    """DocumentType.from_string 테스트"""
    print("=" * 60)
    print("테스트: DocumentType.from_string")
    print("=" * 60)
    
    # 정확한 매칭
    assert DocumentType.from_string("statute_article") == DocumentType.STATUTE_ARTICLE
    assert DocumentType.from_string("precedent_content") == DocumentType.PRECEDENT_CONTENT
    # 레거시 호환: case_paragraph는 precedent_content로 매핑
    assert DocumentType.from_string("case_paragraph") == DocumentType.PRECEDENT_CONTENT
    print("✅ 정확한 매칭 테스트 통과")
    
    # 대소문자 무시
    assert DocumentType.from_string("STATUTE_ARTICLE") == DocumentType.STATUTE_ARTICLE
    assert DocumentType.from_string("Precedent_Content") == DocumentType.PRECEDENT_CONTENT
    print("✅ 대소문자 무시 테스트 통과")
    
    # 레거시 호환성
    assert DocumentType.from_string("case") == DocumentType.PRECEDENT_CONTENT
    assert DocumentType.from_string("precedent") == DocumentType.PRECEDENT_CONTENT
    print("✅ 레거시 호환성 테스트 통과")
    
    # 빈 문자열
    assert DocumentType.from_string("") == DocumentType.UNKNOWN
    assert DocumentType.from_string(None) == DocumentType.UNKNOWN
    print("✅ 빈 문자열 테스트 통과")
    
    print("✅ 모든 from_string 테스트 통과\n")


def test_document_type_from_metadata():
    """DocumentType.from_metadata 테스트"""
    print("=" * 60)
    print("테스트: DocumentType.from_metadata")
    print("=" * 60)
    
    # type 필드가 있는 경우
    doc = {"type": "statute_article"}
    assert DocumentType.from_metadata(doc) == DocumentType.STATUTE_ARTICLE
    print("✅ type 필드 테스트 통과")
    
    # source_type 필드가 있는 경우
    doc = {"source_type": "precedent_content"}
    assert DocumentType.from_metadata(doc) == DocumentType.PRECEDENT_CONTENT
    # 레거시 호환: case_paragraph는 precedent_content로 매핑
    doc = {"source_type": "case_paragraph"}
    assert DocumentType.from_metadata(doc) == DocumentType.PRECEDENT_CONTENT
    print("✅ source_type 필드 테스트 통과")
    
    # metadata.source_type이 있는 경우
    doc = {"metadata": {"source_type": "precedent_content"}}
    assert DocumentType.from_metadata(doc) == DocumentType.PRECEDENT_CONTENT
    print("✅ metadata.source_type 필드 테스트 통과")
    
    # statute_article 메타데이터 필드 기반 추론
    doc = {"metadata": {"statute_name": "민법"}}
    assert DocumentType.from_metadata(doc) == DocumentType.STATUTE_ARTICLE
    doc = {"metadata": {"law_name": "형법"}}
    assert DocumentType.from_metadata(doc) == DocumentType.STATUTE_ARTICLE
    doc = {"metadata": {"article_no": "750"}}
    assert DocumentType.from_metadata(doc) == DocumentType.STATUTE_ARTICLE
    print("✅ statute_article 메타데이터 필드 추론 테스트 통과")
    
    # precedent_content 메타데이터 필드 기반 추론
    doc = {"metadata": {"case_id": "12345"}}
    assert DocumentType.from_metadata(doc) == DocumentType.PRECEDENT_CONTENT
    doc = {"metadata": {"court": "대법원"}}
    assert DocumentType.from_metadata(doc) == DocumentType.PRECEDENT_CONTENT
    doc = {"metadata": {"casenames": "손해배상청구 사건"}}
    assert DocumentType.from_metadata(doc) == DocumentType.PRECEDENT_CONTENT
    doc = {"metadata": {"doc_id": "2021다12345"}}
    assert DocumentType.from_metadata(doc) == DocumentType.PRECEDENT_CONTENT
    print("✅ precedent_content 메타데이터 필드 추론 테스트 통과")
    
    
    
    # 알 수 없는 타입
    doc = {}
    assert DocumentType.from_metadata(doc) == DocumentType.UNKNOWN
    doc = {"metadata": {}}
    assert DocumentType.from_metadata(doc) == DocumentType.UNKNOWN
    print("✅ 알 수 없는 타입 테스트 통과")
    
    # 잘못된 입력
    assert DocumentType.from_metadata(None) == DocumentType.UNKNOWN
    assert DocumentType.from_metadata("not a dict") == DocumentType.UNKNOWN
    print("✅ 잘못된 입력 테스트 통과")
    
    print("✅ 모든 from_metadata 테스트 통과\n")


def test_document_type_config():
    """DocumentTypeConfig 테스트"""
    print("=" * 60)
    print("테스트: DocumentTypeConfig")
    print("=" * 60)
    
    # 최소 개수
    assert DocumentTypeConfig.get_min_count(DocumentType.STATUTE_ARTICLE) == 1
    assert DocumentTypeConfig.get_min_count(DocumentType.PRECEDENT_CONTENT) == 2
    print("✅ 최소 개수 테스트 통과")
    
    # 우선순위 부스팅
    assert DocumentTypeConfig.get_priority_boost(DocumentType.STATUTE_ARTICLE) == 1.0
    assert DocumentTypeConfig.get_priority_boost(DocumentType.PRECEDENT_CONTENT) == 0.9
    print("✅ 우선순위 부스팅 테스트 통과")
    
    # 필수 메타데이터 키
    keys = DocumentTypeConfig.get_required_metadata_keys(DocumentType.STATUTE_ARTICLE)
    assert "statute_name" in keys
    assert "law_name" in keys
    assert "article_no" in keys
    print("✅ 필수 메타데이터 키 테스트 통과")
    
    # 선택적 메타데이터 키
    keys = DocumentTypeConfig.get_optional_metadata_keys(DocumentType.STATUTE_ARTICLE)
    assert "clause_no" in keys
    assert "item_no" in keys
    print("✅ 선택적 메타데이터 키 테스트 통과")
    
    # 메타데이터 유효성 검증
    metadata = {"statute_name": "민법"}
    assert DocumentTypeConfig.is_valid_metadata(DocumentType.STATUTE_ARTICLE, metadata) == True
    metadata = {"some_other_field": "value"}
    assert DocumentTypeConfig.is_valid_metadata(DocumentType.STATUTE_ARTICLE, metadata) == False
    print("✅ 메타데이터 유효성 검증 테스트 통과")
    
    print("✅ 모든 DocumentTypeConfig 테스트 통과\n")


def test_real_world_examples():
    """실제 문서 예시 테스트"""
    print("=" * 60)
    print("테스트: 실제 문서 예시")
    print("=" * 60)
    
    # 법령 문서
    doc = {
        "type": "statute_article",
        "metadata": {
            "statute_name": "민법",
            "article_no": "750",
            "clause_no": "1",
            "item_no": "1"
        },
        "content": "불법행위로 인한 손해배상의무..."
    }
    doc_type = DocumentType.from_metadata(doc)
    assert doc_type == DocumentType.STATUTE_ARTICLE
    assert DocumentTypeConfig.is_valid_metadata(doc_type, doc["metadata"]) == True
    print("✅ 법령 문서 예시 테스트 통과")
    
    # 판례 문서
    doc = {
        "source_type": "precedent_content",
        "metadata": {
            "doc_id": "2021다12345",
            "court": "대법원",
            "casenames": "손해배상청구 사건",
            "case_id": "12345"
        },
        "content": "원고는 피고의 불법행위로 인한 손해를 입었다고 주장한다..."
    }
    doc_type = DocumentType.from_metadata(doc)
    assert doc_type == DocumentType.PRECEDENT_CONTENT
    assert DocumentTypeConfig.is_valid_metadata(doc_type, doc["metadata"]) == True
    print("✅ 판례 문서 예시 테스트 통과")
    
    
    print("✅ 모든 실제 문서 예시 테스트 통과\n")


def main():
    """메인 테스트 실행 함수"""
    print("\n" + "=" * 60)
    print("DocumentType 및 DocumentTypeConfig 테스트 시작")
    print("=" * 60 + "\n")
    
    try:
        test_document_type_from_string()
        test_document_type_from_metadata()
        test_document_type_config()
        test_real_world_examples()
        
        print("=" * 60)
        print("✅ 모든 테스트 통과!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

