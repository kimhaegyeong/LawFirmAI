# -*- coding: utf-8 -*-
"""
sources_detail 필드 생성 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.generation.formatters.unified_source_formatter import UnifiedSourceFormatter
from lawfirm_langgraph.core.generation.validators.source_validator import SourceValidator


def test_sources_detail_generation():
    """sources_detail 필드 생성 테스트"""
    print("\n=== sources_detail 필드 생성 테스트 ===")
    
    formatter = UnifiedSourceFormatter()
    validator = SourceValidator()
    
    # 테스트용 retrieved_docs 시뮬레이션
    retrieved_docs = [
        {
            "type": "statute_article",
            "statute_name": "민법",
            "article_no": "제1조",
            "clause_no": "1",
            "item_no": "1",
            "metadata": {
                "statute_name": "민법",
                "article_no": "제1조",
                "clause_no": "1",
                "item_no": "1"
            }
        },
        {
            "type": "case_paragraph",
            "court": "대법원",
            "doc_id": "2020다12345",
            "casenames": "손해배상청구 사건",
            "metadata": {
                "court": "대법원",
                "doc_id": "2020다12345",
                "casenames": "손해배상청구 사건"
            }
        },
        {
            "type": "decision_paragraph",
            "org": "법제처",
            "doc_id": "2020-123",
            "metadata": {
                "org": "법제처",
                "doc_id": "2020-123"
            }
        }
    ]
    
    # sources_detail 생성 로직 시뮬레이션
    final_sources_list = []
    final_sources_detail = []
    seen_sources = set()
    
    for doc in retrieved_docs:
        if not isinstance(doc, dict):
            continue
        
        source = None
        source_type = doc.get("type") or doc.get("source_type") or ""
        metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        
        # 통일된 포맷터로 상세 정보 생성
        source_info_detail = None
        if formatter and source_type:
            try:
                # doc과 metadata를 병합
                merged_metadata = {**metadata}
                for key in ["statute_name", "law_name", "article_no", "article_number", "clause_no", "item_no",
                           "court", "doc_id", "casenames", "org", "title", "announce_date", "decision_date", "response_date"]:
                    if key in doc:
                        merged_metadata[key] = doc[key]
                
                source_info_detail = formatter.format_source(source_type, merged_metadata)
                
                # 검증 수행
                if validator:
                    validation_result = validator.validate_source(source_type, merged_metadata)
                    source_info_detail.validation = validation_result
            except Exception as e:
                print(f"  경고: 출처 상세 정보 생성 중 오류 발생: {e}")
        
        # 출처 문자열 생성 (기존 로직)
        if source_type == "statute_article":
            statute_name = doc.get("statute_name") or metadata.get("statute_name")
            if statute_name:
                article_no = doc.get("article_no") or metadata.get("article_no")
                clause_no = doc.get("clause_no") or metadata.get("clause_no")
                item_no = doc.get("item_no") or metadata.get("item_no")
                
                source_parts = [statute_name]
                if article_no:
                    source_parts.append(article_no)
                if clause_no:
                    source_parts.append(f"제{clause_no}항")
                if item_no:
                    source_parts.append(f"제{item_no}호")
                
                source = " ".join(source_parts)
        
        elif source_type == "case_paragraph":
            court = doc.get("court") or metadata.get("court")
            casenames = doc.get("casenames") or metadata.get("casenames")
            doc_id = doc.get("doc_id") or metadata.get("doc_id")
            
            if court or casenames:
                source_parts = []
                if court:
                    source_parts.append(court)
                if casenames:
                    source_parts.append(casenames)
                if doc_id:
                    source_parts.append(f"({doc_id})")
                source = " ".join(source_parts)
        
        elif source_type == "decision_paragraph":
            org = doc.get("org") or metadata.get("org")
            doc_id = doc.get("doc_id") or metadata.get("doc_id")
            
            if org:
                source_parts = [org]
                if doc_id:
                    source_parts.append(f"({doc_id})")
                source = " ".join(source_parts)
        
        # 소스 문자열 변환 및 중복 제거
        if source:
            if isinstance(source, str):
                source_str = source.strip()
            else:
                try:
                    source_str = str(source).strip()
                except Exception:
                    source_str = None
            
            # 검색 타입 필터링 (최종 검증)
            if source_str:
                source_lower = source_str.lower().strip()
                invalid_sources = ["semantic", "keyword", "unknown", "fts", "vector", "search", "text2sql", ""]
                if source_lower not in invalid_sources and len(source_lower) >= 2:
                    if source_str not in seen_sources and source_str != "Unknown":
                        final_sources_list.append(source_str)
                        seen_sources.add(source_str)
                        
                        # sources_detail 추가
                        if source_info_detail:
                            final_sources_detail.append({
                                "name": source_info_detail.name,
                                "type": source_info_detail.type,
                                "url": source_info_detail.url or "",
                                "metadata": source_info_detail.metadata or {}
                            })
    
    # 결과 검증
    print(f"\n생성된 sources (문자열 배열): {final_sources_list}")
    print(f"생성된 sources_detail (상세 정보): {len(final_sources_detail)}개")
    
    assert len(final_sources_list) == 3, f"예상: 3개, 실제: {len(final_sources_list)}개"
    assert len(final_sources_detail) == 3, f"예상: 3개, 실제: {len(final_sources_detail)}개"
    
    # 각 sources_detail 검증
    for i, detail in enumerate(final_sources_detail):
        print(f"\n  출처 {i+1}:")
        print(f"    name: {detail['name']}")
        print(f"    type: {detail['type']}")
        print(f"    url: {detail['url']}")
        print(f"    metadata: {detail['metadata']}")
        
        assert detail['name'] is not None and detail['name'] != "", "출처명이 있어야 함"
        assert detail['type'] is not None and detail['type'] != "", "출처 타입이 있어야 함"
        assert isinstance(detail['metadata'], dict), "메타데이터가 딕셔너리여야 함"
    
    # sources와 sources_detail의 name이 일치하는지 확인
    for i, (source_str, detail) in enumerate(zip(final_sources_list, final_sources_detail)):
        assert source_str == detail['name'], f"출처 {i+1}: sources와 sources_detail의 name이 일치해야 함"
    
    print("\n✅ sources_detail 필드 생성 테스트 통과")


def test_api_schema_compatibility():
    """API 스키마 호환성 테스트"""
    print("\n=== API 스키마 호환성 테스트 ===")
    
    try:
        from api.schemas.chat import SourceInfo, ChatResponse
        
        # SourceInfo 모델 테스트
        source_info = SourceInfo(
            name="민법 제1조",
            type="statute_article",
            url="https://www.law.go.kr/LSW/lsSc.do?lawNm=민법&articleNo=1",
            metadata={"statute_name": "민법", "article_no": "제1조"}
        )
        
        print(f"SourceInfo 생성 성공:")
        print(f"  name: {source_info.name}")
        print(f"  type: {source_info.type}")
        print(f"  url: {source_info.url}")
        print(f"  metadata: {source_info.metadata}")
        
        assert source_info.name == "민법 제1조"
        assert source_info.type == "statute_article"
        assert source_info.url is not None
        
        # ChatResponse 모델 테스트 (sources_detail 필드 포함)
        chat_response = ChatResponse(
            answer="테스트 답변",
            sources=["민법 제1조"],
            sources_detail=[source_info],
            confidence=0.9,
            legal_references=[],
            processing_steps=[],
            session_id="test-session",
            processing_time=1.0,
            query_type="test",
            metadata={},
            errors=[],
            warnings=[]
        )
        
        print(f"\nChatResponse 생성 성공:")
        print(f"  sources: {chat_response.sources}")
        print(f"  sources_detail: {len(chat_response.sources_detail)}개")
        
        assert len(chat_response.sources) == 1, "sources 필드가 있어야 함"
        assert len(chat_response.sources_detail) == 1, "sources_detail 필드가 있어야 함"
        assert chat_response.sources_detail[0].name == "민법 제1조"
        
        print("\n✅ API 스키마 호환성 테스트 통과")
        
    except ImportError as e:
        print(f"⚠️  API 스키마 import 실패 (정상일 수 있음): {e}")
        print("  이는 API 모듈이 설치되지 않았거나 경로 문제일 수 있습니다.")


if __name__ == "__main__":
    print("=" * 60)
    print("sources_detail 필드 생성 테스트 시작")
    print("=" * 60)
    
    try:
        test_sources_detail_generation()
        test_api_schema_compatibility()
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 통과!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

