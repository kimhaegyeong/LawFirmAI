# -*- coding: utf-8 -*-
"""
textToSql 간단 테스트 스크립트
LegalDataConnectorV2를 직접 사용하여 textToSql 검색 결과 확인

Usage:
    python scripts/tests/test_text2sql_simple.py "민법 제15조에 대해서 설명해줘"
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_dir = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_dir))

import logging
from lawfirm_langgraph.core.agents.legal_data_connector_v2 import LegalDataConnectorV2, route_query

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_text2sql(query: str, limit: int = 10):
    """textToSql 검색 테스트"""
    print("=" * 80)
    print("textToSql 검색 테스트")
    print("=" * 80)
    print(f"\n📋 질의: {query}\n")
    
    # 1. 라우팅 확인
    route = route_query(query)
    print(f"🔍 라우팅 결과: {route}")
    if route != "text2sql":
        print(f"⚠️  경고: 이 쿼리는 'text2sql'로 라우팅되지 않았습니다. (실제: '{route}')")
        print("   '제XX조' 패턴이 있는 쿼리만 text2sql로 라우팅됩니다.")
        return
    
    # 2. LegalDataConnectorV2 초기화
    print("\n2️⃣  LegalDataConnectorV2 초기화 중...")
    try:
        connector = LegalDataConnectorV2()
        print("   ✅ 초기화 완료")
    except Exception as e:
        print(f"   ❌ 초기화 실패: {e}")
        return
    
    # 3. 검색 실행
    print(f"\n3️⃣  검색 실행 중... (limit={limit})")
    try:
        results = connector.search_documents(query, limit=limit)
        print(f"   ✅ 검색 완료: {len(results)}개 결과")
    except Exception as e:
        print(f"   ❌ 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 결과 출력
    print("\n4️⃣  검색 결과:")
    print("=" * 80)
    
    if not results:
        print("   ⚠️  검색 결과가 없습니다.")
        return
    
    # 타입별 분류
    type_counts = {}
    for doc in results:
        doc_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "unknown")
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    print(f"\n📊 타입별 분포: {type_counts}")
    
    # 상세 결과 출력
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] " + "-" * 76)
        
        # 기본 정보
        doc_id = doc.get("doc_id") or doc.get("id") or doc.get("_id") or f"doc_{i}"
        doc_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "unknown")
        title = doc.get("title") or doc.get("name") or doc.get("content", "")[:50] or "제목 없음"
        source = doc.get("source") or doc.get("source_name") or "N/A"
        
        print(f"   타입: {doc_type}")
        print(f"   ID: {doc_id}")
        print(f"   제목: {title}")
        print(f"   출처: {source}")
        
        # 점수 정보
        score = doc.get("score") or doc.get("similarity") or doc.get("relevance_score")
        if score is not None:
            print(f"   점수: {score:.4f}")
        
        # statute_article 타입인 경우 상세 정보
        if doc_type == "statute_article":
            statute_name = doc.get("statute_name") or doc.get("law_name") or doc.get("metadata", {}).get("statute_name") or doc.get("metadata", {}).get("law_name")
            article_no = doc.get("article_no") or doc.get("article_number") or doc.get("metadata", {}).get("article_no") or doc.get("metadata", {}).get("article_number")
            clause_no = doc.get("clause_no") or doc.get("metadata", {}).get("clause_no")
            item_no = doc.get("item_no") or doc.get("metadata", {}).get("item_no")
            
            print(f"   법령명: {statute_name}")
            print(f"   조문번호: {article_no}")
            if clause_no:
                print(f"   항번호: {clause_no}")
            if item_no:
                print(f"   호번호: {item_no}")
        
        # 내용 미리보기
        content = doc.get("content") or doc.get("text") or ""
        if content:
            preview = content[:200] if len(content) > 200 else content
            print(f"   내용 미리보기: {preview}...")
        
        # 메타데이터
        metadata = doc.get("metadata", {})
        if metadata and doc_type != "statute_article":
            print(f"   메타데이터: {metadata}")
    
    print("\n" + "=" * 80)
    print("✅ 테스트 완료!")
    print("=" * 80)


def main():
    """메인 실행 함수"""
    # 명령줄 인자에서 질의 추출
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # 기본 질의
        query = "민법 제15조에 대해서 설명해줘"
        print(f"⚠️  질의가 지정되지 않아 기본 질의를 사용합니다: {query}")
        print(f"   사용법: python {sys.argv[0]} \"질의 내용\"\n")
    
    # limit 옵션 확인
    limit = 10
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            try:
                limit = int(sys.argv[idx + 1])
            except ValueError:
                print(f"⚠️  잘못된 limit 값: {sys.argv[idx + 1]}, 기본값 10 사용")
    
    test_text2sql(query, limit=limit)


if __name__ == "__main__":
    main()

