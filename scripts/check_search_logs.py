# -*- coding: utf-8 -*-
"""
검색 로그 확인 스크립트
검색 단계별 로그를 확인하여 문제 진단
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

from core.agents.legal_data_connector_v2 import LegalDataConnectorV2
from source.services.semantic_search_engine_v2 import SemanticSearchEngineV2
from source.utils.config import Config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_search_components():
    """검색 컴포넌트 직접 테스트"""
    print("=" * 80)
    print("검색 컴포넌트 테스트")
    print("=" * 80)

    config = Config()
    db_path = "./data/lawfirm_v2.db"

    if not os.path.exists(db_path):
        print(f"❌ 데이터베이스가 없습니다: {db_path}")
        return

    print(f"\n✅ 데이터베이스 존재: {db_path}")

    # LegalDataConnectorV2 테스트
    print("\n[1] LegalDataConnectorV2 테스트")
    try:
        connector = LegalDataConnectorV2(db_path)
        print(f"   ✅ LegalDataConnectorV2 초기화 성공")

        query = "계약 해지"

        # FTS 검색 테스트
        print(f"\n   검색 쿼리: '{query}'")

        statute_results = connector.search_statutes_fts(query, limit=5)
        print(f"   - 법령 FTS 검색: {len(statute_results)}개")

        case_results = connector.search_cases_fts(query, limit=5)
        print(f"   - 판례 FTS 검색: {len(case_results)}개")

        decision_results = connector.search_decisions_fts(query, limit=5)
        print(f"   - 심결례 FTS 검색: {len(decision_results)}개")

        interp_results = connector.search_interpretations_fts(query, limit=5)
        print(f"   - 유권해석 FTS 검색: {len(interp_results)}개")

        total_fts = len(statute_results) + len(case_results) + len(decision_results) + len(interp_results)
        print(f"   - 총 FTS 검색 결과: {total_fts}개")

    except Exception as e:
        print(f"   ❌ LegalDataConnectorV2 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

    # SemanticSearchEngineV2 테스트
    print("\n[2] SemanticSearchEngineV2 테스트")
    try:
        engine = SemanticSearchEngineV2(db_path)
        print(f"   ✅ SemanticSearchEngineV2 초기화 성공")

        if not engine.embedder:
            print(f"   ⚠️ 임베딩 모델이 로드되지 않았습니다")
            return

        query = "계약 해지"
        print(f"\n   검색 쿼리: '{query}'")

        results = engine.search(query, k=5, similarity_threshold=0.2)
        print(f"   - 벡터 검색 결과: {len(results)}개")

        if len(results) > 0:
            for i, r in enumerate(results[:3], 1):
                print(f"     [{i}] score={r.get('score', 0):.3f}, type={r.get('type')}, source={r.get('source', 'N/A')[:50]}")

    except Exception as e:
        print(f"   ❌ SemanticSearchEngineV2 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)

if __name__ == "__main__":
    test_search_components()
