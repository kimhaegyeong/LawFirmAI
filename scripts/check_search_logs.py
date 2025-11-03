# -*- coding: utf-8 -*-
"""
ê²€??ë¡œê·¸ ?•ì¸ ?¤í¬ë¦½íŠ¸
ê²€???¨ê³„ë³?ë¡œê·¸ë¥??•ì¸?˜ì—¬ ë¬¸ì œ ì§„ë‹¨
"""

import os
import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

from source.agents.legal_data_connector_v2 import LegalDataConnectorV2
from source.services.semantic_search_engine_v2 import SemanticSearchEngineV2
from source.utils.config import Config

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_search_components():
    """ê²€??ì»´í¬?ŒíŠ¸ ì§ì ‘ ?ŒìŠ¤??""
    print("=" * 80)
    print("ê²€??ì»´í¬?ŒíŠ¸ ?ŒìŠ¤??)
    print("=" * 80)

    config = Config()
    db_path = "./data/lawfirm_v2.db"

    if not os.path.exists(db_path):
        print(f"???°ì´?°ë² ?´ìŠ¤ê°€ ?†ìŠµ?ˆë‹¤: {db_path}")
        return

    print(f"\n???°ì´?°ë² ?´ìŠ¤ ì¡´ìž¬: {db_path}")

    # LegalDataConnectorV2 ?ŒìŠ¤??
    print("\n[1] LegalDataConnectorV2 ?ŒìŠ¤??)
    try:
        connector = LegalDataConnectorV2(db_path)
        print(f"   ??LegalDataConnectorV2 ì´ˆê¸°???±ê³µ")

        query = "ê³„ì•½ ?´ì?"

        # FTS ê²€???ŒìŠ¤??
        print(f"\n   ê²€??ì¿¼ë¦¬: '{query}'")

        statute_results = connector.search_statutes_fts(query, limit=5)
        print(f"   - ë²•ë ¹ FTS ê²€?? {len(statute_results)}ê°?)

        case_results = connector.search_cases_fts(query, limit=5)
        print(f"   - ?ë? FTS ê²€?? {len(case_results)}ê°?)

        decision_results = connector.search_decisions_fts(query, limit=5)
        print(f"   - ?¬ê²°ë¡€ FTS ê²€?? {len(decision_results)}ê°?)

        interp_results = connector.search_interpretations_fts(query, limit=5)
        print(f"   - ? ê¶Œ?´ì„ FTS ê²€?? {len(interp_results)}ê°?)

        total_fts = len(statute_results) + len(case_results) + len(decision_results) + len(interp_results)
        print(f"   - ì´?FTS ê²€??ê²°ê³¼: {total_fts}ê°?)

    except Exception as e:
        print(f"   ??LegalDataConnectorV2 ?ŒìŠ¤???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()

    # SemanticSearchEngineV2 ?ŒìŠ¤??
    print("\n[2] SemanticSearchEngineV2 ?ŒìŠ¤??)
    try:
        engine = SemanticSearchEngineV2(db_path)
        print(f"   ??SemanticSearchEngineV2 ì´ˆê¸°???±ê³µ")

        if not engine.embedder:
            print(f"   ? ï¸ ?„ë² ??ëª¨ë¸??ë¡œë“œ?˜ì? ?Šì•˜?µë‹ˆ??)
            return

        query = "ê³„ì•½ ?´ì?"
        print(f"\n   ê²€??ì¿¼ë¦¬: '{query}'")

        results = engine.search(query, k=5, similarity_threshold=0.2)
        print(f"   - ë²¡í„° ê²€??ê²°ê³¼: {len(results)}ê°?)

        if len(results) > 0:
            for i, r in enumerate(results[:3], 1):
                print(f"     [{i}] score={r.get('score', 0):.3f}, type={r.get('type')}, source={r.get('source', 'N/A')[:50]}")

    except Exception as e:
        print(f"   ??SemanticSearchEngineV2 ?ŒìŠ¤???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("?ŒìŠ¤???„ë£Œ")
    print("=" * 80)

if __name__ == "__main__":
    test_search_components()
