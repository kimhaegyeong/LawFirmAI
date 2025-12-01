# -*- coding: utf-8 -*-
"""
lawfirm_v2.db ?µí•© ?ŒìŠ¤???¤í¬ë¦½íŠ¸
?¤ì œ ?°ì´?°ë² ?´ìŠ¤?€ ?°ë™?˜ì—¬ ê²€??ê¸°ëŠ¥ ê²€ì¦?
"""

import sys
import os
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from source.utils.config import Config
from source.agents.legal_data_connector_v2 import LegalDataConnectorV2, route_query
from source.services.semantic_search_engine_v2 import SemanticSearchEngineV2
from source.services.hybrid_search_engine_v2 import HybridSearchEngineV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_query_routing():
    """Query Router ?ŒìŠ¤??""
    print("\n=== Query Router ?ŒìŠ¤??===")

    test_queries = [
        ("??ì¡??Œë ¤ì¤?, "text2sql"),
        ("2023-10-19 ?œí–‰", "text2sql"),
        ("?€ë²•ì› ?ë?", "text2sql"),
        ("ë¶€?¹ì´??ë°˜í™˜ ?”ê±´", "vector"),
        ("ê³„ì•½ ?´ì? ?ˆì°¨", "vector"),
    ]

    for query, expected in test_queries:
        result = route_query(query)
        status = "?? if result == expected else "??
        print(f"{status} '{query}' ??{result} (expected: {expected})")


def test_fts_search(config):
    """FTS5 ê²€???ŒìŠ¤??""
    print("\n=== FTS5 ê²€???ŒìŠ¤??===")

    db_path = config.database_path
    if not os.path.exists(db_path):
        print(f"? ï¸ ?°ì´?°ë² ?´ìŠ¤ê°€ ?†ìŠµ?ˆë‹¤: {db_path}")
        print("ë¨¼ì? ?°ì´?°ë? ?ì¬?˜ì„¸??")
        print("  python scripts/ingest/ingest_statutes.py --json <path> --domain ë¯¼ì‚¬ë²?)
        return

    connector = LegalDataConnectorV2(db_path)

    # ë²•ë ¹ ê²€??
    print("\n1. ë²•ë ¹ FTS ê²€??")
    results = connector.search_statutes_fts("?¸ì?", limit=5)
    print(f"   ê²°ê³¼: {len(results)}ê°?)
    for i, r in enumerate(results[:3], 1):
        print(f"   {i}. [{r['relevance_score']:.3f}] {r['source']} - {r['content'][:50]}...")

    # ?ë? ê²€??
    print("\n2. ?ë? FTS ê²€??")
    results = connector.search_cases_fts("?í•´ë°°ìƒ", limit=5)
    print(f"   ê²°ê³¼: {len(results)}ê°?)
    for i, r in enumerate(results[:3], 1):
        print(f"   {i}. [{r['relevance_score']:.3f}] {r['source']} - {r['content'][:50]}...")

    # ?µí•© ê²€??(?¼ìš°??
    print("\n3. ?µí•© ê²€??(?ë™ ?¼ìš°??:")
    results = connector.search_documents("??ì¡?, limit=5)
    print(f"   Text2SQL ?¼ìš°??ê²°ê³¼: {len(results)}ê°?)

    results = connector.search_documents("ë¶€?¹ì´??, limit=5)
    print(f"   Vector ?¼ìš°??ê²°ê³¼: {len(results)}ê°?(SemanticSearchEngineV2ë¡??„ì„)")


def test_semantic_search(config):
    """ë²¡í„° ê²€???ŒìŠ¤??""
    print("\n=== ë²¡í„° ?˜ë? ê²€???ŒìŠ¤??===")

    db_path = config.database_path
    if not os.path.exists(db_path):
        print(f"? ï¸ ?°ì´?°ë² ?´ìŠ¤ê°€ ?†ìŠµ?ˆë‹¤: {db_path}")
        return

    try:
        engine = SemanticSearchEngineV2(db_path)

        if not engine.embedder:
            print("? ï¸ ?„ë² ??ëª¨ë¸??ë¡œë“œ?????†ìŠµ?ˆë‹¤. ëª¨ë¸ ?¤ìš´ë¡œë“œê°€ ?„ìš”?????ˆìŠµ?ˆë‹¤.")
            return

        print("\në²¡í„° ê²€???¤í–‰ ì¤?..")
        results = engine.search(
            query="ë¶€?¹ì´??ë°˜í™˜ ì²?µ¬ ?”ê±´",
            k=5,
            similarity_threshold=0.3
        )

        print(f"ê²°ê³¼: {len(results)}ê°?)
        for i, r in enumerate(results[:3], 1):
            print(f"   {i}. [{r['score']:.3f}] {r['source']}")
            print(f"      {r['text'][:80]}...")

    except Exception as e:
        print(f"??ë²¡í„° ê²€???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()


def test_hybrid_search(config):
    """?˜ì´ë¸Œë¦¬??ê²€???ŒìŠ¤??""
    print("\n=== ?˜ì´ë¸Œë¦¬??ê²€???ŒìŠ¤??===")

    db_path = config.database_path
    if not os.path.exists(db_path):
        print(f"? ï¸ ?°ì´?°ë² ?´ìŠ¤ê°€ ?†ìŠµ?ˆë‹¤: {db_path}")
        return

    try:
        engine = HybridSearchEngineV2(db_path)

        print("\n?˜ì´ë¸Œë¦¬??ê²€???¤í–‰ ì¤?..")
        result = engine.search(
            query="?¸ì?",
            search_types=["law"],
            max_results=10,
            include_exact=True,
            include_semantic=True
        )

        print(f"ì´?ê²°ê³¼: {result['total']}ê°?)
        print(f"FTS5 ê²°ê³¼: {result['exact_count']}ê°?)
        print(f"ë²¡í„° ê²°ê³¼: {result['semantic_count']}ê°?)

        for i, r in enumerate(result['results'][:3], 1):
            print(f"\n   {i}. [{r.get('search_type', 'unknown')}] {r.get('relevance_score', 0):.3f}")
            text = r.get('text', r.get('content', ''))[:80]
            print(f"      {text}...")

    except Exception as e:
        print(f"???˜ì´ë¸Œë¦¬??ê²€???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()


def check_database_status(config):
    """?°ì´?°ë² ?´ìŠ¤ ?íƒœ ?•ì¸"""
    print("\n=== ?°ì´?°ë² ?´ìŠ¤ ?íƒœ ?•ì¸ ===")

    db_path = config.database_path

    if not os.path.exists(db_path):
        print(f"???°ì´?°ë² ?´ìŠ¤ ?Œì¼???†ìŠµ?ˆë‹¤: {db_path}")
        print("\nì´ˆê¸°??ë°©ë²•:")
        print("  python scripts/init_lawfirm_v2_db.py")
        return False

    print(f"???°ì´?°ë² ?´ìŠ¤ ì¡´ì¬: {db_path}")

    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # ?Œì´ë¸??•ì¸
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\n?Œì´ë¸??? {len(tables)}")

        # ì£¼ìš” ?Œì´ë¸??°ì´???•ì¸
        checks = [
            ("domains", "SELECT COUNT(*) FROM domains"),
            ("statutes", "SELECT COUNT(*) FROM statutes"),
            ("statute_articles", "SELECT COUNT(*) FROM statute_articles"),
            ("cases", "SELECT COUNT(*) FROM cases"),
            ("case_paragraphs", "SELECT COUNT(*) FROM case_paragraphs"),
            ("text_chunks", "SELECT COUNT(*) FROM text_chunks"),
            ("embeddings", "SELECT COUNT(*) FROM embeddings"),
        ]

        print("\n?°ì´???µê³„:")
        for name, query in checks:
            try:
                cursor.execute(query)
                count = cursor.fetchone()[0]
                print(f"  {name}: {count}ê°?)
            except:
                print(f"  {name}: ?Œì´ë¸??†ìŒ")

        # FTS5 ?Œì´ë¸??•ì¸
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'")
        fts_tables = [row[0] for row in cursor.fetchall()]
        print(f"\nFTS5 ?Œì´ë¸? {len(fts_tables)}ê°?)
        for table in fts_tables:
            print(f"  - {table}")

        conn.close()
        return True

    except Exception as e:
        print(f"???°ì´?°ë² ?´ìŠ¤ ?‘ê·¼ ?¤íŒ¨: {e}")
        return False


def main():
    """ë©”ì¸ ?ŒìŠ¤???¤í–‰"""
    print("=" * 60)
    print("lawfirm_v2.db ?µí•© ?ŒìŠ¤??)
    print("=" * 60)

    # V2 DB ê²½ë¡œ ëª…ì‹œ???¤ì •
    v2_db_path = "./data/lawfirm_v2.db"
    print(f"\n?ŒìŠ¤???€???°ì´?°ë² ?´ìŠ¤: {v2_db_path}")

    # Config ê°ì²´ ?ì„± (?´ë??ìœ¼ë¡??¬ìš©)
    config = Config()
    # ?ŒìŠ¤?¸ì—?œëŠ” v2 ê²½ë¡œë¥?ì§ì ‘ ?¬ìš©
    config.database_path = v2_db_path

    # ?°ì´?°ë² ?´ìŠ¤ ?íƒœ ?•ì¸
    if not check_database_status(config):
        print("\n? ï¸ ?°ì´?°ë² ?´ìŠ¤ê°€ ì¤€ë¹„ë˜ì§€ ?Šì•˜?µë‹ˆ?? ?ŒìŠ¤?¸ë? ê±´ë„ˆ?ë‹ˆ??")
        return

    # ?ŒìŠ¤???¤í–‰
    test_query_routing()
    test_fts_search(config)
    test_semantic_search(config)
    test_hybrid_search(config)

    print("\n" + "=" * 60)
    print("?ŒìŠ¤???„ë£Œ")
    print("=" * 60)


if __name__ == "__main__":
    main()
