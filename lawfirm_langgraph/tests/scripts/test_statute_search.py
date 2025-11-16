# -*- coding: utf-8 -*-
"""법령 검색 문제 확인 테스트"""

import sys
import os
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

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from lawfirm_langgraph.core.utils.config import Config
import logging
import sqlite3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 설정
os.environ['USE_EXTERNAL_VECTOR_STORE'] = 'true'
os.environ['EXTERNAL_VECTOR_STORE_BASE_PATH'] = str(project_root / "data" / "vector_store" / "v2.0.0-dynamic-dynamic-ivfpq")

logger.info("=" * 60)
logger.info("법령 검색 문제 확인 테스트")
logger.info("=" * 60)

# SemanticSearchEngineV2 초기화
config = Config()
search_engine = SemanticSearchEngineV2(
    db_path=config.database_path,
    use_external_index=config.use_external_vector_store,
    external_index_path=config.external_vector_store_base_path
)

# 검색 쿼리
query = "임대차 보증금 반환"
logger.info(f"\n검색 쿼리: {query}")

# 1. 법령만 검색
logger.info("\n1️⃣  법령만 검색 (source_types=['statute_article'])")
statute_results = search_engine.search(
    query, 
    k=10, 
    similarity_threshold=0.15,
    source_types=["statute_article"]
)

logger.info(f"법령 검색 결과: {len(statute_results)}개")

if statute_results:
    for i, result in enumerate(statute_results[:5], 1):
        logger.info(f"\n결과 {i}:")
        logger.info(f"  score: {result.get('score', 0):.4f}")
        logger.info(f"  text 길이: {len(result.get('text', ''))}자")
        logger.info(f"  statute_name: {result.get('statute_name') or result.get('metadata', {}).get('statute_name')}")
        logger.info(f"  article_no: {result.get('article_no') or result.get('metadata', {}).get('article_no')}")
        logger.info(f"  text 미리보기: {result.get('text', '')[:100]}...")
else:
    logger.warning("⚠️  법령 검색 결과가 없습니다")

# 2. 모든 타입 검색 (필터링 전)
logger.info("\n2️⃣  모든 타입 검색 (필터링 전)")
all_results = search_engine.search(
    query, 
    k=20, 
    similarity_threshold=0.15
)

logger.info(f"전체 검색 결과: {len(all_results)}개")

# 소스 타입별 분류
type_counts = {}
for result in all_results:
    source_type = result.get('type') or result.get('metadata', {}).get('source_type')
    type_counts[source_type] = type_counts.get(source_type, 0) + 1

logger.info(f"소스 타입별 분포: {type_counts}")

# 3. 법령 텍스트 길이 확인
logger.info("\n3️⃣  법령 텍스트 길이 확인")
conn = sqlite3.connect(config.database_path)
conn.row_factory = sqlite3.Row

cursor = conn.execute("""
    SELECT tc.id, tc.text, tc.source_id, sa.article_no, s.name as statute_name
    FROM text_chunks tc
    JOIN statute_articles sa ON tc.source_id = sa.id
    JOIN statutes s ON sa.statute_id = s.id
    WHERE tc.source_type = 'statute_article'
    LIMIT 100
""")

text_lengths = []
for row in cursor.fetchall():
    text_length = len(row['text']) if row['text'] else 0
    text_lengths.append(text_length)
    if text_length < 100:
        logger.info(f"  짧은 법령: chunk_id={row['id']}, length={text_length}, "
                   f"statute={row['statute_name']}, article={row['article_no']}")

if text_lengths:
    logger.info(f"\n법령 텍스트 길이 통계:")
    logger.info(f"  평균: {sum(text_lengths) / len(text_lengths):.1f}자")
    logger.info(f"  최소: {min(text_lengths)}자")
    logger.info(f"  최대: {max(text_lengths)}자")
    logger.info(f"  100자 미만: {sum(1 for l in text_lengths if l < 100)}개 ({sum(1 for l in text_lengths if l < 100)/len(text_lengths)*100:.1f}%)")
    logger.info(f"  50자 미만: {sum(1 for l in text_lengths if l < 50)}개 ({sum(1 for l in text_lengths if l < 50)/len(text_lengths)*100:.1f}%)")

conn.close()

# 4. 판례에서 인용된 법령 확인
logger.info("\n4️⃣  판례에서 인용된 법령 확인")
conn = sqlite3.connect(config.database_path)
conn.row_factory = sqlite3.Row

# 판례 텍스트에서 법령 인용 패턴 찾기
cursor = conn.execute("""
    SELECT DISTINCT c.id, c.doc_id, c.casenames, tc.text
    FROM cases c
    JOIN text_chunks tc ON tc.source_type = 'case_paragraph' AND tc.source_id = c.id
    WHERE tc.text LIKE '%제%조%' OR tc.text LIKE '%법%조%' OR tc.text LIKE '%법률%'
    LIMIT 10
""")

logger.info(f"법령 인용이 있는 판례: {len(cursor.fetchall())}개")

# 판례 텍스트 샘플 확인
cursor = conn.execute("""
    SELECT tc.text
    FROM text_chunks tc
    JOIN cases c ON tc.source_type = 'case_paragraph' AND tc.source_id = c.id
    WHERE tc.text LIKE '%임대차%' AND (tc.text LIKE '%제%조%' OR tc.text LIKE '%법%조%')
    LIMIT 5
""")

logger.info("\n판례에서 법령 인용 예시:")
for i, row in enumerate(cursor.fetchall(), 1):
    text = row['text']
    # 법령 인용 패턴 찾기
    import re
    patterns = re.findall(r'[가-힣]+법\s*제\d+조|제\d+조', text)
    if patterns:
        logger.info(f"  예시 {i}: {patterns[:3]}")

conn.close()

