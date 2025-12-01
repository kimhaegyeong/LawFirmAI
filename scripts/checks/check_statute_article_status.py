#!/usr/bin/env python3
"""statute_article 처리 상태 확인"""
import sqlite3

conn = sqlite3.connect('data/lawfirm_v2.db')
conn.row_factory = sqlite3.Row

# statute_article 처리 상태
cursor = conn.execute("""
    SELECT COUNT(DISTINCT source_id) as count
    FROM text_chunks
    WHERE source_type = 'statute_article'
    AND embedding_version_id = 5
    AND chunking_strategy = 'dynamic'
""")
processed = cursor.fetchone()['count']

# 전체 statute_article 수
cursor = conn.execute("SELECT COUNT(*) FROM statute_articles")
total = cursor.fetchone()[0]

# 생성된 청크 수
cursor = conn.execute("""
    SELECT COUNT(*) as count
    FROM text_chunks
    WHERE source_type = 'statute_article'
    AND embedding_version_id = 5
    AND chunking_strategy = 'dynamic'
""")
chunks = cursor.fetchone()['count']

print("=" * 60)
print("statute_article 처리 상태")
print("=" * 60)
print(f"전체 statute_article: {total:,}개")
print(f"처리된 문서: {processed:,}개 ({processed/total*100:.1f}%)")
print(f"생성된 청크: {chunks:,}개")
print(f"문서당 평균 청크: {chunks/processed if processed > 0 else 0:.1f}개")
print(f"남은 문서: {total - processed:,}개")
print("=" * 60)

conn.close()

