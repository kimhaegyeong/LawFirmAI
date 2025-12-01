"""
지식재산권법 데이터 적재 진행 상황 확인
"""
import sqlite3
from pathlib import Path

db_path = "data/lawfirm_v2.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 도메인 ID 확인
cursor.execute("SELECT id FROM domains WHERE name = ?", ("지식재산권법",))
domain_row = cursor.fetchone()

if not domain_row:
    print("❌ 지식재산권법 도메인이 아직 생성되지 않았습니다.")
    conn.close()
    exit(0)

domain_id = domain_row[0]
print(f"✅ 도메인 ID: {domain_id}\n")

# 통계 조회
cursor.execute("""
    SELECT COUNT(*) FROM cases WHERE domain_id = ?
""", (domain_id,))
case_count = cursor.fetchone()[0]

cursor.execute("""
    SELECT COUNT(*) FROM case_paragraphs cp
    JOIN cases c ON cp.case_id = c.id
    WHERE c.domain_id = ?
""", (domain_id,))
para_count = cursor.fetchone()[0]

cursor.execute("""
    SELECT COUNT(*) FROM text_chunks tc
    JOIN cases c ON tc.source_id = c.id
    WHERE tc.source_type = 'case_paragraph' AND c.domain_id = ?
""", (domain_id,))
chunk_count = cursor.fetchone()[0]

cursor.execute("""
    SELECT COUNT(*) FROM embeddings e
    JOIN text_chunks tc ON e.chunk_id = tc.id
    JOIN cases c ON tc.source_id = c.id
    WHERE tc.source_type = 'case_paragraph' AND c.domain_id = ?
""", (domain_id,))
embedding_count = cursor.fetchone()[0]

# 예상 파일 수 (판결문 폴더)
expected_files = 8004  # 데이터 형식 확인에서 확인한 수

print("=" * 60)
print("📊 지식재산권법 데이터 적재 진행 상황")
print("=" * 60)
print(f"📁 판례 수:        {case_count:,} / {expected_files:,} ({case_count/expected_files*100:.1f}%)")
print(f"📄 문단 수:        {para_count:,}")
print(f"🔤 청크 수:        {chunk_count:,}")
print(f"🔢 임베딩 수:      {embedding_count:,}")
print("=" * 60)

# 최근 적재된 데이터 샘플
cursor.execute("""
    SELECT doc_id, casenames, court, announce_date
    FROM cases
    WHERE domain_id = ?
    ORDER BY id DESC
    LIMIT 5
""", (domain_id,))

print("\n📋 최근 적재된 판례 (최대 5개):")
for row in cursor.fetchall():
    print(f"   - {row[0]}: {row[1]} ({row[2]}) - {row[3][:10] if row[3] else 'N/A'}")

# 법원별 통계
cursor.execute("""
    SELECT court, COUNT(*) as cnt
    FROM cases
    WHERE domain_id = ?
    GROUP BY court
    ORDER BY cnt DESC
    LIMIT 10
""", (domain_id,))

print("\n🏛️  법원별 통계 (상위 10개):")
for row in cursor.fetchall():
    print(f"   - {row[0] or 'NULL'}: {row[1]:,}건")

conn.close()




