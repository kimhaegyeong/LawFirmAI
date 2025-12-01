import sqlite3
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

db_path = os.path.join(os.path.dirname(__file__), '../../data/lawfirm_v2.db')

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 전체 법령명 샘플 확인
print("=== 전체 법령명 샘플 (상위 30개) ===")
cursor.execute("SELECT DISTINCT name, abbrv FROM statutes LIMIT 30")
rows = cursor.fetchall()
for row in rows:
    print(f"  이름: {row['name']}, 약어: {row['abbrv']}")

# "법"으로 끝나는 법령명 확인
print("\n=== '법'으로 끝나는 법령명 샘플 (상위 20개) ===")
cursor.execute("SELECT DISTINCT name, abbrv FROM statutes WHERE name LIKE '%법' LIMIT 20")
rows = cursor.fetchall()
for row in rows:
    print(f"  이름: {row['name']}, 약어: {row['abbrv']}")

# statute_articles에서 실제 사용되는 법령명 확인
print("\n=== statute_articles에 실제 존재하는 법령명 샘플 (상위 20개) ===")
cursor.execute("""
    SELECT DISTINCT s.name, s.abbrv 
    FROM statute_articles sa
    JOIN statutes s ON sa.statute_id = s.id
    LIMIT 20
""")
rows = cursor.fetchall()
for row in rows:
    print(f"  이름: {row['name']}, 약어: {row['abbrv']}")

conn.close()

