import sqlite3
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

db_path = os.path.join(os.path.dirname(__file__), '../../data/lawfirm_v2.db')

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 형법, 상법, 민사소송법, 근로기준법 관련 법령명 찾기
search_terms = ['형법', '상법', '민사소송법', '근로기준법']

for term in search_terms:
    print(f"\n=== '{term}' 관련 법령명 ===")
    cursor.execute("""
        SELECT DISTINCT name, abbrv 
        FROM statutes 
        WHERE name LIKE ? OR abbrv LIKE ?
        LIMIT 5
    """, (f'%{term}%', f'%{term}%'))
    
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print(f"  이름: {row['name']}, 약어: {row['abbrv']}")
    else:
        print(f"  찾을 수 없음")

conn.close()

