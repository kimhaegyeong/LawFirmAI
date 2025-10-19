import sqlite3

def check_table_structure():
    conn = sqlite3.connect('data/lawfirm.db')
    cursor = conn.cursor()
    
    # law_metadata 테이블 구조 확인
    cursor.execute('PRAGMA table_info(law_metadata)')
    print('law_metadata columns:')
    for row in cursor.fetchall():
        print(row)
    
    print('\nprecedent_metadata columns:')
    cursor.execute('PRAGMA table_info(precedent_metadata)')
    for row in cursor.fetchall():
        print(row)
    
    conn.close()

if __name__ == "__main__":
    check_table_structure()
