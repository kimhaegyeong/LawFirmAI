import sqlite3

def add_case_name_column():
    """precedent_metadata 테이블에 case_name 컬럼 추가"""
    conn = sqlite3.connect('data/lawfirm.db')
    cursor = conn.cursor()
    
    try:
        # case_name 컬럼 추가
        cursor.execute('ALTER TABLE precedent_metadata ADD COLUMN case_name TEXT')
        print("Added case_name to precedent_metadata")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("case_name already exists in precedent_metadata")
        else:
            print(f"Error adding case_name to precedent_metadata: {e}")
    
    conn.commit()
    conn.close()
    print("Database schema update completed")

if __name__ == "__main__":
    add_case_name_column()
