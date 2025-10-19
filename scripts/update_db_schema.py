import sqlite3

def update_database_schema():
    conn = sqlite3.connect('data/lawfirm.db')
    cursor = conn.cursor()
    
    try:
        # law_metadata에 updated_at 컬럼 추가
        cursor.execute('ALTER TABLE law_metadata ADD COLUMN updated_at TIMESTAMP')
        print("Added updated_at to law_metadata")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("updated_at already exists in law_metadata")
        else:
            print(f"Error adding updated_at to law_metadata: {e}")
    
    try:
        # precedent_metadata에 필요한 컬럼들 추가
        cursor.execute('ALTER TABLE precedent_metadata ADD COLUMN is_active BOOLEAN DEFAULT 1')
        print("Added is_active to precedent_metadata")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("is_active already exists in precedent_metadata")
        else:
            print(f"Error adding is_active to precedent_metadata: {e}")
    
    try:
        cursor.execute('ALTER TABLE precedent_metadata ADD COLUMN updated_at TIMESTAMP')
        print("Added updated_at to precedent_metadata")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("updated_at already exists in precedent_metadata")
        else:
            print(f"Error adding updated_at to precedent_metadata: {e}")
    
    try:
        cursor.execute('ALTER TABLE precedent_metadata ADD COLUMN precedent_number TEXT')
        print("Added precedent_number to precedent_metadata")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("precedent_number already exists in precedent_metadata")
        else:
            print(f"Error adding precedent_number to precedent_metadata: {e}")
    
    conn.commit()
    conn.close()
    print("Database schema update completed")

if __name__ == "__main__":
    update_database_schema()
