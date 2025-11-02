import os
import sqlite3


def run_sql_script(db_path: str, sql_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.enable_load_extension(True)
        with open(sql_path, "r", encoding="utf-8") as f:
            sql = f.read()
        conn.executescript(sql)
        conn.commit()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    db_path = os.path.join(base_dir, "data", "lawfirm_v2.db")
    sql_path = os.path.join(base_dir, "scripts", "migrations", "001_create_lawfirm_v2.sql")
    run_sql_script(db_path, sql_path)
    print(f"Initialized {db_path} using {sql_path}")
