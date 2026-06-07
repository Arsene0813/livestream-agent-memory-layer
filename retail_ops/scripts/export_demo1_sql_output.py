from pathlib import Path
import os
import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)

SQL_PATH = REPO_ROOT / "retail_ops" / "sql" / "01_store_a_month_over_month_diagnostic.sql"
OUTPUT_PATH = REPO_ROOT / "retail_ops" / "outputs" / "store_a_demo1_sql_output.csv"

sql = SQL_PATH.read_text(encoding="utf-8").strip()

if sql.endswith(";"):
    sql = sql[:-1]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(database=":memory:")

copy_sql = f"""
COPY (
{sql}
) TO '{OUTPUT_PATH.as_posix()}'
WITH (HEADER, DELIMITER ',');
"""

con.execute(copy_sql)

print(f"Generated: {OUTPUT_PATH}")
