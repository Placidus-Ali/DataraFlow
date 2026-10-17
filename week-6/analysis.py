import sys
import pathlib

# 1. Pathfinder to find the root db module from subfolders (like week-6)
_p = pathlib.Path.cwd()
while _p != _p.parent and not (_p / "db" / "dataraflow_db.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))

from db.dataraflow_db import run_query
import pandas as pd

print("--- 1. Testing Single Dataset (London) ---")
london = run_query("SELECT TOP 3 * FROM week06.london_2014")
print(london.head(3))
print("London columns:", london.columns.tolist())

print("\n--- 2. Inspecting All City Table Columns ---")
cities_list = [
    "week06.london_2014", 
    "week06.beijing_pek_2014", 
    "week06.brasilia_bsb_2014", 
    "week06.capetown_cpt_2014", 
    "week06.delhi_del_2014", 
    "week06.moscow_svo_2014"
]

for table in cities_list:
    df_sample = run_query(f"SELECT TOP 1 * FROM {table}")
    print(f"Table {table} columns -> Date col found: {[c for c in df_sample.columns if 'date' in c.lower() or 'gmt' in c.lower()]}")

print("\n--- 3. Combining All Six Cities via UNION ALL ---")
cities = run_query("""
    SELECT 'London'    AS city, GMT AS [Date], [Mean TemperatureC] FROM week06.london_2014
    UNION ALL SELECT 'Beijing',  Date, [Mean TemperatureC] FROM week06.beijing_pek_2014
    UNION ALL SELECT 'Brasilia', Date, [Mean TemperatureC] FROM week06.brasilia_bsb_2014
    UNION ALL SELECT 'CapeTown', Date, [Mean TemperatureC] FROM week06.capetown_cpt_2014
    UNION ALL SELECT 'Delhi',    Date, [Mean TemperatureC] FROM week06.delhi_del_2014
    UNION ALL SELECT 'Moscow',   Date, [Mean TemperatureC] FROM week06.moscow_svo_2014
""")
print("Combined shape:", cities.shape)

print("\n--- 4. Saving to CSV ---")
cities.to_csv("cities_mean_temp.csv", index=False)
print("Saved cities_mean_temp.csv successfully!")

print("\n--- 5. Pulling WHO Data ---")
who_all = run_query("SELECT * FROM week06.who_pop_tb_all")
print("WHO All Countries Shape:", who_all.shape)
print(who_all.head(3))