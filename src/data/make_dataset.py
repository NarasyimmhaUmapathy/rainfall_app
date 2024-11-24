
import sqlite3 

import pandas as pd

with sqlite3.connect("rainfall.db") as conn:
    # interact with database
    print(f"Opened SQLite database with version {sqlite3.sqlite_version} successfully.")

  
# Load CSV data into Pandas DataFrame 
stud_data = pd.read_csv('weatherAUS.csv',
                        skiprows=7,delimiter=",",
                        encoding = 'ISO-8859-1') 
# Write the data to a sqlite table 
stud_data.to_sql('rainfall', conn, if_exists='replace', index=False) 
  
# Create a cursor object 
cur = conn.cursor() 
# Fetch and display result 
for row in cur.execute('SELECT * FROM rainfall LIMIT 20'): 
    print(row) 
# Close connection to SQLite database 
conn.close() 
