import os
import sys

import pandas as pd


# Add the main project folder to Python's path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


from app import app
from database import db


CSV_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "ecommerce_events.csv"
)


print("Loading dataset...")

df = pd.read_csv(CSV_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"Rows found: {len(df):,}")


with app.app_context():

    print("Connecting to MySQL...")

    df.to_sql(
        "events",
        con=db.engine,
        if_exists="append",
        index=False,
        chunksize=1000
    )


print()
print("===================================")
print("DATA SUCCESSFULLY LOADED!")
print("===================================")
print(f"Inserted events: {len(df):,}")
print("Database table: events")
print("===================================")