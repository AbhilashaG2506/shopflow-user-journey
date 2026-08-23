import os
import sys
import pandas as pd
from sqlalchemy import text


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


from app import app
from database import db


# ---------------------------------------------------------
# LOAD DATA FROM MYSQL
# ---------------------------------------------------------

with app.app_context():

    query = text("""
        SELECT
            user_id,
            timestamp,
            event_type,
            device,
            location,
            traffic_source
        FROM events
    """)

    df = pd.read_sql(
        query,
        db.engine
    )


print("\n==========================================")
print("USER JOURNEY FUNNEL ANALYSIS")
print("==========================================")

print(f"\nTotal events: {len(df):,}")
print(f"Unique users: {df['user_id'].nunique():,}")


# ---------------------------------------------------------
# FUNNEL
# ---------------------------------------------------------

funnel_events = [
    "visit",
    "signup",
    "add_to_cart",
    "purchase"
]


funnel_counts = {}

for event in funnel_events:

    users = df.loc[
        df["event_type"] == event,
        "user_id"
    ].nunique()

    funnel_counts[event] = users


print("\n========== FUNNEL ==========")

for event, count in funnel_counts.items():

    print(
        f"{event.upper():15} {count:,}"
    )


# ---------------------------------------------------------
# DROP-OFF CALCULATION
# ---------------------------------------------------------

print("\n========== DROP-OFF ==========")

dropoffs = {}

for i in range(len(funnel_events) - 1):

    current_stage = funnel_events[i]
    next_stage = funnel_events[i + 1]

    current_users = funnel_counts[current_stage]
    next_users = funnel_counts[next_stage]

    dropoff = (
        (current_users - next_users)
        / current_users
    ) * 100

    conversion = (
        next_users / current_users
    ) * 100

    dropoffs[
        f"{current_stage} -> {next_stage}"
    ] = dropoff

    print(
        f"{current_stage.upper()} -> "
        f"{next_stage.upper()}"
    )

    print(
        f"Drop-off: {dropoff:.2f}%"
    )

    print(
        f"Conversion: {conversion:.2f}%\n"
    )


# ---------------------------------------------------------
# OVERALL CONVERSION
# ---------------------------------------------------------

overall_conversion = (
    funnel_counts["purchase"]
    / funnel_counts["visit"]
) * 100


print("========== OVERALL ==========")

print(
    f"Overall Conversion Rate: "
    f"{overall_conversion:.2f}%"
)


# ---------------------------------------------------------
# BIGGEST LEAKAGE
# ---------------------------------------------------------

biggest_leakage = max(
    dropoffs,
    key=dropoffs.get
)

biggest_leakage_value = dropoffs[
    biggest_leakage
]


print("\n========== BIGGEST LEAKAGE ==========")

print(
    f"Leakage Point: {biggest_leakage}"
)

print(
    f"Drop-off: {biggest_leakage_value:.2f}%"
)


# ---------------------------------------------------------
# SEGMENT ANALYSIS FUNCTION
# ---------------------------------------------------------

def segment_funnel(data, column):

    print("\n")
    print("=" * 60)
    print(f"SEGMENTATION BY {column.upper()}")
    print("=" * 60)

    segments = data[column].unique()

    results = []

    for segment in segments:

        segment_data = data[
            data[column] == segment
        ]

        counts = {}

        for event in funnel_events:

            counts[event] = segment_data.loc[
                segment_data["event_type"] == event,
                "user_id"
            ].nunique()

        visit = counts["visit"]

        purchase = counts["purchase"]

        if visit > 0:

            conversion_rate = (
                purchase / visit
            ) * 100

        else:

            conversion_rate = 0


        results.append({

            column: segment,

            "visits": counts["visit"],

            "signups": counts["signup"],

            "add_to_cart": counts["add_to_cart"],

            "purchases": counts["purchase"],

            "conversion_rate": round(
                conversion_rate,
                2
            )

        })


    result_df = pd.DataFrame(results)

    print(
        result_df.to_string(index=False)
    )

    return result_df


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

device_analysis = segment_funnel(
    df,
    "device"
)


# ---------------------------------------------------------
# LOCATION
# ---------------------------------------------------------

location_analysis = segment_funnel(
    df,
    "location"
)


# ---------------------------------------------------------
# TRAFFIC SOURCE
# ---------------------------------------------------------

traffic_analysis = segment_funnel(
    df,
    "traffic_source"
)


# ---------------------------------------------------------
# SAVE ANALYSIS RESULTS
# ---------------------------------------------------------

os.makedirs(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "analysis"
    ),
    exist_ok=True
)


device_analysis.to_csv(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "analysis",
        "device_funnel.csv"
    ),
    index=False
)


location_analysis.to_csv(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "analysis",
        "location_funnel.csv"
    ),
    index=False
)


traffic_analysis.to_csv(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "analysis",
        "traffic_funnel.csv"
    ),
    index=False
)


print("\n==========================================")
print("ANALYSIS COMPLETED")
print("==========================================")