import os
import sys
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


from config import Config
from sqlalchemy import create_engine


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "data",
    "ml"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "prediction_dataset.csv"
)


# ============================================================
# CONNECT TO MYSQL
# ============================================================

print("\n==========================================")
print("CREATING PREDICTION DATASET")
print("==========================================\n")


engine = create_engine(
    Config.SQLALCHEMY_DATABASE_URI
)


# ============================================================
# READ EVENTS
# ============================================================

query = """
SELECT
    user_id,
    event_type,
    device,
    location,
    traffic_source,
    timestamp
FROM events
ORDER BY user_id, timestamp
"""


df = pd.read_sql(
    query,
    engine
)


print(
    "Events loaded:",
    len(df)
)

print(
    "Users:",
    df["user_id"].nunique()
)


# ============================================================
# SORT EVENTS
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    ["user_id", "timestamp"]
)


# ============================================================
# CREATE PRE-OUTCOME SNAPSHOTS
# ============================================================
#
# IMPORTANT:
#
# We create a prediction snapshot BEFORE the user's
# final outcome is known.
#
# Example:
#
# User:
#
# visit
# signup
# add_to_cart
# purchase
#
# Snapshot at add_to_cart:
#
# device
# location
# traffic source
# number of visits
# number of signups
# number of carts
# current stage
#
# Target:
#
# Did this user eventually purchase?
#
# ============================================================


records = []


for user_id, group in df.groupby(
    "user_id"
):

    group = group.sort_values(
        "timestamp"
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # DID USER EVENTUALLY PURCHASE?
    # --------------------------------------------------------

    eventually_purchased = int(
        (group["event_type"] == "purchase").any()
    )


    # --------------------------------------------------------
    # CREATE SNAPSHOTS
    # --------------------------------------------------------

    for i, row in group.iterrows():

        event_type = row["event_type"]


        # We only make predictions at meaningful
        # journey stages.

        if event_type not in [
            "visit",
            "signup",
            "add_to_cart",
            "checkout"
        ]:
            continue


        history = group.iloc[
            :i + 1
        ]


        # ----------------------------------------------------
        # EVENT COUNTS AVAILABLE AT THIS MOMENT
        # ----------------------------------------------------

        visit_count = int(
            (
                history["event_type"]
                == "visit"
            ).sum()
        )


        signup_count = int(
            (
                history["event_type"]
                == "signup"
            ).sum()
        )


        cart_count = int(
            (
                history["event_type"]
                == "add_to_cart"
            ).sum()
        )


        checkout_count = int(
            (
                history["event_type"]
                == "checkout"
            ).sum()
        )


        # ----------------------------------------------------
        # CURRENT STAGE
        # ----------------------------------------------------

        if event_type == "checkout":

            current_stage = "checkout"

        elif event_type == "add_to_cart":

            current_stage = "add_to_cart"

        elif event_type == "signup":

            current_stage = "signup"

        else:

            current_stage = "visit"


        # ----------------------------------------------------
        # TIME SINCE FIRST EVENT
        # ----------------------------------------------------

        first_timestamp = group[
            "timestamp"
        ].iloc[0]

        current_timestamp = row[
            "timestamp"
        ]

        session_duration_seconds = (
            current_timestamp
            - first_timestamp
        ).total_seconds()


        # ----------------------------------------------------
        # CREATE RECORD
        # ----------------------------------------------------

        records.append({

            "user_id": user_id,

            "device": row["device"],

            "location": row["location"],

            "traffic_source": row[
                "traffic_source"
            ],

            "visit_count": visit_count,

            "signup_count": signup_count,

            "cart_count": cart_count,

            "checkout_count": checkout_count,

            "current_stage": current_stage,

            "session_duration_seconds":
                session_duration_seconds,

            "event_number":
                i + 1,

            # TARGET
            #
            # 1 = eventually purchased
            # 0 = eventually dropped off

            "eventually_purchased":
                eventually_purchased

        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

prediction_df = pd.DataFrame(
    records
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

prediction_df = prediction_df.drop_duplicates()


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


# ============================================================
# SAVE DATASET
# ============================================================

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("PREDICTION DATASET CREATED")
print("==========================================")


print(
    "\nDataset shape:"
)

print(
    prediction_df.shape
)


print(
    "\nColumns:"
)

print(
    prediction_df.columns.tolist()
)


print(
    "\nTarget distribution:"
)

print(
    prediction_df[
        "eventually_purchased"
    ].value_counts()
)


print(
    "\nTarget percentage:"
)

print(
    prediction_df[
        "eventually_purchased"
    ]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)


print(
    "\nSample:"
)

print(
    prediction_df.head(10)
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)


print("\n==========================================")
print("PREDICTION DATASET COMPLETED")
print("==========================================\n")