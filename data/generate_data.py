import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

TOTAL_EVENTS = 50000

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------
# BASIC VALUES
# ---------------------------------------------------------

devices = [
    "mobile",
    "desktop",
    "tablet"
]

locations = [
    "US",
    "UK",
    "Germany",
    "India"
]

traffic_sources = [
    "organic_search",
    "paid_ads",
    "social_media"
]

products = [
    ("P001", "Wireless Headphones", "Electronics", 79.99),
    ("P002", "Smart Watch", "Electronics", 129.99),
    ("P003", "Running Shoes", "Fashion", 89.99),
    ("P004", "Backpack", "Fashion", 49.99),
    ("P005", "Bluetooth Speaker", "Electronics", 59.99),
    ("P006", "Coffee Maker", "Home", 99.99),
    ("P007", "Desk Lamp", "Home", 39.99),
    ("P008", "Skin Care Set", "Beauty", 69.99)
]


# ---------------------------------------------------------
# TIMESTAMP GENERATOR
# ---------------------------------------------------------

def random_timestamp():

    now = datetime.now()

    random_seconds = random.randint(
        0,
        30 * 24 * 60 * 60
    )

    return now - timedelta(seconds=random_seconds)


# ---------------------------------------------------------
# EVENT GENERATOR
# ---------------------------------------------------------

events = []

user_counter = 1

while len(events) < TOTAL_EVENTS:

    user_id = f"U{user_counter:05d}"

    device = random.choices(
        devices,
        weights=[0.60, 0.30, 0.10]
    )[0]

    location = random.choices(
        locations,
        weights=[0.35, 0.20, 0.15, 0.30]
    )[0]

    traffic_source = random.choices(
        traffic_sources,
        weights=[0.45, 0.30, 0.25]
    )[0]

    product = random.choice(products)

    product_id = product[0]

    timestamp = random_timestamp()


    # -----------------------------------------------------
    # VISIT
    # -----------------------------------------------------

    events.append({
        "user_id": user_id,
        "timestamp": timestamp,
        "event_type": "visit",
        "device": device,
        "location": location,
        "traffic_source": traffic_source,
        "product_id": product_id
    })

    if len(events) >= TOTAL_EVENTS:
        break


    # -----------------------------------------------------
    # SIGNUP PROBABILITY
    # -----------------------------------------------------

    signup_probability = 0.55


    # Social traffic drops quickly
    if traffic_source == "social_media":
        signup_probability *= 0.45


    # Mobile slightly lower signup
    if device == "mobile":
        signup_probability *= 0.85


    if random.random() < signup_probability:

        timestamp += timedelta(
            seconds=random.randint(20, 300)
        )

        events.append({
            "user_id": user_id,
            "timestamp": timestamp,
            "event_type": "signup",
            "device": device,
            "location": location,
            "traffic_source": traffic_source,
            "product_id": product_id
        })


        if len(events) >= TOTAL_EVENTS:
            break


        # -------------------------------------------------
        # ADD TO CART
        # -------------------------------------------------

        cart_probability = 0.55


        if traffic_source == "social_media":
            cart_probability *= 0.50


        if device == "mobile":
            cart_probability *= 0.85


        if random.random() < cart_probability:

            timestamp += timedelta(
                seconds=random.randint(30, 600)
            )

            events.append({
                "user_id": user_id,
                "timestamp": timestamp,
                "event_type": "add_to_cart",
                "device": device,
                "location": location,
                "traffic_source": traffic_source,
                "product_id": product_id
            })


            if len(events) >= TOTAL_EVENTS:
                break


            # ---------------------------------------------
            # PURCHASE
            # ---------------------------------------------

            purchase_probability = 0.45


            # MASSIVE MOBILE CHECKOUT LEAKAGE
            if device == "mobile":
                purchase_probability *= 0.25


            if device == "desktop":
                purchase_probability *= 1.25


            if traffic_source == "social_media":
                purchase_probability *= 0.60


            if traffic_source == "organic_search":
                purchase_probability *= 1.15


            if location == "India":
                purchase_probability *= 0.85


            if location == "US":
                purchase_probability *= 1.10


            purchase_probability = min(
                purchase_probability,
                0.90
            )


            if random.random() < purchase_probability:

                timestamp += timedelta(
                    seconds=random.randint(60, 900)
                )

                events.append({
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "event_type": "purchase",
                    "device": device,
                    "location": location,
                    "traffic_source": traffic_source,
                    "product_id": product_id
                })


    user_counter += 1


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------

df = pd.DataFrame(events)

df = df.head(TOTAL_EVENTS)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ---------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------

df.to_csv(
    "data/ecommerce_events.csv",
    index=False
)


print("\nDataset generated successfully!")
print("--------------------------------")
print(f"Total events: {len(df):,}")
print(f"Unique users: {df['user_id'].nunique():,}")

print("\nEvent distribution:")
print(df["event_type"].value_counts())

print("\nDevice distribution:")
print(df["device"].value_counts())

print("\nTraffic source:")
print(df["traffic_source"].value_counts())

print("\nSaved to:")
print("data/ecommerce_events.csv")