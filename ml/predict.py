import os
import sys
import pandas as pd
import joblib


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "ml",
    "models",
    "dropoff_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==========================================")
print("USER DROP-OFF PREDICTION")
print("==========================================\n")


model = joblib.load(
    MODEL_FILE
)


print(
    "ML model loaded successfully."
)


# ============================================================
# SAMPLE USER
# ============================================================

user_data = {

    "device": "mobile",

    "location": "India",

    "traffic_source": "social_media",

    "visit_count": 1,

    "signup_count": 1,

    "cart_count": 1,

    "checkout_count": 0,

    "current_stage": "add_to_cart",

    "session_duration_seconds": 180,

    "event_number": 3

}


# ============================================================
# CREATE DATAFRAME
# ============================================================

input_data = pd.DataFrame(
    [user_data]
)


# ============================================================
# PREDICT PURCHASE PROBABILITY
# ============================================================

purchase_probability = model.predict_proba(
    input_data
)[0][1]


# ============================================================
# DROP-OFF PROBABILITY
# ============================================================

dropoff_probability = (
    1 - purchase_probability
)


# ============================================================
# RISK LEVEL
# ============================================================

if dropoff_probability >= 0.70:

    risk_level = "HIGH"

elif dropoff_probability >= 0.40:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# ============================================================
# RECOMMENDATION
# ============================================================

if risk_level == "HIGH":

    recommendation = (
        "High-risk user. "
        "Simplify checkout, improve mobile UX "
        "and consider a limited-time incentive."
    )

elif risk_level == "MEDIUM":

    recommendation = (
        "Medium-risk user. "
        "Provide product reassurance, "
        "reviews and a clear checkout experience."
    )

else:

    recommendation = (
        "Low-risk user. "
        "Continue the current experience "
        "and encourage purchase completion."
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n==========================================")
print("PREDICTION RESULT")
print("==========================================\n")


print(
    "Device:",
    user_data["device"]
)


print(
    "Location:",
    user_data["location"]
)


print(
    "Traffic Source:",
    user_data["traffic_source"]
)


print(
    "Current Stage:",
    user_data["current_stage"]
)


print(
    "\nPurchase Probability:",
    f"{purchase_probability * 100:.2f}%"
)


print(
    "Drop-off Probability:",
    f"{dropoff_probability * 100:.2f}%"
)


print(
    "Risk Level:",
    risk_level
)


print(
    "\nRecommendation:"
)

print(
    recommendation
)


print("\n==========================================")