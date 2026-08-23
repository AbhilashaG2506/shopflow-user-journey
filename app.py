from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import datetime
import sqlite3
import os
import csv

# ============================================================
# OPTIONAL ML LIBRARIES
# ============================================================

try:
    import joblib
    import pandas as pd

    ML_LIBRARIES_AVAILABLE = True

except Exception as e:
    joblib = None
    pd = None
    ML_LIBRARIES_AVAILABLE = False

    print("Warning: ML libraries are not available.")
    print("Reason:", e)


# ============================================================
# SHOPFLOW APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE = os.path.join(
    BASE_DIR,
    "shopflow.db"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "dropoff_model.pkl"
)

ECOMMERCE_DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ecommerce_events.csv"
)


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_database():

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id TEXT NOT NULL,

            event_type TEXT NOT NULL,

            product_id TEXT,

            device TEXT,

            location TEXT,

            traffic_source TEXT,

            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()

    conn.close()


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

ml_model = None


if ML_LIBRARIES_AVAILABLE:

    try:

        if os.path.exists(MODEL_PATH):

            ml_model = joblib.load(
                MODEL_PATH
            )

            print()
            print("==============================================")
            print("          SHOPFLOW ML MODEL LOADED")
            print("==============================================")
            print()
            print("Model:")
            print(MODEL_PATH)
            print()

        else:

            print()
            print("WARNING: ML model file not found.")
            print()
            print("Expected:")
            print(MODEL_PATH)
            print()

    except Exception as e:

        print()
        print("WARNING: Could not load ML model.")
        print("Reason:", e)
        print()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "healthy",

        "application": "ShopFlow",

        "database": DATABASE,

        "ml_model_loaded":
            ml_model is not None

    })


# ============================================================
# GET USER EVENTS
# ============================================================

def get_user_events(user_id):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM events
        WHERE user_id = ?
        ORDER BY id ASC
    """, (
        user_id,
    ))

    rows = cursor.fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# CURRENT USER STAGE
# ============================================================

def get_current_stage(events):

    if not events:

        return "visit"

    latest_event = events[-1]["event_type"]

    stage_map = {

        "visit":
            "visit",

        "signup":
            "signup",

        "product_view":
            "product_view",

        "add_to_cart":
            "add_to_cart",

        "view_cart":
            "view_cart",

        "checkout":
            "checkout",

        "purchase":
            "purchase",

        "remove_from_cart":
            "remove_from_cart"

    }

    return stage_map.get(
        latest_event,
        "visit"
    )


# ============================================================
# SESSION DURATION
# ============================================================

def get_session_duration(events):

    if len(events) < 2:

        return 0

    try:

        first_time = datetime.fromisoformat(
            events[0]["timestamp"]
        )

        last_time = datetime.fromisoformat(
            events[-1]["timestamp"]
        )

        duration = (
            last_time - first_time
        ).total_seconds()

        return max(
            0,
            int(duration)
        )

    except Exception:

        return 0


# ============================================================
# USER FEATURES
# ============================================================

def get_user_features(events):

    event_types = [
        event["event_type"]
        for event in events
    ]

    return {

        "visit_count":
            event_types.count("visit"),

        "signup_count":
            event_types.count("signup"),

        "product_view_count":
            event_types.count("product_view"),

        "cart_count":
            event_types.count("add_to_cart"),

        "view_cart_count":
            event_types.count("view_cart"),

        "checkout_count":
            event_types.count("checkout"),

        "purchase_count":
            event_types.count("purchase"),

        "remove_count":
            event_types.count("remove_from_cart"),

        "current_stage":
            get_current_stage(events),

        "session_duration_seconds":
            get_session_duration(events),

        "event_number":
            len(events)

    }


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def generate_recommendation(
    dropoff_probability,
    purchase_probability,
    current_stage,
    features,
    device
):

    if current_stage == "purchase":

        return {

            "action":
                "Purchase completed",

            "recommendation":
                "User completed the purchase successfully. "
                "Consider post-purchase engagement, "
                "cross-selling and retention campaigns.",

            "priority":
                "LOW"

        }


    if dropoff_probability >= 70:

        if current_stage == "checkout":

            recommendation = (
                "High checkout drop-off risk. "
                "Simplify checkout, reduce form friction, "
                "show trust signals and offer quick payment options."
            )

        elif current_stage in [
            "view_cart",
            "add_to_cart"
        ]:

            recommendation = (
                "High cart abandonment risk. "
                "Show cart reminders, delivery information, "
                "trust signals and a limited-time incentive."
            )

        elif current_stage == "product_view":

            recommendation = (
                "High browsing drop-off risk. "
                "Show related products, reviews, offers "
                "and clearer Add to Cart actions."
            )

        else:

            recommendation = (
                "High drop-off risk. "
                "Improve engagement and guide the user "
                "towards the next journey stage."
            )

        return {

            "action":
                "Immediate intervention",

            "recommendation":
                recommendation,

            "priority":
                "HIGH"

        }


    if dropoff_probability >= 40:

        if current_stage == "product_view":

            recommendation = (
                "User is evaluating products. "
                "Highlight product benefits, ratings, "
                "reviews and relevant recommendations."
            )

        elif current_stage in [
            "add_to_cart",
            "view_cart"
        ]:

            recommendation = (
                "User has shopping intent. "
                "Use cart reminders and make checkout "
                "easy and transparent."
            )

        elif current_stage == "checkout":

            recommendation = (
                "User reached checkout but still has "
                "moderate drop-off risk. "
                "Provide clear payment and delivery information."
            )

        else:

            recommendation = (
                "Increase engagement and provide "
                "clear guidance towards the next stage."
            )

        return {

            "action":
                "Engagement intervention",

            "recommendation":
                recommendation,

            "priority":
                "MEDIUM"

        }


    if purchase_probability >= 70:

        return {

            "action":
                "Encourage conversion",

            "recommendation":
                "User shows strong purchase intent. "
                "Keep the journey simple and encourage "
                "purchase completion.",

            "priority":
                "LOW"

        }


    return {

        "action":
            "Continue monitoring",

        "recommendation":
            "User journey is progressing normally. "
            "Continue monitoring behaviour.",

        "priority":
            "LOW"

    }


# ============================================================
# FALLBACK PREDICTION
# ============================================================

def fallback_prediction(features):

    stage = features[
        "current_stage"
    ]

    event_number = features[
        "event_number"
    ]

    duration = features[
        "session_duration_seconds"
    ]


    if stage == "purchase":

        purchase_probability = 98
        dropoff_probability = 2

    elif stage == "checkout":

        purchase_probability = 80
        dropoff_probability = 20

    elif stage in [
        "add_to_cart",
        "view_cart"
    ]:

        purchase_probability = 65
        dropoff_probability = 35

    elif stage == "product_view":

        purchase_probability = 45
        dropoff_probability = 55

    else:

        purchase_probability = 30
        dropoff_probability = 70


    if event_number >= 5:

        purchase_probability += 5
        dropoff_probability -= 5


    if (
        duration > 600
        and stage != "purchase"
    ):

        purchase_probability -= 5
        dropoff_probability += 5


    purchase_probability = max(
        0,
        min(
            100,
            purchase_probability
        )
    )

    dropoff_probability = max(
        0,
        min(
            100,
            dropoff_probability
        )
    )


    if dropoff_probability >= 70:

        risk_level = "HIGH"

    elif dropoff_probability >= 40:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    return {

        "dropoff_probability":
            round(
                dropoff_probability,
                2
            ),

        "purchase_probability":
            round(
                purchase_probability,
                2
            ),

        "risk_level":
            risk_level,

        "prediction_source":
            "fallback"

    }


# ============================================================
# RANDOM FOREST ML PREDICTION
# ============================================================

def generate_ml_prediction(user_id):

    events = get_user_events(
        user_id
    )


    if not events:

        return {

            "user_id":
                user_id,

            "current_stage":
                "visit",

            "dropoff_probability":
                0,

            "purchase_probability":
                0,

            "risk_level":
                "LOW",

            "prediction_source":
                "waiting",

            "recommendation":
                "Waiting for user activity.",

            "action":
                "Wait",

            "priority":
                "LOW"

        }


    features = get_user_features(
        events
    )


    latest_event = events[-1]


    device = (
        latest_event.get("device")
        or "desktop"
    )

    location = (
        latest_event.get("location")
        or "India"
    )

    traffic_source = (
        latest_event.get("traffic_source")
        or "direct"
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    if (
        ml_model is not None
        and ML_LIBRARIES_AVAILABLE
    ):

        try:

            input_data = pd.DataFrame([{

                "device":
                    device,

                "location":
                    location,

                "traffic_source":
                    traffic_source,

                "visit_count":
                    features[
                        "visit_count"
                    ],

                "signup_count":
                    features[
                        "signup_count"
                    ],

                "cart_count":
                    features[
                        "cart_count"
                    ],

                "checkout_count":
                    features[
                        "checkout_count"
                    ],

                "current_stage":
                    features[
                        "current_stage"
                    ],

                "session_duration_seconds":
                    features[
                        "session_duration_seconds"
                    ],

                "event_number":
                    features[
                        "event_number"
                    ]

            }])


            probabilities = (
                ml_model.predict_proba(
                    input_data
                )[0]
            )


            classes = list(
                ml_model.classes_
            )


            if 1 in classes:

                purchase_probability = (
                    probabilities[
                        classes.index(1)
                    ] * 100
                )

            else:

                purchase_probability = 0


            dropoff_probability = (
                100 -
                purchase_probability
            )


            prediction_source = (
                "Random Forest ML model"
            )


        except Exception as e:

            print()
            print("ML prediction error:")
            print(e)
            print()

            fallback = fallback_prediction(
                features
            )

            purchase_probability = (
                fallback[
                    "purchase_probability"
                ]
            )

            dropoff_probability = (
                fallback[
                    "dropoff_probability"
                ]
            )

            prediction_source = (
                "fallback_after_ml_error"
            )

    else:

        fallback = fallback_prediction(
            features
        )

        purchase_probability = (
            fallback[
                "purchase_probability"
            ]
        )

        dropoff_probability = (
            fallback[
                "dropoff_probability"
            ]
        )

        prediction_source = (
            fallback[
                "prediction_source"
            ]
        )


    purchase_probability = round(
        max(
            0,
            min(
                100,
                purchase_probability
            )
        ),
        2
    )


    dropoff_probability = round(
        max(
            0,
            min(
                100,
                dropoff_probability
            )
        ),
        2
    )


    if dropoff_probability >= 70:

        risk_level = "HIGH"

    elif dropoff_probability >= 40:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    recommendation_data = (
        generate_recommendation(

            dropoff_probability,

            purchase_probability,

            features[
                "current_stage"
            ],

            features,

            device

        )
    )


    return {

        "user_id":
            user_id,

        "current_stage":
            features[
                "current_stage"
            ],

        "event_number":
            features[
                "event_number"
            ],

        "session_duration_seconds":
            features[
                "session_duration_seconds"
            ],

        "visit_count":
            features[
                "visit_count"
            ],

        "signup_count":
            features[
                "signup_count"
            ],

        "product_view_count":
            features[
                "product_view_count"
            ],

        "cart_count":
            features[
                "cart_count"
            ],

        "view_cart_count":
            features[
                "view_cart_count"
            ],

        "checkout_count":
            features[
                "checkout_count"
            ],

        "purchase_count":
            features[
                "purchase_count"
            ],

        "remove_count":
            features[
                "remove_count"
            ],

        "device":
            device,

        "location":
            location,

        "traffic_source":
            traffic_source,

        "dropoff_probability":
            dropoff_probability,

        "purchase_probability":
            purchase_probability,

        "risk_level":
            risk_level,

        "prediction_source":
            prediction_source,

        "action":
            recommendation_data[
                "action"
            ],

        "recommendation":
            recommendation_data[
                "recommendation"
            ],

        "priority":
            recommendation_data[
                "priority"
            ]

    }


# ============================================================
# TRACK EVENT
# ============================================================

@app.route(
    "/track-event",
    methods=["POST"]
)
def track_event():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )


    user_id = data.get(
        "user_id"
    )

    event_type = data.get(
        "event_type"
    )

    product_id = data.get(
        "product_id"
    )

    device = data.get(
        "device",
        "desktop"
    )

    location = data.get(
        "location",
        "India"
    )

    traffic_source = data.get(
        "traffic_source",
        "direct"
    )


    if not user_id:

        return jsonify({

            "success":
                False,

            "message":
                "user_id is required"

        }), 400


    if not event_type:

        return jsonify({

            "success":
                False,

            "message":
                "event_type is required"

        }), 400


    timestamp = (
        datetime.now().isoformat()
    )


    conn = get_db_connection()

    cursor = conn.cursor()


    cursor.execute("""
        INSERT INTO events
        (
            user_id,
            event_type,
            product_id,
            device,
            location,
            traffic_source,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (

        user_id,

        event_type,

        product_id,

        device,

        location,

        traffic_source,

        timestamp

    ))


    conn.commit()

    event_id = (
        cursor.lastrowid
    )

    conn.close()


    print(
        f"[EVENT] {event_type} | "
        f"User: {user_id} | "
        f"Product: {product_id}"
    )


    prediction = (
        generate_ml_prediction(
            user_id
        )
    )


    return jsonify({

        "success":
            True,

        "message":
            "Event tracked successfully",

        "event_id":
            event_id,

        "user_id":
            user_id,

        "event_type":
            event_type,

        "product_id":
            product_id,

        "timestamp":
            timestamp,

        "prediction":
            prediction

    })


# ============================================================
# ALL LIVE EVENTS
# ============================================================

@app.route(
    "/api/events"
)
def get_events():

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM events
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    conn.close()


    return jsonify([
        dict(row)
        for row in rows
    ])


# ============================================================
# ANALYTICS SUMMARY
# ============================================================

@app.route(
    "/api/analytics/summary"
)
def analytics_summary():

    conn = get_db_connection()

    cursor = conn.cursor()


    cursor.execute(
        "SELECT COUNT(*) AS count FROM events"
    )

    total_events = (
        cursor.fetchone()["count"]
    )


    cursor.execute(
        "SELECT COUNT(DISTINCT user_id) AS count FROM events"
    )

    unique_users = (
        cursor.fetchone()["count"]
    )


    event_types = [
        "product_view",
        "add_to_cart",
        "view_cart",
        "checkout",
        "purchase",
        "remove_from_cart"
    ]


    counts = {}


    for event_type in event_types:

        cursor.execute("""
            SELECT COUNT(*) AS count
            FROM events
            WHERE event_type = ?
        """, (
            event_type,
        ))

        counts[event_type] = (
            cursor.fetchone()["count"]
        )


    conn.close()


    return jsonify({

        "success":
            True,

        "total_events":
            total_events,

        "unique_users":
            unique_users,

        "product_views":
            counts["product_view"],

        "add_to_cart":
            counts["add_to_cart"],

        "view_cart":
            counts["view_cart"],

        "checkout":
            counts["checkout"],

        "purchases":
            counts["purchase"],

        "remove_from_cart":
            counts["remove_from_cart"],

        "ml_model_loaded":
            ml_model is not None

    })


# ============================================================
# FUNNEL ANALYTICS
# ============================================================

@app.route(
    "/api/analytics/funnel"
)
def funnel():

    try:

        conn = get_db_connection()

        cursor = conn.cursor()


        stages = [
            "visit",
            "product_view",
            "add_to_cart",
            "view_cart",
            "checkout",
            "purchase"
        ]


        funnel_data = {}

        for event_type in stages:

            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) AS count
                FROM events
                WHERE event_type = ?
            """, (
                event_type,
            ))

            funnel_data[event_type] = (
                cursor.fetchone()["count"]
            )


        conn.close()


        return jsonify({

            "success":
                True,

            "funnel":
                funnel_data

        })


    except Exception as e:

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# DASHBOARD DATA
#
# IMPORTANT:
# This uses ONLY live shopflow.db events.
# It DOES NOT include the 50,014 historical dataset.
# ============================================================

@app.route(
    "/dashboard-data"
)
def dashboard_data():

    conn = get_db_connection()

    cursor = conn.cursor()


    cursor.execute(
        "SELECT COUNT(*) AS count FROM events"
    )

    total_events = (
        cursor.fetchone()["count"]
    )


    cursor.execute(
        "SELECT COUNT(DISTINCT user_id) AS count FROM events"
    )

    unique_users = (
        cursor.fetchone()["count"]
    )


    # --------------------------------------------------------
    # EVENT COUNTS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            event_type,
            COUNT(*) AS count
        FROM events
        GROUP BY event_type
        ORDER BY count DESC
    """)

    rows = cursor.fetchall()


    event_counts = {

        row["event_type"]:
            row["count"]

        for row in rows

    }


    # --------------------------------------------------------
    # FUNNEL
    # --------------------------------------------------------

    funnel = {}

    funnel_events = [
        "visit",
        "product_view",
        "add_to_cart",
        "view_cart",
        "checkout",
        "purchase"
    ]


    for event_type in funnel_events:

        cursor.execute("""
            SELECT COUNT(DISTINCT user_id) AS count
            FROM events
            WHERE event_type = ?
        """, (
            event_type,
        ))

        funnel[event_type] = (
            cursor.fetchone()["count"]
        )


    # --------------------------------------------------------
    # DEVICES
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            device,
            COUNT(*) AS count
        FROM events
        WHERE device IS NOT NULL
        GROUP BY device
    """)

    devices = {

        row["device"]:
            row["count"]

        for row in cursor.fetchall()

    }


    # --------------------------------------------------------
    # TRAFFIC SOURCES
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            traffic_source,
            COUNT(*) AS count
        FROM events
        WHERE traffic_source IS NOT NULL
        GROUP BY traffic_source
    """)

    traffic_sources = {

        row["traffic_source"]:
            row["count"]

        for row in cursor.fetchall()

    }


    # --------------------------------------------------------
    # PRODUCT VIEWS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            product_id,
            COUNT(*) AS count
        FROM events
        WHERE event_type = 'product_view'
        AND product_id IS NOT NULL
        GROUP BY product_id
        ORDER BY count DESC
    """)

    product_views = {

        row["product_id"]:
            row["count"]

        for row in cursor.fetchall()

    }


    # --------------------------------------------------------
    # ADD TO CART PRODUCTS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            product_id,
            COUNT(*) AS count
        FROM events
        WHERE event_type = 'add_to_cart'
        AND product_id IS NOT NULL
        GROUP BY product_id
        ORDER BY count DESC
    """)

    add_to_cart_products = {

        row["product_id"]:
            row["count"]

        for row in cursor.fetchall()

    }


    # --------------------------------------------------------
    # REMOVE FROM CART PRODUCTS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            product_id,
            COUNT(*) AS count
        FROM events
        WHERE event_type = 'remove_from_cart'
        AND product_id IS NOT NULL
        GROUP BY product_id
        ORDER BY count DESC
    """)

    remove_from_cart_products = {

        row["product_id"]:
            row["count"]

        for row in cursor.fetchall()

    }


    # --------------------------------------------------------
    # RECENT LIVE EVENTS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT *
        FROM events
        ORDER BY id DESC
        LIMIT 20
    """)

    recent_events = [

        dict(row)

        for row in cursor.fetchall()

    ]


    # --------------------------------------------------------
    # LATEST USER
    # --------------------------------------------------------

    cursor.execute("""
        SELECT user_id
        FROM events
        ORDER BY id DESC
        LIMIT 1
    """)

    row = cursor.fetchone()


    latest_user = (
        row["user_id"]
        if row
        else None
    )


    conn.close()


    # --------------------------------------------------------
    # LIVE ML PREDICTION
    # --------------------------------------------------------

    prediction = None


    if latest_user:

        prediction = (
            generate_ml_prediction(
                latest_user
            )
        )


    return jsonify({

        "success":
            True,

        "total_events":
            total_events,

        "unique_users":
            unique_users,

        "event_counts":
            event_counts,

        "funnel":
            funnel,

        "devices":
            devices,

        "traffic_sources":
            traffic_sources,

        "product_views":
            product_views,

        "add_to_cart_products":
            add_to_cart_products,

        "remove_from_cart_products":
            remove_from_cart_products,

        "recent_events":
            recent_events,

        "prediction":
            prediction,

        "ml_model_loaded":
            ml_model is not None,

        "latest_user":
            latest_user

    })


# ============================================================
# ML PREDICTION API
# ============================================================

@app.route(
    "/api/ml/prediction"
)
def ml_prediction():

    user_id = request.args.get(
        "user_id"
    )


    if not user_id:

        conn = get_db_connection()

        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id
            FROM events
            ORDER BY id DESC
            LIMIT 1
        """)

        row = cursor.fetchone()

        conn.close()


        if row:

            user_id = row["user_id"]

        else:

            return jsonify({

                "success":
                    True,

                "prediction":
                    None,

                "message":
                    "No user activity available yet."

            })


    prediction = (
        generate_ml_prediction(
            user_id
        )
    )


    return jsonify({

        "success":
            True,

        "prediction":
            prediction

    })


# ============================================================
# LATEST PREDICTION
# ============================================================

@app.route(
    "/api/latest-prediction"
)
def latest_prediction():

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT user_id
        FROM events
        ORDER BY id DESC
        LIMIT 1
    """)

    row = cursor.fetchone()

    conn.close()


    if not row:

        return jsonify({

            "success":
                True,

            "prediction":
                None

        })


    prediction = (
        generate_ml_prediction(
            row["user_id"]
        )
    )


    return jsonify({

        "success":
            True,

        "prediction":
            prediction

    })


# ============================================================
# ML STATUS
# ============================================================

@app.route(
    "/api/ml-status"
)
def ml_status():

    if ml_model is not None:

        return jsonify({

            "success":
                True,

            "ml_model_loaded":
                True,

            "model_file":
                MODEL_PATH,

            "message":
                "Random Forest ML model loaded successfully."

        })


    return jsonify({

        "success":
            False,

        "ml_model_loaded":
            False,

        "model_file":
            MODEL_PATH,

        "message":
            "ML model is not loaded. "
            "The system is using fallback prediction."

    })


# ============================================================
# RECOMMENDATION API
# ============================================================

@app.route(
    "/api/recommendation"
)
def recommendation_api():

    response = latest_prediction()

    data = response.get_json()

    prediction = data.get(
        "prediction"
    )


    if not prediction:

        return jsonify({

            "success":
                True,

            "recommendation":
                None

        })


    return jsonify({

        "success":
            True,

        "user_id":
            prediction["user_id"],

        "risk_level":
            prediction["risk_level"],

        "action":
            prediction["action"],

        "recommendation":
            prediction["recommendation"],

        "priority":
            prediction["priority"]

    })


# ============================================================
# E-COMMERCE DATASET
#
# THIS IS SEPARATE FROM LIVE EVENTS.
#
# File:
# data/ecommerce_events.csv
#
# Expected:
# 50,014 events
# ============================================================

def get_ecommerce_dataset_stats():

    result = {

        "success":
            False,

        "total_events":
            0,

        "unique_users":
            0,

        "event_counts":
            {},

        "dataset_file":
            ECOMMERCE_DATASET_PATH

    }


    if not os.path.exists(
        ECOMMERCE_DATASET_PATH
    ):

        result["message"] = (
            "ecommerce_events.csv not found."
        )

        return result


    try:

        unique_users = set()

        event_counts = {}

        total_events = 0


        with open(
            ECOMMERCE_DATASET_PATH,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(
                file
            )


            for row in reader:

                total_events += 1


                user_id = (
                    row.get(
                        "user_id",
                        ""
                    )
                    .strip()
                )

                event_type = (
                    row.get(
                        "event_type",
                        ""
                    )
                    .strip()
                    .lower()
                )


                if user_id:

                    unique_users.add(
                        user_id
                    )


                if event_type:

                    event_counts[
                        event_type
                    ] = (
                        event_counts.get(
                            event_type,
                            0
                        ) + 1
                    )


        result.update({

            "success":
                True,

            "total_events":
                total_events,

            "unique_users":
                len(unique_users),

            "event_counts":
                event_counts,

            "message":
                "E-commerce dataset loaded successfully."

        })


        return result


    except Exception as e:

        result["message"] = str(e)

        return result


# ============================================================
# E-COMMERCE DATASET API
#
# IMPORTANT:
# This does NOT insert anything into shopflow.db.
# Therefore it cannot change the live 23 events.
# ============================================================

@app.route(
    "/api/ecommerce-dataset"
)
def ecommerce_dataset_api():

    stats = (
        get_ecommerce_dataset_stats()
    )

    return jsonify(
        stats
    )


# ============================================================
# HEARTBEAT
#
# ONLY ONE HEARTBEAT ROUTE.
#
# DO NOT ADD ANOTHER /heartbeat ROUTE.
# ============================================================

@app.route(
    "/heartbeat",
    methods=["POST"]
)
def heartbeat():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )


    user_id = data.get(
        "user_id",
        "dashboard"
    )


    return jsonify({

        "success":
            True,

        "message":
            "Heartbeat received",

        "user_id":
            user_id,

        "timestamp":
            datetime.now().isoformat()

    })


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    init_database()


    print()
    print("==============================================")
    print("          SHOPFLOW APPLICATION STARTED")
    print("==============================================")
    print()

    print("Website:")
    print(
        "http://127.0.0.1:5000"
    )

    print()

    print("Dashboard:")
    print(
        "http://127.0.0.1:5000/dashboard"
    )

    print()

    print("Health:")
    print(
        "http://127.0.0.1:5000/health"
    )

    print()

    print("Events:")
    print(
        "http://127.0.0.1:5000/api/events"
    )

    print()

    print("Analytics:")
    print(
        "http://127.0.0.1:5000/api/analytics/summary"
    )

    print()

    print("Funnel:")
    print(
        "http://127.0.0.1:5000/api/analytics/funnel"
    )

    print()

    print("Dashboard Data:")
    print(
        "http://127.0.0.1:5000/dashboard-data"
    )

    print()

    print("ML Prediction:")
    print(
        "http://127.0.0.1:5000/api/ml/prediction"
    )

    print()

    print("ML Status:")
    print(
        "http://127.0.0.1:5000/api/ml-status"
    )

    print()

    print("Recommendation:")
    print(
        "http://127.0.0.1:5000/api/recommendation"
    )

    print()

    print("E-commerce Dataset:")
    print(
        "http://127.0.0.1:5000/api/ecommerce-dataset"
    )

    print()

    print("Heartbeat:")
    print(
        "http://127.0.0.1:5000/heartbeat"
    )

    print()

    print("==============================================")
    print()

    app.run(
    debug=True,
    host="127.0.0.1",
    port=5000
)