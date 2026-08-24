from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import os
import re
import math
import pandas as pd
import mysql.connector
import joblib


# =========================================================
# APP
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "ecommerce_events.csv"
MODEL_FILE = BASE_DIR / "models" / "dropout_model.pkl"
METRICS_FILE = BASE_DIR / "models" / "model_metrics.pkl"


# =========================================================
# MYSQL CONFIG
# =========================================================

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "funnel_analysis"),
    "connection_timeout": 8
}


def get_mysql_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)


# =========================================================
# ML
# =========================================================

FEATURES = [
    "age",
    "pages_visited",
    "session_duration",
    "clicks",
    "previous_visits"
]

model = None
metrics = {}


if MODEL_FILE.exists():
    try:
        model = joblib.load(MODEL_FILE)
        print("Random Forest model loaded successfully.")
    except Exception as e:
        print("Model loading error:", e)
        model = None


if METRICS_FILE.exists():
    try:
        metrics = joblib.load(METRICS_FILE)
    except Exception as e:
        print("Metrics loading error:", e)
        metrics = {}


# =========================================================
# CORS
# =========================================================

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# =========================================================
# HELPERS
# =========================================================

def clean_value(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def normalize_key(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower()
    )


def normalize_event(value):

    x = clean_value(value).lower()

    x = x.replace("_", " ")
    x = x.replace("-", " ")
    x = re.sub(r"\s+", " ", x).strip()

    mapping = {
        "visit": "visit",
        "visited": "visit",
        "page visit": "visit",
        "session start": "visit",
        "landing": "visit",
        "home": "visit",

        "product view": "product_view",
        "product viewed": "product_view",
        "view product": "product_view",
        "viewed product": "product_view",
        "product": "product_view",

        "add to cart": "add_to_cart",
        "added to cart": "add_to_cart",
        "cart": "add_to_cart",

        "view cart": "view_cart",
        "cart view": "view_cart",

        "checkout": "checkout",
        "checkout start": "checkout",
        "checkout started": "checkout",

        "purchase": "purchase",
        "purchased": "purchase",
        "order": "purchase",
        "payment success": "purchase",
        "transaction": "purchase"
    }

    return mapping.get(
        x,
        x.replace(" ", "_")
    )


def event_display(value):

    normalized = normalize_event(value)

    mapping = {
        "visit": "Visit",
        "product_view": "Product View",
        "add_to_cart": "Add to Cart",
        "view_cart": "View Cart",
        "checkout": "Checkout",
        "purchase": "Purchase"
    }

    return mapping.get(
        normalized,
        clean_value(value)
    )


def detect_device():

    user_agent = request.headers.get(
        "User-Agent",
        ""
    ).lower()

    if (
        "ipad" in user_agent
        or "tablet" in user_agent
    ):
        return "Tablet"

    if (
        "mobile" in user_agent
        or "iphone" in user_agent
        or "android" in user_agent
    ):
        return "Mobile"

    return "Desktop"


# =========================================================
# LIVE DATABASE
# =========================================================

def ensure_mysql_tables():

    connection = None
    cursor = None

    try:

        connection = get_mysql_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS live_events (

                id INT AUTO_INCREMENT PRIMARY KEY,

                timestamp DATETIME NOT NULL,

                user_id VARCHAR(100) NOT NULL,

                event_type VARCHAR(100),

                product_id VARCHAR(255),

                product VARCHAR(255),

                device VARCHAR(100),

                location VARCHAR(255),

                traffic_source VARCHAR(100),

                current_page VARCHAR(100),

                age INT DEFAULT 25,

                pages_visited INT DEFAULT 1,

                session_duration INT DEFAULT 0,

                clicks INT DEFAULT 0,

                previous_visits INT DEFAULT 0,

                dropout_probability DECIMAL(10,8) DEFAULT 0,

                risk VARCHAR(20)
            )
            """
        )

        connection.commit()

        cursor.execute(
            "SHOW COLUMNS FROM live_events"
        )

        existing = {
            row[0]
            for row in cursor.fetchall()
        }

        additions = {

            "event_type":
                "VARCHAR(100)",

            "product_id":
                "VARCHAR(255)",

            "product":
                "VARCHAR(255)",

            "device":
                "VARCHAR(100)",

            "location":
                "VARCHAR(255)",

            "traffic_source":
                "VARCHAR(100)",

            "current_page":
                "VARCHAR(100)",

            "age":
                "INT DEFAULT 25",

            "pages_visited":
                "INT DEFAULT 1",

            "session_duration":
                "INT DEFAULT 0",

            "clicks":
                "INT DEFAULT 0",

            "previous_visits":
                "INT DEFAULT 0",

            "dropout_probability":
                "DECIMAL(10,8) DEFAULT 0",

            "risk":
                "VARCHAR(20)"
        }

        for column, definition in additions.items():

            if column not in existing:

                cursor.execute(
                    f"""
                    ALTER TABLE live_events
                    ADD COLUMN {column} {definition}
                    """
                )

        connection.commit()

        print("MySQL live_events table ready.")

    except Exception as e:

        print(
            "MySQL setup error:",
            e
        )

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# =========================================================
# LOAD ONLY LIVE EVENTS
# =========================================================

def load_live_events():

    connection = None
    cursor = None

    try:

        connection = get_mysql_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT *
            FROM live_events
            ORDER BY timestamp DESC, id DESC
            LIMIT 2000
            """
        )

        rows = cursor.fetchall()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    except Exception as e:

        print(
            "Live event loading error:",
            e
        )

        return pd.DataFrame()

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# =========================================================
# NEXT USER ID
# =========================================================

def next_user_id():

    connection = None
    cursor = None

    try:

        connection = get_mysql_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT MAX(
                CAST(
                    REPLACE(user_id, 'U', '')
                    AS UNSIGNED
                )
            )
            FROM live_events
            """
        )

        result = cursor.fetchone()

        maximum = 1000

        if result and result[0]:
            maximum = max(
                maximum,
                int(result[0])
            )

        return f"U{maximum + 1}"

    except Exception as e:

        print(
            "Next user ID error:",
            e
        )

        return "U1001"

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# =========================================================
# PREDICTION
# =========================================================

def heuristic_prediction(
    pages,
    duration,
    clicks,
    event
):

    score = 0.65

    if pages >= 5:
        score -= 0.15

    if duration >= 180:
        score -= 0.12

    if clicks >= 5:
        score -= 0.10

    stage = normalize_event(event)

    if stage == "checkout":
        score -= 0.10

    elif stage == "add_to_cart":
        score -= 0.05

    elif stage == "product_view":
        score += 0.05

    elif stage == "visit":
        score += 0.10

    score = max(
        0.05,
        min(score, 0.95)
    )

    return score


def calculate_prediction(
    age,
    pages,
    duration,
    clicks,
    previous_visits,
    page
):

    stage = normalize_event(page)

    if stage == "purchase":
        return 0.0, "LOW"

    probability = None

    if model is not None:

        try:

            data = pd.DataFrame(
                [{
                    "age": age,
                    "pages_visited": pages,
                    "session_duration": duration,
                    "clicks": clicks,
                    "previous_visits": previous_visits
                }]
            )

            probability = float(
                model.predict_proba(
                    data[FEATURES]
                )[0][1]
            )

            if not math.isfinite(
                probability
            ):
                probability = None

        except Exception as e:

            print(
                "ML prediction error:",
                e
            )

            probability = None

    if probability is None:

        probability = heuristic_prediction(
            pages,
            duration,
            clicks,
            page
        )

    probability = max(
        0.0,
        min(
            probability,
            1.0
        )
    )

    if probability >= 0.70:
        risk = "HIGH"

    elif probability >= 0.40:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    return probability, risk


# =========================================================
# SAVE LIVE EVENT
# =========================================================

def save_live_event(user, data):

    connection = None
    cursor = None

    try:

        connection = get_mysql_connection()
        cursor = connection.cursor()

        user_id = str(
            user["user_id"]
        )

        event_type = event_display(
            user["current_page"]
        )

        product_id = str(
            data.get(
                "product_id",
                ""
            ) or ""
        )

        product = str(
            data.get(
                "product",
                data.get(
                    "product_name",
                    data.get(
                        "product_viewed",
                        ""
                    )
                )
            ) or ""
        )

        location = str(
            data.get(
                "location",
                ""
            ) or ""
        )

        traffic_source = str(
            data.get(
                "traffic_source",
                "Direct"
            ) or "Direct"
        )

        query = """
            INSERT INTO live_events
            (
                timestamp,
                user_id,
                event_type,
                product_id,
                product,
                device,
                location,
                traffic_source,
                current_page,
                age,
                pages_visited,
                session_duration,
                clicks,
                previous_visits,
                dropout_probability,
                risk
            )
            VALUES
            (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
        """

        values = (
            datetime.now(),
            user_id,
            event_type,
            product_id,
            product,
            user["device"],
            location,
            traffic_source,
            user["current_page"],
            user["age"],
            user["pages_visited"],
            user["session_duration"],
            user["clicks"],
            user["previous_visits"],
            user["dropout_probability"],
            user["risk"]
        )

        cursor.execute(
            query,
            values
        )

        connection.commit()

        return True

    except Exception as e:

        print(
            "MySQL live event save error:",
            e
        )

        return False

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# =========================================================
# LIVE DASHBOARD DATA
# =========================================================
#
# IMPORTANT:
# THIS DOES NOT USE ecommerce_events.csv.
#
# EVERYTHING HERE COMES FROM MySQL live_events.
# =========================================================

@app.route("/dashboard-data")
def dashboard_data():

    df = load_live_events()

    if df.empty:

        return jsonify({

            "total_events": 0,

            "unique_users": 0,

            "product_views": 0,

            "purchases": 0,

            "funnel": {
                "visit": 0,
                "product_view": 0,
                "add_to_cart": 0,
                "checkout": 0,
                "purchase": 0
            },

            "prediction": {

                "current_user": "—",

                "current_stage": "—",

                "dropoff_probability": 0,

                "purchase_probability": 0,

                "risk_level": "LOW"
            },

            "dropoff": {
                "biggest": "—"
            },

            "recent_events": [],

            "ml_model_loaded": model is not None

        })


    # -----------------------------------------------------
    # NORMALIZE EVENTS
    # -----------------------------------------------------

    if "event_type" in df.columns:

        df["normalized_event"] = (
            df["event_type"]
            .apply(normalize_event)
        )

    else:

        df["normalized_event"] = ""


    # -----------------------------------------------------
    # BASIC LIVE COUNTS
    # -----------------------------------------------------

    total_events = len(df)

    if "user_id" in df.columns:

        unique_users = int(
            df["user_id"]
            .astype(str)
            .nunique()
        )

    else:

        unique_users = 0


    product_views = int(
        (
            df["normalized_event"]
            == "product_view"
        ).sum()
    )

    add_to_cart = int(
        (
            df["normalized_event"]
            == "add_to_cart"
        ).sum()
    )

    checkout = int(
        (
            df["normalized_event"]
            == "checkout"
        ).sum()
    )

    purchases = int(
        (
            df["normalized_event"]
            == "purchase"
        ).sum()
    )

    visits = int(
        (
            df["normalized_event"]
            == "visit"
        ).sum()
    )


    # -----------------------------------------------------
    # LIVE FUNNEL
    # -----------------------------------------------------

    funnel = {

        "visit":
            visits,

        "product_view":
            product_views,

        "add_to_cart":
            add_to_cart,

        "checkout":
            checkout,

        "purchase":
            purchases
    }


    # -----------------------------------------------------
    # DROP-OFF
    # -----------------------------------------------------

    funnel_pairs = [

        (
            "Visit",
            visits,
            "Product View",
            product_views
        ),

        (
            "Product View",
            product_views,
            "Add to Cart",
            add_to_cart
        ),

        (
            "Add to Cart",
            add_to_cart,
            "Checkout",
            checkout
        ),

        (
            "Checkout",
            checkout,
            "Purchase",
            purchases
        )
    ]


    biggest_dropoff = "—"
    biggest_drop = -1

    for (
        current_name,
        current_count,
        next_name,
        next_count
    ) in funnel_pairs:

        if current_count > 0:

            drop = (
                current_count
                - next_count
            )

            if drop > biggest_drop:

                biggest_drop = drop

                biggest_dropoff = (
                    f"{current_name} → "
                    f"{next_name}"
                )


    # -----------------------------------------------------
    # LATEST LIVE USER
    # -----------------------------------------------------

    latest = df.iloc[0]

    current_user = str(
        latest.get(
            "user_id",
            "—"
        )
    )

    current_stage = event_display(
        latest.get(
            "event_type",
            latest.get(
                "current_page",
                ""
            )
        )
    )


    # -----------------------------------------------------
    # LIVE PREDICTION
    # -----------------------------------------------------

    try:

        age = int(
            latest.get(
                "age",
                25
            )
        )

    except Exception:

        age = 25


    try:

        pages = int(
            latest.get(
                "pages_visited",
                1
            )
        )

    except Exception:

        pages = 1


    try:

        duration = int(
            latest.get(
                "session_duration",
                0
            )
        )

    except Exception:

        duration = 0


    try:

        clicks = int(
            latest.get(
                "clicks",
                0
            )
        )

    except Exception:

        clicks = 0


    try:

        previous_visits = int(
            latest.get(
                "previous_visits",
                0
            )
        )

    except Exception:

        previous_visits = 0


    probability, risk = calculate_prediction(

        age,

        pages,

        duration,

        clicks,

        previous_visits,

        current_stage
    )


    # If a prediction was stored in MySQL,
    # use that actual live prediction.

    try:

        stored_probability = float(
            latest.get(
                "dropout_probability",
                probability
            )
        )

        if math.isfinite(
            stored_probability
        ):

            probability = max(
                0,
                min(
                    stored_probability,
                    1
                )
            )

    except Exception:

        pass


    if current_stage == "Purchase":

        purchase_probability = 1.0

    else:

        purchase_probability = (
            1.0 - probability
        )


    # -----------------------------------------------------
    # LIVE RECENT EVENTS
    # -----------------------------------------------------

    recent_events = []

    recent_df = df.head(10)

    for _, row in recent_df.iterrows():

        event = {

            "user_id":
                str(
                    row.get(
                        "user_id",
                        ""
                    )
                ),

            "event_type":
                event_display(
                    row.get(
                        "event_type",
                        ""
                    )
                ),

            "product_id":
                str(
                    row.get(
                        "product_id",
                        ""
                    ) or ""
                ),

            "product":
                str(
                    row.get(
                        "product",
                        ""
                    ) or ""
                ),

            "device":
                str(
                    row.get(
                        "device",
                        ""
                    ) or ""
                ),

            "location":
                str(
                    row.get(
                        "location",
                        ""
                    ) or ""
                ),

            "traffic_source":
                str(
                    row.get(
                        "traffic_source",
                        ""
                    ) or ""
                )
        }


        timestamp = row.get(
            "timestamp",
            ""
        )

        if pd.notna(timestamp):

            try:

                event["timestamp"] = (
                    pd.to_datetime(
                        timestamp
                    ).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

            except Exception:

                event["timestamp"] = str(
                    timestamp
                )

        else:

            event["timestamp"] = ""


        recent_events.append(
            event
        )


    # -----------------------------------------------------
    # FINAL LIVE RESPONSE
    # -----------------------------------------------------

    return jsonify({

        "total_events":
            total_events,

        "unique_users":
            unique_users,

        "product_views":
            product_views,

        "add_to_cart":
            add_to_cart,

        "checkout":
            checkout,

        "purchases":
            purchases,

        "funnel":
            funnel,

        "prediction": {

            "current_user":
                current_user,

            "current_stage":
                current_stage,

            "dropoff_probability":
                round(
                    probability,
                    4
                ),

            "purchase_probability":
                round(
                    purchase_probability,
                    4
                ),

            "risk_level":
                risk
        },

        "dropoff": {

            "biggest":
                biggest_dropoff
        },

        "recent_events":
            recent_events,

        "ml_model_loaded":
            model is not None
    })


# =========================================================
# MAIN DASHBOARD
# =========================================================

@app.route("/")
@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        site="ShopFlow",
        page="Overview"
    )


# =========================================================
# LIVE USER JOURNEY PAGE
# =========================================================

@app.route("/live")
def live():

    return render_template(
        "dashboard.html",
        site="ShopFlow",
        page="Live User Journey",
        next_user=next_user_id()
    )


# =========================================================
# FUNNEL PAGE
# =========================================================

@app.route("/funnel")
def funnel_page():

    return render_template(
        "dashboard.html",
        site="ShopFlow",
        page="Funnel Analysis"
    )


# =========================================================
# DROPOFF PAGE
# =========================================================

@app.route("/dropoff")
def dropoff_page():

    return render_template(
        "dashboard.html",
        site="ShopFlow",
        page="Drop-off Prediction"
    )


# =========================================================
# USER INSIGHTS PAGE
# =========================================================

@app.route("/insights")
def insights_page():

    return render_template(
        "dashboard.html",
        site="ShopFlow",
        page="User Insights"
    )


# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================

@app.route("/model")
def model_page():

    return render_template(
        "dashboard.html",
        site="ShopFlow",
        page="Model Performance"
    )


# =========================================================
# LIVE API
# =========================================================

@app.route("/api/live")
def api_live():

    df = load_live_events()

    if df.empty:

        return jsonify({
            "events": [],
            "count": 0
        })


    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.sort_values(
            ["timestamp", "id"]
            if "id" in df.columns
            else ["timestamp"],
            ascending=False
        )


    recent_df = df.head(50)

    events = []


    for _, row in recent_df.iterrows():

        timestamp = row.get(
            "timestamp",
            ""
        )

        if pd.notna(timestamp):

            try:

                timestamp = pd.to_datetime(
                    timestamp
                ).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            except Exception:

                timestamp = str(timestamp)

        else:

            timestamp = ""


        events.append({

            "id":
                str(
                    row.get(
                        "id",
                        ""
                    )
                ),

            "user_id":
                str(
                    row.get(
                        "user_id",
                        ""
                    )
                ),

            "event_type":
                event_display(
                    row.get(
                        "event_type",
                        ""
                    )
                ),

            "product_id":
                str(
                    row.get(
                        "product_id",
                        ""
                    ) or ""
                ),

            "product":
                str(
                    row.get(
                        "product",
                        ""
                    ) or ""
                ),

            "device":
                str(
                    row.get(
                        "device",
                        ""
                    ) or ""
                ),

            "location":
                str(
                    row.get(
                        "location",
                        ""
                    ) or ""
                ),

            "traffic_source":
                str(
                    row.get(
                        "traffic_source",
                        ""
                    ) or ""
                ),

            "timestamp":
                timestamp
        })


    return jsonify({

        "events":
            events,

        "count":
            len(events)
    })


# =========================================================
# NEXT USER API
# =========================================================

@app.route("/api/next-user")
def api_next_user():

    return jsonify({

        "user_id":
            next_user_id()
    })


# =========================================================
# PREDICTION API
# =========================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    data = request.get_json(
        force=True
    ) or {}


    try:

        age = int(
            data.get(
                "age",
                25
            )
        )

    except Exception:

        age = 25


    try:

        pages = int(
            data.get(
                "pages_visited",
                1
            )
        )

    except Exception:

        pages = 1


    try:

        duration = int(
            data.get(
                "session_duration",
                0
            )
        )

    except Exception:

        duration = 0


    try:

        clicks = int(
            data.get(
                "clicks",
                0
            )
        )

    except Exception:

        clicks = 0


    try:

        previous_visits = int(
            data.get(
                "previous_visits",
                0
            )
        )

    except Exception:

        previous_visits = 0


    current_page = data.get(
        "current_page",
        "Visit"
    )


    probability, risk = calculate_prediction(

        age,

        pages,

        duration,

        clicks,

        previous_visits,

        current_page
    )


    return jsonify({

        "probability":
            probability,

        "percentage":
            round(
                probability * 100,
                2
            ),

        "risk":
            risk,

        "model":
            "Random Forest"
            if model is not None
            else "Heuristic fallback"
    })


# =========================================================
# SHOPPING WEBSITE → LIVE EVENT API
# =========================================================

@app.route(
    "/api/event",
    methods=["POST"]
)
def api_event():

    data = request.get_json(
        force=True
    ) or {}


    # -----------------------------------------------------
    # USER
    # -----------------------------------------------------

    user_id = str(
        data.get(
            "user_id",
            next_user_id()
        )
    )


    # -----------------------------------------------------
    # EVENT
    # -----------------------------------------------------

    current_page = str(
        data.get(
            "current_page",
            data.get(
                "event",
                "Visit"
            )
        )
    )


    # -----------------------------------------------------
    # DEVICE
    # -----------------------------------------------------

    device = str(
        data.get(
            "device",
            detect_device()
        )
    )


    # -----------------------------------------------------
    # USER FEATURES
    # -----------------------------------------------------

    try:
        age = int(
            data.get(
                "age",
                25
            )
        )
    except Exception:
        age = 25


    try:
        pages_visited = int(
            data.get(
                "pages_visited",
                1
            )
        )
    except Exception:
        pages_visited = 1


    try:
        session_duration = int(
            data.get(
                "session_duration",
                0
            )
        )
    except Exception:
        session_duration = 0


    try:
        clicks = int(
            data.get(
                "clicks",
                0
            )
        )
    except Exception:
        clicks = 0


    try:
        previous_visits = int(
            data.get(
                "previous_visits",
                0
            )
        )
    except Exception:
        previous_visits = 0


    # -----------------------------------------------------
    # LIVE ML PREDICTION
    # -----------------------------------------------------

    probability, risk = calculate_prediction(

        age,

        pages_visited,

        session_duration,

        clicks,

        previous_visits,

        current_page
    )


    # -----------------------------------------------------
    # USER OBJECT
    # -----------------------------------------------------

    user = {

        "user_id":
            user_id,

        "current_page":
            current_page,

        "event":
            event_display(
                current_page
            ),

        "age":
            age,

        "pages_visited":
            pages_visited,

        "session_duration":
            session_duration,

        "clicks":
            clicks,

        "previous_visits":
            previous_visits,

        "device":
            device,

        "dropout_probability":
            probability,

        "risk":
            risk
    }


    # -----------------------------------------------------
    # SAVE ONLY TO LIVE DATABASE
    # -----------------------------------------------------

    saved = save_live_event(
        user,
        data
    )


    return jsonify({

        "ok":
            saved,

        "live_saved":
            saved,

        "user":
            user,

        "percentage":
            round(
                probability * 100,
                2
            )
    })


# =========================================================
# OPTIONAL DATASET API
# =========================================================
#
# These are NOT used by Overview.
#
# The Overview does NOT call ecommerce_events.csv.
#
# They are kept only so existing pages/API calls don't break.
# =========================================================

@app.route("/api/historical")
def historical():

    return jsonify({

        "message":
            "Historical dataset is not used by the live Overview.",

        "available":
            DATA_FILE.exists()
    })


@app.route("/api/overview")
def api_overview():

    # IMPORTANT:
    # Overview is LIVE ONLY.

    return dashboard_data()


@app.route("/api/ecommerce-dataset-detailed")
def ecommerce_dataset_detailed():

    if not DATA_FILE.exists():

        return jsonify({

            "available":
                False,

            "message":
                "Dataset not available."
        })


    try:

        df = pd.read_csv(
            DATA_FILE
        )

        return jsonify({

            "available":
                True,

            "total_events":
                len(df),

            "columns":
                [
                    str(c)
                    for c in df.columns
                ]
        })

    except Exception as e:

        return jsonify({

            "available":
                False,

            "error":
                str(e)
        })


@app.route("/api/dataset-columns")
def dataset_columns():

    if not DATA_FILE.exists():

        return jsonify({

            "columns": []
        })


    try:

        df = pd.read_csv(
            DATA_FILE,
            nrows=1
        )

        return jsonify({

            "columns":
                [
                    str(c)
                    for c in df.columns
                ]
        })

    except Exception:

        return jsonify({

            "columns": []
        })


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    mysql_status = False

    connection = None

    try:

        connection = get_mysql_connection()

        mysql_status = (
            connection.is_connected()
        )

    except Exception as e:

        print(
            "Health MySQL error:",
            e
        )

    finally:

        if connection is not None:

            try:
                connection.close()
            except Exception:
                pass


    return jsonify({

        "status":
            "ok",

        "mysql":
            mysql_status,

        "csv":
            DATA_FILE.exists(),

        "model":
            model is not None
    })


# =========================================================
# STARTUP
# =========================================================

def startup():

    print(
        "===================================="
    )

    print(
        "ShopFlow starting..."
    )

    print(
        "MySQL host:",
        MYSQL_CONFIG["host"]
    )

    print(
        "MySQL database:",
        MYSQL_CONFIG["database"]
    )

    print(
        "LIVE MODE: MySQL live_events ONLY"
    )

    print(
        "Historical CSV is NOT used by Overview."
    )

    print(
        "Model loaded:",
        model is not None
    )

    print(
        "===================================="
    )

    ensure_mysql_tables()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    startup()

    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )