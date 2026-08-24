from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import pandas as pd
import mysql.connector
import joblib
import math
import re

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "user_journey_data.csv"
MODEL_FILE = BASE_DIR / "models" / "dropout_model.pkl"
METRICS_FILE = BASE_DIR / "models" / "model_metrics.pkl"

# =========================================================
# MYSQL
# =========================================================

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Abhilasha@123",
    "database": "funnel_analysis"
}


def mysql_connection():
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
        print("Random Forest model loaded.")
    except Exception as e:
        print("Model loading error:", e)

if METRICS_FILE.exists():
    try:
        metrics = joblib.load(METRICS_FILE)
    except Exception:
        metrics = {}


# =========================================================
# HELPERS
# =========================================================

def clean_value(value):
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_event(value):
    """
    Converts different CSV/event names into dashboard stages.
    """

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

    return mapping.get(x, x.replace(" ", "_"))


def find_column(df, candidates):
    """
    Finds actual CSV column regardless of case,
    spaces or underscores.
    """

    if df.empty:
        return None

    normalized = {}

    for col in df.columns:
        key = re.sub(
            r"[^a-z0-9]",
            "",
            str(col).lower()
        )
        normalized[key] = col

    for candidate in candidates:

        key = re.sub(
            r"[^a-z0-9]",
            "",
            candidate.lower()
        )

        if key in normalized:
            return normalized[key]

    return None


def load_csv():
    if not DATA_FILE.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(DATA_FILE)
    except Exception as e:
        print("CSV error:", e)
        return pd.DataFrame()


def detect_columns(df):
    return {
        "user": find_column(
            df,
            [
                "user_id",
                "userid",
                "customer_id",
                "customerid",
                "visitor_id",
                "visitorid"
            ]
        ),

        "event": find_column(
            df,
            [
                "event",
                "event_type",
                "eventtype",
                "action",
                "activity",
                "stage"
            ]
        ),

        "product": find_column(
            df,
            [
                "product_id",
                "productid",
                "product",
                "product_name",
                "productname",
                "product_viewed",
                "item_id",
                "item"
            ]
        ),

        "device": find_column(
            df,
            [
                "device",
                "device_type",
                "devicetype"
            ]
        ),

        "location": find_column(
            df,
            [
                "location",
                "city",
                "country",
                "region"
            ]
        ),

        "traffic": find_column(
            df,
            [
                "traffic_source",
                "trafficsource",
                "source",
                "channel",
                "marketing_channel"
            ]
        ),

        "timestamp": find_column(
            df,
            [
                "timestamp",
                "datetime",
                "date_time",
                "event_time",
                "event_datetime",
                "created_at"
            ]
        ),

        "date": find_column(
            df,
            [
                "date",
                "event_date",
                "visit_date"
            ]
        ),

        "time": find_column(
            df,
            [
                "time",
                "event_time_only"
            ]
        )
    }


def normalize_dataset(df):
    """
    Creates internal normalized columns while preserving
    ALL original CSV columns.
    """

    if df.empty:
        return df.copy(), detect_columns(df)

    result = df.copy()
    cols = detect_columns(result)

    # User
    if cols["user"]:
        result["_user"] = result[cols["user"]].astype(str)
    else:
        result["_user"] = [
            f"ROW_{i+1}"
            for i in range(len(result))
        ]

    # Event
    if cols["event"]:
        result["_event"] = result[cols["event"]].apply(
            normalize_event
        )
    else:
        result["_event"] = ""

    # Product
    if cols["product"]:
        result["_product"] = result[cols["product"]].apply(
            clean_value
        )
    else:
        result["_product"] = ""

    # Device
    if cols["device"]:
        result["_device"] = result[cols["device"]].apply(
            clean_value
        )
    else:
        result["_device"] = ""

    # Location
    if cols["location"]:
        result["_location"] = result[cols["location"]].apply(
            clean_value
        )
    else:
        result["_location"] = ""

    # Traffic
    if cols["traffic"]:
        result["_traffic"] = result[cols["traffic"]].apply(
            clean_value
        )
    else:
        result["_traffic"] = ""

    # Timestamp
    timestamp_series = pd.Series(
        pd.NaT,
        index=result.index
    )

    if cols["timestamp"]:
        timestamp_series = pd.to_datetime(
            result[cols["timestamp"]],
            errors="coerce"
        )

    elif cols["date"] and cols["time"]:

        timestamp_series = pd.to_datetime(
            result[cols["date"]].astype(str)
            + " "
            + result[cols["time"]].astype(str),
            errors="coerce"
        )

    elif cols["date"]:

        timestamp_series = pd.to_datetime(
            result[cols["date"]],
            errors="coerce"
        )

    result["_timestamp"] = timestamp_series

    return result, cols


# =========================================================
# EVENT COUNTING
# =========================================================

def event_counts(df):
    if df.empty:
        return {
            "visit": 0,
            "product_view": 0,
            "add_to_cart": 0,
            "view_cart": 0,
            "checkout": 0,
            "purchase": 0
        }

    return {
        "visit": int(
            (df["_event"] == "visit").sum()
        ),
        "product_view": int(
            (df["_event"] == "product_view").sum()
        ),
        "add_to_cart": int(
            (df["_event"] == "add_to_cart").sum()
        ),
        "view_cart": int(
            (df["_event"] == "view_cart").sum()
        ),
        "checkout": int(
            (df["_event"] == "checkout").sum()
        ),
        "purchase": int(
            (df["_event"] == "purchase").sum()
        )
    }


# =========================================================
# USER FUNNEL
# =========================================================

def user_funnel(df):
    if df.empty:
        return {
            "visit": 0,
            "product_view": 0,
            "add_to_cart": 0,
            "view_cart": 0,
            "checkout": 0,
            "purchase": 0
        }

    stages = [
        "visit",
        "product_view",
        "add_to_cart",
        "view_cart",
        "checkout",
        "purchase"
    ]

    users = {}

    for user, group in df.groupby("_user"):

        events = set(
            group["_event"].astype(str)
        )

        # If a user exists in the dataset but there is
        # no explicit Visit event, count the user as a visit.
        if events:
            events.add("visit")

        users[str(user)] = events

    result = {}

    for stage in stages:

        result[stage] = sum(
            1
            for events in users.values()
            if stage in events
        )

    return result


# =========================================================
# DROP-OFF
# =========================================================

def dropoff_analysis(df):

    funnel = user_funnel(df)

    visit_users = 0
    view_users = 0
    cart_users = 0
    checkout_users = 0
    purchase_users = 0

    if not df.empty:

        for user, group in df.groupby("_user"):

            events = set(
                group["_event"].astype(str)
            )

            events.add("visit")

            if "visit" in events:
                visit_users += 1

            if "product_view" in events:
                view_users += 1

            if "add_to_cart" in events:
                cart_users += 1

            if "checkout" in events:
                checkout_users += 1

            if "purchase" in events:
                purchase_users += 1

    visit_drop = max(
        visit_users - view_users,
        0
    )

    view_drop = max(
        view_users - cart_users,
        0
    )

    cart_drop = max(
        cart_users - checkout_users,
        0
    )

    checkout_drop = max(
        checkout_users - purchase_users,
        0
    )

    stages = {
        "Visit → Product View": visit_drop,
        "Product View → Add to Cart": view_drop,
        "Add to Cart → Checkout": cart_drop,
        "Checkout → Purchase": checkout_drop
    }

    biggest = (
        max(
            stages,
            key=stages.get
        )
        if stages
        else "—"
    )

    return {
        "visit": visit_drop,
        "view": view_drop,
        "cart": cart_drop,
        "checkout": checkout_drop,

        "visit_pct": round(
            visit_drop / visit_users * 100,
            2
        ) if visit_users else 0,

        "view_pct": round(
            view_drop / view_users * 100,
            2
        ) if view_users else 0,

        "cart_pct": round(
            cart_drop / cart_users * 100,
            2
        ) if cart_users else 0,

        "checkout_pct": round(
            checkout_drop / checkout_users * 100,
            2
        ) if checkout_users else 0,

        "biggest": biggest,

        "biggest_value": stages.get(
            biggest,
            0
        ),

        "funnel": funnel
    }


# =========================================================
# TIME ANALYSIS
# =========================================================

def time_analysis(df):

    if df.empty:
        return {
            "date_range": "—",
            "peak_hour": "—",
            "peak_hour_count": 0,
            "peak_purchase_hour": "—",
            "events_by_date": {},
            "events_by_hour": {}
        }

    t = df[df["_timestamp"].notna()].copy()

    if t.empty:
        return {
            "date_range": "Timestamp not available",
            "peak_hour": "—",
            "peak_hour_count": 0,
            "peak_purchase_hour": "—",
            "events_by_date": {},
            "events_by_hour": {}
        }

    t["_date"] = t["_timestamp"].dt.strftime(
        "%Y-%m-%d"
    )

    t["_hour"] = t["_timestamp"].dt.hour

    date_counts = (
        t.groupby("_date")
        .size()
        .sort_index()
    )

    hour_counts = (
        t.groupby("_hour")
        .size()
        .sort_index()
    )

    purchase_df = t[
        t["_event"] == "purchase"
    ]

    if not purchase_df.empty:

        purchase_hours = (
            purchase_df
            .groupby(
                purchase_df["_timestamp"].dt.hour
            )
            .size()
        )

        purchase_peak = int(
            purchase_hours.idxmax()
        )

    else:
        purchase_peak = None

    hour_data = {}

    for hour in range(24):

        subset = t[
            t["_hour"] == hour
        ]

        hour_data[str(hour)] = {
            "events": int(len(subset)),
            "dropoffs": int(
                len(
                    subset[
                        subset["_event"].isin([
                            "visit",
                            "product_view",
                            "add_to_cart",
                            "checkout"
                        ])
                    ]
                )
            ),
            "purchases": int(
                (
                    subset["_event"]
                    == "purchase"
                ).sum()
            )
        }

    peak_hour = int(
        hour_counts.idxmax()
    ) if not hour_counts.empty else None

    return {
        "date_range":
            f"{t['_timestamp'].min().strftime('%Y-%m-%d')} → "
            f"{t['_timestamp'].max().strftime('%Y-%m-%d')}",

        "peak_hour":
            f"{peak_hour:02d}:00 - {peak_hour:02d}:59"
            if peak_hour is not None
            else "—",

        "peak_hour_count":
            int(
                hour_counts.max()
            ) if not hour_counts.empty else 0,

        "peak_purchase_hour":
            f"{purchase_peak:02d}:00 - {purchase_peak:02d}:59"
            if purchase_peak is not None
            else "No purchases",

        "events_by_date":
            {
                str(k): int(v)
                for k, v in date_counts.items()
            },

        "events_by_hour":
            hour_data
    }


# =========================================================
# MYSQL TABLES
# =========================================================

def ensure_mysql_tables():

    connection = None
    cursor = None

    try:

        connection = mysql_connection()
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_events (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                user_id INT NOT NULL,
                event_type VARCHAR(100),
                product_id VARCHAR(255),
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
        """)

        connection.commit()

        # Add columns to an older live_events table.
        existing = set()

        cursor.execute("""
            SHOW COLUMNS FROM live_events
        """)

        for row in cursor.fetchall():
            existing.add(row[0])

        additions = {
            "event_type":
                "VARCHAR(100)",
            "product_id":
                "VARCHAR(255)",
            "location":
                "VARCHAR(255)",
            "traffic_source":
                "VARCHAR(100)"
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

        print("MySQL tables ready.")

    except Exception as e:
        print("MySQL setup error:", e)

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# =========================================================
# LIVE EVENTS
# =========================================================

def load_live_events():

    connection = None
    cursor = None

    try:

        connection = mysql_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute("""
            SELECT *
            FROM live_events
            ORDER BY timestamp DESC, id DESC
            LIMIT 500
        """)

        rows = cursor.fetchall()

        return pd.DataFrame(rows)

    except Exception as e:

        print(
            "Live event loading error:",
            e
        )

        return pd.DataFrame()

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# =========================================================
# PREDICTION
# =========================================================

def predict_dropout(
    age,
    pages,
    duration,
    clicks,
    previous_visits,
    event
):

    # Completed purchase = zero drop-off.
    if normalize_event(event) == "purchase":
        return 0.0, "LOW"

    if model is not None:

        try:

            x = pd.DataFrame([{
                "age": age,
                "pages_visited": pages,
                "session_duration": duration,
                "clicks": clicks,
                "previous_visits": previous_visits
            }])

            probability = float(
                model.predict_proba(
                    x[FEATURES]
                )[0][1]
            )

            probability = max(
                0,
                min(
                    probability,
                    1
                )
            )

        except Exception as e:

            print(
                "ML prediction error:",
                e
            )

            probability = heuristic_prediction(
                pages,
                duration,
                clicks,
                event
            )

    else:

        probability = heuristic_prediction(
            pages,
            duration,
            clicks,
            event
        )

    if probability >= 0.70:
        risk = "HIGH"

    elif probability >= 0.40:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    return probability, risk


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
        score -= 0.04

    elif stage == "product_view":
        score += 0.03

    return max(
        0.05,
        min(
            score,
            0.95
        )
    )


def recommendation_for(
    probability,
    stage
):

    stage = normalize_event(stage)

    if probability >= 0.70:

        if stage == "checkout":
            return (
                "High-risk checkout user. "
                "Reduce checkout friction and "
                "provide assistance or a reminder."
            )

        if stage == "add_to_cart":
            return (
                "High drop-off risk after cart activity. "
                "Consider product reassurance or an incentive."
            )

        return (
            "High drop-off risk. "
            "Increase engagement and show relevant recommendations."
        )

    if probability >= 0.40:

        return (
            "Medium-risk user. "
            "Continue monitoring behaviour and improve engagement."
        )

    return (
        "User journey is progressing normally. "
        "Continue monitoring behaviour."
    )


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html"
    )


# =========================================================
# DASHBOARD DATA
# =========================================================

@app.route("/dashboard-data")
def dashboard_data():

    df = load_csv()

    if not df.empty:

        df, cols = normalize_dataset(
            df
        )

    else:
        cols = {}

    counts = event_counts(df)
    funnel = user_funnel(df)
    dropoff = dropoff_analysis(df)

    total_events = len(df)

    unique_users = (
        df["_user"].nunique()
        if not df.empty
        else 0
    )

    # -----------------------------------------------------
    # Recent historical events
    # -----------------------------------------------------

    recent = []

    if not df.empty:

        recent_df = df.copy()

        recent_df = recent_df.sort_values(
            "_timestamp",
            ascending=False,
            na_position="last"
        ).head(30)

        for _, row in recent_df.iterrows():

            ts = row["_timestamp"]

            recent.append({
                "user_id":
                    clean_value(row["_user"]),

                "event_type":
                    clean_value(row["_event"]),

                "product_id":
                    clean_value(row["_product"]),

                "device":
                    clean_value(row["_device"]),

                "location":
                    clean_value(row["_location"]),

                "traffic_source":
                    clean_value(row["_traffic"]),

                "timestamp":
                    ts.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if pd.notna(ts)
                    else ""
            })

    # -----------------------------------------------------
    # Latest live prediction
    # -----------------------------------------------------

    live = load_live_events()

    prediction = None

    if not live.empty:

        latest = live.iloc[0]

        drop = float(
            latest.get(
                "dropout_probability",
                0
            ) or 0
        )

        prediction = {
            "user_id":
                f"U{latest['user_id']}",

            "current_stage":
                latest.get(
                    "event_type"
                )
                or latest.get(
                    "current_page",
                    "—"
                ),

            "dropoff_probability":
                round(
                    drop * 100,
                    2
                ),

            "purchase_probability":
                round(
                    (1 - drop) * 100,
                    2
                ),

            "risk_level":
                latest.get(
                    "risk",
                    "LOW"
                ),

            "device":
                latest.get(
                    "device",
                    "—"
                ),

            "location":
                latest.get(
                    "location",
                    "—"
                ),

            "traffic_source":
                latest.get(
                    "traffic_source",
                    "—"
                ),

            "recommendation":
                recommendation_for(
                    drop,
                    latest.get(
                        "event_type",
                        "Visit"
                    )
                ),

            "prediction_source":
                "Random Forest ML"
                if model is not None
                else "ML fallback"
        }

    return jsonify({

        "total_events":
            total_events,

        "unique_users":
            unique_users,

        "event_counts":
            counts,

        "funnel":
            funnel,

        "dropoff":
            dropoff,

        "recent_events":
            recent,

        "prediction":
            prediction,

        "ml_model_loaded":
            model is not None,

        "csv_columns":
            list(df.columns)
            if not df.empty
            else []
    })


# =========================================================
# LIVE DATA
# =========================================================

@app.route("/api/live")
def api_live():

    df = load_live_events()

    if df.empty:
        return jsonify({
            "events": [],
            "count": 0
        })

    records = []

    for _, row in df.iterrows():

        uid = clean_value(
            row.get(
                "user_id"
            )
        )

        if uid.isdigit():
            uid = f"U{uid}"

        timestamp = row.get(
            "timestamp"
        )

        if isinstance(
            timestamp,
            datetime
        ):
            timestamp = timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            timestamp = clean_value(
                timestamp
            )

        records.append({

            "user_id":
                uid,

            "event_type":
                clean_value(
                    row.get(
                        "event_type"
                    )
                    or row.get(
                        "current_page"
                    )
                ),

            "product_id":
                clean_value(
                    row.get(
                        "product_id"
                    )
                ),

            "device":
                clean_value(
                    row.get(
                        "device"
                    )
                ),

            "location":
                clean_value(
                    row.get(
                        "location"
                    )
                ),

            "traffic_source":
                clean_value(
                    row.get(
                        "traffic_source"
                    )
                ),

            "timestamp":
                timestamp
        })

    return jsonify({
        "events": records[:100],
        "count": len(records)
    })


# =========================================================
# NEXT USER
# =========================================================

@app.route("/api/next-user")
def next_user():

    connection = None
    cursor = None

    try:

        connection = mysql_connection()
        cursor = connection.cursor()

        cursor.execute("""
            SELECT COALESCE(
                MAX(user_id),
                1000
            )
            FROM live_events
        """)

        result = cursor.fetchone()

        maximum = int(
            result[0]
            if result and result[0]
            else 1000
        )

        return jsonify({
            "user_id":
                f"U{maximum + 1}"
        })

    except Exception:

        return jsonify({
            "user_id":
                "U1001"
        })

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# =========================================================
# PREDICT API
# =========================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    data = request.get_json(
        force=True
    ) or {}

    probability, risk = predict_dropout(

        int(
            data.get(
                "age",
                25
            )
        ),

        int(
            data.get(
                "pages_visited",
                1
            )
        ),

        int(
            data.get(
                "session_duration",
                0
            )
        ),

        int(
            data.get(
                "clicks",
                0
            )
        ),

        int(
            data.get(
                "previous_visits",
                0
            )
        ),

        data.get(
            "event",
            "Visit"
        )
    )

    return jsonify({

        "dropoff_probability":
            round(
                probability * 100,
                2
            ),

        "purchase_probability":
            round(
                (1 - probability) * 100,
                2
            ),

        "risk":
            risk,

        "prediction_source":
            "Random Forest ML"
            if model is not None
            else "ML fallback"
    })


# =========================================================
# LIVE EVENT
# =========================================================

@app.route(
    "/api/event",
    methods=["POST"]
)
def receive_event():

    data = request.get_json(
        force=True
    ) or {}

    event = data.get(
        "event_type",
        data.get(
            "event",
            data.get(
                "current_page",
                "Visit"
            )
        )
    )

    user_id = str(
        data.get(
            "user_id",
            "1001"
        )
    )

    if user_id.startswith("U"):
        user_id = user_id[1:]

    try:
        user_id_int = int(user_id)
    except:
        user_id_int = 1001

    product_id = data.get(
        "product_id",
        data.get(
            "product",
            ""
        )
    )

    device = data.get(
        "device",
        "Desktop"
    )

    location = data.get(
        "location",
        "India"
    )

    traffic_source = data.get(
        "traffic_source",
        "Direct"
    )

    age = int(
        data.get(
            "age",
            25
        )
    )

    pages = int(
        data.get(
            "pages_visited",
            1
        )
    )

    duration = int(
        data.get(
            "session_duration",
            0
        )
    )

    clicks = int(
        data.get(
            "clicks",
            0
        )
    )

    previous_visits = int(
        data.get(
            "previous_visits",
            0
        )
    )

    probability, risk = predict_dropout(

        age,
        pages,
        duration,
        clicks,
        previous_visits,
        event
    )

    connection = None
    cursor = None

    try:

        connection = mysql_connection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO live_events
            (
                timestamp,
                user_id,
                event_type,
                product_id,
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
                %s, %s, %s, %s, %s
            )
        """, (

            datetime.now(),

            user_id_int,

            event,

            product_id,

            device,

            location,

            traffic_source,

            event,

            age,

            pages,

            duration,

            clicks,

            previous_visits,

            probability,

            risk
        ))

        connection.commit()

        return jsonify({

            "success":
                True,

            "user_id":
                f"U{user_id_int}",

            "event":
                event,

            "product_id":
                product_id,

            "device":
                device,

            "location":
                location,

            "traffic_source":
                traffic_source,

            "dropoff_probability":
                round(
                    probability * 100,
                    2
                ),

            "risk":
                risk
        })

    except Exception as e:

        print(
            "Event insert error:",
            e
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# =========================================================
# COMPLETE DATASET ANALYSIS
# =========================================================

@app.route(
    "/api/ecommerce-dataset-detailed"
)
def ecommerce_dataset_detailed():

    df = load_csv()

    if df.empty:

        return jsonify({
            "success": True,
            "message":
                "CSV dataset not found or empty.",
            "total_events": 0,
            "unique_users": 0,
            "columns": [],
            "rows": [],
            "event_counts": {},
            "dropoff": {},
            "traffic_sources": {},
            "devices": {},
            "time": {}
        })

    normalized, detected = normalize_dataset(
        df
    )

    counts = event_counts(
        normalized
    )

    unique_users = int(
        normalized["_user"].nunique()
    )

    # -----------------------------------------------------
    # Actual CSV rows
    # -----------------------------------------------------

    rows = []

    display_df = df.copy().head(300)

    for _, row in display_df.iterrows():

        item = {}

        for col in df.columns:

            value = row[col]

            if pd.isna(value):
                item[str(col)] = ""
            else:
                item[str(col)] = str(value)

        rows.append(item)

    # -----------------------------------------------------
    # Traffic sources
    # -----------------------------------------------------

    traffic = {}

    if detected["traffic"]:

        values = (
            normalized[
                "_traffic"
            ]
            .replace(
                "",
                "Unknown"
            )
        )

        traffic = {
            str(k): int(v)
            for k, v in
            values.value_counts().items()
        }

    # -----------------------------------------------------
    # Devices
    # -----------------------------------------------------

    devices = {}

    if detected["device"]:

        values = (
            normalized[
                "_device"
            ]
            .replace(
                "",
                "Unknown"
            )
        )

        devices = {
            str(k): int(v)
            for k, v in
            values.value_counts().items()
        }

    # -----------------------------------------------------
    # Dataset date/time
    # -----------------------------------------------------

    time_data = time_analysis(
        normalized
    )

    # -----------------------------------------------------
    # Actual detected columns
    # -----------------------------------------------------

    actual_columns = [
        str(c)
        for c in df.columns
    ]

    return jsonify({

        "success":
            True,

        "total_events":
            len(df),

        "unique_users":
            unique_users,

        "event_counts":
            counts,

        "columns":
            actual_columns,

        "rows":
            rows,

        "detected_columns": {
            k:
                str(v)
                if v
                else None
            for k, v in detected.items()
        },

        "dropoff":
            dropoff_analysis(
                normalized
            ),

        "traffic_sources":
            traffic,

        "devices":
            devices,

        "time":
            time_data
    })


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    ensure_mysql_tables()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )