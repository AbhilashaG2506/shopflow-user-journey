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

DATA_FILE = os.path.join(BASE_DIR, "data", "ecommerce_events.csv")
MODEL_FILE = BASE_DIR / "models" / "dropout_model.pkl"
METRICS_FILE = BASE_DIR / "models" / "model_metrics.pkl"


# =========================================================
# MYSQL CONFIGURATION
#
# LOCAL:
#   MYSQL_HOST=localhost
#
# RENDER:
#   Add these in Render Environment Variables:
#
#   MYSQL_HOST
#   MYSQL_PORT
#   MYSQL_USER
#   MYSQL_PASSWORD
#   MYSQL_DATABASE
#
# DO NOT put your real password in this file.
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
# BASIC HELPERS
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


def find_column(df, candidates):
    """
    Finds the actual CSV column even when naming differs.

    Example:
        User ID
        user_id
        USERID
        UserID

    are treated as the same logical field.
    """

    if df is None or df.empty:
        return None

    normalized = {}

    for column in df.columns:
        normalized[
            normalize_key(column)
        ] = column

    for candidate in candidates:
        key = normalize_key(candidate)

        if key in normalized:
            return normalized[key]

    return None


def get_column_series(df, candidates, default=""):
    column = find_column(df, candidates)

    if column is None:
        return pd.Series(
            [default] * len(df),
            index=df.index
        )

    return df[column]


# =========================================================
# EVENT NORMALIZATION
# =========================================================

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


# =========================================================
# CSV
# =========================================================

def load_data():

    if not os.path.exists(DATA_FILE):
        print("CSV file not found:", DATA_FILE)
        return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_FILE)

        print(
            "CSV loaded:",
            len(df),
            "rows"
        )

        return df

    except Exception as e:

        print(
            "CSV read error:",
            e
        )

        return pd.DataFrame()


# =========================================================
# MODEL METRICS
# =========================================================

def load_metrics():

    if not METRICS_FILE.exists():
        return {}

    try:
        return joblib.load(METRICS_FILE)

    except Exception as e:

        print(
            "Metrics read error:",
            e
        )

        return {}


# =========================================================
# TIMESTAMP HANDLING
# =========================================================

def get_timestamp_column(df):

    return find_column(
        df,
        [
            "timestamp",
            "date_time",
            "datetime",
            "date time",
            "event_time",
            "event_timestamp",
            "time_stamp",
            "time"
        ]
    )


def get_date_column(df):

    return find_column(
        df,
        [
            "date",
            "event_date",
            "visit_date",
            "event day"
        ]
    )


def get_time_column(df):

    return find_column(
        df,
        [
            "time",
            "event_time",
            "eventtime"
        ]
    )


def build_timestamp(df):

    if df.empty:
        return pd.Series(
            dtype="datetime64[ns]"
        )

    timestamp_col = get_timestamp_column(df)

    if timestamp_col:

        ts = pd.to_datetime(
            df[timestamp_col],
            errors="coerce"
        )

        return ts

    date_col = get_date_column(df)
    time_col = get_time_column(df)

    if date_col:

        if time_col:

            combined = (
                df[date_col].astype(str)
                + " "
                + df[time_col].astype(str)
            )

            return pd.to_datetime(
                combined,
                errors="coerce"
            )

        return pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

    return pd.Series(
        pd.NaT,
        index=df.index
    )


# =========================================================
# DEVICE DETECTION
# =========================================================

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
# MYSQL TABLES
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

                dropout_probability
                    DECIMAL(10,8)
                    DEFAULT 0,

                risk VARCHAR(20)

            )
            """
        )

        connection.commit()

        # -------------------------------------------------
        # Add missing columns to older table
        # -------------------------------------------------

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

        print(
            "MySQL live_events table ready."
        )

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
# LOAD LIVE EVENTS
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

        maximum = 1000

        # funnel_data may already exist
        try:

            cursor.execute(
                """
                SELECT MAX(user_id)
                FROM funnel_data
                """
            )

            result = cursor.fetchone()

            if result and result[0]:

                maximum = max(
                    maximum,
                    int(result[0])
                )

        except Exception:
            pass

        # live_events
        try:

            cursor.execute(
                """
                SELECT MAX(
                    CAST(
                        REPLACE(user_id,'U','')
                        AS UNSIGNED
                    )
                )
                FROM live_events
                """
            )

            result = cursor.fetchone()

            if result and result[0]:

                maximum = max(
                    maximum,
                    int(result[0])
                )

        except Exception:
            pass

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

    # Completed purchase
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

        raw_user_id = str(
            user["user_id"]
        )

        # Keep numeric ID when possible,
        # otherwise keep original string.
        mysql_user_id = raw_user_id

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
            mysql_user_id,
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

        print(
            "Live event saved:",
            mysql_user_id,
            event_type
        )

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
# SAVE FUNNEL DATA
# =========================================================

def save_event_to_funnel(data, user):

    connection = None
    cursor = None

    try:

        connection = get_mysql_connection()

        cursor = connection.cursor()

        raw_user_id = str(
            user["user_id"]
        )

        try:

            numeric_user_id = int(
                raw_user_id.replace(
                    "U",
                    ""
                )
            )

        except Exception:

            numeric_user_id = 0

        current_page = str(
            user["current_page"]
        ).strip()

        traffic_source = str(
            data.get(
                "traffic_source",
                "Direct"
            ) or "Direct"
        )

        landing_page = str(
            data.get(
                "landing_page",
                "Home"
            ) or "Home"
        )

        product_viewed = data.get(
            "product_viewed",
            data.get(
                "product",
                data.get(
                    "product_name",
                    ""
                )
            )
        )

        if not product_viewed:
            product_viewed = ""

        added_to_cart = int(
            data.get(
                "added_to_cart",
                1
                if normalize_event(
                    current_page
                ) == "add_to_cart"
                else 0
            )
        )

        checkout_started = int(
            data.get(
                "checkout_started",
                1
                if normalize_event(
                    current_page
                ) == "checkout"
                else 0
            )
        )

        purchase_completed = int(
            data.get(
                "purchase_completed",
                1
                if normalize_event(
                    current_page
                ) == "purchase"
                else 0
            )
        )

        purchase_amount = float(
            data.get(
                "purchase_amount",
                0
            ) or 0
        )

        # Make sure funnel_data exists.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS funnel_data
            (
                id INT AUTO_INCREMENT PRIMARY KEY,

                user_id INT,

                visit_date DATE,

                traffic_source VARCHAR(100),

                device VARCHAR(100),

                landing_page VARCHAR(255),

                product_viewed VARCHAR(255),

                added_to_cart INT DEFAULT 0,

                checkout_started INT DEFAULT 0,

                purchase_completed INT DEFAULT 0,

                purchase_amount DECIMAL(12,2)
                    DEFAULT 0
            )
            """
        )

        query = """
            INSERT INTO funnel_data
            (
                user_id,
                visit_date,
                traffic_source,
                device,
                landing_page,
                product_viewed,
                added_to_cart,
                checkout_started,
                purchase_completed,
                purchase_amount
            )
            VALUES
            (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
        """

        values = (
            numeric_user_id,
            datetime.now().date(),
            traffic_source,
            user["device"],
            landing_page,
            product_viewed,
            added_to_cart,
            checkout_started,
            purchase_completed,
            purchase_amount
        )

        cursor.execute(
            query,
            values
        )

        connection.commit()

        return True

    except Exception as e:

        print(
            "MySQL funnel save error:",
            e
        )

        return False

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# =========================================================
# DATASET ANALYSIS
#
# THIS IS THE DETAILED DATASET SECTION.
#
# It uses ACTUAL CSV columns dynamically.
# =========================================================

def analyze_dataset():

    df = load_data()

    if df.empty:

        return {
            "available": False,
            "message": "Dataset not available",
            "columns": [],
            "total_events": 0,
            "unique_users": 0,
            "product_views": 0,
            "add_to_cart": 0,
            "checkout": 0,
            "purchases": 0,
            "rows": []
        }

    original_columns = [
        str(c)
        for c in df.columns
    ]

    # -----------------------------------------------------
    # IDENTIFY IMPORTANT COLUMNS
    # -----------------------------------------------------

    user_col = find_column(
        df,
        [
            "user_id",
            "userid",
            "user id",
            "customer_id",
            "customerid"
        ]
    )

    event_col = find_column(
        df,
        [
            "event",
            "event_type",
            "event type",
            "action",
            "current_page",
            "page"
        ]
    )

    product_id_col = find_column(
        df,
        [
            "product_id",
            "product id",
            "productid",
            "item_id",
            "sku"
        ]
    )

    product_col = find_column(
        df,
        [
            "product",
            "product_name",
            "product name",
            "product_viewed",
            "item",
            "item_name"
        ]
    )

    device_col = find_column(
        df,
        [
            "device",
            "device_type",
            "device type"
        ]
    )

    location_col = find_column(
        df,
        [
            "location",
            "city",
            "country",
            "region"
        ]
    )

    traffic_col = find_column(
        df,
        [
            "traffic_source",
            "traffic source",
            "source",
            "channel",
            "marketing_channel"
        ]
    )

    # -----------------------------------------------------
    # TIMESTAMP
    # -----------------------------------------------------

    timestamps = build_timestamp(df)

    # -----------------------------------------------------
    # EVENT COUNTS
    # -----------------------------------------------------

    if event_col:

        normalized_events = (
            df[event_col]
            .apply(normalize_event)
        )

    else:

        normalized_events = pd.Series(
            [""] * len(df),
            index=df.index
        )

    total_events = len(df)

    unique_users = 0

    if user_col:

        unique_users = int(
            df[user_col]
            .dropna()
            .astype(str)
            .nunique()
        )

    product_views = int(
        (
            normalized_events
            == "product_view"
        ).sum()
    )

    add_to_cart = int(
        (
            normalized_events
            == "add_to_cart"
        ).sum()
    )

    checkout = int(
        (
            normalized_events
            == "checkout"
        ).sum()
    )

    purchases = int(
        (
            normalized_events
            == "purchase"
        ).sum()
    )

    visits = int(
        (
            normalized_events
            == "visit"
        ).sum()
    )

    view_cart = int(
        (
            normalized_events
            == "view_cart"
        ).sum()
    )

    # -----------------------------------------------------
    # USER IDs
    # -----------------------------------------------------

    if user_col:

        user_values = (
            df[user_col]
            .fillna("")
            .astype(str)
        )

    else:

        user_values = pd.Series(
            [""] * len(df),
            index=df.index
        )

    # -----------------------------------------------------
    # DROP-OFF ANALYSIS
    # -----------------------------------------------------

    stage_counts = {
        "Visit": visits,
        "Product View": product_views,
        "Add to Cart": add_to_cart,
        "View Cart": view_cart,
        "Checkout": checkout,
        "Purchase": purchases
    }

    # Sequential user-based stage analysis
    user_stage_sets = {}

    if user_col:

        for stage_name, stage_key in [
            ("Visit", "visit"),
            ("Product View", "product_view"),
            ("Add to Cart", "add_to_cart"),
            ("View Cart", "view_cart"),
            ("Checkout", "checkout"),
            ("Purchase", "purchase")
        ]:

            mask = (
                normalized_events
                == stage_key
            )

            user_stage_sets[
                stage_name
            ] = set(
                user_values[mask]
                .dropna()
                .astype(str)
            )

    else:

        user_stage_sets = {
            key: set()
            for key in stage_counts
        }

    visit_users = len(
        user_stage_sets["Visit"]
    )

    product_users = len(
        user_stage_sets["Product View"]
    )

    cart_users = len(
        user_stage_sets["Add to Cart"]
    )

    checkout_users = len(
        user_stage_sets["Checkout"]
    )

    purchase_users = len(
        user_stage_sets["Purchase"]
    )

    # If CSV does not have Visit event,
    # use all unique users as visit base.
    if visit_users == 0:
        visit_users = unique_users

    # -----------------------------------------------------
    # STOPPED AFTER EACH STAGE
    # -----------------------------------------------------

    stopped_after_visit = max(
        visit_users - product_users,
        0
    )

    stopped_after_product = max(
        product_users - cart_users,
        0
    )

    stopped_after_cart = max(
        cart_users - checkout_users,
        0
    )

    stopped_after_checkout = max(
        checkout_users - purchase_users,
        0
    )

    # -----------------------------------------------------
    # DROP-OFF PERCENTAGES
    # -----------------------------------------------------

    def percentage(part, whole):

        if whole <= 0:
            return 0.0

        return round(
            part / whole * 100,
            2
        )

    dropoff_analysis = {

        "visit_to_product": percentage(
            stopped_after_visit,
            visit_users
        ),

        "product_to_cart": percentage(
            stopped_after_product,
            product_users
        ),

        "cart_to_checkout": percentage(
            stopped_after_cart,
            cart_users
        ),

        "checkout_to_purchase": percentage(
            stopped_after_checkout,
            checkout_users
        )
    }

    biggest_dropoff = "—"

    if dropoff_analysis:

        biggest_key = max(
            dropoff_analysis,
            key=dropoff_analysis.get
        )

        labels = {

            "visit_to_product":
                "Visit → Product View",

            "product_to_cart":
                "Product View → Add to Cart",

            "cart_to_checkout":
                "Add to Cart → Checkout",

            "checkout_to_purchase":
                "Checkout → Purchase"
        }

        biggest_dropoff = (
            labels[biggest_key]
            + " "
            + str(
                dropoff_analysis[
                    biggest_key
                ]
            )
            + "%"
        )

    # -----------------------------------------------------
    # TRAFFIC ANALYSIS
    # ONLY ACTUAL CSV VALUES
    # -----------------------------------------------------

    traffic_analysis = {}

    if traffic_col:

        values = (
            df[traffic_col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        values = values[
            values != ""
        ]

        traffic_analysis = {
            str(k): int(v)
            for k, v in
            values.value_counts().items()
        }

    # -----------------------------------------------------
    # DEVICE ANALYSIS
    # ONLY ACTUAL CSV VALUES
    # -----------------------------------------------------

    device_analysis = {}

    if device_col:

        values = (
            df[device_col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        values = values[
            values != ""
        ]

        device_analysis = {
            str(k): int(v)
            for k, v in
            values.value_counts().items()
        }

    # -----------------------------------------------------
    # LOCATION ANALYSIS
    # -----------------------------------------------------

    location_analysis = {}

    if location_col:

        values = (
            df[location_col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        values = values[
            values != ""
        ]

        location_analysis = {
            str(k): int(v)
            for k, v in
            values.value_counts().head(20).items()
        }

    # -----------------------------------------------------
    # PRODUCT ANALYSIS
    # -----------------------------------------------------

    product_analysis = {}

    product_field = (
        product_col
        if product_col
        else product_id_col
    )

    if product_field:

        values = (
            df[product_field]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        values = values[
            values != ""
        ]

        product_analysis = {
            str(k): int(v)
            for k, v in
            values.value_counts().head(30).items()
        }

    # -----------------------------------------------------
    # TIME ANALYSIS
    # -----------------------------------------------------

    events_by_date = {}
    events_by_hour = {}

    peak_activity_time = "—"
    peak_purchase_time = "—"

    dropoffs_by_time = {}
    purchases_by_time = {}

    if timestamps.notna().any():

        valid_ts = timestamps[
            timestamps.notna()
        ]

        date_counts = (
            valid_ts
            .dt.strftime("%Y-%m-%d")
            .value_counts()
            .sort_index()
        )

        events_by_date = {
            str(k): int(v)
            for k, v in
            date_counts.items()
        }

        hour_counts = (
            valid_ts
            .dt.hour
            .value_counts()
            .sort_index()
        )

        events_by_hour = {
            f"{int(k):02d}:00":
            int(v)
            for k, v in
            hour_counts.items()
        }

        if not hour_counts.empty:

            peak_hour = int(
                hour_counts.idxmax()
            )

            peak_activity_time = (
                f"{peak_hour:02d}:00 - "
                f"{peak_hour:02d}:59"
            )

        purchase_mask = (
            normalized_events
            == "purchase"
        )

        purchase_ts = timestamps[
            purchase_mask
        ]

        purchase_ts = purchase_ts[
            purchase_ts.notna()
        ]

        if not purchase_ts.empty:

            purchase_hours = (
                purchase_ts
                .dt.hour
                .value_counts()
                .sort_index()
            )

            peak_purchase_hour = int(
                purchase_hours.idxmax()
            )

            peak_purchase_time = (
                f"{peak_purchase_hour:02d}:00 - "
                f"{peak_purchase_hour:02d}:59"
            )

            purchases_by_time = {
                f"{int(k):02d}:00":
                int(v)
                for k, v in
                purchase_hours.items()
            }

        # Hourly drop-off approximation
        dropoff_mask = (
            normalized_events.isin(
                [
                    "visit",
                    "product_view",
                    "add_to_cart",
                    "view_cart",
                    "checkout"
                ]
            )
        )

        dropoff_ts = timestamps[
            dropoff_mask
        ]

        dropoff_ts = dropoff_ts[
            dropoff_ts.notna()
        ]

        if not dropoff_ts.empty:

            drop_hours = (
                dropoff_ts
                .dt.hour
                .value_counts()
                .sort_index()
            )

            dropoffs_by_time = {
                f"{int(k):02d}:00":
                int(v)
                for k, v in
                drop_hours.items()
            }

    else:

        events_by_hour = {}

    # -----------------------------------------------------
    # DETAILED ACTUAL DATASET ROWS
    # -----------------------------------------------------

    detailed_rows = []

    display_df = df.copy()

    display_df["_dashboard_timestamp"] = timestamps

    for index, row in display_df.iterrows():

        record = {}

        # Keep every ORIGINAL CSV column.
        for column in original_columns:

            value = row.get(column, "")

            if pd.isna(value):
                value = ""

            elif hasattr(value, "item"):

                try:
                    value = value.item()
                except Exception:
                    value = str(value)

            record[str(column)] = value

        # Add normalized dashboard fields.
        user_value = (
            row[user_col]
            if user_col
            else ""
        )

        event_value = (
            row[event_col]
            if event_col
            else ""
        )

        product_id_value = (
            row[product_id_col]
            if product_id_col
            else ""
        )

        product_value = (
            row[product_col]
            if product_col
            else ""
        )

        device_value = (
            row[device_col]
            if device_col
            else ""
        )

        location_value = (
            row[location_col]
            if location_col
            else ""
        )

        traffic_value = (
            row[traffic_col]
            if traffic_col
            else ""
        )

        timestamp_value = (
            timestamps.loc[index]
            if index in timestamps.index
            else pd.NaT
        )

        if pd.notna(timestamp_value):

            timestamp_text = (
                timestamp_value.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

        else:

            timestamp_text = ""

        record[
            "_dashboard_user_id"
        ] = clean_value(user_value)

        record[
            "_dashboard_event"
        ] = event_display(event_value)

        record[
            "_dashboard_product_id"
        ] = clean_value(product_id_value)

        record[
            "_dashboard_product"
        ] = clean_value(product_value)

        record[
            "_dashboard_device"
        ] = clean_value(device_value)

        record[
            "_dashboard_location"
        ] = clean_value(location_value)

        record[
            "_dashboard_traffic_source"
        ] = clean_value(traffic_value)

        record[
            "_dashboard_timestamp"
        ] = timestamp_text

        detailed_rows.append(record)

    # newest first
    detailed_rows.reverse()

    # -----------------------------------------------------
    # DATASET SUMMARY
    # -----------------------------------------------------

    return {

        "available": True,

        "columns": original_columns,

        "column_mapping": {

            "user_id": user_col,

            "event": event_col,

            "product_id":
                product_id_col,

            "product":
                product_col,

            "device":
                device_col,

            "location":
                location_col,

            "traffic_source":
                traffic_col,

            "timestamp":
                get_timestamp_column(df),

            "date":
                get_date_column(df),

            "time":
                get_time_column(df)
        },

        # Overview metrics
        "total_events": total_events,

        "unique_users": unique_users,

        "product_views": product_views,

        "add_to_cart": add_to_cart,

        "checkout": checkout,

        "purchases": purchases,

        # Extra stages
        "visits": visits,

        "view_cart": view_cart,

        # Drop-off
        "stopped_after_visit":
            stopped_after_visit,

        "stopped_after_product_view":
            stopped_after_product,

        "stopped_after_add_to_cart":
            stopped_after_cart,

        "checkout_without_purchase":
            stopped_after_checkout,

        "dropoff_percentages":
            dropoff_analysis,

        "biggest_dropoff":
            biggest_dropoff,

        # Traffic
        "traffic_analysis":
            traffic_analysis,

        # Device
        "device_analysis":
            device_analysis,

        # Location
        "location_analysis":
            location_analysis,

        # Product
        "product_analysis":
            product_analysis,

        # Time
        "events_by_date":
            events_by_date,

        "events_by_hour":
            events_by_hour,

        "peak_activity_time":
            peak_activity_time,

        "dropoffs_by_time":
            dropoffs_by_time,

        "purchases_by_time":
            purchases_by_time,

        "peak_purchase_time":
            peak_purchase_time,

        # Actual dataset
        "rows":
            detailed_rows
    }


# =========================================================
# MAIN DASHBOARD
# =========================================================

@app.route("/")
@app.route("/dashboard")
def dashboard():

    df = load_data()

    total = len(df)

    return render_template(
        "dashboard.html",
        site="ShopFlow",
        page="Overview",
        total=total,
        metrics=load_metrics()
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
# HISTORICAL API
# =========================================================

@app.route("/api/historical")
def historical():

    analysis = analyze_dataset()

    return jsonify({

        "total":
            analysis["total_events"],

        "unique_users":
            analysis["unique_users"],

        "product_views":
            analysis["product_views"],

        "add_to_cart":
            analysis["add_to_cart"],

        "checkout":
            analysis["checkout"],

        "purchases":
            analysis["purchases"],

        "visits":
            analysis["visits"],

        "dropoff_rate":
            analysis[
                "dropoff_percentages"
            ].get(
                "checkout_to_purchase",
                0
            ),

        "biggest_dropoff":
            analysis["biggest_dropoff"]
    })


# =========================================================
# OVERVIEW DATA API
# =========================================================

@app.route("/api/overview")
def api_overview():

    analysis = analyze_dataset()

    return jsonify(analysis)


# =========================================================
# DETAILED DATASET API
#
# This is what your dashboard can use for the final
# "Detailed Dataset Analysis" section.
# =========================================================

@app.route("/api/ecommerce-dataset-detailed")
def ecommerce_dataset_detailed():

    analysis = analyze_dataset()

    return jsonify(analysis)


# =========================================================
# DATASET COLUMNS API
# =========================================================

@app.route("/api/dataset-columns")
def dataset_columns():

    analysis = analyze_dataset()

    return jsonify({

        "columns":
            analysis.get(
                "columns",
                []
            ),

        "mapping":
            analysis.get(
                "column_mapping",
                {}
            )
    })


# =========================================================
# LIVE EVENTS API
# =========================================================

@app.route("/api/live")
def api_live():

    all_events = load_live_events()

    if all_events.empty:

        return jsonify({
            "events": [],
            "count": 0
        })

    all_events[
        "timestamp"
    ] = pd.to_datetime(
        all_events[
            "timestamp"
        ],
        errors="coerce"
    )

    all_events = all_events.sort_values(
        ["timestamp", "id"],
        ascending=[False, False]
    )

    records = []

    # -----------------------------------------------------
    # Latest state for every user
    # -----------------------------------------------------

    for user_id, user_events in (
        all_events.groupby(
            "user_id",
            sort=False
        )
    ):

        user_events = user_events.sort_values(
            ["timestamp", "id"],
            ascending=[False, False]
        )

        latest = user_events.iloc[
            0
        ].copy()

        # Purchase history
        purchase_events = user_events[
            user_events[
                "current_page"
            ]
            .astype(str)
            .apply(normalize_event)
            .eq("purchase")
        ]

        purchase_count = len(
            purchase_events
        )

        has_purchased = (
            purchase_count > 0
        )

        latest_purchase_time = ""

        if has_purchased:

            purchase_time = (
                purchase_events.iloc[
                    0
                ]["timestamp"]
            )

            if pd.notna(
                purchase_time
            ):

                latest_purchase_time = (
                    purchase_time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

        # Product from current event
        product = str(
            latest.get(
                "product",
                ""
            )
            or ""
        )

        product_id = str(
            latest.get(
                "product_id",
                ""
            )
            or ""
        )

        # Format record
        record = latest.to_dict()

        record[
            "user_id"
        ] = (
            f"U{int(user_id)}"
            if str(user_id).isdigit()
            else str(user_id)
        )

        record[
            "event"
        ] = event_display(
            latest.get(
                "event_type",
                latest.get(
                    "current_page",
                    ""
                )
            )
        )

        record[
            "product_id"
        ] = product_id

        record[
            "product"
        ] = product

        record[
            "has_purchased"
        ] = has_purchased

        record[
            "purchase_count"
        ] = purchase_count

        record[
            "last_purchase_time"
        ] = latest_purchase_time

        timestamp = latest.get(
            "timestamp"
        )

        if pd.notna(timestamp):

            record[
                "timestamp"
            ] = timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        else:

            record[
                "timestamp"
            ] = ""

        for key, value in list(
            record.items()
        ):

            try:

                if pd.isna(value):

                    record[key] = ""

                elif hasattr(
                    value,
                    "item"
                ):

                    record[key] = (
                        value.item()
                    )

            except Exception:

                record[key] = str(value)

        records.append(record)

    records.sort(
        key=lambda x:
            str(
                x.get(
                    "timestamp",
                    ""
                )
            ),
        reverse=True
    )

    records = records[:50]

    return jsonify({
        "events": records,
        "count": len(records)
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

    probability, risk = (
        calculate_prediction(

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
                "current_page",
                "Home"
            )
        )
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
# LIVE ECOMMERCE EVENT API
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
    # PAGE / EVENT
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
    # USER DATA
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
    # ML
    # -----------------------------------------------------

    probability, risk = (
        calculate_prediction(

            age,

            pages_visited,

            session_duration,

            clicks,

            previous_visits,

            current_page
        )
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
    # SAVE
    # -----------------------------------------------------

    live_saved = save_live_event(
        user,
        data
    )

    funnel_saved = save_event_to_funnel(
        data,
        user
    )

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return jsonify({

        "ok": True,

        "live_saved":
            live_saved,

        "funnel_saved":
            funnel_saved,

        "user":
            user,

        "percentage":
            round(
                probability * 100,
                2
            )
    })


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    mysql_status = False

    connection = None

    try:

        connection = (
            get_mysql_connection()
        )

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

    "csv": DATA_FILE.exists(),

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
        "CSV:",
        DATA_FILE
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