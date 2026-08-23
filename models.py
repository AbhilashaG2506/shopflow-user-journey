from database import db
from datetime import datetime


# ============================================================
# USER
# ============================================================

class User(db.Model):

    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    user_id = db.Column(
        db.String(50),
        unique=True,
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


# ============================================================
# PRODUCT
# ============================================================

class Product(db.Model):

    __tablename__ = "products"

    id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    product_id = db.Column(
        db.String(50),
        unique=True,
        nullable=False
    )

    product_name = db.Column(
        db.String(100),
        nullable=False
    )

    category = db.Column(
        db.String(50)
    )

    price = db.Column(
        db.Float,
        nullable=False
    )


# ============================================================
# EVENT
# ============================================================

class Event(db.Model):

    __tablename__ = "events"

    id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    user_id = db.Column(
        db.String(50),
        nullable=False
    )

    timestamp = db.Column(
        db.DateTime,
        nullable=False
    )

    event_type = db.Column(
        db.String(30),
        nullable=False
    )

    device = db.Column(
        db.String(20),
        nullable=False
    )

    location = db.Column(
        db.String(30),
        nullable=False
    )

    traffic_source = db.Column(
        db.String(30),
        nullable=False
    )

    product_id = db.Column(
        db.String(50),
        nullable=True
    )


# ============================================================
# ORDER
# ============================================================

class Order(db.Model):

    __tablename__ = "orders"

    id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    user_id = db.Column(
        db.String(50),
        nullable=False
    )

    product_id = db.Column(
        db.String(50),
        nullable=False
    )

    amount = db.Column(
        db.Float,
        nullable=False
    )

    timestamp = db.Column(
        db.DateTime,
        nullable=False
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

class Prediction(db.Model):

    __tablename__ = "predictions"

    id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    user_id = db.Column(
        db.String(50),
        nullable=False
    )

    device = db.Column(
        db.String(20),
        nullable=False
    )

    location = db.Column(
        db.String(30),
        nullable=False
    )

    traffic_source = db.Column(
        db.String(30),
        nullable=False
    )

    current_stage = db.Column(
        db.String(30),
        nullable=False
    )

    event_number = db.Column(
        db.Integer,
        nullable=False
    )

    session_duration_seconds = db.Column(
        db.Float,
        nullable=False
    )

    dropoff_probability = db.Column(
        db.Float,
        nullable=False
    )

    purchase_probability = db.Column(
        db.Float,
        nullable=False
    )

    risk_level = db.Column(
        db.String(20),
        nullable=False
    )

    recommendation = db.Column(
        db.Text,
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )