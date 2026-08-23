import os
import sys

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


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
# FILE PATHS
# ============================================================

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "ml",
    "prediction_dataset.csv"
)

MODEL_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "ml",
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIRECTORY,
    "dropoff_model.pkl"
)


# Create model directory if it does not exist
os.makedirs(
    MODEL_DIRECTORY,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("\n==========================================")
print("TRAINING USER JOURNEY PREDICTION MODEL")
print("==========================================\n")


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATA_FILE):

    print("ERROR: Prediction dataset not found.")

    print(
        "\nExpected file:"
    )

    print(
        DATA_FILE
    )

    print(
        "\nRun this first:"
    )

    print(
        "python ml/create_ml_dataset.py"
    )

    sys.exit(1)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATA_FILE
)


print(
    "Prediction dataset loaded successfully."
)

print(
    "Rows:",
    len(df)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

features = [

    "device",

    "location",

    "traffic_source",

    "visit_count",

    "signup_count",

    "cart_count",

    "checkout_count",

    "current_stage",

    "session_duration_seconds",

    "event_number"

]


target = "eventually_purchased"


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = features + [target]

missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    print(
        "\nERROR: Missing columns:"
    )

    print(
        missing_columns
    )

    sys.exit(1)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=required_columns
).copy()


print(
    "Rows after cleaning:",
    len(df)
)


# ============================================================
# FEATURES
# ============================================================

X = df[features]


# ============================================================
# TARGET
# ============================================================

# 1 = Eventually Purchased
# 0 = Eventually Dropped Off

y = df[target]


# ============================================================
# TARGET CHECK
# ============================================================

print(
    "\nTarget distribution:"
)

print(
    y.value_counts()
)


print(
    "\nTarget percentage:"
)

print(
    y.value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = [

    "device",

    "location",

    "traffic_source",

    "current_stage"

]


numerical_features = [

    "visit_count",

    "signup_count",

    "cart_count",

    "checkout_count",

    "session_duration_seconds",

    "event_number"

]


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numerical",

            "passthrough",

            numerical_features
        )

    ]

)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=14,

    min_samples_leaf=4,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1

)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",

            preprocessor
        ),

        (
            "model",

            model
        )

    ]

)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "\nTraining rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# ============================================================
# TRAIN MODEL
# ============================================================

print(
    "\nTraining Random Forest model..."
)


pipeline.fit(

    X_train,

    y_train

)


print(
    "Model training completed."
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = pipeline.predict(

    X_test

)


y_probability = pipeline.predict_proba(

    X_test

)[:, 1]


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


# ============================================================
# ROC-AUC
# ============================================================

auc = roc_auc_score(

    y_test,

    y_probability

)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,

    y_pred,

    target_names=[

        "Dropped Off",

        "Purchased"

    ]

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(

    y_test,

    y_pred

)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

print(
    "\n=========================================="
)

print(
    "MODEL PERFORMANCE"
)

print(
    "=========================================="
)


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print(
    f"ROC-AUC: {auc:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    report
)


print(
    "Confusion Matrix:"
)

print(
    matrix
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    pipeline,

    MODEL_FILE

)


# ============================================================
# MODEL SAVED
# ============================================================

print(
    "\n=========================================="
)

print(
    "MODEL SAVED"
)

print(
    "=========================================="
)


print(
    "\nSaved to:"
)

print(
    MODEL_FILE
)


print(
    "\n=========================================="
)

print(
    "ML TRAINING COMPLETED"
)

print(
    "==========================================\n"
)