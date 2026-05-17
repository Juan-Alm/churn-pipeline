import subprocess
subprocess.run(["pip", "install", "xgboost"], check=True, capture_output=True)

import dlt
import json
import pickle
import pandas as pd
from pyspark.sql.functions import pandas_udf, col, when, current_timestamp, struct
from pyspark.sql.types import DoubleType, IntegerType


catalog = "workspace"
silver_schema = "silver"
gold_schema = "gold"

MODEL_PATH = "/Volumes/main/default/churn_vol/models/churn_model.pkl"
MAPPING_PATH = "/Volumes/main/default/churn_vol/models/label_mapping.json"

# ── Load feature column order from mapping ──
# read outside UDF so column order is consistent with training
with open(MAPPING_PATH, "r") as f:
    mapping = json.load(f)

feature_cols = mapping["feature_cols"]

# ── Prediction UDF ──

@pandas_udf("struct<churn_probability:double, churned_predicted:integer>")
def predict_churn_udf(batch: pd.DataFrame) -> pd.DataFrame:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # ensure column order matches training
    batch = batch[feature_cols]

    proba = model.predict_proba(batch)[:, 1]
    predicted = (proba >= 0.5).astype(int)

    return pd.DataFrame({
        "churn_probability": proba.round(4),
        "churned_predicted": predicted.tolist()
    })

# ── churn_predictions DLT table ──

@dlt.table(
    name=f"{catalog}.{gold_schema}.churn_predictions",
    comment="Churn predictions with risk segmentation — updated daily",
    table_properties={"quality": "gold"}
)

def churn_predictions():
    df = dlt.read(f"{catalog}.{silver_schema}.silver_churn_features")

    # run prediction UDF on feature columns only
    predictions = df.withColumn(
        "prediction",
        predict_churn_udf(struct(*[col(c) for c in feature_cols]))
    )

    return (
        predictions
        .withColumn("churn_probability",
            col("prediction.churn_probability"))
        .withColumn("churned_predicted",
            col("prediction.churned_predicted"))

        # ── Risk segmentation ──
        .withColumn("churn_risk_segment",
            when(col("churn_probability") < 0.3, "low risk")
            .when(col("churn_probability") < 0.6, "medium risk")
            .otherwise("high risk")
        )

        # ── Metadata ──
        .withColumn("prediction_date", current_timestamp())

        # ── Final column selection ──
        .select(
            "customer_id",
            "age",
            "tenure",
            "balance",
            "products_number",
            "active_member",
            "is_germany",
            "is_spain",
            "is_inactive_with_balance",
            "balance_per_product",
            "churn_probability",
            "churned_predicted",
            "churn_risk_segment",
            "churn",              # actual label — useful for monitoring drift
            "prediction_date"
        )
    )