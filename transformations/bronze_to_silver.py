import dlt
from pyspark.sql.functions import (
    col, when, lit, round as spark_round,
    to_timestamp
)

catalog = "workspace"
bronze_schema = "bronze"
silver_schema = "silver"

# ── Table 1: silver_churn_clean ──

@dlt.table(
    name=f"{catalog}.{silver_schema}.silver_churn_clean",
    comment="Cleaned bank customers — cast types, nulls dropped",
    table_properties={"quality": "silver"}
)

@dlt.expect_all_or_drop({
    "valid_customer_id": "customer_id IS NOT NULL",
    "valid_credit_score": "credit_score BETWEEN 300 AND 850",
    "valid_age": "age BETWEEN 18 AND 100",
    "valid_tenure": "tenure BETWEEN 0 AND 10",
    "valid_churn": "churn IN (0, 1)"
})

def silver_churn_clean():
    return (
        dlt.read(f"{catalog}.{bronze_schema}.bronze_churn_raw")
        .select(
            col("customer_id").cast("long"),
            col("credit_score").cast("integer"),
            col("country").cast("string"),
            col("gender").cast("string"),
            col("age").cast("integer"),
            col("tenure").cast("integer"),
            col("balance").cast("double"),
            col("products_number").cast("integer"),
            col("credit_card").cast("integer"),
            col("active_member").cast("integer"),
            col("estimated_salary").cast("double"),
            col("churn").cast("integer"),
            col("created_at").cast("timestamp")
        )
    )

# ── Table 2: silver_churn_features ──

@dlt.table(
    name=f"{catalog}.{silver_schema}.silver_churn_features",
    comment="Engineered features ready for ML training and inference",
    table_properties={"quality": "silver"}
)

def silver_churn_features():
    df = dlt.read(f"{catalog}.{silver_schema}.silver_churn_clean")

    return (
        df
        # ── Encode Geography ──
        .withColumn("is_germany",
            when(col("country") == "Germany", 1).otherwise(0))
        .withColumn("is_spain",
            when(col("country") == "Spain", 1).otherwise(0))
        # France is the baseline — no column needed to avoid multicollinearity

        # ── Encode Gender ──
        .withColumn("is_male",
            when(col("gender") == "Male", 1).otherwise(0))

        # ── Derived features ──
        .withColumn("is_zero_balance",
            when(col("balance") == 0, 1).otherwise(0))

        .withColumn("balance_per_product",
            when(col("products_number") > 0,
                spark_round(col("balance") / col("products_number"), 2)
            ).otherwise(0))

        .withColumn("is_inactive_with_balance",
            when(
                (col("active_member") == 0) & (col("balance") > 0), 1
            ).otherwise(0))

        .withColumn("tenure_ratio",
            when(col("age") > 0,
                spark_round(col("tenure") / col("age"), 4)
            ).otherwise(0))

        .withColumn("age_group",
            when(col("age") < 35, "young")
            .when(col("age") < 55, "middle")
            .otherwise("senior"))

        # ── Encode age_group ──
        .withColumn("is_young",
            when(col("age_group") == "young", 1).otherwise(0))
        .withColumn("is_senior",
            when(col("age_group") == "senior", 1).otherwise(0))
        # middle is the baseline

        # ── Select final columns ──
        # keep customer_id and churn for joining and labelling
        # drop raw categorical columns replaced by encoded versions
        .select(
            "customer_id",
            "credit_score",
            "age",
            "tenure",
            "balance",
            "products_number",
            "credit_card",
            "active_member",
            "estimated_salary",
            "is_germany",
            "is_spain",
            "is_male",
            "is_zero_balance",
            "balance_per_product",
            "is_inactive_with_balance",
            "tenure_ratio",
            "is_young",
            "is_senior",
            "churn",
            "created_at"
        )
    )