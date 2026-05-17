# ── Export silver_churn_features to Volume as Parquet ──

export_path = "/Volumes/main/default/churn_vol/exports/silver_churn_features"

(spark.read.table("workspace.silver.silver_churn_features")
    .drop("created_at")   # not needed for training
    .write
    .mode("overwrite")
    .parquet(export_path)
)

print(f"Export complete: {export_path}")

# ── Verify ──

exported = spark.read.parquet(export_path)
print(f"Rows exported:   {exported.count()}")
print(f"Columns:         {exported.columns}")