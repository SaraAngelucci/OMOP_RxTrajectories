from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("turnover_threshold_validation") \
    .master("local[*]") \
    .getOrCreate()

# Use the already-computed person-month summary — no recalculation needed
pm = spark.read.parquet(
    "outputs/synthetic_50k/baseline_cohort_person_months.parquet"
)

# How much of the data is zero-turnover (stable)?
total     = pm.count()
zero_t    = pm.filter(F.col("turnover") == 0.0).count()
nonzero_t = pm.filter(F.col("turnover") >  0.0).count()

print(f"Total person-months:          {total:,}")
print(f"Zero turnover (T=0):          {zero_t:,}  ({100*zero_t/total:.1f}%)")
print(f"Non-zero turnover:            {nonzero_t:,}  ({100*nonzero_t/total:.1f}%)")

# Distribution of non-zero turnover
print("\nDistribution of non-zero T_pt values:")
pm.filter(F.col("turnover") > 0.0) \
  .select("turnover") \
  .describe() \
  .show()

# Key percentiles of all non-zero transitions
quantiles = pm.filter(F.col("turnover") > 0.0) \
              .stat.approxQuantile("turnover",
                                   [0.10, 0.25, 0.50, 0.75, 0.90],
                                   0.005)

labels = ["P10", "P25 (Q1)", "P50 (median)", "P75 (Q3)", "P90"]
for label, q in zip(labels, quantiles):
    print(f"  {label}: {q:.4f}")

# Proportion of non-zero months classified as low / moderate / high
# under the current thresholds
low_t  = pm.filter((F.col("turnover") > 0.0) &
                   (F.col("turnover") <  0.20)).count()
mid_t  = pm.filter((F.col("turnover") >= 0.20) &
                   (F.col("turnover") <  0.50)).count()
high_t = pm.filter( F.col("turnover") >= 0.50 ).count()

print(f"\nAmong non-zero transitions (n={nonzero_t:,}):")
print(f"  Low   (T < 0.20):          {low_t:,}  ({100*low_t/nonzero_t:.1f}%)")
print(f"  Moderate (0.20 <= T < 0.50): {mid_t:,}  ({100*mid_t/nonzero_t:.1f}%)")
print(f"  High  (T >= 0.50):         {high_t:,}  ({100*high_t/nonzero_t:.1f}%)")

# What does T < 0.20 actually look like? Show distinct values below 0.20
print("\nDistinct T_pt values below 0.20 (should be sparse — only T=0 excluded):")
pm.filter((F.col("turnover") > 0.0) & (F.col("turnover") < 0.20)) \
  .groupBy("turnover") \
  .count() \
  .orderBy("turnover") \
  .show(20)