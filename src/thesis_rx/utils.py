"""
utils.py — shared helpers
"""
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


def jaccard_distance(set_col_a, set_col_b):
    """
    Jaccard distance between two array<long> columns.
    Returns 0.0 when both sets are empty (no change = no distance).
    Returns 1.0 when one is empty and the other is not (total change).
    Handles null inputs safely.
    """
    a = F.coalesce(set_col_a, F.array().cast("array<long>"))
    b = F.coalesce(set_col_b, F.array().cast("array<long>"))

    intersection_size = F.size(F.array_intersect(a, b)).cast(DoubleType())
    union_size        = F.size(F.array_union(a, b)).cast(DoubleType())

    return F.when(
        union_size == 0, F.lit(0.0)          # both empty → no distance
    ).otherwise(
        F.lit(1.0) - (intersection_size / union_size)
    )
