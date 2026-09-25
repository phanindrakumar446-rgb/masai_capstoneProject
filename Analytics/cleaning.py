"""
Shared cleaning helpers for the analytics module.

01_eda.py applies the full EDA cleaning (row drops + column decisions + age imputation).
02_modeling.py re-applies only the *row/column-level* decisions (which learn nothing from the data),
and leaves age imputation to the train-only SimpleImputer inside its sklearn Pipeline, so no
statistic from the test split leaks into training.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "titanic.csv"
CHART_DIR = HERE / "charts"

# Palette (validated reference palette): identity follows the entity, not the rank.
SURVIVED_COLOR = "#2a78d6"   # blue   -> survived
DIED_COLOR = "#eb6834"       # orange -> did not survive
NEUTRAL = "#f0efec"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"

# Columns with < 5% missing -> drop the affected rows (see README / 01_eda.py output for the %).
DROP_ROW_COLUMNS = ["embarked", "embark_town"]
# Column with > 30% missing -> imputation unreliable -> keep it but encode "Missing" as its own category.
MISSING_AS_CATEGORY = "deck"


def row_level_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Decisions that are deterministic per row/column (nothing is *learned* from the data)."""
    out = df.dropna(subset=DROP_ROW_COLUMNS).copy()
    out[MISSING_AS_CATEGORY] = out[MISSING_AS_CATEGORY].astype("object").fillna("Missing")
    return out.reset_index(drop=True)


def eda_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full EDA cleaning: row-level rules + age imputation (5-30% missing -> impute)."""
    out = row_level_clean(df)
    # age: impute with the median age of the passenger's (sex, pclass) group - age differs a lot
    # between e.g. 1st-class women and 3rd-class men, so a group median is more faithful than one global median.
    out["age"] = out.groupby(["sex", "pclass"])["age"].transform(lambda s: s.fillna(s.median()))
    return out


def apply_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "font.size": 10,
            "savefig.dpi": 120,
            "savefig.bbox": "tight",
        }
    )
