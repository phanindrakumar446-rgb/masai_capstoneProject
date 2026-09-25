"""
Module 2 / Part A - Profiling, cleaning and the data story (Titanic).

    python analytics/01_eda.py

This is the ONLY place the raw dataset is loaded (sns.load_dataset('titanic')). It is saved immediately
to analytics/titanic.csv, which 02_modeling.py reads - the network is never hit a second time.
Charts are written to analytics/charts/; written interpretations live in analytics/README.md.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler

from cleaning import (
    CHART_DIR, CSV_PATH, DIED_COLOR, INK_2, MUTED, NEUTRAL, SURVIVED_COLOR,
    apply_style, eda_clean,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
CHART_DIR.mkdir(exist_ok=True)
apply_style()


def section(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


def save(fig, name):
    fig.savefig(CHART_DIR / name)
    plt.close(fig)
    print(f"[chart saved] charts/{name}")


# ======================================================================================
# Task 1 - Load ONCE, save the offline fallback immediately, profile
# ======================================================================================
section("TASK 1 - Load & profile")
try:
    df = sns.load_dataset("titanic")          # the one and only network/cache load
    print("Loaded via sns.load_dataset('titanic')")
except Exception as exc:                      # offline grading fallback
    print(f"sns.load_dataset failed ({exc}); falling back to committed titanic.csv")
    df = pd.read_csv(CSV_PATH)
df.to_csv(CSV_PATH, index=False)              # committed offline fallback, used by 02_modeling.py
print(f"Saved raw DataFrame -> {CSV_PATH.name}")

print(f"\ndf.shape = {df.shape}\n")
df.info()
print("\ndf.describe():\n", df.describe())
print("\ndf.describe(exclude='number'):\n", df.describe(exclude="number"))

missing_pct = (df.isna().mean() * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
print("\nMissing values (% of rows), columns with any missing:\n", missing_pct.to_string())

balance = df["survived"].value_counts(normalize=True).mul(100).round(2)
print(f"\nClass balance of 'survived' (%):\n{balance.to_string()}")

# ======================================================================================
# Task 2 - Missing-value handling via the threshold rule
# ======================================================================================
section("TASK 2 - Missing-value handling")
for col, pct in missing_pct.items():
    if pct < 5:
        rule = "< 5%  -> DROP affected rows"
    elif pct <= 30:
        rule = "5-30% -> IMPUTE (median age within each sex x pclass group)"
    else:
        rule = "> 30% -> too sparse to impute reliably -> encode 'Missing' as its own category"
    print(f"{col:<12} {pct:6.2f}%   {rule}")

clean = eda_clean(df)
print(f"\nShape before cleaning {df.shape} -> after cleaning {clean.shape}")
print("Remaining missing values:", int(clean.isna().sum().sum()))
print("\nDeck value counts after encoding:\n", clean["deck"].value_counts().to_string())
print("\nSurvival rate: deck known vs 'Missing' (why missing is informative):")
print(clean.assign(deck_known=clean["deck"] != "Missing").groupby("deck_known")["survived"].mean().round(3).to_string())

# ======================================================================================
# Task 3 - Univariate analysis: age & fare
# ======================================================================================
section("TASK 3 - Univariate analysis (age, fare)")
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for row, col in enumerate(["age", "fare"]):
    axes[row, 0].hist(clean[col], bins=30, color=SURVIVED_COLOR, edgecolor="#fcfcfb", linewidth=1)
    axes[row, 0].set(title=f"{col} - histogram", xlabel=col, ylabel="passengers")
    axes[row, 1].boxplot(clean[col], orientation="horizontal", widths=0.5, patch_artist=True,
                         boxprops=dict(facecolor="#b7d3f6", color=SURVIVED_COLOR),
                         medianprops=dict(color=INK_2, linewidth=2),
                         flierprops=dict(marker="o", markersize=5, markerfacecolor=DIED_COLOR,
                                         markeredgecolor="#fcfcfb", alpha=0.8))
    axes[row, 1].set(title=f"{col} - box plot (orange = IQR outliers)", xlabel=col, yticks=[])
fig.tight_layout()
save(fig, "01_univariate_age_fare.png")


def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum()), q1, q3, iqr, lo, hi


for col in ["age", "fare"]:
    n, q1, q3, iqr, lo, hi = iqr_outliers(clean[col])
    print(f"{col:<5} Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} bounds=[{lo:.2f}, {hi:.2f}] -> {n} outliers")
n_raw_age = iqr_outliers(df["age"].dropna())[0]
print(f"(for reference: age BEFORE imputation, non-null rows only -> {n_raw_age} outliers)")

fare_mean, fare_median, fare_mode = clean["fare"].mean(), clean["fare"].median(), clean["fare"].mode()[0]
print(f"\nfare: mean={fare_mean:.2f}  median={fare_median:.2f}  mode={fare_mode:.2f}  skewness={clean['fare'].skew():.2f}")
if fare_mean > fare_median > fare_mode:
    print("mean > median > mode  -> RIGHT-skewed (long tail of expensive fares)")
elif fare_mean < fare_median < fare_mode:
    print("mean < median < mode  -> LEFT-skewed")
else:
    print("mean ~ median ~ mode  -> roughly symmetric")

# ======================================================================================
# Task 4 - Bivariate analysis with boolean masks + correlation heatmap
# ======================================================================================
section("TASK 4 - Bivariate analysis")
overall = clean["survived"].mean()
print(f"Overall survival rate: {overall:.2%}\n")

print("(a) by sex")
for sex in ["female", "male"]:
    mask = clean["sex"] == sex
    print(f"   {sex:<7} n={mask.sum():>3}  survival={clean.loc[mask, 'survived'].mean():.2%}")

print("(b) by pclass")
for p in [1, 2, 3]:
    mask = clean["pclass"] == p
    print(f"   class {p}  n={mask.sum():>3}  survival={clean.loc[mask, 'survived'].mean():.2%}")

print("(c) by sex AND pclass (mask_sex & mask_class)")
for sex in ["female", "male"]:
    for p in [1, 2, 3]:
        mask = (clean["sex"] == sex) & (clean["pclass"] == p)
        print(f"   {sex:<7} class {p}  n={mask.sum():>3}  survival={clean.loc[mask, 'survived'].mean():.2%}")

mask_priority = (clean["sex"] == "female") | (clean["age"] < 16)
print(f"\n'Women OR children (<16)' (| mask): survival={clean.loc[mask_priority, 'survived'].mean():.2%} "
      f"vs everyone else {clean.loc[~mask_priority, 'survived'].mean():.2%}")

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]   # adult_male / alone excluded on purpose
corr = clean[corr_cols].corr()
print("\nCorrelation matrix (6 x 6):\n", corr.round(3))

pairs = [(corr_cols[i], corr_cols[j], corr.iloc[i, j])
         for i in range(len(corr_cols)) for j in range(i + 1, len(corr_cols))]
pairs.sort(key=lambda t: abs(t[2]), reverse=True)
print("\nOff-diagonal pairs ranked by |r|:")
for a, b, r in pairs:
    print(f"   {a:>8} ~ {b:<8} r={r:+.3f}")
print(f"\nTwo strongest: {pairs[0][0]}~{pairs[0][1]} ({pairs[0][2]:+.3f}), {pairs[1][0]}~{pairs[1][1]} ({pairs[1][2]:+.3f})")

diverging = LinearSegmentedColormap.from_list("orange_gray_blue", [DIED_COLOR, NEUTRAL, SURVIVED_COLOR])
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap=diverging, vmin=-1, vmax=1, center=0, square=True,
            linewidths=2, linecolor="#fcfcfb", cbar_kws={"label": "Pearson r"}, ax=ax)
ax.set_title("Correlation matrix - survived, pclass, age, sibsp, parch, fare")
ax.grid(False)
save(fig, "02_correlation_heatmap.png")

# ======================================================================================
# Task 5 - Multivariate data story (5 charts)
# ======================================================================================
section("TASK 5 - Multivariate data story")
story = clean.copy()
story["outcome"] = story["survived"].map({1: "Survived", 0: "Did not survive"})
outcome_palette = {"Survived": SURVIVED_COLOR, "Did not survive": DIED_COLOR}

# Chart 1: survival rate by class, split by sex
rates = story.groupby(["pclass", "sex"])["survived"].mean().unstack()
fig, ax = plt.subplots(figsize=(8, 4.8))
x = np.arange(3)
for k, (sex, color) in enumerate([("female", SURVIVED_COLOR), ("male", DIED_COLOR)]):
    bars = ax.bar(x + (k - 0.5) * 0.38, rates[sex], width=0.36, color=color, label=sex)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in rates[sex]], padding=3, color=INK_2, fontsize=9)
ax.set(xticks=x, xticklabels=["1st class", "2nd class", "3rd class"], ylim=(0, 1.08),
       ylabel="survival rate", title="Chart 1 - Survival rate by class and sex")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(title="sex")
save(fig, "03_story_survival_class_sex.png")
print("Chart 1 data:\n", rates.round(3))

# Chart 2: age distribution by outcome, per sex
fig, ax = plt.subplots(figsize=(8, 4.8))
sns.boxplot(data=story, x="sex", y="age", hue="outcome", palette=outcome_palette, width=0.6, gap=0.1,
            flierprops=dict(markersize=4), ax=ax)
ax.set_title("Chart 2 - Age distribution by sex and outcome")
ax.legend(title="")
save(fig, "04_story_age_by_sex_outcome.png")
print("Chart 2 data (median age):\n", story.groupby(["sex", "outcome"])["age"].median())

# Chart 3: survival rate by age band and sex (the 'children first' check)
story["age_band"] = pd.cut(story["age"], bins=[0, 12, 18, 30, 45, 60, 80],
                           labels=["0-12", "13-18", "19-30", "31-45", "46-60", "61-80"])
band = story.groupby(["age_band", "sex"], observed=True)["survived"].mean().unstack()
fig, ax = plt.subplots(figsize=(8, 4.8))
for sex, color, marker in [("female", SURVIVED_COLOR, "o"), ("male", DIED_COLOR, "s")]:
    ax.plot(band.index.astype(str), band[sex], color=color, marker=marker, markersize=8, linewidth=2,
            markeredgecolor="#fcfcfb", label=sex)
ax.set(ylim=(0, 1.05), xlabel="age band", ylabel="survival rate",
       title="Chart 3 - Survival rate by age band and sex")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(title="sex")
save(fig, "05_story_age_band_sex.png")
print("Chart 3 data:\n", band.round(3))

# Chart 4: fare vs age scatter coloured by outcome (log fare), faceted by class
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
for ax, p in zip(axes, [1, 2, 3]):
    sub = story[story["pclass"] == p]
    for outcome in ["Did not survive", "Survived"]:
        s = sub[sub["outcome"] == outcome]
        ax.scatter(s["age"], s["fare"] + 1, s=22, color=outcome_palette[outcome], alpha=0.7,
                   edgecolor="#fcfcfb", linewidth=0.6, label=outcome)
    ax.set(yscale="log", xlabel="age", title=f"class {p}")
axes[0].set_ylabel("fare + 1 (log scale)")
axes[0].legend(title="")
fig.suptitle("Chart 4 - Fare vs age by class, coloured by outcome", fontweight="bold")
fig.tight_layout()
save(fig, "06_story_fare_age_class.png")
print("Chart 4 data (median fare by class/outcome):\n", story.groupby(["pclass", "outcome"])["fare"].median())

# Chart 5: family size
story["family_size"] = story["sibsp"] + story["parch"] + 1
fam = story.groupby("family_size").agg(rate=("survived", "mean"), n=("survived", "size"))
fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.bar(fam.index.astype(str), fam["rate"], color=SURVIVED_COLOR, width=0.7)
ax.bar_label(bars, labels=[f"n={n}" for n in fam["n"]], padding=3, color=INK_2, fontsize=8)
ax.axhline(overall, color=MUTED, linestyle="--", linewidth=1)
ax.text(len(fam) - 0.5, overall + 0.02, f"overall {overall:.0%}", color=INK_2, ha="right", fontsize=9)
ax.set(ylim=(0, 1.0), xlabel="family size aboard (sibsp + parch + 1)", ylabel="survival rate",
       title="Chart 5 - Survival rate by family size")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
save(fig, "07_story_family_size.png")
print("Chart 5 data:\n", fam.round(3))

# ======================================================================================
# Task 6 - Exploratory z-score standardization check (NOT used by the modeling pipeline)
# ======================================================================================
section("TASK 6 - Exploratory standardization (age, fare)")
z = clean[["age", "fare"]].copy()
z_manual = (z - z.mean()) / z.std(ddof=0)
z_sklearn = pd.DataFrame(StandardScaler().fit_transform(z), columns=z.columns)
summary = pd.DataFrame({
    "before_mean": z.mean(), "before_std": z.std(ddof=0),
    "after_mean (manual)": z_manual.mean(), "after_std (manual)": z_manual.std(ddof=0),
    "after_mean (StandardScaler)": z_sklearn.mean(), "after_std (StandardScaler)": z_sklearn.std(ddof=0),
}).round(4)
print(summary.to_string())
print("Manual z-score == StandardScaler:", np.allclose(z_manual.values, z_sklearn.values))

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for row, col in enumerate(["age", "fare"]):
    axes[row, 0].hist(z[col], bins=30, color=MUTED, edgecolor="#fcfcfb")
    axes[row, 0].set(title=f"{col} - original (mean {z[col].mean():.2f}, std {z[col].std(ddof=0):.2f})")
    axes[row, 1].hist(z_manual[col], bins=30, color=SURVIVED_COLOR, edgecolor="#fcfcfb")
    axes[row, 1].set(title=f"{col} - z-score (mean {z_manual[col].mean():.2f}, std {z_manual[col].std(ddof=0):.2f})")
fig.suptitle("Before / after z-score standardization (shape unchanged, centre 0, spread 1)", fontweight="bold")
fig.tight_layout()
save(fig, "08_zscore_before_after.png")

print("\nEDA complete.")
