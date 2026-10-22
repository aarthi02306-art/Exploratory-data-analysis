"""
Exploratory Data Analysis (EDA) Project - starter pipeline
Usage: python eda_project.py your_data.csv [target_column]
Outputs: eda_charts/*.png and eda_report.md (structured report with auto-generated insights)
The target column is optional; if given, the script ranks which factors influence it most.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
os.makedirs("eda_charts", exist_ok=True)

path = sys.argv[1] if len(sys.argv) > 1 else "your_data.csv"
target = sys.argv[2] if len(sys.argv) > 2 else None

report = []


def add(text=""):
    report.append(text)


# 1. LOAD + OVERVIEW ----------------------------------------------------------
df = pd.read_csv(path)
df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]+", "_", regex=True)
if target:
    target = target.strip().lower().replace(" ", "_")

add("# Exploratory Data Analysis Report")
add(f"\n**Dataset:** `{os.path.basename(path)}`  ")
add(f"**Rows:** {df.shape[0]:,} | **Columns:** {df.shape[1]}  ")
add(f"**Duplicate rows:** {df.duplicated().sum()}\n")

add("## 1. Data Overview")
add("\n" + pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "missing_%": (df.isna().mean() * 100).round(1),
    "unique": df.nunique(),
}).to_markdown())

# Light cleaning so charts work
df = df.drop_duplicates()
for c in df.select_dtypes(include="object"):
    if "date" in c or "time" in c:
        df[c] = pd.to_datetime(df[c], errors="coerce")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in df.select_dtypes(include="object").columns if df[c].nunique() <= 30]

# 2. STATISTICAL SUMMARY ------------------------------------------------------
add("\n## 2. Statistical Summary (numeric)")
if num_cols:
    summary = df[num_cols].describe().T
    summary["skew"] = df[num_cols].skew()
    add("\n" + summary.round(2).to_markdown())

if cat_cols:
    add("\n## 3. Categorical Summary")
    for c in cat_cols[:5]:
        top = df[c].value_counts(normalize=True).head(5) * 100
        add(f"\n**{c}** - top categories (%):\n")
        add(top.round(1).to_frame("share_%").to_markdown())

# 3. VISUALIZATIONS -----------------------------------------------------------
# Missing values
if df.isna().any().any():
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False)
    plt.title("Missing Values Map")
    plt.tight_layout()
    plt.savefig("eda_charts/01_missing.png", dpi=150)
    plt.close()

# Distributions
if num_cols:
    df[num_cols].hist(figsize=(12, 8), bins=25)
    plt.suptitle("Distributions of Numeric Features")
    plt.tight_layout()
    plt.savefig("eda_charts/02_distributions.png", dpi=150)
    plt.close()

    # Outlier boxplots
    cols = num_cols[:8]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.5 * len(cols), 4), squeeze=False)
    for ax, c in zip(axes[0], cols):
        sns.boxplot(y=df[c], ax=ax)
        ax.set_title(c)
    plt.tight_layout()
    plt.savefig("eda_charts/03_boxplots.png", dpi=150)
    plt.close()

# Categorical counts
for c in cat_cols[:4]:
    plt.figure(figsize=(8, 4))
    df[c].value_counts().head(10).plot(kind="barh")
    plt.title(f"Top categories: {c}")
    plt.tight_layout()
    plt.savefig(f"eda_charts/04_{c}_counts.png", dpi=150)
    plt.close()

# 4. CORRELATIONS + KEY FACTORS ----------------------------------------------
add("\n## 4. Correlations")
if len(num_cols) > 1:
    corr = df[num_cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=len(num_cols) <= 12, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("eda_charts/05_correlation.png", dpi=150)
    plt.close()

    pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
             .stack().sort_values(key=abs, ascending=False).head(5))
    add("\nStrongest feature-to-feature correlations:\n")
    for (a, b), v in pairs.items():
        add(f"- **{a}** vs **{b}**: {v:.2f}")

if target and target in df.columns:
    add(f"\n## 5. Key Factors Influencing `{target}`")
    if target in num_cols:
        tcorr = df[num_cols].corr()[target].drop(target).sort_values(key=abs, ascending=False)
        add("\nNumeric features ranked by correlation with the target:\n")
        for k, v in tcorr.head(8).items():
            direction = "positive" if v > 0 else "negative"
            add(f"- **{k}**: {v:.2f} ({direction})")
        plt.figure(figsize=(7, 5))
        tcorr.head(10).sort_values().plot(kind="barh", title=f"Correlation with {target}")
        plt.tight_layout()
        plt.savefig("eda_charts/06_target_correlation.png", dpi=150)
        plt.close()
        for c in cat_cols[:3]:
            plt.figure(figsize=(8, 4))
            df.groupby(c)[target].mean().sort_values().tail(10).plot(kind="barh")
            plt.title(f"Average {target} by {c}")
            plt.tight_layout()
            plt.savefig(f"eda_charts/07_{target}_by_{c}.png", dpi=150)
            plt.close()
    else:
        add("\nAverage of numeric features by target class:\n")
        add(df.groupby(target)[num_cols].mean().round(2).T.to_markdown())
        for c in num_cols[:4]:
            plt.figure(figsize=(7, 4))
            sns.boxplot(data=df, x=target, y=c)
            plt.title(f"{c} by {target}")
            plt.tight_layout()
            plt.savefig(f"eda_charts/07_{c}_by_{target}.png", dpi=150)
            plt.close()

# 5. TIME TREND (if a date column exists) -------------------------------------
date_cols = df.select_dtypes(include="datetime").columns
if len(date_cols) and num_cols:
    d = date_cols[0]
    v = target if target in num_cols else num_cols[0]
    trend = df.set_index(d)[v].resample("M").mean()
    plt.figure(figsize=(10, 4))
    trend.plot(marker="o")
    plt.title(f"Monthly average {v}")
    plt.tight_layout()
    plt.savefig("eda_charts/08_trend.png", dpi=150)
    plt.close()
    add(f"\n## 6. Trend\n\nMonthly average of **{v}** saved as `eda_charts/08_trend.png`.")

# 6. INSIGHTS TEMPLATE --------------------------------------------------------
add("\n## Key Insights (write your own here)")
add("\n1. ...\n2. ...\n3. ...")
add("\n## Conclusion and Recommendations")
add("\n...")

with open("eda_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("Done. Report: eda_report.md | Charts: eda_charts/")
