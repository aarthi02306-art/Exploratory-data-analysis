import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 1. Load data
df = pd.read_csv("your_data.csv")   # <- change this
target = "target_column"            # <- optional: the column you want to explain

print(df.shape)
print(df.head())
print(df.info())

# 2. Quick cleaning
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.drop_duplicates()
print("Missing values:\n", df.isna().sum())

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in df.select_dtypes(include="object").columns
            if df[c].nunique() <= 30]

# 3. Statistical summary
print(df[num_cols].describe().T)
print("Skewness:\n", df[num_cols].skew())

# 4. Distributions
df[num_cols].hist(figsize=(12, 8), bins=25)
plt.suptitle("Distributions")
plt.show()

# 5. Outliers
fig, axes = plt.subplots(1, min(len(num_cols), 6), figsize=(14, 4))
for ax, c in zip(np.atleast_1d(axes), num_cols[:6]):
    sns.boxplot(y=df[c], ax=ax)
    ax.set_title(c)
plt.tight_layout()
plt.show()

# 6. Categorical columns
for c in cat_cols[:4]:
    plt.figure(figsize=(8, 4))
    df[c].value_counts().head(10).plot(kind="barh")
    plt.title(f"Top categories: {c}")
    plt.show()

# 7. Correlations
corr = df[num_cols].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation heatmap")
plt.show()

# 8. Key factors influencing the target
if target in num_cols:
    tcorr = corr[target].drop(target).sort_values(key=abs, ascending=False)
    print(tcorr.head(8))
    tcorr.head(10).sort_values().plot(kind="barh",
                                      title=f"Correlation with {target}")
    plt.show()
    for c in cat_cols[:3]:
        df.groupby(c)[target].mean().sort_values().tail(10).plot(
            kind="barh", title=f"Average {target} by {c}")
        plt.show()
elif target in df.columns:
    print(df.groupby(target)[num_cols].mean().round(2).T)
    for c in num_cols[:4]:
        sns.boxplot(data=df, x=target, y=c)
        plt.title(f"{c} by {target}")
        plt.show()
