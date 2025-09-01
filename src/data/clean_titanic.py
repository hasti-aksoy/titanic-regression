import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

RAW_PATH = "data/raw/titanic.csv"
OUT_PATH = "data/processed/titanic_clean.csv"

KEEP_COLS_SEABORN = [
    "survived","pclass","sex","age","sibsp","parch","fare","embarked","class","who","adult_male","alone"
]
KEEP_COLS_KAGGLE = [
    "Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Name","Ticket","Cabin"
]

def to_snake(s: str) -> str:
    return (
        s.strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )

def load_titanic_dataframe() -> pd.DataFrame:
    if os.path.exists(RAW_PATH):
        logging.info(f"Reading Kaggle-style Titanic from {RAW_PATH}")
        df = pd.read_csv(RAW_PATH)
        keep = [c for c in KEEP_COLS_KAGGLE if c in df.columns]
        df = df[keep]
    else:
        logging.info("Local CSV not found. Falling back to seaborn.titanic dataset.")
        import seaborn as sns
        df = sns.load_dataset("titanic")
        keep = [c for c in KEEP_COLS_SEABORN if c in df.columns]
        df = df[keep]
    df.columns = [to_snake(c) for c in df.columns]
    return df

def report_missing(df: pd.DataFrame, title: str):
    miss = df.isna().sum().sort_values(ascending=False)
    rate = (miss / len(df)).round(3)
    rep = pd.DataFrame({"missing": miss, "rate": rate})
    logging.info(f"\nMissing Report — {title}\n{rep[rep['missing']>0]}")

def iqr_cap(series: pd.Series, k: float = 3.0) -> pd.Series:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return series.clip(lower=lower, upper=upper)

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = load_titanic_dataframe()
    logging.info(f"Loaded shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")

    report_missing(df, "Before Cleaning")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    logging.info(f"Removed {before - len(df)} duplicate rows.")

    # Drop 'cabin' and 'ticket' if exist (too many missings; feature eng later)
    for col in ["cabin", "ticket"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure correct dtypes
    cat_cols = [c for c in ["sex", "embarked", "class", "who", "alone"] if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    # Impute 'embarked' by mode (most frequent)
    if "embarked" in df.columns and df["embarked"].isna().any():
        mode_val = df["embarked"].mode(dropna=True)
        if len(mode_val) > 0:
            df["embarked"] = df["embarked"].fillna(mode_val.iloc[0])

    # Group-wise median imputation for 'age' and 'fare'
    group_keys = [c for c in ["sex", "pclass"] if c in df.columns]
    if "age" in df.columns and df["age"].isna().any() and group_keys:
        df["age"] = df.groupby(group_keys)["age"].transform(lambda s: s.fillna(s.median()))
        df["age"] = df["age"].fillna(df["age"].median())

    if "fare" in df.columns and df["fare"].isna().any():
        if group_keys:
            df["fare"] = df.groupby(group_keys)["fare"].transform(lambda s: s.fillna(s.median()))
        df["fare"] = df["fare"].fillna(df["fare"].median())

    # Soft outlier capping
    if "fare" in df.columns:
        df["fare"] = iqr_cap(df["fare"], k=3.0)

    # Make sure target 'survived' exists and is int/binary
    if "survived" in df.columns:
        df["survived"] = df["survived"].astype(int)

    report_missing(df, "After Cleaning")

    df.to_csv(OUT_PATH, index=False)
    logging.info(f"Saved cleaned Titanic to {OUT_PATH}")
    logging.info(f"Final shape: {df.shape}")

if __name__ == "__main__":
    main()