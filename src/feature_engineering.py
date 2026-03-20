import pandas as pd
import numpy as np
from preprocessing import run_preprocessing


# setting the target values to 0 or 1
def map_income_target(df):
    df = df.copy()

    df["income"] = df["income"].astype(str).str.strip()

    df["income_binary"] = df["income"].map({
         "- 50000.": 0,
        "50000+.": 1,
    })

    if df["income_binary"].isna().any():
        raise ValueError("Income mapping failed for some rows")

    return df


def get_grouping_values(df, col, target_col="income_binary"):
    positive_values = set(df.loc[df[target_col] == 1, col].dropna().unique())
    all_values = set(df[col].dropna().unique())

    zero_positive_values = all_values - positive_values

    return zero_positive_values


def apply_grouping(df, col, values_to_group):
    df = df.copy()

    if len(values_to_group) == 0:
        return df

    df[col] = df[col].apply(lambda x: "other" if x in values_to_group else x)

    return df


def add_simple_flags(df):
    df = df.copy()

    df["has_capital_gains"] = (df["capital gains"] > 0).astype(int)
    df["has_capital_losses"] = (df["capital losses"] > 0).astype(int)
    df["has_dividends"] = (df["divdends from stocks"] > 0).astype(int)
    df["worked_last_year"] = (df["weeks worked in year"] > 0).astype(int)

    return df


def add_log_features(df):
    df = df.copy()

    df["capital gains log"] = np.log1p(df["capital gains"])
    df["capital losses log"] = np.log1p(df["capital losses"])
    df["dividends log"] = np.log1p(df["divdends from stocks"])

    return df


def feature_engineering(df, grouping_rules=None):
    df = df.copy()

    df = map_income_target(df)

    if grouping_rules is not None:
        for col, values_to_group in grouping_rules.items():
            df = apply_grouping(df, col, values_to_group)

    df = add_simple_flags(df)
    df = add_log_features(df)

    return df


if __name__ == "__main__":
    learn_clean, test_clean = run_preprocessing()

    # map target first on train so grouping rules can be learned from train only
    learn_temp = map_income_target(learn_clean)

    grouping_rules = {
        "detailed household and family stat": get_grouping_values(learn_temp, "detailed household and family stat" ),
        "family members under 18": get_grouping_values( learn_temp, "family members under 18"),
    }

    learn_fe = feature_engineering(learn_clean, grouping_rules=grouping_rules)
    test_fe = feature_engineering(test_clean, grouping_rules=grouping_rules)

    print("train shape:", learn_fe.shape)
    print("test shape:", test_fe.shape)

    print("\n new columns:")
    new_cols = [col for col in learn_fe.columns if col not in learn_clean.columns]
    print("\n number of new columns:", len(new_cols))
    print("\n new columns added:")
    for x in new_cols:
        print("-", x)