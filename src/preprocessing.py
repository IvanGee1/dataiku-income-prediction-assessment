import pandas as pd
import numpy as np
from pathlib import Path



def get_metadata_cols(path):
    # The metadata file contains a mapping of feature names to internal census codes.


    with open(path, 'r') as file:
        metadata = file.read()


    lines = metadata.split("\n")
    columns = []
    capture = False

    for line in lines:
        line = line.strip()
        
        # Start capturing from the first actual column
        if line.startswith("| age"):
            capture = True
        
        # Stop when we reach the stats section
        if line.startswith("| Basic statistics"):
            break
        
        if not capture:
            continue
        
        if line.startswith("|"):
            line = line.replace("|", "").strip()
            parts = line.split()
            
            # last value is internal code, rest is the name to be used
            if len(parts) > 1:
                name = " ".join(parts[:-1])
                columns.append(name)
    

    return columns



def fetch_data(path):

    df = pd.read_csv(path,header=None,)

    return df


def column_cleaning(df,columns):
#    manually setitng the columns to be added and removed based on the eda 
    missing_columns = [
    "adjusted gross income",
    "federal income tax liability",
    "total person earnings",
    "total person income",
    "taxable income amount",
    ]


    cols_to_drop = [
    "migration prev res in sunbelt",
    "migration code-change in msa",
    "migration code-change in reg",
    "migration code-move within reg"
    ]


    columns = [col for col in columns if col not in missing_columns]

    # Fix spelling mistake
    columns = ["race" if col == "mace" else col for col in columns]

    # Add missing real column
    columns.append("year")


    expected_cols = columns + ["income"]

    if df.shape[1] != len(expected_cols):
        raise ValueError(
            f"Column mismatch: dataframe has {df.shape[1]} columns but expected {len(expected_cols)}"
        )

    df.columns = expected_cols

    
    # removal of ? from data as done in eda
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    df.replace(r'^\s*\?\s*$', np.nan, regex=True, inplace=True)
   
    df.drop(columns=cols_to_drop, inplace=True)

    return df



def remove_duplicates_and_fill(df):


    
    cat_fill_cols = [
    "country of birth father",
    "country of birth mother",
    "state of previous residence",
    "country of birth self"
    ]

    for col in cat_fill_cols:
        df[col] = df[col].fillna("Unknown")

    df = df.drop_duplicates()
    feature_cols = df.columns[:-1]

    # Find duplicate groups
    dup_groups = df.groupby(list(feature_cols))

    # Keep only groups where size is 2 and the income differs 
    to_drop_index = []

    for _, group in dup_groups:
        if len(group) == 2 and group["income"].nunique() > 1:
            to_drop_index.extend(group.index)

    df = df.drop(index=to_drop_index)


    return df




def data_casting(df):


    cols_to_category = [
    "industry code",
    "occupation code",
    "own business or self employed",
    "veterans benefits",
    "year"
    ]

    for col in cols_to_category:
        df[col] = df[col].astype("category")



    return df



def run_preprocessing():

    BASE_DIR = Path(__file__).resolve().parents[1]

    learn_data = BASE_DIR / "data" / "raw" / "census_income_learn.csv"
    test_data = BASE_DIR / "data" / "raw" / "census_income_test.csv"
    metadata = BASE_DIR / "data" / "raw" / "census_income_metadata.txt"


    learn_df= fetch_data(learn_data)
    test_df =fetch_data(test_data)
    metadata_cols =get_metadata_cols(metadata)


    learn_clean= column_cleaning(learn_df,metadata_cols)
    test_clean =column_cleaning(test_df,metadata_cols)




    learn_clean = remove_duplicates_and_fill(learn_clean)
    test_clean = remove_duplicates_and_fill(test_clean)

    learn_clean = data_casting(learn_clean)
    test_clean = data_casting(test_clean)

    return learn_clean, test_clean