import os

import pandas as pd

from enums import Ron, TCFDDomain

"""
This module contains helper functions for pandas dataframes required for handling 
the dataframes created in the pipeline. 
"""


def store_df(df: pd.DataFrame, store_at: str = False, path: str = False):
    print(f"Store file")
    if path:
        df.to_csv(path, sep=";")
    elif not store_at:
        df.to_csv(f"outputs/{df['pdf_name'][0]}.csv", sep=";")
    else:
        df.to_csv(f"{store_at}/{df['pdf_name'][0]}.csv", sep=";")


def filter(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[
        (df["climate_related"] == True)
        & (df["domain"] == TCFDDomain.Strategy.value)
        & (df["ron"] == Ron.Risk.value)
    ]


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter only texts that are in ron field risk
    """
    print(df.head())
    return df.loc[df["ron_risk"] > 0.4]


def create_filterd_outputs():

    folder_path = "outputs"
    files = os.listdir(folder_path)
    csv_names = [file.split(".")[-2] for file in files if file.endswith(".csv")]

    print(csv_names)

    for csv_name in csv_names:
        df = pd.read_csv(
            f"outputs/{csv_name}.csv", index_col=0, on_bad_lines="skip", sep=";"
        )
        df = filter_df(df)
        store_df(df, path=f"filtered_outputs/{csv_name.split('.')[0]}_filtered.csv")
