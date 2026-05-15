"""
The bathrooms column in the raw dataset is almost entirely null.
Instead, bathrooms_text contains strings like "1 bath", "1.5 baths",
"Half-bath", "2 shared baths" — we extract the numeric value and
create a flag for shared bathrooms.
"""

import pandas as pd

def parse_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the bathrooms_text column into a numeric feature and a shared bath flag.

    :param df: DataFrame containing a raw `bathrooms_text` column.
    :type df: pd.DataFrame

    :returns: DataFrame with `bathrooms_parsed` and `is_shared_bath` columns added.
    :rtype: pd.DataFrame
    """
    # Replace "Half-bath" with "0.5" before extracting — otherwise the regex
    # finds no digits and returns null instead of 0.5
    df["bathrooms_parsed"] = (
        df["bathrooms_text"]
        .str.replace("half-bath", "0.5", case=False, regex=False)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    # Shared bathrooms tend to be priced lower than private bathrooms
    df["is_shared_bath"] = df["bathrooms_text"].str.contains(
        "shared", case=False, na=False
    ).astype(int)

    return df