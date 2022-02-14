import numpy as np
import pandas as pd


def drop_duplicates_and_nan_values(pandas_df: pd.DataFrame):
    # drop null values and duplicate values
    df_selected = pandas_df.dropna()
    df_selected.drop_duplicates(inplace=True)
    df_selected.info()

    return df_selected


def detect_outliers(data: pd.DataFrame, col_name: str, p=1.5):
    """
    this function detects outliers based on 3 time IQR and
    returns the number of lower and upper limit and number of outliers respectively
    """
    first_quartile = np.percentile(np.array(data[col_name].tolist()), 25)
    third_quartile = np.percentile(np.array(data[col_name].tolist()), 75)
    IQR = third_quartile - first_quartile

    upper_limit = third_quartile + (p * IQR)
    lower_limit = first_quartile - (p * IQR)
    outlier_count = 0

    for value in data[col_name].tolist():
        if (value < lower_limit) | (value > upper_limit):
            outlier_count += 1
    return lower_limit, upper_limit, outlier_count


def get_columns_outliers(pandas_df: pd.DataFrame, iqr: int = 2, label_column_name: str = "Class"):
    """
    This method show number of outliers in columns
    :param label_column_name:
    :param pandas_df:
    :param iqr:
    :return:
    """
    features = pandas_df.columns.tolist()
    features.remove(label_column_name)

    print(f"Number of Outliers for {iqr}*IQR after Logarithmed\n")

    total = 0
    for col in features:
        if detect_outliers(pandas_df, col)[2] > 0:
            outliers = detect_outliers(pandas_df, col, iqr)[2]
            total += outliers
            print(f"{outliers} outliers in '{col}'")
    print(f"\n{total} OUTLIERS TOTALLY")
