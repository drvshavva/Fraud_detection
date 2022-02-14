from os.path import dirname
import warnings

from pandas import DataFrame
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline

warnings.filterwarnings(action='ignore')
DATA_PATH = dirname(dirname(dirname(__file__))) + "/data/"
RESULTS_PATH = dirname(dirname(dirname(__file__))) + "/results/"


def read_from_csv(csv_name: str, sep: str = ",") -> DataFrame:
    """
    This method read data from csv file and  returns DataFrame

    Args:
         sep: csv seperator, :type str
         csv_name: name of the csv, :type str
    Returns:
         DataFrame
    """
    df = pd.read_csv(f"{DATA_PATH}{csv_name}", sep=sep)
    print(f"Data is read. Len of the data {len(df)} and columns {df.columns}")
    return df


def read_from_excel(file_name: str, cols) -> DataFrame:
    """
    This method read data from xlsx file and  returns DataFrame

    Args:
         file_name: name of the csv, :type str
    Returns:
         DataFrame
    """
    df = pd.read_excel(f"{DATA_PATH}{file_name}", names=cols)
    print(f"Data is read. Len of the data {len(df)} and columns {df.columns}")
    return df


def write_to_csv(csv_name: str, data: DataFrame):
    """
    This method write data from csv file and  returns DataFrame

    Args:
         data: data to save, :type str
         csv_name: name of the csv, :type str
    Returns:
         None
    """
    data.to_csv(f"{DATA_PATH}{csv_name}", index=False)
    print(f"Data is wrote to path {DATA_PATH}, with name {csv_name}")


def load_model(path: str):
    """
    This method loads the pipeline

    :param path: path of the mode, :type str
    :return:
    """
    with open(path, 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model


def save_model(steps: list, name: str):
    pipeline = Pipeline(steps=steps)
    pickle.dump(pipeline, open(f"{RESULTS_PATH}/{name}.pkl", "wb"))
    print(f"Model saved to path: {RESULTS_PATH} with name {name}")
