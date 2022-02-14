import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import numpy as np

from os.path import dirname
import warnings

warnings.filterwarnings(action='ignore')
plt.style.use('ggplot')

RESULTS_PATH = dirname(dirname(dirname(__file__))) + "/results/"


def plot_correlation_relation_with_class(pandas_df: pd.DataFrame, label_column_name: str = "Class"):
    """
    This method provides correlation relation with class column
    :param pandas_df: data frame name
    :param label_column_name:  label class name
    :return: plt
    """
    corr = pandas_df.corrwith(pandas_df[label_column_name], method='spearman').reset_index()

    feature_columns, correlations = 'Feature Columns', 'Correlations'
    corr.columns = [feature_columns, correlations]
    corr = corr.set_index(feature_columns)
    corr = corr.sort_values(by=[correlations], ascending=False).head(10)

    plt.figure(figsize=(24.0, 16.0))
    fig = sns.heatmap(corr, annot=True, fmt="g", cmap='Set3', linewidths=0.4, linecolor='green')
    plt.title("Correlation of Features with Class", fontsize=20)
    plt.savefig(f"{RESULTS_PATH}correlation_relation_with_class")
    return plt


def plot_correlation_matrix(pandas_df: pd.DataFrame, title: str = "CreditCard Data"):
    """
    This method provides correlation matrix for given data frame

    :param title: name of the dataset
    :param pandas_df: data frame
    :return:
    """
    df = pandas_df[[col for col in pandas_df if
                    pandas_df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
    corr = df.corr()
    plt.figure(num=None, figsize=(24.0, 16.0), dpi=80, facecolor='w', edgecolor='k')
    corr_matrix = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corr_matrix)
    plt.title(f'Correlation Matrix for {title}', fontsize=15)
    plt.savefig(f"{RESULTS_PATH}correlation_matrix_{title}")
    return plt


def box_plot(pandas_df: pd.DataFrame, title: str = "CreditCart Data"):
    """
    This method provides box plot for outlier analysis

    :param pandas_df: pandas data frame
    :param title: title of the box plot
    :return: plot
    """
    plt.figure(figsize=(24.0, 16.0))
    sns.boxplot(data=pandas_df)
    plt.title(f'Box Plot for {title}', fontsize=15)
    plt.savefig(f"{RESULTS_PATH}box_plot_{title}")
    return plt


def plot_histogram(pandas_df: pd.DataFrame, title: str = "CreditCart Data"):
    """
    This method provides plot histogram for each column of the given data frame
    :param pandas_df: pandas data frame
    :param title: title of the box plot
    :return: plot
    """
    pandas_df.hist(figsize=(24.0, 16.0), ec='w')
    plt.title(f'Histogram for {title}', fontsize=15)
    plt.savefig(f"{RESULTS_PATH}histogram_{title}")
    return plt


def plot_bar_chart(labels: list, values: list, title: str):
    """
    This method plot bar chart

    :param labels: list of labels, :type list
    :param values: count of each label values, :type list
    :param title: title of plot
    :return: plot
    """
    y_pos = np.arange(len(labels))
    plt.figure(figsize=(24.0, 16.0))
    plt.bar(y_pos, values, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel('Count')
    plt.title(title)
    plt.savefig(f"{RESULTS_PATH}bar_chart_{title}")
    return plt


def plot_pie_chart(labels: list, values: list, title: str):
    """
    This method plot pie chart

    :param labels: list of labels, :type list
    :param values: count of each label values, :type list
    :param title: title of plot
    :return: plot
    """
    plt.figure(figsize=(24.0, 16.0))
    plt.pie(values, labels=labels, startangle=90, autopct='%.1f%%')
    plt.title(title)
    plt.savefig(f"{RESULTS_PATH}pie_chart_{title}")
    return plt


def plot_count_plot(label_name: str, data: DataFrame, title: str):
    """
    This method returns count plot of the dataset

    :param label_name: name of the class, :type str
    :param data: input dataFrame, :type DataFrame
    :param title: title of plot
    :return plt
    """
    plt.figure(figsize=(24.0, 16.0))
    sns.countplot(x=label_name, data=data)
    plt.title(title)
    plt.savefig(f"{RESULTS_PATH}plot_count_{title}")
    return plt
