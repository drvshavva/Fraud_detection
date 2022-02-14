import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_ml_model_results(x_train, y_train, x_test, y_test):
    """
    This method provides machine learning model results
    :param x_train: train features
    :param y_train: train labels
    :param x_test: test features
    :param y_test: test labels
    :return: results
    """
    ml_models = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), GaussianNB(),
                 RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), XGBClassifier()]
    results_columns = list()
    ml_compare_results = pd.DataFrame(columns=results_columns)

    row_index = 0
    for model in ml_models:
        predicted = model.fit(x_train, y_train).predict(x_test)
        model_name = model.__class__.__name__
        ml_compare_results.loc[row_index, 'Model Name'] = model_name
        ml_compare_results.loc[row_index, 'Train Accuracy'] = round(model.score(x_train, y_train), 2)
        ml_compare_results.loc[row_index, 'Test Accuracy'] = round(model.score(x_test, y_test), 2)
        ml_compare_results.loc[row_index, 'Precision'] = round(precision_score(y_test, predicted), 2)
        ml_compare_results.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 2)
        ml_compare_results.loc[row_index, 'F1 score'] = round(f1_score(y_test, predicted), 2)
        row_index += 1

    ml_compare_results.sort_values(by=['Test Accuracy'], ascending=False, inplace=True)
    return ml_compare_results
