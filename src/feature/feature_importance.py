import pandas as pd
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, f_classif, SelectKBest
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px


# resource: https://machinelearningmastery.com/calculate-feature-importance-with-python/
class Model:
    decision_tree = "DecisionTreeClassifier"
    random_forest = "RandomForestClassifier"


def model_feature_importance(pandas_df: pd.DataFrame, label_column_name: str = "Class",
                             model_name: str = Model.decision_tree):
    """
    This method provides model feature importance
    :param pandas_df: data frame
    :param label_column_name:
    :param model_name: model name to get feature importance results
    :return:
    """
    x, y = pandas_df.drop([label_column_name], axis=1), pandas_df[[label_column_name]]
    model = eval(model_name)()
    # fit the model
    model.fit(x, y)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i, v in zip(x.columns.tolist(), importance):
        print(f'Feature: {i}, Score: %.5f' % v)
    # plot feature importance
    fi = {'features': x.columns.tolist(), 'feature_importance': importance}
    df_fi = pd.DataFrame(fi)
    df_fi.sort_values(by=['feature_importance'], ascending=True, inplace=True)
    fig = px.bar(df_fi, x='feature_importance', y='features', title=f"{model_name} Feature Importance", height=500)
    return fig


def model_feature_selection(x_train, y_train, x_test, model_name: str = Model.random_forest, max_features: int = 20):
    """
    This method provides feature selection operation
    :param model_name:
    :param x_train:
    :param y_train:
    :param x_test:
    :param max_features: max feature number to select
    :return:
    """
    # configure to select a subset of features
    fs = SelectFromModel(eval(model_name)(), max_features=max_features)
    # learn relationship from training data
    fs.fit(x_train, y_train)
    # transform train input data
    x_train_fs = fs.transform(x_train)
    # transform test input data
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs


def anova_feature_selection(x_train, y_train, x_test, max_features):
    # configure to select a subset of features
    fs = SelectKBest(f_classif, k=max_features)
    # learn relationship from training data
    fs.fit(x_train, y_train)
    # transform train input data
    x_train_fs = fs.transform(x_train)
    # transform test input data
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs
