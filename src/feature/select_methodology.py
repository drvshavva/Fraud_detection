import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.feature.feature_importance import anova_feature_selection, model_feature_selection, Model
from src.feature.over_samplers import OverSampler
from src.feature.under_samplers import UnderSampler


# todo burayı refactor et, kod tekrarları var !!
def get_feature_engineering_results(pandas_df: pd.DataFrame, label_column_name: str = "Class"):
    print("Feature engineering & selection results on data with Logistic Regression model (test size:0.3)")
    model = LogisticRegression()

    x, y = pandas_df.drop([label_column_name], axis=1), pandas_df[[label_column_name]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # scaler
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # anova
    x_train_anova_max_features_25, x_test_anova_max_features_25 = anova_feature_selection(x_train=x_train,
                                                                                          y_train=y_train,
                                                                                          x_test=x_test,
                                                                                          max_features=25)
    x_train_anova_max_features_20, x_test_anova_max_features_20 = anova_feature_selection(x_train=x_train,
                                                                                          y_train=y_train,
                                                                                          x_test=x_test,
                                                                                          max_features=20)
    # model - 1 feature selected
    x_train_model_1_max_features_20, x_test_model_1_max_features_20 = model_feature_selection(x_train=x_train,
                                                                                              y_train=y_train,
                                                                                              x_test=x_test,
                                                                                              model_name=Model.random_forest)
    x_train_model_1_max_features_15, x_test_model_1_max_features_15 = model_feature_selection(x_train=x_train,
                                                                                              y_train=y_train,
                                                                                              x_test=x_test,
                                                                                              model_name=Model.random_forest,
                                                                                              max_features=15)
    x_train_model_1_max_features_25, x_test_model_1_max_features_25 = model_feature_selection(x_train=x_train,
                                                                                              y_train=y_train,
                                                                                              x_test=x_test,
                                                                                              model_name=Model.random_forest,
                                                                                              max_features=25)
    # model - 2 feature selected
    x_train_model_2_max_features_20, x_test_model_2_max_features_20 = model_feature_selection(x_train=x_train,
                                                                                              y_train=y_train,
                                                                                              x_test=x_test,
                                                                                              model_name=Model.decision_tree)
    x_train_model_2_max_features_15, x_test_model_2_max_features_15 = model_feature_selection(x_train=x_train,
                                                                                              y_train=y_train,
                                                                                              x_test=x_test,
                                                                                              model_name=Model.decision_tree,
                                                                                              max_features=15)
    x_train_model_2_max_features_25, x_test_model_2_max_features_25 = model_feature_selection(x_train=x_train,
                                                                                              y_train=y_train,
                                                                                              x_test=x_test,
                                                                                              model_name=Model.decision_tree,
                                                                                              max_features=25)
    results_columns = ["Without Selection&Scaling", "Standard Scaler", "Anova features 25", "Anova features 20",
                       "Random Forest Feature Select feature 20", "Random Forest Feature Select feature 15",
                       "Random Forest Feature Select feature 25", "Decision Tree Feature Select feature 20",
                       "Decision Tree Feature Select feature 15", "Decision Tree Feature Select feature 25"]

    data = [(x_train, x_test), (x_train_scaled, x_test_scaled),
            (x_train_anova_max_features_25, x_test_anova_max_features_25),
            (x_train_anova_max_features_20, x_test_anova_max_features_20),
            (x_train_model_1_max_features_20, x_test_model_1_max_features_20),
            (x_train_model_1_max_features_15, x_test_model_1_max_features_15),
            (x_train_model_1_max_features_25, x_test_model_1_max_features_25),
            (x_train_model_2_max_features_20, x_test_model_2_max_features_20),
            (x_train_model_2_max_features_15, x_test_model_2_max_features_15),
            (x_train_model_2_max_features_25, x_test_model_2_max_features_25)]
    ml_compare_results = pd.DataFrame(columns=list())

    row_index = 0

    for d, m in zip(data, results_columns):
        train, test = d[0], d[1]
        predicted = model.fit(X=train, y=y_train).predict(X=test)
        ml_compare_results.loc[row_index, 'Methodology'] = m
        ml_compare_results.loc[row_index, 'Train Accuracy'] = round(model.score(train, y_train), 2)
        ml_compare_results.loc[row_index, 'Test Accuracy'] = round(model.score(test, y_test), 2)
        ml_compare_results.loc[row_index, 'Precision'] = round(precision_score(y_test, predicted), 2)
        ml_compare_results.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 2)
        ml_compare_results.loc[row_index, 'F1 score'] = round(f1_score(y_test, predicted), 2)
        row_index += 1

    ml_compare_results.sort_values(by=['Test Accuracy'], ascending=False, inplace=True)
    return ml_compare_results


def get_over_sampling_results(pandas_df: pd.DataFrame, label_column_name: str = "Class"):
    print("Over Sampling on data with Logistic Regression model (test size:0.3)")
    model = LogisticRegression()

    x, y = pandas_df.drop([label_column_name], axis=1), pandas_df[[label_column_name]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # over sampler -1
    random_sampler = OverSampler(OverSampler.Samplers.RandomOver)
    x_train_random_sampled, y_train_random_sampled = random_sampler.resample(x_train, y_train)

    # over sampler -2
    Smote_sampler = OverSampler(OverSampler.Samplers.Smote)
    x_train_smote_sampled, y_train_smote_sampled = Smote_sampler.resample(x_train, y_train)

    # over sampler -3
    Smoten_sampler = OverSampler(OverSampler.Samplers.Smoten)
    x_train_smoten_sampled, y_train_smoten_sampled = Smoten_sampler.resample(x_train, y_train)

    # over sampler -4
    SvmSmote_sampler = OverSampler(OverSampler.Samplers.SvmSmote)
    x_train_SvmSmote_sampled, y_train_SvmSmote_sampled = SvmSmote_sampler.resample(x_train, y_train)

    # over sampler -6
    BorderlineSmote_sampler = OverSampler(OverSampler.Samplers.BorderlineSmote)
    x_train_BorderlineSmote_sampled, y_train_BorderlineSmote_sampled = BorderlineSmote_sampler.resample(x_train,
                                                                                                        y_train)

    # over sampler -8
    Adasyn_sampler = OverSampler(OverSampler.Samplers.Adasyn)
    x_train_Adasyn_sampled, y_train_Adasyn_sampled = Adasyn_sampler.resample(x_train, y_train)

    results_columns = ["Without Selection&Scaling", "RandomOverSampler", "SMOTE", "SMOTEN",
                       "SVMSMOTE",
                       "BorderlineSMOTE",
                       "ADASYN"]

    data = [(x_train, y_train), (x_train_random_sampled, y_train_random_sampled),
            (x_train_smote_sampled, y_train_smote_sampled),
            (x_train_smoten_sampled, y_train_smoten_sampled),
            (x_train_SvmSmote_sampled, y_train_SvmSmote_sampled),
            (x_train_BorderlineSmote_sampled, y_train_BorderlineSmote_sampled),
            (x_train_Adasyn_sampled, y_train_Adasyn_sampled)]
    ml_compare_results = pd.DataFrame(columns=list())

    row_index = 0

    for d, m in zip(data, results_columns):
        train, y = d[0], d[1]
        predicted = model.fit(X=train, y=y).predict(X=x_test)
        ml_compare_results.loc[row_index, 'Over Sampler Method'] = m
        ml_compare_results.loc[row_index, 'Train Accuracy'] = round(model.score(train, y), 2)
        ml_compare_results.loc[row_index, 'Test Accuracy'] = round(model.score(x_test, y_test), 2)
        ml_compare_results.loc[row_index, 'Precision'] = round(precision_score(y_test, predicted), 2)
        ml_compare_results.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 2)
        ml_compare_results.loc[row_index, 'F1 score'] = round(f1_score(y_test, predicted), 2)
        row_index += 1

    ml_compare_results.sort_values(by=['Recall'], ascending=False, inplace=True)
    return ml_compare_results


def get_under_sampling_results(pandas_df: pd.DataFrame, label_column_name: str = "Class"):
    print("Under Sampling on data with Logistic Regression model (test size:0.3)")
    model = LogisticRegression()

    x, y = pandas_df.drop([label_column_name], axis=1), pandas_df[[label_column_name]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # under sampler -1
    UnderClusterCentroids_sampler = UnderSampler(UnderSampler.Samplers.UnderClusterCentroids)
    x_train_UnderClusterCentroids_sampled, y_train_UnderClusterCentroids_sampled = \
        UnderClusterCentroids_sampler.resample(x_train, y_train)

    # under sampler -2
    UnderRandomUnderSampler_sampler = UnderSampler(UnderSampler.Samplers.UnderRandomUnderSampler)
    x_train_UnderRandomUnderSampler_sampled, y_train_UnderRandomUnderSampler_sampled = \
        UnderRandomUnderSampler_sampler.resample(x_train, y_train)

    # under sampler -3
    UnderInstanceHardnessThreshold_sampler = UnderSampler(UnderSampler.Samplers.UnderInstanceHardnessThreshold)
    x_train_UnderInstanceHardnessThreshold_sampled, y_train_UnderInstanceHardnessThreshold_sampled = \
        UnderInstanceHardnessThreshold_sampler.resample(x_train, y_train)

    # under sampler -4
    UnderNearMiss_sampler = UnderSampler(UnderSampler.Samplers.UnderNearMiss)
    x_train_UnderNearMiss_sampled, y_train_UnderNearMiss_sampled = UnderNearMiss_sampler.resample(x_train, y_train)

    # under sampler -5
    UnderTomekLinks_sampler = UnderSampler(UnderSampler.Samplers.UnderTomekLinks)
    x_train_UnderTomekLinks_sampled, y_train_UnderTomekLinks_sampled = \
        UnderTomekLinks_sampler.resample(x_train, y_train)

    # under sampler -6
    UnderEditedNearestNeighbours_sampler = UnderSampler(UnderSampler.Samplers.UnderEditedNearestNeighbours)
    x_train_UnderEditedNearestNeighbours_sampled, y_train_UnderEditedNearestNeighbours_sampled = \
        UnderEditedNearestNeighbours_sampler.resample(x_train, y_train)

    # under sampler -7
    UnderAllKNN_sampler = UnderSampler(UnderSampler.Samplers.UnderAllKNN)
    x_train_UnderAllKNN_sampled, y_train_UnderAllKNN_sampled = UnderAllKNN_sampler.resample(x_train, y_train)

    results_columns = ["Without Selection&Scaling", "UnderClusterCentroids", "UnderRandomUnderSampler"
                                                                             "UnderInstanceHardnessThreshold",
                       "UnderNearMiss",
                       "UnderTomekLinks", "UnderEditedNearestNeighbours",
                       "UnderAllKNN"]

    data = [(x_train, y_train),
            (x_train_UnderClusterCentroids_sampled, y_train_UnderClusterCentroids_sampled),
            (x_train_UnderRandomUnderSampler_sampled, y_train_UnderRandomUnderSampler_sampled),
            (x_train_UnderInstanceHardnessThreshold_sampled, y_train_UnderInstanceHardnessThreshold_sampled),
            (x_train_UnderNearMiss_sampled, y_train_UnderNearMiss_sampled),
            (x_train_UnderTomekLinks_sampled, y_train_UnderTomekLinks_sampled),
            (x_train_UnderEditedNearestNeighbours_sampled, y_train_UnderEditedNearestNeighbours_sampled ),
            (x_train_UnderAllKNN_sampled, y_train_UnderAllKNN_sampled)]
    ml_compare_results = pd.DataFrame(columns=list())

    row_index = 0

    for d, m in zip(data, results_columns):
        train, y = d[0], d[1]
        predicted = model.fit(X=train, y=y).predict(X=x_test)
        ml_compare_results.loc[row_index, 'Under Sampler Method'] = m
        ml_compare_results.loc[row_index, 'Train Accuracy'] = round(model.score(train, y), 2)
        ml_compare_results.loc[row_index, 'Test Accuracy'] = round(model.score(x_test, y_test), 2)
        ml_compare_results.loc[row_index, 'Precision'] = round(precision_score(y_test, predicted), 2)
        ml_compare_results.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 2)
        ml_compare_results.loc[row_index, 'F1 score'] = round(f1_score(y_test, predicted), 2)
        row_index += 1

    ml_compare_results.sort_values(by=['Recall'], ascending=False, inplace=True)
    return ml_compare_results
