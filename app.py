import os
import pickle
from os.path import dirname

from imblearn.under_sampling import NearMiss, EditedNearestNeighbours
from imblearn.over_sampling import SMOTEN, SVMSMOTE
from imblearn.pipeline import Pipeline

import plotly.express as px
import streamlit as st
import pandas as pd
import base64

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings(action='ignore')

RESULTS_PATH = dirname(__file__) + '/models'
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache
def download(name):
    with open(f"{RESULTS_PATH}/{name}", 'rb') as file:
        st.download_button(
            label="Download Model",
            data=file,
            file_name=f"{name}",
            mime="model/pkl"
        )


def file_selector():
    filenames = os.listdir(RESULTS_PATH)
    models = [f for f in filenames if f.endswith(".pkl")]
    m = [""] + models
    selected_filename = st.sidebar.selectbox('Select a model', m)
    return selected_filename


@st.cache(persist=True)
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
    max_score = 0
    max_model = None
    for model in ml_models:
        predicted = model.fit(x_train, y_train).predict(x_test)
        model_name = model.__class__.__name__
        precision = round(precision_score(y_test, predicted), 2)
        if precision > max_score:
            max_score = precision
            max_model = model
        ml_compare_results.loc[row_index, 'Model Name'] = model_name
        ml_compare_results.loc[row_index, 'Train Accuracy'] = round(model.score(x_train, y_train), 2)
        ml_compare_results.loc[row_index, 'Test Accuracy'] = round(model.score(x_test, y_test), 2)
        ml_compare_results.loc[row_index, 'Precision'] = precision
        ml_compare_results.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 2)
        ml_compare_results.loc[row_index, 'F1 score'] = round(f1_score(y_test, predicted), 2)
        row_index += 1

    ml_compare_results.sort_values(by=['Test Accuracy'], ascending=False, inplace=True)
    return ml_compare_results, max_model


def prediction_res(model, x, y):
    predicted = model.predict(x)
    st.write("Accuracy ", round(model.score(x, y), 2))
    st.write("Precision: ", round(precision_score(y, predicted), 2))
    st.write("Recall: ", round(recall_score(y, predicted), 2))
    st.write("F1 score: ", round(f1_score(y, predicted), 2))
    plot_confusion_matrix(model, x, y)
    st.pyplot()


def get_class_info(y):
    df_label_counts = y.value_counts()
    labels = list(df_label_counts.to_frame().index)
    counts = df_label_counts.values
    for l, c in zip(labels, counts):
        st.info(f"Number of samples in class {str(l)}: {c}")


@st.cache
def plot_histogram(pandas_df: pd.DataFrame, label):
    """
    This method provides plot histogram for each column of the given data frame
    :param label:
    :param pandas_df: pandas data frame
    :return: plot
    """
    fig = px.histogram(pandas_df, x=label)
    fig.show()
    return fig


@st.cache
def plot_correlation_matrix(pandas_df: pd.DataFrame):
    """
    This method provides correlation matrix for given data frame

    :param pandas_df: data frame
    :return:
    """
    df = pandas_df[[col for col in pandas_df if
                    pandas_df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
    corr = df.corr()
    fig = px.imshow(corr)
    return fig


@st.cache
def drop_duplicates_and_nan_values(pandas_df: pd.DataFrame):
    # drop null values and duplicate values
    df_selected = pandas_df.dropna()
    df_selected.drop_duplicates(inplace=True)
    df_selected.info()

    return df_selected


@st.cache
def model_feature_importance(pandas_df: pd.DataFrame, label_column_name: str = "Class"):
    x, y = pandas_df.drop([label_column_name], axis=1), pandas_df[[label_column_name]]
    model = RandomForestClassifier()
    # fit the model
    model.fit(x, y)
    # get importance
    importance = model.feature_importances_
    # plot feature importance
    fi = {'features': x.columns.tolist(), 'feature_importance': importance}
    df_fi = pd.DataFrame(fi)
    df_fi.sort_values(by=['feature_importance'], ascending=True, inplace=True)
    fig = px.bar(df_fi, x='feature_importance', y='features', title=f"RandomForestClassifier Feature Importance",
                 height=500)
    return fig


def main():
    st.set_page_config(
        page_title="Web Mining Project",
        page_icon=":shark:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title('Credit Card Fraud Detection Web App')

    file_ = open("fraud.PNG", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    page_bg_img = '''
    <img src="fraud.PNG">
    <style>
    body {
        color: #fff;
        background-color: #FFFFFF;
    }
    .stButton>button {
        color: #4F8BF9;
    }
    .stTextInput>div>div>input {
        color: #4F8BF9;
    }
    </style>
    '''

    st.markdown(f'<img src="data:image/gif;base64,{data_url}">', unsafe_allow_html=True)
    st.info(
        r"""
        ** Problem Tanımı: ** 
        Geçmişteki kredi kartı işlemlerinin, dolandırıcılık olduğu ortaya çıkanların bilgisi ile 
        modellenmesini içerir. Bu model daha sonra yeni bir işlemin dolandırıcılık olup olmadığını belirlemek 
        için kullanılır. Bu arayüz ile yüklenen veri seti üzerinde model eğitimi yapılabilir veya yüklenen veri ve model
        ile tahmin sonuçları alınabilir.
        """
    )

    add_select_box_side = st.sidebar.selectbox(
        'Aim of the UI Usage:',
        ("Train", "Predict"), help=r"**Train:** Veri setinin model eğitimi için kullanılması."
                                   "**Predict:** Veri setinin yüklenen model üzerinden test verisi olarak kullanılması."
    )

    if add_select_box_side == "Train":
        st.sidebar.header("Select Operations and Parameters")
        # upload csv file
        uploaded_file_csv = st.file_uploader("Choose a CSV file for training")
        if uploaded_file_csv is not None:
            # show the data
            st.write("**Uploaded data frame:**")
            df = pd.read_csv(uploaded_file_csv)
            st.dataframe(df.iloc[:10000])

            # drop duplicated and nones
            data = drop_duplicates_and_nan_values(df)
            cols = [""] + df.columns.tolist()
            label_column = st.sidebar.selectbox("Select label column:",
                                                cols, help="Veri setinde kullanılan etiket sütunun adı.")

            if label_column:
                # plots
                st.sidebar.subheader("Plots:")
                if st.sidebar.checkbox("Class Histogram"):
                    with st.container():
                        st.write("** Class Histogram **")
                        hists = plot_histogram(df, label=label_column)
                        st.plotly_chart(hists)

                        with st.expander("See explanation"):
                            st.write("""
                                Bu histogram ile sınıflardaki örnek sayılarını görebiliyoruz.
                            """)
                if st.sidebar.checkbox("Correlation Matrix"):
                    with st.container():
                        st.write("** Correlation Matrix **")
                        corr = plot_correlation_matrix(df)
                        st.plotly_chart(corr)

                        with st.expander("See explanation"):
                            st.write("""
                                Bir korelasyon matrisi, basitçe farklı değişkenler için korelasyon 
                                katsayılarını gösteren bir tablodur. Matris, bir tablodaki tüm 
                                olası değer çiftleri arasındaki korelasyonu gösterir. Büyük bir 
                                veri kümesini özetlemek ve verilen verilerdeki kalıpları belirlemek
                                 ve görselleştirmek için güçlü bir araçtır.
                            """)
                # feature selection
                # initialize parameters
                x, y = data.drop([label_column], axis=1), data[[label_column]]
                selected_columns = None
                sampling_strategy = None
                ratio = 1.0

                over = None
                over_ratio = 0.2

                under = None
                under_ratio = 1.0

                st.sidebar.subheader("Feature Engineering:")
                if st.sidebar.checkbox('Feature Selection'):
                    with st.container():
                        st.write("**Random Forest Classifier Model Feature Scores**")
                        # feature histograms
                        feature_importance = model_feature_importance(df, label_column)
                        st.plotly_chart(feature_importance)

                        with st.expander("See explanation"):
                            st.write("""
                                Özniteliklerin skorlanması ile model için sadece önemli olan parametrelerini
                                görmüş oluruz ve ayrıca önemli olan öznitelikleri kullanarak model eğitiminin 
                                yapılması performans açısındanda fayda sağlar.
                            """)

                        selected_columns = st.sidebar.multiselect(
                            'Select columns:', data.columns.tolist())

                if st.sidebar.checkbox("Sampling:"):
                    with st.sidebar.container():
                        strategy = st.sidebar.radio(label="Select Sampling Strategy:",
                                                    options=["Over", "Under", "Combine"],
                                                    help="Combine strategy combines one over and one under sampler method.",
                                                    index=2)
                        if strategy == "Over":
                            sampling_strategy = st.sidebar.selectbox(label="Over sampling method:",
                                                                     options=[None, "SMOTEN", "SVMSMOTE"])
                            ratio = st.sidebar.number_input("Ratio of the sampling:", min_value=0.2, max_value=1.1,
                                                            value=0.5)
                        elif strategy == "Under":
                            sampling_strategy = st.sidebar.selectbox(label="Under sampling method:",
                                                                     options=[None, "NearMiss",
                                                                              "EditedNearestNeighbours"])
                            ratio = st.sidebar.number_input("Ratio of the sampling:", min_value=0.2, max_value=1.1,
                                                            value=0.5)
                        elif strategy == "Combine":
                            over = st.sidebar.selectbox(label="Over sampling method:",
                                                        options=[None, "SMOTEN", "SVMSMOTE"])
                            over_ratio = st.sidebar.number_input("Ratio of the over sampling:", min_value=0.2,
                                                                 max_value=1.1,
                                                                 value=0.2)
                            under = st.sidebar.selectbox(label="Under sampling method:",
                                                         options=[None, "NearMiss", "EditedNearestNeighbours"])
                            under_ratio = st.sidebar.number_input("Ratio of the under sampling:", min_value=0.2,
                                                                  max_value=1.1,
                                                                  value=1.0)

                scale = st.sidebar.checkbox("Standard Scaler")

                if st.sidebar.button("Get Results"):

                    st.subheader("Training and Evaluation")
                    if selected_columns is not None:
                        st.info(f"Selected columns: {selected_columns}")
                        x = x[selected_columns]
                        st.success("Feature selection operation is successful.")

                    if sampling_strategy is not None:
                        st.info(f"Selected strategy: {sampling_strategy}")
                        x, y = eval(sampling_strategy)(sampling_strategy=ratio).fit_resample(x, y)
                        st.success("Sampling operation is successful.")
                        get_class_info(y)

                    elif over is not None and under is not None:
                        st.info(f"Selected strategies: {over} and {under}")
                        over_sampler = eval(over)(sampling_strategy=over_ratio)
                        under_sampler = eval(under)(sampling_strategy=under_ratio)
                        steps = [('o', over_sampler), ('u', under_sampler)]
                        pipeline = Pipeline(steps=steps)

                        x, y = pipeline.fit_resample(x, y)

                        st.success(f"Sampling operation is successful.")
                        get_class_info(y)

                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

                    if scale:
                        scaler = StandardScaler()
                        x_train = scaler.fit_transform(x_train)
                        x_test = scaler.transform(x_test)
                        st.success("Scaling operation is successful.")

                    with st.container():
                        st.info("For performance metrics 0.3 test size is used.")
                        model_results, model = get_ml_model_results(x_train, y_train, x_test, y_test)
                        if isinstance(model_results, pd.DataFrame):
                            st.balloons()
                            st.dataframe(model_results)
                            st.write(f"**{model.__class__.__name__} Confusion Matrix: **")
                            plot_confusion_matrix(model, x_test, y_test, cmap="BuPu")
                            st.pyplot()
                            if scale:
                                pipeline = Pipeline(steps=[('StandardScaler', scaler), ('MLModel', model)])
                                name = f"{model.__class__.__name__}_with_scaler.pkl"
                                pickle.dump(pipeline, open(f"{RESULTS_PATH}/{name}", "wb"))
                                download(f"{name}")
                            else:
                                name = f"{model.__class__.__name__}.pkl"
                                pickle.dump(model, open(f"{RESULTS_PATH}/{name}", "wb"))
                                download(f"{name}")

                            os.remove(f"{RESULTS_PATH}/{name}")

    else:
        # upload csv file
        uploaded_file_csv = st.file_uploader("Choose a CSV file for testing")
        if uploaded_file_csv is not None:
            st.write("**Uploaded data frame:**")
            df = pd.read_csv(uploaded_file_csv)
            st.dataframe(df.iloc[:10000])

            # drop duplicated and nones
            data = drop_duplicates_and_nan_values(df)
            cols = [""] + df.columns.tolist()
            label_column = st.sidebar.selectbox("Select label column:",
                                                cols, help="Veri setinde kullanılan etiket sütunun adı.")
            if label_column:
                st.sidebar.subheader("Upload pre-trained model")
                model_name = file_selector()

                if model_name:
                    with open(f"{RESULTS_PATH}/{model_name}", 'rb') as file:
                        pickle_model = pickle.load(file)

                    x, y = data.drop([label_column], axis=1), data[[label_column]]

                    if st.sidebar.button("Predict"):
                        prediction_res(pickle_model, x, y)
                        st.balloons()


if __name__ == "__main__":
    # streamlit run app.py --logger.level "error"
    main()
