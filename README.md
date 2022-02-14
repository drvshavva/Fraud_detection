Fraud Detection Project
=======================

 * Python version: 3.7
 * conda environment
 * data : data/creditcard.csv (https://www.kaggle.com/mlg-ulb/creditcardfraud)
 * For using run the command below in the project folder:
    $ pip install -r requirements.txt
 * For frontend usage run the command below in the project folder: (app.py uı geliştirmesi)
    $ streamlit run app.py --logger.level "error"
 * Notebooks directory is /notebooks:
    * denemeler ve çalışmada alınan sonuçlar bu notebooklarda
        * notebooks/data_analysis&cleaning.ipynb: veri kalitesi analizi ve veri önişleme adımları
        * notebooks/feature_selection&engineering.ipynb: öznitelik çıkarımı yöntemlerinin karşılaştırılması
        * notebooks/machine_learning_results.ipynb: seçilen yöntemler üzerinde makine öğrenmesi algoritmaları sonuçları
        * notebooks/auto_encoder.ipynb: autoencoder denemesi
 * Operations directory is /src
        * /src/utils: Utility methods and file operations (plot, model/data save/load, data quality operations)
        * /src/feature: Preprocess & feature engineering methods (over/under samplers, feature selections)
        * /src/pipeline: Model results, auto encoder class
 * Docker for UI
        * $ docker build -t fraud_detection_app:latest .
        * $ docker run fraud_detection_app:latest

NOTE
------

 * frontend kullanım video örnekleri video dizininde
 * Results dizini kodun içerisinde elde edilen resim ve dosyaların kaydedildiği dizin
 * Models dizini app çalıştırıldığında eğitilen modelin kaydedildiği dizin