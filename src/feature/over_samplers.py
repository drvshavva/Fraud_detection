import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE, SMOTENC, \
    ADASYN


class OverSampler:
    class Samplers:
        RandomOver = "RandomOverSampler"
        Smote = "SMOTE"
        Smoten = "SMOTEN"
        SvmSmote = "SVMSMOTE"
        KMeansSmote = "KMeansSMOTE"
        BorderlineSmote = "BorderlineSMOTE"
        Smotenc = "SMOTENC"
        Adasyn = "ADASYN"

    def __init__(self, sampling_name: str = Samplers.RandomOver):
        self.sampler = sampling_name

    @property
    def over_sampler(self):
        return eval(self.sampler)

    def resample(self, x: pd.DataFrame, y: pd.DataFrame or pd.Series):
        # burada seçilen sampling yöntemi uygulayarak resample edilir veri
        x_resampled, y_resampled = eval(self.sampler)().fit_resample(x, y)
        return x_resampled, y_resampled
