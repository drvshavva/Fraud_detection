import pandas as pd
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, InstanceHardnessThreshold, NearMiss, \
    TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN


class UnderSampler:
    class Samplers:
        UnderClusterCentroids = "ClusterCentroids"
        UnderRandomUnderSampler = "RandomUnderSampler"
        UnderInstanceHardnessThreshold = "InstanceHardnessThreshold"
        UnderNearMiss = "NearMiss"
        UnderTomekLinks = "TomekLinks"
        UnderEditedNearestNeighbours = "EditedNearestNeighbours"
        UnderRepeatedEditedNearestNeighbours = "RepeatedEditedNearestNeighbours"
        UnderAllKNN = "AllKNN"

    def __init__(self, sampling_name: str = Samplers.UnderRandomUnderSampler):
        self.sampler = sampling_name

    @property
    def over_sampler(self):
        return eval(self.sampler)

    def resample(self, x: pd.DataFrame, y: pd.DataFrame or pd.Series):
        # burada seçilen sampling yöntemi uygulayarak resample edilir veri
        x_resampled, y_resampled = eval(self.sampler)().fit_resample(x, y)
        return x_resampled, y_resampled
