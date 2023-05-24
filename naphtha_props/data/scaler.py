import numpy as np
from naphtha_props.data.data import DatapointList


class Scaler:
    """Class used to scale targets and features, either standard scaling or minmax scaling
    It also allows for inverse operation of the scaling
    The scaled targets and features are saved in the Datapoint object"""

    def __init__(self, data: DatapointList = None, scale_features: bool = True, mean=0.0, mean_f=0.0, std=0.0, std_f=0.0,
                 min=0.0, min_f=0.0, max=0.0, max_f=0.0, use_same_scaler_for_features: bool = False):
        self.scale_features = scale_features
        self.use_same_scaler_for_features = use_same_scaler_for_features
        if data is not None:
            targets = data.get_targets()
            self.mean = 0
            self.std = 0
            self.mean = np.nanmean(targets, axis=0)
            self.std = np.nanstd(targets, axis=0)
            self.min = np.nanmin(targets, axis=0)
            self.max = np.nanmax(targets, axis=0)
            if self.scale_features and not self.use_same_scaler_for_features:
                features = data.get_features()
                self.mean_features = 0
                self.std_features = 0
                self.mean_features = np.mean(features, axis=0)
                self.std_features = np.std(features, axis=0)
                self.min_features = np.min(features, axis=0)
                self.max_features = np.max(features, axis=0)
            elif self.scale_features and self.use_same_scaler_for_features:
                self.mean_features = self.mean
                self.std_features = self.std
                self.min_features = self.min
                self.max_features = self.max
            else:
                self.mean_features = 0
                self.std_features = 0
                self.min_features = 0
                self.max_features = 0
            self.type = "None"
        else:
            self.mean = mean
            self.std = std
            self.min = min
            self.max = max
            self.mean_features = mean_f
            self.std_features = std_f
            self.min_features = min_f
            self.max_features = max_f

    def transform_standard(self, data):
        targets = data.get_targets()
        scaled_targets = (targets-self.mean)/self.std
        data.set_scaled_targets(scaled_targets)
        if self.scale_features:
            features = data.get_features()
            scaled_features = (features-self.mean_features)/self.std_features
            data.set_scaled_features(scaled_features)
        else:
            data.set_scaled_features(data.get_features())
        self.type = "standard"

    # def inverse_transform_standard(self, data):
    #     scaled_predictions = data.get_scaled_predictions()
    #     predictions = (self.std*scaled_predictions+self.mean)
    #     data.set_predictions(predictions)

    def inverse_transform_standard(self, preds):
        scaled_predictions = preds
        predictions = (self.std*scaled_predictions+self.mean)
        return predictions

    def transform_minmax(self, data):
        targets = data.get_targets()
        scaled_targets = (targets-self.min)/(self.max-self.min)
        data.set_scaled_targets(scaled_targets)
        if self.scale_features:
            features = data.get_features()
            scaled_features = (features-self.min_features)/(self.max_features-self.min_features)
            data.set_scaled_features(scaled_features)
        else:
            data.set_scaled_features(data.get_features())
        self.type = "minmax"

    def inverse_transform_minmax(self, preds):
        scaled_predictions = preds
        predictions = (self.max-self.min)*scaled_predictions+self.min
        return predictions

    # def inverse_transform_minmax(self, data):
    #     scaled_predictions = data.get_scaled_predictions()
    #     predictions = (self.max-self.min)*scaled_predictions+self.min
    #     data.set_predictions(predictions)

    def inverse_transform(self, data):
        if self.type == "standard":
            return self.inverse_transform_standard(data)
        elif self.type == "minmax":
            return self.inverse_transform_minmax(data)
        elif self.type == "None":
            print("no scaler transformation")
        else:
            raise ValueError(f'Type of scaler transformation"{self.type}" not supported.')

