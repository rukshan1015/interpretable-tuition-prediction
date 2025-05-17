# Creating a custom encoder for frequency encoding 

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)  # Ensure DataFrame
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.freq_maps[col] = freq
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            if col in self.freq_maps:
                X[col] = X[col].map(self.freq_maps[col]).fillna(0)
            else:
                X[col] = 0  # or keep original if preferred
        return X.values  # Return numpy array for sklearn compatibility
        
    def get_feature_names_out(self, input_features=None):
        return input_features

