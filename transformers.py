import lightgbm
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from utils import get_numeric_columns_indices, correlation_calc

DEFAULT_NAN_FILLER = '0'


class DiscreteDistribution:
    def __init__(self, samples, probs):
        assert np.abs(sum(probs) - 1) < 1e-2
        assert len(samples) == len(probs)
        self.samples = samples
        self.probs = probs

    def sample(self, n=1):
        return np.random.choice(a=self.samples, p=self.probs, size=n, replace=True)


def append_col(nd, col):
    return np.append(nd, col.reshape(-1, 1), axis=1)


class OHESplitMetricsInjector(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        res = X.copy()
        numeric_columns_non_ohe = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) > 0]
        numeric_columns_ohe = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) == 2]
        relevant_data_ohe = X[:, numeric_columns_ohe]
        relevant_data_nonohe = X[:, numeric_columns_non_ohe]
        precentiles = [25, 50, 75]
        res=append_col(res, np.mean(relevant_data_ohe, axis=1))
        res=append_col(res, np.sum(relevant_data_ohe, axis=1))
        res=append_col(res, np.mean(relevant_data_nonohe, axis=1))
        res=append_col(res, np.sum(relevant_data_nonohe, axis=1))
        for p in precentiles:
            res=append_col(res, np.percentile(relevant_data_nonohe, p, axis=1))
            res=append_col(res, np.percentile(relevant_data_ohe, p, axis=1))
        res=append_col(res, np.min(relevant_data_nonohe, axis=1))
        res=append_col(res, np.max(relevant_data_nonohe, axis=1))

        return res


class SimpleMetricsInjector(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        res = X.copy()
        res = append_col(res, np.mean(X, axis=1))
        res = append_col(res, np.median(X, axis=1))
        res = append_col(res, np.sum(X, axis=1))
        return res


class SumInjector(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        res = X.copy()
        res[:, -1] = np.sum(X, axis=1)
        return res


class LowNaNCommonByLabelImputer(TransformerMixin, BaseEstimator):
    def __init__(self, nan_ratio):
        self.nan_ratio = nan_ratio
        self.impute_map = None
        self.low_nan_features = None

    def fit(self, X=None, y=None):
        if self.impute_map is not None:
            return self

        # Attributes with NaN ratio lower than nan_ratio
        self.low_nan_features = list(X.columns[(X.isnull().sum() / len(X)) < self.nan_ratio])
        self.impute_map = {}
        # Go over relevant attributes
        for attr in X.loc[:, self.low_nan_features]:
            # Fo over all possible possible target labels
            for target_label in y.unique():
                # Value is not null and labeled target_label
                row_non_nan_same_label = (X[attr].notnull()) & (y == target_label)
                # Mapping between attribute value to number of appearances
                attr_dict = X.loc[row_non_nan_same_label, attr].value_counts().to_dict()
                # Map between the attribute and target label to the most common atribute value
                self.impute_map[(attr, target_label)] = max(attr_dict.items(), key=lambda x: x[1])[0]

        return self

    def transform(self, X, y=None):
        res = X.copy()
        for feature in self.low_nan_features:
            for target_label in y.unique():
                # Value is null and labeled target_label
                row_nan_same_label = (X[feature].isnull()) & (y == target_label)
                # Impute with the most common value
                res.loc[row_nan_same_label, feature] = self.impute_map[(feature, target_label)]
        return res


class LowVarianceDropper(TransformerMixin, BaseEstimator):
    def __init__(self, var_drop_threshold=0):
        self.var_drop_threshold = var_drop_threshold
        self.cols_to_drop = []

    def fit(self, X, y=None):
        vars = {}
        columns = X.columns
        numeric_columns = columns[get_numeric_columns_indices(X)]
        for col in numeric_columns:
            nn_mask = X.loc[:, col].notnull()
            var = np.std(X.loc[nn_mask, col]) ** 2
            vars[col] = np.abs(var)
        sorted_vars = sorted(list(filter(lambda x: not np.isnan(x[1]), vars.items())), key=lambda x: x[1])
        for col, var in sorted_vars:
            if var <= self.var_drop_threshold:
                self.cols_to_drop.append(col)
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.cols_to_drop)


class ConvertToLGBDataset(TransformerMixin, BaseEstimator):
    def __init__(self, nominal_features_indices):
        self.nominal_features_indices = nominal_features_indices

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return lightgbm.Dataset(X, categorical_feature=self.nominal_features_indices)


class LowCorrelationDropper(TransformerMixin, BaseEstimator):
    def __init__(self, corr_drop_threshold=0.1):
        self.corr_drop_threshold = corr_drop_threshold
        self.cols_to_drop = []

    def fit(self, X, y):
        corrs = {}
        columns = X.columns
        numeric_columns = columns[get_numeric_columns_indices(X)]
        target = y
        for col in numeric_columns:
            nn_mask = X.loc[:, col].notnull()
            corr = correlation_calc(X.loc[nn_mask, col], target[nn_mask])
            corrs[col] = np.abs(corr)
        sorted_corrs = sorted(list(filter(lambda x: not np.isnan(x[1]), corrs.items())), key=lambda x: x[1])
        for col, corr in sorted_corrs:
            if corr < self.corr_drop_threshold:
                self.cols_to_drop.append(col)
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.cols_to_drop)



class LowCorrelationDropperPost(TransformerMixin, BaseEstimator):
    def __init__(self, corr_drop_threshold=0.1):
        self.corr_drop_threshold = corr_drop_threshold
        self.cols_to_drop = []

    def fit(self, X, y):
        corrs = {}
        target = y.to_numpy()
        for col_idx in range(X.shape[1]):
            corr = correlation_calc(X[:, col_idx], target)
            corrs[col_idx] = np.abs(corr)
        for col_idx, corr in corrs.items():
            if corr < self.corr_drop_threshold:
                self.cols_to_drop.append(col_idx)
        return self

    def transform(self, X, y=None):
        return np.delete(X, self.cols_to_drop, axis=1)

class BalancedHighNanFeaturesDropper(TransformerMixin, BaseEstimator):
    def __init__(self, high_nans_threshold=0.8, balance_threhsold=0.1):
        self.high_nan_threshold = high_nans_threshold
        self.balanced_threshold = balance_threhsold
        self.features_to_drop = []
        self.features_to_impute = []

    def fit(self, X, y):
        total_balance = y.mean()
        for col in X.columns[X.isnull().sum() / len(X) > self.high_nan_threshold]:
            balance = y[X[col].isnull()].mean()
            # print("For col %s balance is %f" % (col, balance))
            if np.abs(1 - (balance / total_balance)) < self.balanced_threshold:
                self.features_to_drop.append(col)
            else:
                raise Exception("High nans with high imbalance feature encountered %s" % col)
        return self

    def transform(self, X, y=None):
        res = X.copy()
        res.drop(columns=self.features_to_drop, inplace=True)
        return X.drop(columns=self.features_to_drop)


class LowNanImputer(TransformerMixin, BaseEstimator):
    def __init__(self, low_nans_threshold=0.8, balance_threshold=0.1, new_class_sign=DEFAULT_NAN_FILLER):
        self.low_nans_threshold = low_nans_threshold
        self.balance_threshold = balance_threshold
        self.balanced_features_to_impute = []
        self.imbalanced_features_to_impute_to_imbalance = {}
        self.imbalanced_numeric_dists = {}
        self.imbalanced_nominal_features = []
        self.balanced_numeric_means = {}
        self.balanced_nominal_dists = {}
        self.new_class_sign = new_class_sign

    def fit(self, X, y):
        total_balance = y.mean()
        for col in X.columns[X.isnull().sum() / len(X) <= self.low_nans_threshold]:
            if sum(X[col].isnull()) == 0:
                continue
            balance = y[X[col].isnull()].mean()
            # print("For col %s balance is %f" % (col, balance))
            imbalance_factor = 1 - (balance / total_balance)
            if np.abs(imbalance_factor) > self.balance_threshold:
                self.imbalanced_features_to_impute_to_imbalance[col] = imbalance_factor
            else:
                self.balanced_features_to_impute.append(col)
        for col, imbalance_factor in self.imbalanced_features_to_impute_to_imbalance.items():
            if is_string_dtype(X[col]):
                self.imbalanced_nominal_features.append(col)
            elif is_numeric_dtype(X[col]):
                positive_average = X[col][y == 1].mean()
                negative_average = X[col][y == 0].mean()

                samples = [negative_average, positive_average]

                if imbalance_factor > 0:  # balance is smaller than total
                    probs = [imbalance_factor, 1 - imbalance_factor]
                else:
                    inverse_imbalance = imbalance_factor * (imbalance_factor - 1)
                    probs = [inverse_imbalance, 1 - inverse_imbalance]

                self.imbalanced_numeric_dists[col] = DiscreteDistribution(samples=samples, probs=probs)
            else:
                raise Exception("Unhandled dtype found for col %s" % col)
        for col in self.balanced_features_to_impute:
            if is_string_dtype(X[col]):
                probs = (X[col].value_counts() / X[col].count()).values
                samples = list(X[col].value_counts().index)
                self.balanced_nominal_dists[col] = DiscreteDistribution(samples=samples, probs=probs)
            elif is_numeric_dtype(X[col]):
                self.balanced_numeric_means[col] = X[col].mean()
            else:
                raise Exception("Unhandled dtype found for col %s" % col)
        return self

    def transform(self, X, y=None):
        res = X.copy()
        for feature, mean in self.balanced_numeric_means.items():
            res.loc[res[feature].isnull(), feature] = mean
        for feature, dist in self.balanced_nominal_dists.items():
            res.loc[res[feature].isnull(), feature] = dist.sample(n=sum(res[feature].isnull()))
        for feature, dist in self.imbalanced_numeric_dists.items():
            res.loc[res[feature].isnull(), feature] = dist.sample(n=sum(res[feature].isnull()))
        for feature in self.imbalanced_nominal_features:
            res.loc[res[feature].isnull(), feature] = self.new_class_sign
        return res


class HighNaNConstantImputer(TransformerMixin, BaseEstimator):
    def __init__(self, nan_ratio=0.5, nan_filler=DEFAULT_NAN_FILLER):
        self.nan_ratio = nan_ratio
        self.nan_filler = nan_filler
        self.high_nan_features = None

    def fit(self, X=None, y=None):
        if self.high_nan_features is not None:
            return self
        # Attributes with NaN ratio higher than nan_ratio
        self.high_nan_features = list(X.columns[(X.isnull().sum() / len(X)) >= self.nan_ratio])
        return self

    def transform(self, X, y=None):
        res = X.copy()
        # Replace NaNs with self.nan_filler
        res.loc[:, self.high_nan_features] = X.loc[:, self.high_nan_features].fillna(value=self.nan_filler)
        return res


class LowNaNMFImputer(TransformerMixin, BaseEstimator):
    def __init__(self, nan_ratio=0.5):
        self.nan_ratio = nan_ratio
        self.low_nan_features = None
        self.feature_to_most_frequent = {}

    def fit(self, X=None, y=None):
        if self.low_nan_features is not None:
            return self
        # Attributes with NaN ratio higher than nan_ratio
        self.low_nan_features = list(X.columns[(X.isnull().sum() / len(X)) < self.nan_ratio])
        for feature in self.low_nan_features:
            most_frequest_val = sorted(X[feature].value_counts().to_dict().items(), key=lambda x: x[1])[-1][0]
            self.feature_to_most_frequent[feature] = most_frequest_val
        return self

    def transform(self, X, y=None):
        res = X.copy()
        # Replace NaNs with most frequents
        res.fillna(value=self.feature_to_most_frequent, inplace=True)
        return res
