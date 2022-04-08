# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from datetime import time, datetime

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score as scorer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import IncrementalPCA, PCA
from catboost import CatBoostClassifier
import xgboost as xgb
import autosklearn.classification
import sklearn.datasets
import sklearn.metrics

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

import os
import random

def get_pca_num_components(cv):
    try:
        return cv.best_estimator_.named_steps.preprocessing.named_steps.pca.n_components_
    except AttributeError:
        try:
            return cv.best_estimator_.named_steps.preprocessing.named_steps.ipca.n_components_
        except:
            return -1  # No PCA


def get_steps_names(pipeline):
    if hasattr(pipeline, 'best_estimator_'):
        return get_steps_names(getattr(pipeline, 'best_estimator_'))
    elif hasattr(pipeline, 'named_steps'):
        res = {}
        for step in pipeline.named_steps:
            res[step] = get_steps_names(getattr(pipeline.named_steps, step))
        return res
    elif hasattr(pipeline, 'named_transformers_'):
        res = {}
        for transformer in getattr(pipeline, 'named_transformers_'):
            res[transformer] = get_steps_names(getattr(pipeline, 'named_transformers_')[transformer])
        return res
    else:
        return pipeline.__str__()
    return None

from transformers import BalancedHighNanFeaturesDropper, LowNanImputer

KAGGLE_ROOT = './input'
TRAIN_PATH = KAGGLE_ROOT + '/train.CSV/train.CSV'
TEST_PATH = KAGGLE_ROOT + '/test.CSV'
LOG_DIR = './logs'
SUBMISSIONS_DIR = './submissions'

NOM_ATTR_RANGE_LOW = 0
NOM_ATTR_RANGE_HIGH = 83
NUM_OF_ATTRS = 452

NAN_AUTO_FILL_RATIO_LIMIT = 0.66
NAN_FILLER = '0'

TEST_SIZE = 0.33
RANDOM_SEED = 42

NUM_EXPLAINED_ROWS = 5

TARGET_COLUMN = -1
POSITIVE_CLASS_INDEX = 1

GENERATE_SUBMISSION = True


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


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


class HighNaNConstantImputer(TransformerMixin, BaseEstimator):
    def __init__(self, nan_ratio=0.5, nan_filler=NAN_FILLER):
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
        # Replace NaNs withh self.nan_filler
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


if '__main__' == __name__:
    train = pd.read_csv(TRAIN_PATH)

    # Split train
    X = train.iloc[:, :TARGET_COLUMN]
    y = train.iloc[:, TARGET_COLUMN]

    y = pd.Series(LabelEncoder().fit_transform(y))  # TODO try to add this to pipeline

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED,
                                                        shuffle=True)

    # Nominal Imputations
    nominal_columns = list(range(NOM_ATTR_RANGE_LOW, NOM_ATTR_RANGE_HIGH))
    base_nominal_imputer = SimpleImputer(strategy='most_frequent')
    nominal_imputer = Pipeline(steps=[
        ("lowimputer", LowNaNMFImputer()),
        ("highimputer", HighNaNConstantImputer()),
    ])

    # Numerical Imputations
    numeric_columns = list(range(NOM_ATTR_RANGE_HIGH, NUM_OF_ATTRS))
    numeric_imputer = SimpleImputer(strategy='mean')
    # Deal with NaNs in the nominal attributes

    # Full Imputers
    base_imputer = ColumnTransformer([
        ("nominal", base_nominal_imputer, nominal_columns),
        ("numerical", numeric_imputer, numeric_columns)
    ])

    imputer = ColumnTransformer([
        ("nominal", nominal_imputer, nominal_columns),
        ("numerical", numeric_imputer, numeric_columns)
    ])

    random_imputer = Pipeline(steps=[
        ("lowimputer", LowNanImputer()),
    ])

    iterative_imputer = ColumnTransformer([
        ("nominal", base_nominal_imputer, nominal_columns),
        ("numerical", IterativeImputer(), numeric_columns)
    ])

    # Encoding
    one_hot_encoder = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown='ignore'), nominal_columns)],
                                        "passthrough")

    # Scale
    scaler = ColumnTransformer([
        ("numerical", StandardScaler(), numeric_columns)
    ])

    # PCA
    pca = PCA()

    # Preprocessing Pipelines
    base_preprocessing = Pipeline([("imputer", base_imputer),
                                   ("ohe", one_hot_encoder)])

    random_preprocessing = Pipeline([("imputer", random_imputer),
                                     ("ohe", one_hot_encoder)])

    preprocessing = Pipeline([("imputer", imputer),
                              ("ohe", one_hot_encoder)])

    iter_preprocessing = Pipeline([("imputer", iterative_imputer),
                                   ("ohe", one_hot_encoder)])

    pca_preprocessing = Pipeline([("imputer", imputer),
                                  ("ohe", one_hot_encoder),
                                  ("scaler", scaler),
                                  ("pca", pca)])

    base_pca_preprocessing = Pipeline([("imputer", base_imputer),
                                       ("ohe", one_hot_encoder),
                                       ("pca", pca)])



    # X_train = preprocessing.fit_transform(X_train, y_train)
    # assert sum(sum(X_train == np.nan)) == 0

    # Models Construction
    explored_n_estimators = [100, 500, 1000, 5000]
    explored_nan_ratios = [0.3, 0.9, 0.95]
    explored_pca_components = [100]
    explored_ipca_components = [100]
    explored_iterations = [1000]
    explored_learning_rate = [0.015, 0.03, 0.06]
    explored_loss_function = ['Logloss']
    explored_max_depth = [3, 10, 20]

    #########

    baseline_model = Pipeline([("preprocessing", base_preprocessing),
                               ("model", RandomForestClassifier())])

    baseline_model_params = {
        'model__n_estimators': explored_n_estimators
    }

    #########

    pca_base_model = Pipeline([("preprocessing", base_pca_preprocessing),
                               ("model", RandomForestClassifier())])

    pca_base_model_params = {
        'model__n_estimators': explored_n_estimators,
        'preprocessing__pca__n_components': explored_pca_components
    }

    #########

    iter_baseline_model = Pipeline([("preprocessing", iter_preprocessing),
                                    ("model", RandomForestClassifier())])

    iter_baseline_model_params = {
        'model__n_estimators': explored_n_estimators,
        'preprocessing__imputer__numerical__estimator': [ExtraTreesRegressor(n_estimators=10, random_state=0)],
        'preprocessing__imputer__numerical__max_iter': [3]
    }

    #########

    pca_forest_model = Pipeline([("preprocessing", pca_preprocessing),
                                 ("model", RandomForestClassifier())])

    pca_forest_model_params = {
        'model__n_estimators': explored_n_estimators,
        'preprocessing__pca__n_components': explored_pca_components
    }

    #########

    nanthresh_model = Pipeline([("preprocessing", preprocessing),
                                ("model", RandomForestClassifier())])

    nanthresh_model_params = [{'model__n_estimators': explored_n_estimators,
                               'preprocessing__imputer__nominal__lowimputer__nan_ratio': [nan_ratio],
                               'preprocessing__imputer__nominal__highimputer__nan_ratio': [nan_ratio]} for nan_ratio in
                              explored_nan_ratios]

    #########

    xgb_object = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=1,
        colsample_bytree=1,
        objective='reg:squarederror',
    )

    #########

    base_xgb_model = Pipeline([("preprocessing", base_preprocessing),
                               ("model", xgb_object)])

    base_xgb_model_params = {
        'model__n_estimators': explored_n_estimators,
        'model__learning_rate': explored_learning_rate,
        'model__max_depth': explored_max_depth
    }

    #########

    xgb_model = Pipeline([("preprocessing", preprocessing),
                          ("model", xgb_object)])

    xgb_model_params = [{'model__n_estimators': explored_n_estimators,
                         'preprocessing__imputer__nominal__lowimputer__nan_ratio': [nan_ratio],
                         'preprocessing__imputer__nominal__highimputer__nan_ratio': [nan_ratio]} for nan_ratio in
                        explored_nan_ratios]

    #########

    pca_xgb_model = Pipeline([("preprocessing", base_pca_preprocessing),
                              ("model", xgb_object)])

    pca_xgb_model_params = {
        'model__n_estimators': explored_n_estimators,
        'preprocessing__pca__n_components': explored_pca_components,
    }

    #########

    catboost = CatBoostClassifier(
        iterations=5,
        learning_rate=0.1,
        loss_function='CrossEntropy'
    )

    #########

    base_catboost_model = Pipeline([("preprocessing", base_preprocessing),
                                    ("model", catboost)])

    base_catboost_model_params = {
        'model__iterations': explored_iterations,
        'model__learning_rate': explored_learning_rate,
        'model__loss_function': explored_loss_function
    }

    #########

    catboost_model = Pipeline([("preprocessing", preprocessing),
                               ("model", catboost)])

    catboost_model_params = [{'model__iterations': explored_iterations,
                              'model__learning_rate': explored_learning_rate,
                              'model__loss_function': explored_loss_function,
                              'preprocessing__imputer__nominal__lowimputer__nan_ratio': [nan_ratio],
                              'preprocessing__imputer__nominal__highimputer__nan_ratio': [nan_ratio]} for nan_ratio in
                             explored_nan_ratios]

    #########

    pca_catboost_model = Pipeline([("preprocessing", base_pca_preprocessing),
                                   ("model", catboost)])

    pca_catboost_model_params = {
        'model__iterations': explored_iterations,
        'model__learning_rate': explored_learning_rate,
        'model__loss_function': explored_loss_function,
        'preprocessing__pca__n_components': explored_pca_components
    }

    #########
    automl = autosklearn.classification.AutoSklearnClassifier()

    #########

    base_automl_model = Pipeline([("preprocessing", base_preprocessing),
                                  ("model", automl)])

    base_automl_model_params = {}
    
    #########

    automl_model = Pipeline([("preprocessing", preprocessing),
                             ("model", automl)])

    automl_model_params = [{'preprocessing__imputer__nominal__lowimputer__nan_ratio': [nan_ratio],
                            'preprocessing__imputer__nominal__highimputer__nan_ratio': [nan_ratio]} for nan_ratio in explored_nan_ratios]

    #########

    pca_automl_model = Pipeline([("preprocessing", base_pca_preprocessing),
                                 ("model", automl)])

    pca_automl_model_params = {
        'preprocessing__pca__n_components': explored_pca_components
    }

    #########

    random_automl_model = Pipeline([("preprocessing", random_preprocessing),
                                    ("model", automl)])

    random_automl_model_params = {}

    #########

    iterative_automl_model = Pipeline([("preprocessing", iter_preprocessing),
                                       ("model", automl)])

    iterative_automl_model_params = {}

 
    # Model Selection
    model_name = 'iterative_automl_model'

    model_params = globals()[model_name + '_params']  # keep the convention
    model = globals()[model_name]

    num_folds = 5
    cv = GridSearchCV(estimator=model, param_grid=model_params, scoring='roc_auc', cv=num_folds)
    cv.fit(X_train, y_train)
    if GENERATE_SUBMISSION:
        submission_test_data = pd.read_csv(TEST_PATH)
        if INITIAL_FEATURE_DROP:
            submission_test_data = feature_dropper.transform(submission_test_data)
        submission_probs_to_yes = cv.predict_proba(submission_test_data)[:, POSITIVE_CLASS_INDEX]
        submission_pd = pd.DataFrame({'ProbToYes': submission_probs_to_yes})
        submission_pd.set_index(submission_pd.index + 1, inplace=True)
        submission_pd.index.name = 'Id'
        submission_path = os.path.join(SUBMISSIONS_DIR + '/' + model_name + '_'
                                       + datetime.now().strftime("%d-%m-%H-%M-%S")
                                       + '.csv')
        submission_pd.to_csv(submission_path)

    test_yes_probs = cv.predict_proba(X_test)[:, POSITIVE_CLASS_INDEX]
    train_yes_probs = cv.predict_proba(X_train)[:, POSITIVE_CLASS_INDEX]
    result_test = scorer(y_true=y_test, y_score=test_yes_probs)
    result_train = scorer(y_true=y_train, y_score=train_yes_probs)
    print("Results for %s" % model_name)
    print("ROC score on test is : %f" % result_test)
    print("ROC score on train is : %f" % result_train)
    print("Best params achieved by: %s" % str(cv.best_params_))

    log_path = os.path.join(LOG_DIR + '/' + model_name + '_'
                            + datetime.now().strftime("%d-%m-%H-%M-%S")
                            + '_Best-%.3f' % result_test + '.log')
    with open(log_path, 'w+') as f:
        f.write("Results for %s" % model_name)
        f.write("ROC score on test is : %f\n" % result_test)
        f.write("ROC score on train is : %f\n" % result_train)
        f.write("Best params achieved by: %s\n" % str(cv.best_params_))
        f.write("Best PCA preserved %d features\n" % get_pca_num_components(cv))
        f.write("Pipeline steps: %s\n" % str(get_steps_names(cv)))
        f.write("All scores: %s\n" % str(cv.cv_results_))
        f.write("Feature drop applied %s" % ("Yes" if INITIAL_FEATURE_DROP else "No"))
        if GENERATE_SUBMISSION:
            f.write("Has corresponding submission: %s\n" % str(submission_path))
