# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from datetime import time, datetime

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
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

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

import os

from transformers import BalancedHighNanFeaturesDropper, LowNanImputer
from utils import get_pca_num_components, get_steps_names

KAGGLE_ROOT = './input'
TRAIN_PATH = KAGGLE_ROOT + '/train.CSV/train.CSV'
TEST_PATH = KAGGLE_ROOT + '/test.CSV'
LOG_DIR = './logs'
SUBMISSIONS_DIR = './submissions'

NUM_OF_ATTRS = 452

NAN_AUTO_FILL_RATIO_LIMIT = 0.66

TEST_SIZE = 0.33
RANDOM_SEED = 42

TARGET_COLUMN = -1
POSITIVE_CLASS_INDEX = 1

INITIAL_FEATURE_DROP = True

GENERATE_SUBMISSION = True

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


if '__main__' == __name__:
    train = pd.read_csv(TRAIN_PATH)
    train.replace({"Yes": 1, "No": 0}, inplace=True)
    #    find_abnormal_nans(train, 'CLASS')

    # Split train
    X = train.iloc[:, :TARGET_COLUMN]
    y = train.iloc[:, TARGET_COLUMN]

    # y = pd.Series(LabelEncoder().fit_transform(y))  # TODO try to add this to pipeline

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED,
                                                        shuffle=True)

    if INITIAL_FEATURE_DROP:
        # Feature Drops, should be done here
        feature_dropper = BalancedHighNanFeaturesDropper().fit(X_train, y_train)
        X_train = feature_dropper.transform(X_train)
        X_test = feature_dropper.transform(X_test)

    # Nominal Imputations
    all_columns = list(X_train.columns)
    nominal_columns = [all_columns.index(col) for col in X_train.select_dtypes(np.object).columns]
    mf_nominal_imputer = SimpleImputer(strategy='most_frequent')

    # Numerical Imputations
    numeric_columns = [all_columns.index(col) for col in X_train.select_dtypes(np.number).columns]
    mean_numeric_imputer = SimpleImputer(strategy='mean')
    # Deal with NaNs in the nominal attributes

    # Full Imputers
    basic_imputer = ColumnTransformer([
        ("nominal", mf_nominal_imputer, nominal_columns),
        ("numerical", mean_numeric_imputer, numeric_columns)
    ])

    random_imputer = Pipeline(steps=[
        ("lowimputer", LowNanImputer()),
    ])

    iterative_imputer = ColumnTransformer([
        ("nominal", mf_nominal_imputer, nominal_columns),
        ("numerical", IterativeImputer(), numeric_columns)
    ])

    # Encoding
    one_hot_encoder = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown='ignore'), nominal_columns)],
                                        "passthrough")

    # Scale
    scaler = StandardScaler()

    # PCA
    pca = PCA()

    # Preprocessing Pipelines
    basic_preprocessing = Pipeline([("imputer", basic_imputer),
                                    ("ohe", one_hot_encoder)])

    random_preprocessing = Pipeline([("imputer", random_imputer),
                                     ("ohe", one_hot_encoder)])

    iter_preprocessing = Pipeline([("imputer", iterative_imputer),
                                   ("ohe", one_hot_encoder)])

    random_pca_preproceesing = Pipeline([("imputer", random_imputer),
                                         ("ohe", one_hot_encoder),
                                         ("scaler", scaler),
                                         ("pca", pca)])

    basic_pca_preprocessing = Pipeline([("imputer", basic_imputer),
                                        ("ohe", one_hot_encoder),
                                        ("scaler", scaler),
                                        ("pca", pca)])

    # X_train = random_preprocessing.fit_transform(X_train, y_train)
    # assert sum(sum(X_train == np.nan)) == 0

    # Models Construction
    explored_n_estimators = list(range(100, 450, 50))
    explored_nan_ratios = [0.3, 0.9, 0.95]
    explored_pca_components = [0.95, 0.99]
    explored_iterations = [10]
    explored_learning_rate = [0.01]
    explored_loss_function = ['Logloss']

    #########

    baseline_model = Pipeline([("preprocessing", basic_preprocessing),
                               ("model", RandomForestClassifier())])

    baseline_model_params = {
        'model__n_estimators': explored_n_estimators
    }

    #########

    baseline2_model = Pipeline([("preprocessing", basic_preprocessing),
                                ("model", ExtraTreesClassifier())])

    baseline2_model_params = {
        'model__n_estimators': explored_n_estimators
    }

    #########

    pca_basic_model = Pipeline([("preprocessing", basic_pca_preprocessing),
                                ("model", RandomForestClassifier())])

    pca_basic_model_params = {
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

    pca_forest_model = Pipeline([("preprocessing", basic_pca_preprocessing),
                                 ("model", RandomForestClassifier())])

    pca_forest_model_params = {
        'model__n_estimators': explored_n_estimators,
        'preprocessing__pca__n_components': explored_pca_components
    }

    #########

    random_model = Pipeline([("preprocessing", random_preprocessing),
                             ("model", RandomForestClassifier())])

    random_model_params = {'model__n_estimators': explored_n_estimators}

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

    basic_xgb_model = Pipeline([("preprocessing", basic_preprocessing),
                                ("model", xgb_object)])

    basic_xgb_model_params = {
        'model__n_estimators': explored_n_estimators,
        "model__learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "model__max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "model__colsample_bytree": [0.3, 0.4, 0.5, 0.7]
    }

    #########

    random_xgb_model = Pipeline([("preprocessing", random_preprocessing),
                                 ("model", xgb_object)])

    random_xgb_model_params = {'model__n_estimators': explored_n_estimators}

    #########

    catboost = CatBoostClassifier(
        iterations=5,
        learning_rate=0.1,
        loss_function='CrossEntropy'
    )

    #########

    basic_catboost_model = Pipeline([("preprocessing", basic_preprocessing),
                                     ("model", catboost)])

    basic_catboost_model_params = {
        'model__iterations': explored_iterations,
        'model__learning_rate': explored_learning_rate,
        'model__loss_function': explored_loss_function
    }

    #########

    random_catboost_model = Pipeline([("preprocessing", random_preprocessing),
                                      ("model", catboost)])

    random_catboost_model_params = {'model__iterations': explored_iterations,
                                    'model__learning_rate': explored_learning_rate,
                                    'model__loss_function': explored_loss_function}

    #########

    # Model Selection
    model_name = 'random_xgb_model'

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
