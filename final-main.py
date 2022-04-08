# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from datetime import time, datetime
from functools import partial

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score as scorer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn import metrics
from catboost import CatBoostClassifier
import xgboost as xgb
from joblib import dump
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from lightgbm import LGBMClassifier, Dataset
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, spearmanr, chisquare
from sklearn.feature_selection import VarianceThreshold

import os

from transformers import BalancedHighNanFeaturesDropper, LowNanImputer, LowCorrelationDropper, LowVarianceDropper, \
    ConvertToLGBDataset, SimpleMetricsInjector, OHESplitMetricsInjector, SumInjector, LowCorrelationDropperPost
from utils import get_pca_num_components, get_steps_names, get_nominal_columns_indices, get_numeric_columns_indices

figure_num = 1

KAGGLE_ROOT = './input'
TRAIN_PATH = KAGGLE_ROOT + '/train.CSV/train.CSV'
TEST_PATH = KAGGLE_ROOT + '/test.CSV'
LOG_DIR = './logs'
SUBMISSIONS_DIR = './submissions'
MODELS_DIR = './models'

PLOTS_FORMAT = 'png'
PLOT_TOP_FEATURES = 5
PLOTS_DIR = './plots'
NUM_EXPLAINED_ROWS = 3

EPSILON = 1

NUM_OF_ATTRS = 452

NAN_AUTO_FILL_RATIO_LIMIT = 0.66

TEST_SIZE = 0.2
MAX_DEGRADES = 10
NUM_FOLDS = 5
NUM_SAMPLES = 1
RANDOM_SEED = 42

TARGET_COLUMN = -1
POSITIVE_CLASS_INDEX = 1

INITIAL_FEATURE_DROP = False

GENERATE_SUBMISSION = True

USE_ORDINAL_INSTEADOF_OHE = False  # Requires some feature dropping to avoid unknown categories

FEATURE_DROP_THRESHOLD = 1  # Features with higher NaN ratio are dropped
BALANCE_THRESHOLD = 0.1  # Feature with lower balance 
CORRELATION_DROP_THRESHOLD = 0.05  # Drop features with lower correlation

GENERATE_SHAP = False

if GENERATE_SHAP:
    import shap
    import random


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

def close_and_increase_plt():
    plt.close(globals()["figure_num"])
    globals()["figure_num"] += 1

def get_truncated_normal(mean=0., sd=1., low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def get_integer_truncated_normal(*args, **kwargs):
    res = get_truncated_normal(*args, **kwargs)

    def integer_wrap(*args, f=None, **kwargs):
        return int(np.round(f(*args, **kwargs)))

    res.rvs = partial(integer_wrap, f=res.rvs)
    return res


def sample_integer(dist):
    return int(np.round(dist.rvs()))


def generate_submission(model_name, model, result_test, result_train, feature_dropper):
    submission_test_data = pd.read_csv(TEST_PATH)
    if INITIAL_FEATURE_DROP:
        submission_test_data = feature_dropper.transform(submission_test_data)
    submission_probs_to_yes = model.predict_proba(submission_test_data)[:, POSITIVE_CLASS_INDEX]
    submission_pd = pd.DataFrame({'ProbToYes': submission_probs_to_yes})
    submission_pd.set_index(submission_pd.index + 1, inplace=True)
    submission_pd.index.name = 'Id'
    submission_path = get_submission_path(model_name, result_test, result_train)
    submission_pd.to_csv(submission_path)
    return submission_path


def generate_model_stamp(model_name, test_score):
    return os.path.join(MODELS_DIR + '/' + model_name + '_'
                        + datetime.now().strftime("%d-%m-%H-%M-%S")
                        + '_%.3f' % test_score + '.model')


def print_results(model_name, result_test, result_train, best_params_str):
    print("Results for %s" % model_name)
    print("ROC score on test is : %f" % result_test)
    print("ROC score on train is : %f" % result_train)
    print("Best params achieved by: %s" % str(cv.best_params_))


def get_log_path(model_name, result_test, result_train):
    return os.path.join(LOG_DIR + '/' + model_name + '_'
                        + datetime.now().strftime("%d-%m-%H-%M-%S")
                        + '_Best-%.3f' % result_test
                        + '_GAP%.3f' % (result_train - result_test) + '.log')


def get_submission_path(model_name, result_test, result_train):
    return os.path.join(SUBMISSIONS_DIR + '/' + model_name + '_'
                        + datetime.now().strftime("%d-%m-%H-%M-%S")
                        + '_%.3f' % result_test
                        + '_GAP%.3f' % (result_train - result_test) + '.csv')


def create_logs(model_name, result_test, result_train, cv):
    log_path = get_log_path(model_name, result_test, result_train)

    with open(log_path, 'w+') as f:
        f.write("Results for %s" % model_name)
        f.write("ROC score on test is : %f\n" % result_test)
        f.write("ROC score on train is : %f\n" % result_train)
        f.write("Best params achieved by: %s\n" % str(cv.best_params_))
        f.write("Best PCA preserved %d features\n" % get_pca_num_components(cv))
        f.write("Pipeline steps: %s\n" % str(get_steps_names(cv)))
        f.write("All scores: %s\n" % str(cv.cv_results_))
        if INITIAL_FEATURE_DROP:
            f.write("NaNFeature drop applied with drop threshold %f and balance threshold %f\n" % (
                FEATURE_DROP_THRESHOLD, BALANCE_THRESHOLD))
        if GENERATE_SUBMISSION:
            f.write("Has corresponding submission: %s\n"
                    % str(get_submission_path(model_name, result_test, result_train)))


def calculate_results(cv, X_test, X_train, y_test, y_train):
    test_yes_probs = cv.predict_proba(X_test)[:, POSITIVE_CLASS_INDEX]
    train_yes_probs = cv.predict_proba(X_train)[:, POSITIVE_CLASS_INDEX]
    result_test = scorer(y_true=y_test, y_score=test_yes_probs)
    result_train = scorer(y_true=y_train, y_score=train_yes_probs)

    return result_test, result_train


def split_train(train):
    # Split train
    X = train.iloc[:, :TARGET_COLUMN]
    y = train.iloc[:, TARGET_COLUMN]

    # y = pd.Series(LabelEncoder().fit_transform(y))  # TODO try to add this to pipeline

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED,
                            shuffle=True)


def result_delta(prev_result_test, result_test):
    return result_test - prev_result_test


def update_model_params(curr_model_params, improvement, cv, params_ranges):
    new_params_grid = {}
    best_params = cv.best_params_
    for param_name, sample_space in curr_model_params.items():
        if type(sample_space) is list:  # Discrete
            new_params_grid[param_name] = sample_space
        else:  # Continuous, we assume it's truncnorm
            old_mean, old_std = sample_space.mean(), sample_space.std()
            alpha = improvement  # TODO
            best_param = best_params[param_name]
            # new_mean = alpha * best_param + (1 - alpha) * old_mean
            new_mean = best_param
            new_std = old_std * 0.95
            if type(sample_space.rvs()) is int:
                new_distribution = get_integer_truncated_normal(mean=new_mean, sd=new_std,
                                                                low=params_ranges[param_name][0],
                                                                upp=params_ranges[param_name][1])
            else:
                new_distribution = get_truncated_normal(mean=new_mean, sd=new_std, low=params_ranges[param_name][0],
                                                        upp=params_ranges[param_name][1])
            new_params_grid[param_name] = new_distribution
    print("After improvement %f, New Grid Is: %s" % (improvement, str(new_params_grid)))
    return new_params_grid


def get_shap_row_path(model_name, result_test, row):
    return os.path.join(PLOTS_DIR + '/' + model_name + '_'
                        + datetime.now().strftime("%d-%m-%H-%M-%S")
                        + '_Best-%.3f' % result_test
                        + 'row-%d' % row)


def get_shap_plot_path(model_name, result_test, row, plot_type):
    return os.path.join(PLOTS_DIR + '/' + model_name + '_'
                        + datetime.now().strftime("%d-%m-%H-%M-%S")
                        + '_Best-%.3f' % result_test
                        + 'shap_'
                        + plot_type
                        + '_plot_row-%s' % str(row)
                        + '.' + PLOTS_FORMAT)


def generate_shap(model, X_train, X_test, result_test, preproc):
    # transform datasets
    preproc = preproc.fit(X_train)
    try:
        cols = preproc[encoder_tuple[0]].get_feature_names()
    # Ordinal encoder has no function get_feature_names
    except AttributeError as e:
        cols = X_test.columns

    X_test_trans = pd.DataFrame.from_records(preproc.transform(X_test), columns=cols)

    # create a SHAP explainer and values
    explainer = shap.TreeExplainer(model)

    # summary_plot
    plt.figure(globals()["figure_num"])
    shap_values = explainer.shap_values(X_test_trans)
    shap.summary_plot(shap_values[1], X_test_trans, plot_type="dot", max_display=PLOT_TOP_FEATURES, show=False)

    shap_summary_plot_path = get_shap_plot_path(model_name, result_test, "all", "summary")
    plt.savefig(shap_summary_plot_path, bbox_inches='tight', format=PLOTS_FORMAT)
    close_and_increase_plt()

    # visualize explanation for NUM_EXPLAINED_ROWS predictions
    rows_to_show = random.sample(range(0, len(X_test)), NUM_EXPLAINED_ROWS)
    for row in range(NUM_EXPLAINED_ROWS):
        plt.figure(globals()["figure_num"])
        row_num = rows_to_show[row]
        # create shap_values
        data_for_prediction = X_test_trans.iloc[row_num]
        # create plots
        shap.force_plot(explainer.expected_value[1], shap_values[1][row_num], data_for_prediction,
                        matplotlib=True, show=False)
        shap_force_plot_path = get_shap_plot_path(model_name, result_test, row, "force")
        plt.savefig(shap_force_plot_path, bbox_inches='tight', format=PLOTS_FORMAT)

        # save row
        row_path = get_shap_row_path(model_name, result_test, row)
        data_for_prediction.to_csv(row_path)
        close_and_increase_plt()
    
    # dependence_plot
    for rank in range(PLOT_TOP_FEATURES):
        plt.figure(globals()["figure_num"])
        shap.dependence_plot("rank(%d)" % rank, shap_values[1], X_test_trans,
                             interaction_index=None, show=False)
        shap_depend_plot_path = get_shap_plot_path(model_name, result_test,
                                                   "all", "depend-rank-%d" % rank)
        plt.savefig(shap_depend_plot_path, bbox_inches='tight', format=PLOTS_FORMAT)
        close_and_increase_plt()



def get_auc_curve_path(model_name, result_test):
    return os.path.join(PLOTS_DIR + '/' + model_name + '_'
                        + datetime.now().strftime("%d-%m-%H-%M-%S")
                        + '_Best-%.3f' % result_test
                        + 'auc_curve'
                        + '.' + PLOTS_FORMAT)


def create_auc_curve(model_name, result_test, X_test, y_test):
    y_pred_proba = cv.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    auc_curve_plot_path = get_auc_curve_path(model_name, result_test)
    plt.savefig(auc_curve_plot_path, bbox_inches='tight', format=PLOTS_FORMAT)


if '__main__' == __name__:
    train = pd.read_csv(TRAIN_PATH)
    train.replace({"Yes": 1, "No": 0}, inplace=True)
    #    find_abnormal_nans(train, 'CLASS')

    X_train, X_test, y_train, y_test = split_train(train)
    feature_dropper = None
    if INITIAL_FEATURE_DROP:
        # Feature Drops, should be done here
        feature_dropper = Pipeline(steps=[
            ("highnans", BalancedHighNanFeaturesDropper(high_nans_threshold=FEATURE_DROP_THRESHOLD,
                                                        balance_threhsold=BALANCE_THRESHOLD)),
            ("lowcor", LowCorrelationDropper(CORRELATION_DROP_THRESHOLD)),
            ("lowvar", LowVarianceDropper())
        ])
        feature_dropper.fit(X_train, y_train)
        X_train = feature_dropper.transform(X_train)
        X_test = feature_dropper.transform(X_test)

    # Nominal Imputations
    nominal_columns = get_nominal_columns_indices(X_train)
    mf_nominal_imputer = SimpleImputer(strategy='most_frequent')

    # Numerical Imputations
    numeric_columns = get_numeric_columns_indices(X_train)
    mean_numeric_imputer = SimpleImputer(strategy='mean')
    # Deal with NaNs in the nominal attributes

    # Full Imputers
    basic_imputer = ColumnTransformer([
        ("nominal", mf_nominal_imputer, nominal_columns),
        ("numerical", mean_numeric_imputer, numeric_columns)
    ])

    # Encoding
    one_hot_encoder = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown='ignore'), nominal_columns)],
                                        "passthrough")

    ordinal_encoder = ColumnTransformer(
        [("ordinal_encoder", OrdinalEncoder(), nominal_columns)],
        "passthrough")

    # Scale
    scaler = ColumnTransformer(
        [("scaler", StandardScaler(), numeric_columns)],
        "passthrough")

    # Injector
    sum_injector = SumInjector()
    basic_injector = SimpleMetricsInjector()
    advanced_injector = OHESplitMetricsInjector()

    # PCA
    pca = PCA()

    dropper = Pipeline(steps=[
        ("corr", LowCorrelationDropperPost(CORRELATION_DROP_THRESHOLD)),
        ("variance", VarianceThreshold())]
    )

    encoder_tuple = ("ordinal", ordinal_encoder) if USE_ORDINAL_INSTEADOF_OHE else ("ohe", one_hot_encoder)
    # Preprocessing Pipelines
    basic_preprocessing = Pipeline([("imputer", basic_imputer),
                                    encoder_tuple,
                                   ("injector", sum_injector),
                                    ])

    # Models Construction
    params_ranges = {
        "model__n_estimators": (5, 10000),
        "model__learning_rate": (1e-10, 1),
        "model__gamma": (0, 2),
        "model__max_depth": (0, 100),
        "model__lambda": (0, 100),
        "model__reg_lambda": (0, 100),
        "model__alpha": (0, 100),
        "model__reg_alpha": (0, 100),
        "model__colsample_bytree": (0, 5),
        "model__num_leaves": (0, 10000),
        "model__subsample": (0, 1)
    }

    #########

    lxgb_object = LGBMClassifier()

    basic_lxgb_model = Pipeline([("preprocessing", basic_preprocessing),
                                 ("model", lxgb_object)])

    basic_lxgb_model_params = {
        "model__n_estimators": [226],
        "model__learning_rate": [0.11902610209163136],
        "model__max_depth": [11],
        "model__reg_lambda": [21.35594809111083],
        "model__num_leaves": [28],
        "model__objective": ['binary'],
    }

    #########

    # Model Selection
    model_name = 'basic_lxgb_model'

    model_params = globals()[model_name + '_params']  # keep the convention
    model = globals()[model_name]

    prev_result_test = 0
    max_degrades = MAX_DEGRADES
    curr_degrades = 0
    # perform grid search
    num_folds = NUM_FOLDS
    n_samples = NUM_SAMPLES
    cv = RandomizedSearchCV(estimator=model, param_distributions=model_params, scoring='roc_auc', cv=num_folds,
                            n_iter=n_samples)

    # cv.fit(X_train, y_train, **{'model__categorical_feature': nominal_columns})
    cv.fit(X_train, y_train)

    # calculate results
    result_test, result_train = calculate_results(cv, X_test, X_train, y_test, y_train)

    if GENERATE_SUBMISSION:
        submission_path = generate_submission(model_name, model=cv, result_test=result_test,
                                              result_train=result_train, feature_dropper=feature_dropper)

    # print results
    print_results(model_name, result_test, result_train, str(cv.best_params_))

    # create logs
    create_logs(model_name, result_test, result_train, cv)

    # create AUC curve
    plt.figure(globals()["figure_num"])
    create_auc_curve(model_name, result_test, X_test, y_test)
    close_and_increase_plt()

    # make SHAP plots
    if GENERATE_SHAP:
        generate_shap(cv.best_estimator_['model'], X_train, X_test, result_test, basic_preprocessing)

    # check if there is still room for improvment
    improvement = result_delta(prev_result_test, result_test)
    print("Curr Improvement %f" % improvement)
