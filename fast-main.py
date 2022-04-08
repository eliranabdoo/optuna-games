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
from sklearn import metrics
from catboost import CatBoostClassifier
import xgboost as xgb
from joblib import dump
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import shap
import matplotlib.pyplot as plt
import os
import random

from transformers import BalancedHighNanFeaturesDropper, LowNanImputer
from utils import get_pca_num_components, get_steps_names

KAGGLE_ROOT = './input'
TRAIN_PATH = KAGGLE_ROOT + '/train.CSV/train.CSV'
TEST_PATH = KAGGLE_ROOT + '/test.CSV'
LOG_DIR = './logs'
PLOTS_DIR = './plots'
SUBMISSIONS_DIR = './submissions'
MODELS_DIR = './models'

PLOTS_FORMAT = 'png'

EPSILON = 1

NUM_OF_ATTRS = 452

NAN_AUTO_FILL_RATIO_LIMIT = 0.66

TEST_SIZE = 0.66
RANDOM_SEED = 42

TARGET_COLUMN = -1
POSITIVE_CLASS_INDEX = 1

PLOT_TOP_FEATURES = 5
NUM_EXPLAINED_ROWS = 3

INITIAL_FEATURE_DROP = True

GENERATE_SUBMISSION = True
SAVE_MODEL = True

FEATURE_DROP_THRESHOLD = 0.8
BALANCE_THRESHOLD = 0.1


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
def get_submission_path(model_name):
    return os.path.join(SUBMISSIONS_DIR + '/' + model_name + '_'
                        + datetime.now().strftime("%d-%m-%H-%M-%S")
                        + '.csv')


def generate_submission(model_name):
    submission_test_data = pd.read_csv(TEST_PATH)
    if INITIAL_FEATURE_DROP:
        submission_test_data = feature_dropper.transform(submission_test_data)
    submission_probs_to_yes = cv.predict_proba(submission_test_data)[:, POSITIVE_CLASS_INDEX]
    submission_pd = pd.DataFrame({'ProbToYes': submission_probs_to_yes})
    submission_pd.set_index(submission_pd.index + 1, inplace=True)
    submission_pd.index.name = 'Id'
    submission_path = get_submission_path(model_name)
    submission_pd.to_csv(submission_path)

def generate_model_stamp(model_name, test_score):
    return os.path.join(MODELS_DIR + '/' + model_name + '_'
                        + datetime.now().strftime("%d-%m-%H-%M-%S")
                        + '_%.3f' % test_score + '.model')
    return os.path.join(path)

def print_results(model_name, result_test, result_train, best_params_str):
    print("Results for %s" % model_name)
    print("ROC score on test is : %f" % result_test)
    print("ROC score on train is : %f" % result_train)
    print("Best params achieved by: %s" % str(cv.best_params_))


def get_log_path(model_name, result_test):
    return os.path.join(LOG_DIR + '/' + model_name + '_'
                        + datetime.now().strftime("%d-%m-%H-%M-%S")
                        + '_Best-%.3f' % result_test + '.log')


def create_logs(model_name, result_test, result_train, cv):
    log_path = get_log_path(model_name, result_test)

    with open(log_path, 'w+') as f:
        f.write("Results for %s" % model_name)
        f.write("ROC score on test is : %f\n" % result_test)
        f.write("ROC score on train is : %f\n" % result_train)
        f.write("Best params achieved by: %s\n" % str(cv.best_params_))
        f.write("Best PCA preserved %d features\n" % get_pca_num_components(cv))
        f.write("Pipeline steps: %s\n" % str(get_steps_names(cv)))
        f.write("All scores: %s\n" % str(cv.cv_results_))
        if INITIAL_FEATURE_DROP:
            f.write("Feature drop applied with drop threshold %f and balance threshold %f\n" % (
                FEATURE_DROP_THRESHOLD, BALANCE_THRESHOLD))
            if GENERATE_SUBMISSION:
                f.write("Has corresponding submission: %s\n"
                        % str(get_submission_path(model_name)))


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


def update_model_params(curr_model_params, improvment, cv):
    pass

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
    cols = preproc["ohe"].get_feature_names()

    X_test_trans = pd.DataFrame.from_records(preproc.transform(X_test), columns=cols)
    X_train_trans = pd.DataFrame.from_records(preproc.transform(X_train), columns=cols)
    
    # create a SHAP explainer and values
    explainer = shap.TreeExplainer(model)
    
    # summary_plot
    shap_values = explainer.shap_values(X_test_trans)
    shap.summary_plot(shap_values, X_test_trans, plot_type="bar", max_display=PLOT_TOP_FEATURES, show=False)

    shap_summary_plot_path = get_shap_plot_path(model_name, result_test, "all", "summary") 
    plt.savefig(shap_summary_plot_path, bbox_inches='tight', format=PLOTS_FORMAT)
    
    # dependence_plot
    for rank in range(PLOT_TOP_FEATURES):
        shap.dependence_plot("rank(%d)" % rank, shap_values, X_test_trans,
                             interaction_index=None, show=False)
        shap_depend_plot_path = get_shap_plot_path(model_name, result_test,
                                                   "all", "depend-rank-%d" % rank) 
        plt.savefig(shap_depend_plot_path, bbox_inches='tight', format=PLOTS_FORMAT)

    # visualize explanation for NUM_EXPLAINED_ROWS predictions
    rows_to_show = random.sample(range(0, len(X_test)), NUM_EXPLAINED_ROWS)
    for row in range(NUM_EXPLAINED_ROWS):
        # create shap_values
        data_for_prediction = X_test_trans.iloc[rows_to_show[row]]
        shap_values = explainer.shap_values(data_for_prediction)
        # create plots
        shap.force_plot(explainer.expected_value, shap_values, data_for_prediction,
                        matplotlib=True, show=False)
        shap_force_plot_path = get_shap_plot_path(model_name, result_test, row, "force") 
        plt.savefig(shap_force_plot_path, bbox_inches='tight', format=PLOTS_FORMAT)
       
        # save row
        row_path = get_shap_row_path(model_name, result_test, row)
        data_for_prediction.to_csv(row_path)




if '__main__' == __name__:
    train = pd.read_csv(TRAIN_PATH)
    #train = pd.read_csv("/tmp/FIFA 2018 Statistics.csv")
    train.replace({"Yes": 1, "No": 0}, inplace=True)
    #    find_abnormal_nans(train, 'CLASS')

    X_train, X_test, y_train, y_test = split_train(train)

    if INITIAL_FEATURE_DROP:
        # Feature Drops, should be done here
        feature_dropper = BalancedHighNanFeaturesDropper(high_nans_threshold=FEATURE_DROP_THRESHOLD,
                                                         balance_threhsold=BALANCE_THRESHOLD).fit(X_train,
                                                                                                  y_train)
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
        ("lowimputer", LowNanImputer(low_nans_threshold=FEATURE_DROP_THRESHOLD,
                                     balance_threshold=BALANCE_THRESHOLD)),
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
    explored_n_estimators = [10]
    explored_nan_ratios = [0.95]
    explored_pca_components = [0.5]
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
        objective='binary:logistic',
    )

    #########

    basic_xgb_model = Pipeline([("preprocessing", basic_preprocessing),
                                ("model", xgb_object)])

    basic_xgb_model_params = {
        'model__n_estimators': explored_n_estimators,
        "model__learning_rate": [0.05],
        "model__max_depth": [2],
        "model__min_child_weight": [3,5],
        "model__gamma": [0.1],
        "model__subsample": [1]
        # "model__colsample_bytree": [0.05, 0.15,0.3]
    }

    #########

    random_xgb_model = Pipeline([("preprocessing", random_preprocessing),
                                 ("model", xgb_object)])

    random_xgb_model_params = basic_xgb_model_params

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
    model_name = 'basic_xgb_model'

    model_params = globals()[model_name + '_params']  # keep the convention
    model = globals()[model_name]

    prev_result_test = 0

    # perform grid search
    num_folds = 5
    cv = GridSearchCV(estimator=model, param_grid=model_params, scoring='roc_auc', cv=num_folds)
    cv.fit(X_train, y_train)


    if GENERATE_SUBMISSION:
        generate_submission(model_name)

    # calculate results
    result_test, result_train = calculate_results(cv, X_test, X_train, y_test, y_train)

    # print results
    print_results(model_name, result_test, result_train, str(cv.best_params_))

    # create logs
    create_logs(model_name, result_test, result_train, cv)

    # make SHAP plots
    generate_shap(cv.best_estimator_['model'], X_train, X_test, result_test, basic_preprocessing)

    # create AUC curver
    y_pred_proba = cv.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.savefig("/tmp/auc.png", bbox_inches='tight', format=PLOTS_FORMAT)
