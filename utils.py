import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype
from scipy.stats import spearmanr


def get_pca_num_components(cv):
    try:
        return cv.best_estimator_.named_steps.preprocessing.named_steps.pca.n_components_
    except AttributeError:
        return -1


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


def find_abnormal_nans(df, target_col, nan_percent=0.95, abnormal_diff=0.05):
    total_balance = df[target_col].mean()
    for col in df.columns[df.isnull().sum() / len(df) > nan_percent]:
        balance = df[target_col][df[col].isnull()].mean()
        # print("For col %s balance is %f" % (col, balance))
        if np.abs(1 - (balance / total_balance)) > abnormal_diff:
            print("Abnormal balance for col %s, balance is %f" % (col, balance))

    print("Total balance is %f" % df[target_col].mean())


def get_numeric_columns_indices(df):
    columns = list(df.columns)
    return [columns.index(col) for col in df.select_dtypes(np.number).columns]


def get_nominal_columns_indices(df):
    columns = list(df.columns)
    return [columns.index(col) for col in df.select_dtypes(np.object).columns]


def correlation_calc(col1, col2):
    type1 = col1.dtype
    type2 = col2.dtype
    #if is_numeric_dtype(type1) and is_numeric_dtype(type2):
    try:
        return spearmanr(col1, col2).correlation
    #else:
    except:
        raise Exception("Supporting only num and num")
