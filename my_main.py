from cProfile import label
from multiprocessing import Pipe
from uuid import uuid4
from venv import create

# from pydantic.dataclasses import dataclass
from dataclasses import dataclass
from logging.config import dictConfig
from typing import Callable, Dict, List, Optional, Tuple
import hydra
import numpy as np
from omegaconf import DictConfig
from pydantic import BaseModel, validator
from sklearn import pipeline
import sklearn
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer
import optuna
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from types import MappingProxyType

EmptyMap = MappingProxyType({})


class StudyConfig(BaseModel):
    data_path: str
    test_data_path: Optional[str]
    debug_mode: bool
    debug_mode_sample_size: int
    cv_num_folds: int
    scoring: str
    num_trials: int
    objective_optim_direction: str
    uid: str = str(uuid4())
    test_size: float
    n_estimators_range: Tuple[int, int]

    @validator("test_size")
    def test_size_valid(cls, test_size):
        assert 0 <= test_size <= 1
        return test_size


@dataclass
class LabeledData:
    data: pd.DataFrame
    labels: pd.Series


def split_data(data: LabeledData, conf: StudyConfig) -> Tuple[LabeledData, LabeledData]:
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.labels,
        test_size=conf.test_size,
        shuffle=conf.shuffle_data,
        stratify=conf.stratify_split,
    )
    train_data = LabeledData(X_train, y_train)
    test_data = LabeledData(X_test, y_test)
    return train_data, test_data


class PipelineFactory:
    def __init__(self, numeric_cols, categorial_cols):
        self.numeric_cols = numeric_cols
        self.categorial_cols = categorial_cols

    def create_from_args(
        self,
        numeric_imputer_cls,
        categorial_imputer_cls,
        col_dropper_cls,
        categorial_encoder_cls,
        scaler_cls,
        compressor_cls,
        model_cls,
        postprocessor_cls=None,
        col_dropper_params=EmptyMap,
        numeric_imputer_params=EmptyMap,
        categorial_imputer_params=EmptyMap,
        categorial_encoder_params=EmptyMap,
        scaler_params=EmptyMap,
        compressor_params=EmptyMap,
        model_params=EmptyMap,
        postprocessor_params=EmptyMap,
    ) -> Pipeline:
        pipe_steps = []

        preprocess = Pipeline(
            steps=[
                (
                    "imputer",
                    ColumnTransformer(
                        [
                            (
                                "numeric_imputer",
                                numeric_imputer_cls(**numeric_imputer_params),
                                self.numeric_cols,
                            ),
                            (
                                "categorial_imputer",
                                categorial_imputer_cls(**categorial_imputer_params),
                                self.categorial_cols,
                            ),
                        ]
                    ),
                ),
                (
                    "categorial_encoder",
                    ColumnTransformer(
                        [
                            "encoder_0",
                            categorial_encoder_cls(**categorial_encoder_params),
                            self.categorial_cols,
                        ]
                    ),
                )(
                    "col_dropper",
                    ColumnTransformer([("col_dropper_0", col_dropper_cls(**col_dropper_params))]),
                ),
                ("scaler", scaler_cls(**scaler_params)),
                ("compressor", compressor_cls(**compressor_params)),
            ]
        )
        pipe_steps.append(preprocess)

        model = Pipeline(steps=[("model", model_cls(**model_params))])
        pipe_steps.append(model)

        if postprocessor_cls is not None:
            postprocess = Pipeline(steps=[("postprocess", postprocessor_cls(**postprocessor_params))])
            pipe_steps.append(postprocess)

        res = Pipeline(steps=pipe_steps)
        return res

    def create_from_optuna(self, trial: optuna.Trial, study_config: StudyConfig):

        res = self.create_from_args(
            numeric_imputer_cls=trial.suggest_categorical(name="numeric_imputer_cls", choices=[SimpleImputer]),
            numeric_imputer_params=trial.suggest_categorical(
                "numeric_imputer_params", [{"strategy": trial.suggest_categorical(name="strategy", choices=["mean"])}]
            ),
            categorial_imputer_cls=trial.suggest_categorical(name="numeric_imputer_cls", choices=[SimpleImputer]),
            categorial_imputer_params=trial.suggest_categorical(
                "categorial_imputer_params",
                [{"strategy": trial.suggest_categorical(name="strategy", choices=["most_frequent"])}],
            ),
            col_dropper_cls=trial.suggest_categorical(name="col_dropper", choices=[VarianceThreshold]),
            col_dropper_params=trial.suggest_categorical(
                name="col_dropper_params",
                choices=[{"threshold": trial.suggest_categorical("threshold", choices=[0.0])}],
            ),
            compressor_cls=trial.suggest_categorical(name="compressor_cls", choices=[PCA]),
            compressor_params=trial.suggest_categorical(
                name="compressor_params",
                choices=[{"n_components": trial.suggest_categorical(name="n_components", choices=[5])}],
            ),
            categorial_encoder_cls=trial.suggest_categorical(name="categorial_encoder_cls", choices=[OneHotEncoder]),
            model_cls=trial.suggest_categorical(name="model_cls", choices=[XGBClassifier]),
            model_params=trial.suggest_categorical(
                name="model_params",
                choices=[
                    {
                        "n_estimators": trial.suggest_int(
                            "n_estimators",
                            low=study_config.n_estimators_range[0],
                            high=study_config.n_estimators_range[1],
                        )
                    }
                ],
            ),
        )
        return res


def load_data(path: str) -> LabeledData:
    raw_data = pd.read_csv(path)
    labeled_data = LabeledData(raw_data.iloc[:, :-1], raw_data.iloc[:, -1])
    return labeled_data


def save_result(best_params, best_model, y_test_preds, test_score, conf: StudyConfig):
    pass


def create_score_func(study_config: StudyConfig) -> Callable[..., float]:
    return metrics.get_scorer(study_config.scoring)


def create_objective_function(
    x_train, y_train, score_func, pipeline_factory: PipelineFactory, study_config: StudyConfig
):
    def objective(trial: optuna.Trial):
        pipeline = pipeline_factory.create_from_optuna(trial=trial, study_config=study_config)
        cv_score = cross_val_score(
            estimator=pipeline,
            X=x_train,
            y=y_train,
            scoring=score_func,
            cv=study_config.cv_num_folds,
        )
        score = cv_score.mean()
        return score

    return objective


@hydra.main(config_name="config", config_path="config")
def main(conf: StudyConfig) -> None:
    data = load_data(conf.data_path)
    if conf.debug_mode:
        sample_indices = np.random.choice(len(data.data), size=conf.debug_mode_sample_size)
        data.data = data.data.iloc[sample_indices]
        data.labels = data.labels.iloc[sample_indices]

    if conf.test_data_path is not None:
        test_data = load_data(conf.test_data_path)
        train_data = data
    else:
        train_data, test_data = split_data(data, conf)

    x_train, y_train = train_data.data, train_data.labels
    x_test, y_test = test_data.data, test_data.labels

    pipeline_factory = PipelineFactory(
        numeric_cols=x_train.select_dtypes("number").columns,
        categorial_cols=x_train.select_dtypes("object").columns,
    )

    score_func = create_score_func(conf)
    objective_func = create_objective_function(train_data.data, train_data.labels, score_func, conf)

    study = optuna.create_study(
        direction=conf.objective_optim_direction,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10),
    )
    study.optimize(objective_func, n_trials=conf.num_trials)

    best_params = study.best_params
    best_model = pipeline_factory.create_from_args(**best_params)

    best_model.fit(X=x_train, y=y_train)
    y_test_preds = best_model.predict(X=x_test)
    test_score = score_func(y_test_preds, y_test)

    save_result(best_params, best_model, y_test_preds, test_score, conf)


if __name__ == "__main__":
    main()
