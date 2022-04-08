from cProfile import label
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
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer
import optuna


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

    @validator("test_size")
    def test_size_valid(cls, test_size):
        assert 0 <= test_size <= 1
        return test_size


@dataclass
class PreprocessConfig:
    imputer: TransformerMixin
    encoders: List[TransformerMixin]
    compresser: TransformerMixin


@dataclass
class ModelConfig(BaseModel):
    model: BaseEstimator


@dataclass
class PostprocessConfig(BaseModel):
    postprocessor: TransformerMixin


@dataclass
class StructuredPipelineConfig:
    preprocess: PreprocessConfig
    model: ModelConfig
    postprocess: PostprocessConfig


class PipelineFactory:
    def create_from_config(pipeline_config: StructuredPipelineConfig) -> Pipeline:
        return Pipeline(
            steps=[
                (f"imputer", pipeline_config.preprocess.imputer),
                *[
                    (f"encoder__{i}", pipeline_config.preprocess.encoders[i])
                    for i in range(len(pipeline_config.preprocess.encoders))
                ]("compresser", pipeline_config.preprocess.compresser),
                ("model", pipeline_config.model.model),
                ("postprocess", pipeline_config.postprocess.postprocessor),
            ]
        )


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


@hydra.main(config_name="config", config_path="config")
def main(conf: StudyConfig) -> None:
    data = load_data(conf.data_path)
    if conf.debug_mode:
        sample_indices = np.random.choice(
            len(data.data), size=conf.debug_mode_sample_size
        )
        data.data = data.data.iloc[sample_indices]
        data.labels = data.labels.iloc[sample_indices]

    if conf.test_data_path is not None:
        test_data = load_data(conf.test_data_path)
        train_data = data
    else:
        train_data, test_data = split_data(data, conf)

    x_train, y_train = train_data.data, train_data.labels
    x_test, y_test = test_data.data, test_data.labels

    score_func = create_score_func(conf)
    objective_func = create_objective_function(
        train_data.data, train_data.labels, score_func, conf
    )
    study = optuna.create_study(
        direction=conf.objective_optim_direction,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=30, interval_steps=10
        ),
    )
    study.optimize(objective_func, n_trials=conf.num_trials)
    best_params = study.best_params
    best_model = PipelineFactory().create_from_config(best_pipeline_config)
    best_model.fit(X=x_train, y=y_train)
    y_test_preds = best_model.predict(X=x_test)
    test_score = score_func(y_test_preds, y_test)
    save_result(best_params, best_model, y_test_preds, test_score, conf)


def load_data(path: str) -> LabeledData:
    raw_data = pd.read_csv(path)
    labeled_data = LabeledData(raw_data.iloc[:, :-1], raw_data.iloc[:, -1])
    return labeled_data


def save_result(best_params, best_model, y_test_preds, test_score, conf: StudyConfig):
    pass


def create_pipeline_from_trial(
    trial: optuna.trial, pipeline_config, study_config
) -> Pipeline:
    pass


def create_score_func(study_config: StudyConfig) -> Callable[..., float]:
    return metrics.get_scorer(study_config.scoring)


def create_objective_function(x_train, y_train, score_func, study_config: StudyConfig):
    def objective(trial: optuna.Trial):
        pipeline = create_pipeline_from_trial(trial)
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


pipeline_config = {
    "model": {
        "name": "xgboost",
        "params": {
            "n_estimators": 5,
            "learning_rate": 1e-1,
        },
    },
    "preprocess": {
        "imputer": {"name": "mean", "params": {}},
        "encoders": [{}, {}],
        "compresser": {"name": "PCA", "params": {}},
    },
    "postprocess": {"name": "identity"},
}

if __name__ == "__main__":
    main()
