import pandas as pd
from typing import Text, Dict, List
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, f1_score
import numpy as np


class UnsupportedClassifier(Exception):
    """Raised when the estimator is not supported"""

    def __init__(self, estimator: Text):
        self.ms = f"{estimator} is not supported"
        super().__init__(self.ms)


def get_supportestimator(self) -> dict():
    """_summary_ : return the supported estimator"""
    return {
        "LogRe": LogisticRegression(),
        "svm": SVC(),
        "RandomForest": RandomForestClassifier(),
        "knn": KNeighborsClassifier(),
    }


def trainModel(
    features: np.array,
    target: pd.Series,
    estimator: Text,
    params: Dict,
    cv: int,
):
    """_summary_ : train the model with the given estimator and parameters
    _params_ :
         df : dataframe
        estimator : estimator name
        params : parameters for the estimator
        cv : number of cross validation
        target_name : target name
    _return_ : the trained model"""

    get_estimator = get_supportestimator(estimator)

    if estimator not in get_estimator.keys():
        raise UnsupportedClassifier(estimator)

    metrics = make_scorer(f1_score, average="weighted")
    estimator = get_estimator[estimator]
    model = GridSearchCV(estimator=estimator, param_grid=params, cv=cv, scoring=metrics)

    model.fit(features, target)

    return model
