import yaml
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import *

# from utils.utils import get_logger
import os
import argparse


def data_processing(congig_path: Text) -> None:
    # load all parameters

    with open(congig_path) as f:
        config = yaml.safe_load(open(congig_path))

    # logger = get_logger("Data Processing", config["base"]["log_level"])

    xtrain = pd.read_csv(config["path"]["xtrain_path"])
    # Data preprocessing pipeline
    numeric_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("power_transformer", PowerTransformer(method="yeo-johnson")),
        ]
    )

    categorical_pipeline = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    transformation = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                xtrain.select_dtypes(include=["number"]).columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                xtrain.select_dtypes(include=["object"]).columns,
            ),
        ],
        remainder="passthrough",
    )

    xtrain_transformed = transformation.fit(xtrain)
    xtrain = xtrain_transformed.transform(xtrain)

    with open(config["path"]["tranform_path"], "wb") as f:
        dump(transformation, f)

    # logger.info("Data Processing Successfully")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config",
        default="params.yaml",
        help="path to config file",
        type=str,
        dest="config_path",
    )
    args = arg_parser.parse_args()
    data_processing(congig_path=args.config_path)
