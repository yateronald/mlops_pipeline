import sys

sys.path.append(
    r"C:\Users\ASSEKEYATE\Desktop\Data Science Project\MLOps\mlops_pipeline\Diabetes_data"
)
from utils.utils import get_logger
from train.train import trainModel
from train.target_name_transform import target_transform
import yaml
import argparse
import warnings
from joblib import load
import pandas as pd
from typing import Text
from joblib import dump

warnings.filterwarnings("ignore")


def predict(config_path: Text) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = get_logger("TRAIN", log_level=config["base"]["log_level"])

    load_trans = load(config["path"]["tranform_path"])

    # load all data
    logger.info("Loading data")

    xtrain = pd.read_csv(config["path"]["xtrain_path"])
    xtest = pd.read_csv(config["path"]["xtest_path"])
    ytrain = pd.read_csv(config["path"]["ytrain_path"])
    ytest = pd.read_csv(config["path"]["ytest_path"])

    # transform xtrain data
    logger.info("Data Tranformation")
    xtrain = load_trans.transform(xtrain)
    xtest = load_trans.transform(xtest)

    # transform target data

    ytrain = target_transform(ytrain)
    ytest = target_transform(ytest)

    ytrain = ytrain.reshape(
        ytrain.shape[0],
    )
    ytest = ytest.reshape(
        ytest.shape[0],
    )

    # train model
    logger.info("Training Model")
    model_name = config["train"]["random_forest"]["name"]
    params = config["train"]["random_forest"]["params"]

    model = trainModel(xtrain, ytrain, model_name, params, cv=5)

    logger.info("Model Training Completed")

    logger.info(f"best score: {model.best_score_}")
    # save model
    logger.info("Saving Model")
    with open(config["path"]["model_path"], "wb") as f:
        dump(model, f)


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
    predict(config_path=args.config_path)
