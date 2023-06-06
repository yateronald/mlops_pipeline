import yaml
from sklearn.model_selection import train_test_split
from typing import Text
from utils.utils import get_logger
import argparse
import pandas as pd
import os


def datasplit(config_file: Text) -> None:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logger = get_logger("Data Split", config["base"]["log_level"])

    df = pd.read_csv(config["data"]["datasets"])
    xtrain, xtest, ytrain, ytest = train_test_split(
        df,
        test_size=config["data"]["split_ratio"],
        random_state=config["base"]["random_state"],
    )

    xtrain.to_csv(config["path"]["xtrain_path"], index=False)
    xtest.to_csv(config["path"]["xtest_path"], index=False)
    ytrain.to_csv(config["path"]["ytrain_path"], index=False)
    ytest.to_csv(config["path"]["ytest_path"], index=False)

    logger.info("Data Split Successfully")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config",
        default="params.yaml",
        help="path to config file",
        type=str,
        dest="config_file",
    )
    args = arg_parser.parse_args()
    datasplit(config_file=args.config_file)
