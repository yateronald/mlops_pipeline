import yaml
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import *
from utils.utils import get_logger


def data_processing(congig_path: Text) -> None:
    # load all parameters

    with open(congig_path) as f:
        config = yaml.safe_load(open(congig_path))

    logger = get_logger("Data Processing", config["base"]["log_level"])
    df = pd.read_csv(config["data"]["datasets"])
