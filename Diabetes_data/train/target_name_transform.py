import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np


def target_transform(target: pd.Series) -> np.array:
    label_enc = LabelEncoder()
    label_enc.fit(target)

    joblib.dump(label_enc, "target_metadata.joblib")

    return label_enc.transform(target)


def inerverse_target_transform(target: pd.Series) -> pd.Series:
    if not os.path.exists("target_metadata.json"):
        raise FileNotFoundError("target_metadata.json not found")

    with open("target_metadata.joblib", "r") as f:
        label_enc = joblib.load("target_metadata.joblib")

    return label_enc.transform(target)
