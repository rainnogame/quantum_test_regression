import argparse
import pickle

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def train(data_path: str, model_save_path: str):
    """
    Train regression model and save it to the file
    :param data_path: Path to the training data
    :param model_save_path: Path to save the model
    """
    data = pd.read_csv(data_path)

    x, y = data.drop("target", axis=1).values, data["target"].values

    # Use features #6^2 and #7 according to EDA
    features = np.vstack((x[:, 6] ** 2, x[:, 7])).T
    model = LinearRegression()
    model.fit(features, y)

    # Save the model
    pickle.dump(model, open(model_save_path, "wb"))


if __name__ == "__main__":
    # Use arguments parser oto for the script for command line usage
    parser = argparse.ArgumentParser(description="Train regression model")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the file with a data. Default: data/train.csv",
        default="data/train.csv",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        help="Path to save the model. Default: data/model.pkl",
        default="data/model.pkl",
    )

    args = parser.parse_args()
    train(args.data_path, args.model_save_path)
