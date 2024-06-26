import argparse
import pickle

import pandas as pd
import numpy as np
import os


def predict(data_path: str, model_path: str, results_path: str):
    """
    Run regression model over prediction data and save results
    :param data_path: Path to the data to predict
    :param model_path: Path to the model
    :param results_path: Path to save the result
    """
    data = pd.read_csv(data_path)
    x = data.values

    # Use features #6^2 and #7 according to EDA
    features = np.vstack((x[:, 6] ** 2, x[:, 7])).T

    # Load the model
    model = pickle.load(open(model_path, "rb"))
    result = model.predict(features)

    # Save the results
    pd.DataFrame({"target": result}).to_csv(results_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression model")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the file with a data. Default: data/hidden_test.csv",
        default="hidden_test.csv",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained model. Default: data/model.pkl",
        default="model.pkl",
    )

    parser.add_argument(
        "--results_path",
        type=str,
        help="Path to save the results. Default: data/results.csv",
        default="data/results.csv",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise ValueError("Data path doesn't exist")

    if not os.path.exists(args.model_path):
        raise ValueError("Model path doesn't exist")

    predict(args.data_path, args.model_path, args.results_path)
