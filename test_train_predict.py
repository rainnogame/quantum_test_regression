import pathlib

import pandas as pd
import numpy as np

from train import train
from predict import predict


def test_train_predict(tmp_path: pathlib.Path):
    """
    Test the whole pipeline
    :param tmp_path: Temporary directory fixture
    """
    train_data_path = "data/train.csv"
    test_data_path = "data/hidden_test.csv"
    model_path = tmp_path / "model.pkl"
    results_path = tmp_path / "results.csv"

    train(train_data_path, str(model_path))

    # Check model saved
    assert model_path.exists()

    predict(test_data_path, str(model_path), str(results_path))

    # Check results saved
    assert results_path.exists()

    # Check results are correct (corresponds to the exact solution)
    test_data = pd.read_csv(test_data_path)
    test_results = pd.read_csv(results_path)

    exact_solution = (test_data["6"] ** 2 + test_data["7"]).values
    predicted_solution = test_results["target"].values
    np.allclose(exact_solution, predicted_solution, atol=1e-2)
