import os
import time

import fire
import keras
import numpy as np
import plotly.graph_objs as go
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error

from . import estimator


def main(data_path, params_path, toy=False, cache=False):

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    if cache and os.path.exists("input/cache.npz"):
        data = np.load("input/cache.npz")

        X = data["X"]
        y = data["y"]
        As = data["As"]
        Es = data["Es"]

    else:
        df = estimator.get_data(data_path, params)

        Xs, As, Es = estimator.process_data(df, params, toy)

        X, y = estimator.process_Xs(Xs)

        np.savez("input/cache.npz", X=X, y=y, As=As, Es=Es)

    print("X", X.shape, X.dtype)
    print("As", As.shape, As.dtype)
    print("Es", Es.shape, Es.dtype)
    print("y", y.shape, y.dtype)

    np.set_printoptions(suppress=True)

    params["X_shape"] = X.shape[1:]
    params["A_shape"] = As.shape[1:]
    params["E_shape"] = Es.shape[1:]

    # baseline
    y_mean = np.mean(y) * np.ones_like(y)

    print("BASELINE")
    print(
        f"MSE: {mean_squared_error(y, y_mean)}, MAE: {mean_absolute_error(y, y_mean)}"
    )

    # model
    model = estimator.get_model(params)
    model.summary()

    model.fit(
        # [X, As, Es],
        [X, As],
        y,
        validation_split=0.1,
        shuffle=True,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        # verbose=1,
        callbacks=[keras.callbacks.TensorBoard(log_dir=f"models/{int(time.time())}")],
    )


if __name__ == "__main__":
    fire.Fire(main)
