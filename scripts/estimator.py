import time

import cytoolz as cz
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras import layers
from keras.layers import Concatenate, Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from spektral.layers import (
    EdgeConditionedConv,
    GlobalAttentionPool,
    GlobalMaxPool,
    GraphConv,
)
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.utils import normalized_laplacian
from tqdm import tqdm


def get_data(data_path, params):

    df = pd.read_csv(data_path)

    df["is_rusher"] = df["NflId"] == df["NflIdRusher"]

    # df["Yards"] = np.log(1 + np.abs(df["Yards"])) * np.sign(df["Yards"])

    return df


def process_data(df, params, toy):

    if toy:
        plays = cz.take(10, df.groupby("PlayId"))
    else:
        plays = df.groupby("PlayId")

    plays = list(plays)

    Xs = []
    As = []
    Es = []

    for play_id, df_play in tqdm(plays, desc="Processing Plays"):

        pos = df_play[["X", "Y"]].to_numpy()
        A = np.sum((pos[:, np.newaxis, :] - pos[np.newaxis, :, :]) ** 2, axis=-1)
        A = 1.0 / (1.0 + A) * (1 - np.eye(22))
        As.append(normalized_laplacian(A))

        E = np.expand_dims(A, axis=-1)
        Es.append(E)

        # features
        features = dict(
            X=df_play["X"].to_numpy(),
            Y=df_play["Y"].to_numpy(),
            S=df_play["S"].to_numpy(),
            A=df_play["A"].to_numpy(),
            Orientation=df_play["Orientation"].to_numpy(),
            Dir=df_play["Dir"].to_numpy(),
            Team=df_play["Team"].to_numpy(),
            NflId=df_play["NflId"].to_numpy(),
            is_rusher=df_play["is_rusher"].to_numpy(),
            Yards=df_play["Yards"].to_numpy(),
        )
        Xs.append(features)

    Xs = {feature: np.stack([x[feature] for x in Xs], axis=0) for feature in Xs[0]}
    Es = np.stack(Es, axis=0).astype(np.float32)
    As = np.stack(As, axis=0).astype(np.float32)

    return Xs, As, Es


def process_Xs(Xs):

    return (
        np.stack(
            [
                Xs["X"].astype(np.float32),
                Xs["Y"].astype(np.float32),
                Xs["S"].astype(np.float32),
                Xs["A"].astype(np.float32),
                Xs["Orientation"].astype(np.float32),
                Xs["Dir"].astype(np.float32),
                Xs["is_rusher"].astype(np.float32),
                (Xs["Team"] == "home").astype(np.float32),
            ],
            axis=-1,
        ),
        np.expand_dims(Xs["Yards"].astype(np.float32), axis=-1)[:, 0],
    )


def get_model(params):

    # Model definition
    X_in = Input(shape=params["X_shape"])
    A_in = Input(shape=params["A_shape"])
    # E_in = Input(shape=params["E_shape"])
    # aux_in = Input(shape=params["aux_shape"])

    net = X_in
    A_exp = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(A_in)

    ################################
    # block
    ################################

    net = RelationalDense(32)([net, A_exp])
    # net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    net = MaxEdges()(net)
    # net = EdgeConditionedConv(32)([X_in, A_in, E_in])
    net = GraphConv(32)([net, A_in])
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)

    ################################
    # block
    ################################

    # net = RelationalDense(64)([net, A_exp])
    # # net = layers.BatchNormalization()(net)
    # net = layers.Activation("relu")(net)
    # net = MaxEdges()(net)
    # # net = EdgeConditionedConv(64)([net, A_in, E_in])
    # net = GraphConv(128)([net, A_in])
    # net = layers.BatchNormalization()(net)
    # net = layers.Activation("relu")(net)

    ################################
    # pooling
    ################################

    net = GlobalAttentionPool(128)(net)
    # net = GlobalMaxPool()(net)
    net = layers.Dropout(0.5)(net)

    ################################
    # block
    ################################

    # concat = Concatenate()([dense1, aux_in])
    net = Dense(128)(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)

    ################################
    # block
    ################################

    output = Dense(1)(net)

    ################################
    # model
    ################################

    # Build model
    # model = Model(inputs=[X_in, A_in, E_in], outputs=output)
    model = Model(inputs=[X_in, A_in], outputs=output)
    optimizer = Adam(lr=params["learning_rate"])
    model.compile(
        optimizer=optimizer, loss="mse", metrics=["mse", "mae", "mape"],
    )

    return model


class RelationalDense(layers.Dense):
    def __init__(
        self,
        channels,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            channels,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        dense_shape = (input_shape[0], input_shape[1], input_shape[1], input_shape[2])
        super().build(dense_shape)

    def call(self, x):

        if isinstance(x, (list, tuple)):
            if len(x) == 1:
                [x] = x
                E = None
            elif len(x) == 2:
                x, E = x
            else:
                raise ValueError(f"Invalid number of arguments, got {len(x)}")
        else:
            E = None

        x1 = K.expand_dims(x, axis=1)
        x1 = K.tile(x1, [1, x.shape[1], 1, 1])

        x2 = K.expand_dims(x, axis=2)
        x2 = K.tile(x2, [1, 1, x.shape[1], 1])

        if E is not None:
            net = K.concatenate([x2, x1, E], axis=-1)
        else:
            net = K.concatenate([x2, x1], axis=-1)

        output = self.dense(net)

        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        return (input_shape[0], input_shape[1], input_shape[1], self.units)


class AggEdges(layers.Layer):
    def __init__(self, reducer, **kwargs):
        self.reducer = reducer
        super().__init__(**kwargs)

    def call(self, x):
        return self.reducer(x, axis=2)


class MaxEdges(AggEdges):
    def __init__(self, **kwargs):
        super().__init__(K.max, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[:1] + input_shape[2:]
