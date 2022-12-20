import importlib
import os
import sys
import time

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

import submission_src.fincrime.solution_centralized as funcs

importlib.reload(funcs)

dirname = os.path.dirname("/mnt/d/fincrime-federated/")
os.chdir(dirname)


start_time = time.time()

model_dir = "/mnt/d/fincrime-federated/model/fincrime"
preds_format_path = (
    "/mnt/d/fincrime-federated/prediction/fincrime/prediction_format"
)
preds_dest_path = "/mnt/d/fincrime-federated/prediction/fincrime/prediction"


# predict on train data
datapathjsonString = 'data/fincrime/centralized/train/trail_data_2.json'
swift_data_path = funcs.json_to_dict(datapathjsonString)['swift_data_path']
bank_data_path = funcs.json_to_dict(datapathjsonString)['bank_data_path']

x_train,y_train = funcs.fit(swift_data_path = swift_data_path,
                    bank_data_path = bank_data_path,
                    model_dir = model_dir,
                    preds_format_path = preds_format_path,
                    preds_dest_path = preds_dest_path,
                    m = 'xgboost')


# predict on test data
datapathjsonString = 'data/fincrime/centralized/test/trail_data_2.json'
swift_data_path = funcs.json_to_dict(datapathjsonString)['swift_data_path']
bank_data_path = funcs.json_to_dict(datapathjsonString)['bank_data_path']

x_test,y_test = funcs.predict(
                    swift_data_path =swift_data_path,
                    bank_data_path = bank_data_path,
                    model_dir = model_dir,
                    preds_format_path = preds_format_path,
                    preds_dest_path = preds_dest_path,
                    m = 'xgboost'
                    )   

if __name__ == "__main__":

    #(X_train, y_train), (X_test, y_test) = utils.load_mnist()

    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(x_train, y_train, 10)[partition_id]

# preds_format_path = '/mnt/d/fincrime-federated/prediction/fincrime/prediction_format'
# preds_dest_path = '/mnt/d/fincrime-federated/prediction/fincrime/prediction'

# Fit the model on data
####model = funcs.train_model(X_train_data=X_train_data, Y_train_data = Y_train_data, m=m)
model = LogisticRegression(
    penalty="l2",
    max_iter=1,  # local epoch
    warm_start=True,  # prevent refreshing weights when fitting
)

utils.set_initial_params(model)

# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def get_parameters(self, config):  # type: ignore
        return utils.get_model_parameters(model)

    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        print(f"Training finished for round {config['server_round']}")
        return utils.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}


# # Start Flower client
# fl.client.start_numpy_client(
#     server_address="localhost:" + str(sys.argv[1]),
#     client=FlowerClient(),
#     grpc_max_message_length=1024 * 1024 * 1024,
# )

fl.client.start_numpy_client("0.0.0.0:8080", client=MnistClient())

