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


## train on data
datapathjsonString = "data/fincrime/centralized/train/trail_data_1.json"
swift_data_path = funcs.json_to_dict(datapathjsonString)["swift_data_path"]
bank_data_path = funcs.json_to_dict(datapathjsonString)["bank_data_path"]

x_train, y_train, model = funcs.fit(
    swift_data_path=swift_data_path,
    bank_data_path=bank_data_path,
    model_dir=model_dir,
    preds_format_path=preds_format_path,
    preds_dest_path=preds_dest_path,
    m="xgboost",
    a=1,
)


# predict on test data
datapathjsonString = "data/fincrime/centralized/test/trail_data_1.json"
swift_data_path = funcs.json_to_dict(datapathjsonString)["swift_data_path"]
bank_data_path = funcs.json_to_dict(datapathjsonString)["bank_data_path"]

x_test, y_test = funcs.predict(
    swift_data_path=swift_data_path,
    bank_data_path=bank_data_path,
    model_dir=model_dir,
    preds_format_path=preds_format_path,
    preds_dest_path=preds_dest_path,
    m="xgboost",
    a=1,
)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# Define Flower client
class FlowerClient(fl.client.NumPyClient, trainloader, valloader):
    def __init__(self, trainloader, valloader):
        # self.cid = cid
        # self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return model.get_weights()

    # def get_parameters(self, config):
    #     # print(f"[Client {self.cid}] get_parameters")
    #     return get_parameters(self)

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(
            x_train,
            y_train,
            epochs=1,
            validation_data=(x_test, y_test),
            verbose=0,
        )
        hist = r.history
        print("Fit history : ", hist)
        # return r, len(x_train), {}
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        # model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:" + str(sys.argv[1]),
    client=FlowerClient(),
    grpc_max_message_length=1024 * 1024 * 1024,
)
