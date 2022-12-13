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
    m="rf",
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
    m="rf",
    a=1,
)

# # Fit the model on data
# model = funcs.train_model(
#     X_train_data=X_train_data, Y_train_data=Y_train_data, m=m
# )

# # score on train data
# pred, pred_proba = funcs.score_data(
#     train_data=train_data_2,
#     X_train_data=X_train_data,
#     Y_train_data=Y_train_data,
#     model=model,
#     m=m,
# )

# # formatted predictions file
# preds = pd.read_csv(
#     preds_format_path + "/train_pred_format.csv"
# )  # ,index_col='MessageId')
# preds.loc[:, "Score"] = pred_proba

# # save below
# pickle.dump(scaler, open(model_dir + "/finalized_scaler_" + m + ".sav", "wb"))
# pickle.dump(model, open(model_dir + "/finalized_model_" + m + ".sav", "wb"))
# # preds.to_csv(preds_dest_path + '/centralized_train_predictions_' + m + '.csv')
# # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


# # load the model from disk
# model = pickle.load(open(model_dir + "/finalized_model_" + m + ".sav", "rb"))

# # score on train data
# pred, pred_proba = score_data(
#     train_data=train_data_2,
#     X_train_data=X_train_data,
#     Y_train_data=Y_train_data,
#     model=model,
#     m=m,
# )

# # format predictions file
# preds = pd.read_csv(
#     preds_format_path + "/test_pred_format.csv"
# )  # ,index_col='MessageId')
# preds.loc[:, "Score"] = pred_proba

# # save below
# preds.to_csv(preds_dest_path + "/centralized_test_predictions_" + m + ".csv")


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    # def get_parameters(self, config):
    #     return model.get_weights()

    # def get_parameters(self, config):
    #     # print(f"[Client {self.cid}] get_parameters")
    #     return get_parameters(self)

    def fit(self, parameters, config):
        # model.set_weights(parameters)
        r = model.fit(
            x_train,
            y_train,
            # epochs=1,
            # validation_data=(x_test, y_test),
            # verbose=0,
        )
        # hist = r.history
        # print("Fit history : ", hist)
        #return len(x_train), {}
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
