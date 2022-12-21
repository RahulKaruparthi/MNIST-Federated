import os
import sys
from typing import Dict

import flwr as fl
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import submission_src.fincrime.solution_centralized as funcs
import utils


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    # _, (X_test, y_test) = utils.load_mnist()

    dirname = os.path.dirname("/mnt/d/fincrime-federated/")
    os.chdir(dirname)

    model_dir = "/mnt/d/fincrime-federated/model/fincrime"
    preds_format_path = (
        "/mnt/d/fincrime-federated/prediction/fincrime/prediction_format"
    )
    preds_dest_path = "/mnt/d/fincrime-federated/prediction/fincrime/prediction"

    datapathjsonString = "data/fincrime/centralized/test/trail_data_2.json"
    swift_data_path = funcs.json_to_dict(datapathjsonString)["swift_data_path"]
    bank_data_path = funcs.json_to_dict(datapathjsonString)["bank_data_path"]

    x_test, y_test = funcs.predict(
        swift_data_path=swift_data_path,
        bank_data_path=bank_data_path,
        model_dir=model_dir,
        preds_format_path=preds_format_path,
        preds_dest_path=preds_dest_path,
        m="rf",
        a=2,
    )

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(x_test))
        accuracy = model.score(x_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    # model = HistGradientBoostingClassifier(
    #     loss="log_loss",
    #     learning_rate=0.1,
    #     max_iter=10,
    #     max_leaf_nodes=31,
    #     max_depth=None,
    #     min_samples_leaf=20,
    #     l2_regularization=0.0,
    #     max_bins=255,
    #     warm_start=False,
    #     early_stopping="auto",
    #     scoring="loss",
    #     validation_fraction=0.1,
    #     n_iter_no_change=10,
    #     tol=1e-07,
    #     # verbose=0,
    #     random_state=None,
    # )
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    # fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=3))

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="localhost:" + str(sys.argv[1]),
        config=fl.server.ServerConfig(num_rounds=3),
        grpc_max_message_length=1024 * 1024 * 1024,
        strategy=strategy,
    )

    # # Start Flower client
    # fl.client.start_numpy_client(
    #     server_address="localhost:" + str(sys.argv[1]),
    #     client=MnistClient(),
    #     grpc_max_message_length=1024 * 1024 * 1024,
    # )
