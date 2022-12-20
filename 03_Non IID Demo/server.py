import flwr as fl
import sys
import numpy as np
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict

# class SaveModelStrategy(fl.server.strategy.FedAvg):
#     def aggregate_fit(
#         self,
#         rnd,
#         results,
#         failures
#     ):
#         aggregated_weights = super().aggregate_fit(rnd, results, failures)
#         if aggregated_weights is not None:
#             # Save aggregated_weights
#             print(f"Saving round {rnd} aggregated_weights...")
#             np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
#         return aggregated_weights

# # Create strategy and run server
# strategy = SaveModelStrategy()

# # Start Flower server for three rounds of federated learning
# fl.server.start_server(
#         server_address = 'localhost:'+str(sys.argv[1]) , 
#         config=fl.server.ServerConfig(num_rounds=3) ,
#         grpc_max_message_length = 1024*1024*1024,
#         strategy = strategy
# )

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

#     _, (X_test, y_test) = utils.load_mnist()

    def evaluate(parameters: fl.common.Weights):
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(x_test))
        accuracy = model.score(x_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=3))
    
    
