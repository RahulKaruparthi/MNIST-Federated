import importlib
import os
import time

dirname = os.path.dirname("D:/fincrime-federated/")
os.chdir(dirname)
print(os.getcwd())

# import submission_src.fincrime.solution_centralized as funcs
import solution_centralized as funcs

importlib.reload(funcs)

start_time = time.time()

model_dir = "D:/fincrime-federated/model/fincrime"
preds_format_path = (
    "D:/fincrime-federated/prediction/fincrime/prediction_format"
)
preds_dest_path = "D:/fincrime-federated/prediction/fincrime/prediction"


## train on data
datapathjsonString = "data/fincrime/centralized/train/data.json"
swift_data_path = funcs.json_to_dict(datapathjsonString)["swift_data_path"]
bank_data_path = funcs.json_to_dict(datapathjsonString)["bank_data_path"]

funcs.fit(
    swift_data_path=swift_data_path,
    bank_data_path=bank_data_path,
    model_dir=model_dir,
    preds_format_path=preds_format_path,
    preds_dest_path=preds_dest_path,
    m="xgboost",
)


# predict on test data

datapathjsonString = "data/fincrime/centralized/test/data.json"
swift_data_path = funcs.json_to_dict(datapathjsonString)["swift_data_path"]
bank_data_path = funcs.json_to_dict(datapathjsonString)["bank_data_path"]

funcs.predict(
    swift_data_path=swift_data_path,
    bank_data_path=bank_data_path,
    model_dir=model_dir,
    preds_format_path=preds_format_path,
    preds_dest_path=preds_dest_path,
    m="xgboost",
)

print("Total time taken ", (time.time() - start_time), " seconds")
