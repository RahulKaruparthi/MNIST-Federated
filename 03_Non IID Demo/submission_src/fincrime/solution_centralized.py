# import json
# # from sklearn.externals import joblib
# import pickle
# from importlib.resources import path
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import sklearn.utils
# from sklearn import metrics
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (accuracy_score, classification_report,
#                              confusion_matrix)
# from sklearn.model_selection import (KFold, ShuffleSplit, StratifiedKFold,
#                                      StratifiedShuffleSplit, cross_val_score,
#                                      train_test_split)
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from xgboost import XGBClassifier


# def json_to_dict(datapathjsonString):
#     datapathJson = open(datapathjsonString)
#     datapathDict = json.load(datapathJson)
#     return datapathDict


# def load_data(swift_data_path, bank_data_path):
#     swift_data = pd.read_csv(swift_data_path, index_col="MessageId")
#     swift_data["Timestamp"] = swift_data["Timestamp"].astype("datetime64[ns]")
#     bank_data = pd.read_csv(bank_data_path)
#     return swift_data, bank_data


# def create_pred_format(preds_format_path, suffix, dt) -> None:
#     ### Creating the pred format csv file
#     pred_file = pd.DataFrame(
#         index=dt.index, columns=["Score"]
#     )  # .reset_index(names = ['MessageId'])
#     pred_file["Score"] = 0
#     pred_file.to_csv(
#         preds_format_path + "/" + suffix + "_pred_format.csv", index=False
#     )


# def freq_by_features(
#     df,
#     level_1,
#     level_2,
#     model_dir,
#     save_as_object=False,
#     map_from_train_set=False,
# ) -> None:

#     if map_from_train_set == True:

#         level_1_level_2_frequency = pickle.load(
#             open(
#                 model_dir
#                 + "/"
#                 + str(level_1 + "_" + level_2 + "_frequency")
#                 + ".sav",
#                 "rb",
#             )
#         )
#         _level_1 = df.loc[:, level_1].unique()
#         _level_2 = df.loc[:, level_2].unique()
#         df[str(level_1 + "_" + level_2)] = df.loc[:, level_1] + df.loc[
#             :, level_2
#         ].astype(str)
#         df.loc[:, str(level_1 + "_" + level_2 + "_frequency")] = df.loc[
#             :, str(level_1 + "_" + level_2)
#         ].map(level_1_level_2_frequency)
#     else:
#         _level_1 = df.loc[:, level_1].unique()
#         _level_2 = df.loc[:, level_2].unique()
#         df[str(level_1 + "_" + level_2)] = df.loc[:, level_1] + df.loc[
#             :, level_2
#         ].astype(str)
#         level_1_level_2_frequency = {}
#         for s in _level_1:
#             level_1_rows = df[df[level_1] == s]
#             for h in _level_2:
#                 level_1_level_2_frequency[s + str(h)] = len(
#                     level_1_rows[level_1_rows.loc[:, level_2] == h]
#                 )

#         df.loc[:, str(level_1 + "_" + level_2 + "_frequency")] = df.loc[
#             :, str(level_1 + "_" + level_2)
#         ].map(level_1_level_2_frequency)

#         if save_as_object == True:
#             pickle.dump(
#                 level_1_level_2_frequency,
#                 open(
#                     model_dir
#                     + "/"
#                     + str(level_1 + "_" + level_2 + "_frequency")
#                     + ".sav",
#                     "wb",
#                 ),
#             )

#     ...


# # def mean_by_features (
# #     df,
# #     level_1,
# #     level_2
# # ) -> None:

# #     _level_1 = df.loc[:,level_1].unique()
# #     _level_2 = df.loc[:,level_2].unique()
# #     df[str(level_1 + "_" + level_2)] = df.loc[:,level_1] + df.loc[:,level_2].astype(str)
# #     level_1_level_2_mean = {}
# #     for s in _level_1:
# #         level_1_rows = df[df[level_1] == s]
# #         for h in _level_2:
# #             level_1_level_2_mean[s + str(h)] = \
# #                 level_1_rows[level_1_rows.loc[:,level_2] == h][level_2].mean()

# #     return level_1_level_2_mean


# def create_features(df, model_dir, map_from_train_set=False):

#     ## Feature Engineering

#     # Hour column
#     df["hour"] = df["Timestamp"].dt.hour

#     # Hour frequency for each sender
#     freq_by_features(
#         df=df,
#         level_1="Sender",
#         level_2="hour",
#         model_dir=model_dir,
#         save_as_object=True,
#         map_from_train_set=map_from_train_set,
#     )
#     # Hour frequency for each receiver
#     freq_by_features(
#         df=df,
#         level_1="Receiver",
#         level_2="hour",
#         model_dir=model_dir,
#         save_as_object=True,
#         map_from_train_set=map_from_train_set,
#     )
#     # Sender-Currency Frequency
#     freq_by_features(
#         df=df,
#         level_1="Sender",
#         level_2="InstructedCurrency",
#         model_dir=model_dir,
#         save_as_object=True,
#         map_from_train_set=map_from_train_set,
#     )
#     # Receiver-SettledCurrency Frequency
#     freq_by_features(
#         df=df,
#         level_1="Receiver",
#         level_2="SettlementCurrency",
#         model_dir=model_dir,
#         save_as_object=True,
#         map_from_train_set=map_from_train_set,
#     )
#     # Sender-Receiver Frequency
#     freq_by_features(
#         df=df,
#         level_1="Sender",
#         level_2="Receiver",
#         model_dir=model_dir,
#         save_as_object=True,
#         map_from_train_set=map_from_train_set,
#     )

#     # # Average Amount per Sender-Currency - not working
#     # Sender_ICurrency_mean = mean_by_features(df = df, level_1="Sender" , level_2="InstructedCurrency")
#     # df.loc[:,"Sender_ICurrency_mean"] = \
#     #     df.loc[:,"Sender_InstructedCurrency"].map(Sender_ICurrency_mean)

#     # Numbering the transactions within a account order - ben - date combination
#     df = df.sort_values(
#         by=[
#             "SettlementDate",
#             "Sender",
#             "Receiver",
#             "Account_order",
#             "Account_ben",
#             "Timestamp",
#         ],
#         ascending=True,
#     )
#     df["seq"] = (
#         df.groupby(
#             [
#                 "SettlementDate",
#                 "Sender",
#                 "Receiver",
#                 "Account_order",
#                 "Account_ben",
#             ]
#         ).cumcount()
#         + 1
#     )
#     df["seq"] = df["seq"].replace(np.NAN, 1)

#     # Flag columns for transactions with missing bank details
#     df[["MissingBenAccount"]] = 0
#     df.loc[df["Flags_ben"].isnull(), "MissingBenAccount"] = 1
#     df[["MissingOrdAccount"]] = 0
#     df.loc[df["Flags_order"].isnull(), "MissingOrdAccount"] = 1

#     # Different sender account number from bank details
#     df["DifferentOrderNum"] = np.where(
#         df["Account_order"] == df["OrderingAccount"], 0, 1
#     )
#     # Different receiver account number from bank details
#     df["DifferentBenNum"] = np.where(
#         df["Account_ben"] == df["BeneficiaryAccount"], 0, 1
#     )

#     # Different sender account name from bank details
#     df["DifferentOrderName"] = np.where(
#         df["Name_order"] == df["OrderingName"], 0, 1
#     )
#     # Different receiver account name from bank details
#     df["DifferentBenName"] = np.where(
#         df["Name_ben"] == df["BeneficiaryName"], 0, 1
#     )

#     # Different sender account ordering street from bank details
#     df["DifferentOrderStreet"] = np.where(
#         df["Street_order"] == df["OrderingStreet"], 0, 1
#     )
#     # Different receiver account ordering street from bank details
#     df["DifferentBenStreet"] = np.where(
#         df["Street_ben"] == df["BeneficiaryStreet"], 0, 1
#     )

#     # Different sender account country code/zip from bank details
#     df["DifferentOrderZip"] = np.where(
#         df["CountryCityZip_order"] == df["OrderingCountryCityZip"], 0, 1
#     )
#     # Different receiver account country code/zip from bank details
#     df["DifferentBenZip"] = np.where(
#         df["CountryCityZip_ben"] == df["BeneficiaryCountryCityZip"], 0, 1
#     )

#     # Some missing value treatment
#     df.loc[df["Flags_ben"].isna(), "Flags_ben"] = 99

#     return df


# def train_model(X_train_data, Y_train_data, m) -> None:

#     if m == "rf":

#         model = RandomForestClassifier(
#             max_depth=7, random_state=0, n_estimators=10
#         )

#     elif m == "xgboost":

#         model = XGBClassifier(n_estimators=100)

#     kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
#     cv_results = cross_val_score(
#         model, X_train_data, Y_train_data, cv=kfold, scoring="f1"
#     )

#     model.fit(X_train_data, Y_train_data)
#     print("Minimum:", cv_results.min())
#     print("Maximum:", cv_results.max())
#     print("StanDev:", cv_results.std())

#     return model


# def score_data(train_data, X_train_data, Y_train_data, model, m) -> None:

#     pred = model.predict(X_train_data)
#     print(
#         m + " Classification Report=\n\n",
#         classification_report(Y_train_data, pred),
#     )
#     print(m + " Confusion Matrix=\n\n", confusion_matrix(Y_train_data, pred))

#     pred_proba = model.predict_proba(X_train_data)[:, 1]
#     print(
#         "AUPRC:",
#         metrics.average_precision_score(
#             y_true=Y_train_data, y_score=pred_proba
#         ),
#     )

#     importances = model.feature_importances_
#     forest_importances = pd.Series(
#         importances, index=train_data.drop(["Label"], axis=1).columns
#     )
#     print(forest_importances)

#     return pred, pred_proba


# def fit(
#     swift_data_path,
#     bank_data_path,
#     model_dir,
#     preds_format_path,
#     preds_dest_path,
#     m,
# ) -> None:
#     """Function that fits your model on the provided training data and saves
#     your model to disk in the provided directory.

#     Args:
#         swift_data_path (Path): Path to CSV data file for the SWIFT transaction
#             dataset.
#         bank_data_path (Path): Path to CSV data file for the bank account
#             dataset.
#         model_dir (Path): Path to a directory that is constant between the train
#             and test stages. You must use this directory to save and reload
#             your trained model between the stages.
#         preds_format_path (Path): Path to CSV file matching the format you must
#             write your predictions with, filled with dummy values.
#         preds_dest_path (Path): Destination path that you must write your test
#             predictions to as a CSV file.

#     Returns: None
#     """

#     train_data, bank_data = load_data(
#         swift_data_path=swift_data_path, bank_data_path=bank_data_path
#     )

#     # Merging with bank details
#     train_data = pd.merge(
#         train_data,
#         bank_data,
#         left_on="OrderingAccount",
#         right_on="Account",
#         how="left",
#     )
#     train_data = pd.merge(
#         train_data,
#         bank_data,
#         left_on="BeneficiaryAccount",
#         right_on="Account",
#         how="left",
#         suffixes=["_order", "_ben"],
#     )

#     # Creating predictions format file
#     create_pred_format(
#         preds_format_path=preds_format_path, dt=train_data, suffix="train"
#     )

#     # Feature engineering

#     train_data = create_features(
#         df=train_data, model_dir=model_dir, map_from_train_set=False
#     )

#     # Keep below columns for training and testing
#     cols_to_keep = [
#         "SettlementAmount",
#         "InstructedAmount",
#         "Label",
#         "hour",
#         "Flags_ben",
#         "MissingBenAccount",
#         "MissingOrdAccount",
#         "Sender_hour_frequency",
#         # 'sender_currency_amount_average',
#         "Sender_Receiver_frequency",
#         "Sender_InstructedCurrency_frequency",
#         "seq",
#         # 'receiver_transactions',
#         "Receiver_SettlementCurrency_frequency",
#         "Receiver_hour_frequency",
#         "DifferentOrderNum",
#         "DifferentBenNum",
#         "DifferentOrderName",
#         "DifferentBenName",
#         "DifferentOrderStreet",
#         "DifferentBenStreet",
#         "DifferentOrderZip",
#         "DifferentBenZip",
#     ]

#     train_data_2 = train_data[cols_to_keep]
#     print(train_data_2.head(3))

#     # Separate DV - IDV series
#     Y_train_data = train_data_2["Label"].values
#     X_train_data = train_data_2.drop(["Label"], axis=1).values

#     # Normalize

#     scaler = StandardScaler()
#     scaler.fit(X_train_data)
#     X_train_data = scaler.transform(X_train_data)

#     # Fit the model on data
#     model = train_model(
#         X_train_data=X_train_data, Y_train_data=Y_train_data, m=m
#     )

#     # score on train data
#     pred, pred_proba = score_data(
#         train_data=train_data_2,
#         X_train_data=X_train_data,
#         Y_train_data=Y_train_data,
#         model=model,
#         m=m,
#     )

#     # formatted predictions file
#     preds = pd.read_csv(
#         preds_format_path + "/train_pred_format.csv"
#     )  # ,index_col='MessageId')
#     preds.loc[:, "Score"] = pred_proba

#     # save below
#     pickle.dump(
#         scaler, open(model_dir + "/finalized_scaler_" + m + ".sav", "wb")
#     )
#     pickle.dump(model, open(model_dir + "/finalized_model_" + m + ".sav", "wb"))
#     preds.to_csv(
#         preds_dest_path + "/centralized_train_predictions_" + m + ".csv"
#     )


# def predict(
#     swift_data_path,
#     bank_data_path,
#     model_dir,
#     preds_format_path,
#     preds_dest_path,
#     m,
# ) -> None:
#     """Function that loads your model from the provided directory and performs
#     inference on the provided test data. Predictions should match the provided
#     format and be written to the provided destination path.

#     Args:
#         swift_data_path (Path): Path to CSV data file for the SWIFT transaction
#             dataset.
#         bank_data_path (Path): Path to CSV data file for the bank account
#             dataset.
#         model_dir (Path): Path to a directory that is constant between the train
#             and test stages. You must use this directory to save and reload
#             your trained model between the stages.
#         preds_format_path (Path): Path to CSV file matching the format you must
#             write your predictions with, filled with dummy values.
#         preds_dest_path (Path): Destination path that you must write your test
#             predictions to as a CSV file.

#     Returns: None

#     """

#     train_data, bank_data = load_data(
#         swift_data_path=swift_data_path, bank_data_path=bank_data_path
#     )

#     # Merging with bank details
#     train_data = pd.merge(
#         train_data,
#         bank_data,
#         left_on="OrderingAccount",
#         right_on="Account",
#         how="left",
#     )
#     train_data = pd.merge(
#         train_data,
#         bank_data,
#         left_on="BeneficiaryAccount",
#         right_on="Account",
#         how="left",
#         suffixes=["_order", "_ben"],
#     )

#     # Creating predictions format file
#     create_pred_format(
#         preds_format_path=preds_format_path, dt=train_data, suffix="test"
#     )

#     # Feature engineering

#     train_data = create_features(
#         df=train_data, model_dir=model_dir, map_from_train_set=True
#     )

#     # Keep below columns for training and testing
#     cols_to_keep = [
#         "SettlementAmount",
#         "InstructedAmount",
#         "Label",
#         "hour",
#         "Flags_ben",
#         "MissingBenAccount",
#         "MissingOrdAccount",
#         "Sender_hour_frequency",
#         # 'sender_currency_amount_average',
#         "Sender_Receiver_frequency",
#         "Sender_InstructedCurrency_frequency",
#         "seq",
#         # 'receiver_transactions',
#         "Receiver_SettlementCurrency_frequency",
#         "Receiver_hour_frequency",
#         "DifferentOrderNum",
#         "DifferentBenNum",
#         "DifferentOrderName",
#         "DifferentBenName",
#         "DifferentOrderStreet",
#         "DifferentBenStreet",
#         "DifferentOrderZip",
#         "DifferentBenZip",
#     ]

#     train_data_2 = train_data[cols_to_keep]
#     print(train_data_2.head(3))

#     # Separate DV - IDV series

#     Y_train_data = train_data_2["Label"].values
#     X_train_data = train_data_2.drop(["Label"], axis=1).values

#     # load scaler from disk
#     scaler = pickle.load(
#         open(model_dir + "/finalized_scaler_" + m + ".sav", "rb")
#     )
#     X_train_data = scaler.transform(X_train_data)

#     # load the model from disk
#     model = pickle.load(
#         open(model_dir + "/finalized_model_" + m + ".sav", "rb")
#     )

#     # score on train data
#     pred, pred_proba = score_data(
#         train_data=train_data_2,
#         X_train_data=X_train_data,
#         Y_train_data=Y_train_data,
#         model=model,
#         m=m,
#     )

#     # format predictions file
#     preds = pd.read_csv(
#         preds_format_path + "/test_pred_format.csv"
#     )  # , index_col="MessageId"

#     preds.loc[:, "Score"] = pred_proba

#     # save below
#     preds.to_csv(
#         preds_dest_path + "/centralized_test_predictions_" + m + ".csv"
#     )


import sys
import warnings

import flwr as fl
import numpy as np

warnings.simplefilter(action="ignore", category=FutureWarning)

# import pandas
import json

# from sklearn.externals import joblib
import pickle
from importlib.resources import path
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.utils
from pandas import Int64Index, MultiIndex
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier


def json_to_dict(datapathjsonString):
    datapathJson = open(datapathjsonString)
    datapathDict = json.load(datapathJson)
    return datapathDict


def load_data(swift_data_path, bank_data_path):
    swift_data = pd.read_csv(swift_data_path, index_col="MessageId")
    swift_data["Timestamp"] = swift_data["Timestamp"].astype("datetime64[ns]")
    bank_data = pd.read_csv(bank_data_path)
    return swift_data, bank_data


def create_pred_format(preds_format_path, suffix, dt) -> None:
    ### Creating the pred format csv file
    pred_file = pd.DataFrame(
        index=dt.index, columns=["Score"]
    )  # .reset_index(names = ['MessageId'])
    pred_file["Score"] = 0
    pred_file.to_csv(
        preds_format_path + "/" + suffix + "_pred_format.csv", index=False
    )


def freq_by_features(
    df,
    level_1,
    level_2,
    model_dir,
    save_as_object=False,
    map_from_train_set=False,
) -> None:

    if map_from_train_set == True:

        level_1_level_2_frequency = pickle.load(
            open(
                model_dir
                + "/"
                + str(level_1 + "_" + level_2 + "_frequency")
                + ".sav",
                "rb",
            )
        )
        _level_1 = df.loc[:, level_1].unique()
        _level_2 = df.loc[:, level_2].unique()
        df[str(level_1 + "_" + level_2)] = df.loc[:, level_1] + df.loc[
            :, level_2
        ].astype(str)
        df.loc[:, str(level_1 + "_" + level_2 + "_frequency")] = df.loc[
            :, str(level_1 + "_" + level_2)
        ].map(level_1_level_2_frequency)
    else:
        _level_1 = df.loc[:, level_1].unique()
        _level_2 = df.loc[:, level_2].unique()
        df[str(level_1 + "_" + level_2)] = df.loc[:, level_1] + df.loc[
            :, level_2
        ].astype(str)
        level_1_level_2_frequency = {}
        for s in _level_1:
            level_1_rows = df[df[level_1] == s]
            for h in _level_2:
                level_1_level_2_frequency[s + str(h)] = len(
                    level_1_rows[level_1_rows.loc[:, level_2] == h]
                )

        df.loc[:, str(level_1 + "_" + level_2 + "_frequency")] = df.loc[
            :, str(level_1 + "_" + level_2)
        ].map(level_1_level_2_frequency)

        if save_as_object == True:
            pickle.dump(
                level_1_level_2_frequency,
                open(
                    model_dir
                    + "/"
                    + str(level_1 + "_" + level_2 + "_frequency")
                    + ".sav",
                    "wb",
                ),
            )

    ...


def create_features(df, model_dir, map_from_train_set=False):

    ## Feature Engineering

    # Hour column
    df["hour"] = df["Timestamp"].dt.hour

    # Hour frequency for each sender
    freq_by_features(
        df=df,
        level_1="Sender",
        level_2="hour",
        model_dir=model_dir,
        save_as_object=True,
        map_from_train_set=map_from_train_set,
    )
    # Hour frequency for each receiver
    freq_by_features(
        df=df,
        level_1="Receiver",
        level_2="hour",
        model_dir=model_dir,
        save_as_object=True,
        map_from_train_set=map_from_train_set,
    )
    # Sender-Currency Frequency
    freq_by_features(
        df=df,
        level_1="Sender",
        level_2="InstructedCurrency",
        model_dir=model_dir,
        save_as_object=True,
        map_from_train_set=map_from_train_set,
    )
    # Receiver-SettledCurrency Frequency
    freq_by_features(
        df=df,
        level_1="Receiver",
        level_2="SettlementCurrency",
        model_dir=model_dir,
        save_as_object=True,
        map_from_train_set=map_from_train_set,
    )
    # Sender-Receiver Frequency
    freq_by_features(
        df=df,
        level_1="Sender",
        level_2="Receiver",
        model_dir=model_dir,
        save_as_object=True,
        map_from_train_set=map_from_train_set,
    )

    # # Average Amount per Sender-Currency - not working
    # Sender_ICurrency_mean = mean_by_features(df = df, level_1="Sender" , level_2="InstructedCurrency")
    # df.loc[:,"Sender_ICurrency_mean"] = \
    #     df.loc[:,"Sender_InstructedCurrency"].map(Sender_ICurrency_mean)

    # Numbering the transactions within a account order - ben - date combination
    df = df.sort_values(
        by=[
            "SettlementDate",
            "Sender",
            "Receiver",
            "Account_order",
            "Account_ben",
            "Timestamp",
        ],
        ascending=True,
    )
    df["seq"] = (
        df.groupby(
            [
                "SettlementDate",
                "Sender",
                "Receiver",
                "Account_order",
                "Account_ben",
            ]
        ).cumcount()
        + 1
    )
    df["seq"] = df["seq"].replace(np.NAN, 1)

    # Flag columns for transactions with missing bank details
    df[["MissingBenAccount"]] = 0
    df.loc[df["Flags_ben"].isnull(), "MissingBenAccount"] = 1
    df[["MissingOrdAccount"]] = 0
    df.loc[df["Flags_order"].isnull(), "MissingOrdAccount"] = 1

    # Different sender account number from bank details
    df["DifferentOrderNum"] = np.where(
        df["Account_order"] == df["OrderingAccount"], 0, 1
    )
    # Different receiver account number from bank details
    df["DifferentBenNum"] = np.where(
        df["Account_ben"] == df["BeneficiaryAccount"], 0, 1
    )

    # Different sender account name from bank details
    df["DifferentOrderName"] = np.where(
        df["Name_order"] == df["OrderingName"], 0, 1
    )
    # Different receiver account name from bank details
    df["DifferentBenName"] = np.where(
        df["Name_ben"] == df["BeneficiaryName"], 0, 1
    )

    # Different sender account ordering street from bank details
    df["DifferentOrderStreet"] = np.where(
        df["Street_order"] == df["OrderingStreet"], 0, 1
    )
    # Different receiver account ordering street from bank details
    df["DifferentBenStreet"] = np.where(
        df["Street_ben"] == df["BeneficiaryStreet"], 0, 1
    )

    # Different sender account country code/zip from bank details
    df["DifferentOrderZip"] = np.where(
        df["CountryCityZip_order"] == df["OrderingCountryCityZip"], 0, 1
    )
    # Different receiver account country code/zip from bank details
    df["DifferentBenZip"] = np.where(
        df["CountryCityZip_ben"] == df["BeneficiaryCountryCityZip"], 0, 1
    )

    # Some missing value treatment
    df.loc[df["Flags_ben"].isna(), "Flags_ben"] = 99

    return df


def train_model(X_train_data, Y_train_data, m) -> None:

    if m == "rf":

        model = RandomForestClassifier(
            max_depth=7, random_state=0, n_estimators=10
        )

    elif m == "xgboost":

        model = XGBClassifier(n_estimators=100)

    kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    cv_results = cross_val_score(
        model, X_train_data, Y_train_data, cv=kfold, scoring="f1"
    )

    model.fit(X_train_data, Y_train_data)
    #     print("Minimum:", cv_results.min())
    #     print("Maximum:", cv_results.max())
    #     print("StanDev:", cv_results.std())

    return model


def score_data(train_data, X_train_data, Y_train_data, model, m) -> None:

    pred = model.predict(X_train_data)
    #     print(m + " Classification Report=\n\n", classification_report(Y_train_data, pred))
    #     print(m + " Confusion Matrix=\n\n", confusion_matrix(Y_train_data, pred))

    pred_proba = model.predict_proba(X_train_data)[:, 1]
    #     print("AUPRC:", metrics.average_precision_score(y_true=Y_train_data, y_score=pred_proba))

    # print(
    #     metrics.average_precision_score(y_true=Y_train_data, y_score=pred_proba)
    # )

    importances = model.feature_importances_
    forest_importances = pd.Series(
        importances, index=train_data.drop(["Label"], axis=1).columns
    )
    #     print(forest_importances)

    return pred, pred_proba


def fit(
    swift_data_path,
    bank_data_path,
    model_dir,
    preds_format_path,
    preds_dest_path,
    m,
) -> None:

    train_data, bank_data = load_data(
        swift_data_path=swift_data_path, bank_data_path=bank_data_path
    )

    # Merging with bank details
    train_data = pd.merge(
        train_data,
        bank_data,
        left_on="OrderingAccount",
        right_on="Account",
        how="left",
    )
    train_data = pd.merge(
        train_data,
        bank_data,
        left_on="BeneficiaryAccount",
        right_on="Account",
        how="left",
        suffixes=["_order", "_ben"],
    )

    # Creating predictions format file
    create_pred_format(
        preds_format_path=preds_format_path, dt=train_data, suffix="train"
    )

    # Feature engineering

    train_data = create_features(
        df=train_data, model_dir=model_dir, map_from_train_set=False
    )

    # Keep below columns for training and testing
    cols_to_keep = [
        "SettlementAmount",
        "InstructedAmount",
        "Label",
        "hour",
        "Flags_ben",
        "MissingBenAccount",
        "MissingOrdAccount",
        "Sender_hour_frequency",
        # 'sender_currency_amount_average',
        "Sender_Receiver_frequency",
        "Sender_InstructedCurrency_frequency",
        "seq",
        # 'receiver_transactions',
        "Receiver_SettlementCurrency_frequency",
        "Receiver_hour_frequency",
        "DifferentOrderNum",
        "DifferentBenNum",
        "DifferentOrderName",
        "DifferentBenName",
        "DifferentOrderStreet",
        "DifferentBenStreet",
        "DifferentOrderZip",
        "DifferentBenZip",
    ]

    train_data_2 = train_data[cols_to_keep]
    #     print(train_data_2.head(3))

    # Separate DV - IDV series
    Y_train_data = train_data_2["Label"].values
    X_train_data = train_data_2.drop(["Label"], axis=1).values

    # Normalize

    scaler = StandardScaler()
    scaler.fit(X_train_data)
    X_train_data = scaler.transform(X_train_data)

    # Fit the model on data
    model = train_model(
        X_train_data=X_train_data, Y_train_data=Y_train_data, m=m
    )

    # score on train data
    pred, pred_proba = score_data(
        train_data=train_data_2,
        X_train_data=X_train_data,
        Y_train_data=Y_train_data,
        model=model,
        m=m,
    )

    # formatted predictions file
    preds = pd.read_csv(
        preds_format_path + "/train_pred_format.csv"
    )  # ,index_col='MessageId')
    preds.loc[:, "Score"] = pred_proba

    # save below
    pickle.dump(
        scaler, open(model_dir + "/finalized_scaler_" + m + ".sav", "wb")
    )
    pickle.dump(model, open(model_dir + "/finalized_model_" + m + ".sav", "wb"))
    # preds.to_csv(preds_dest_path + '/centralized_train_predictions_' + m + '.csv')
    return X_train_data, Y_train_data, model


def predict(
    swift_data_path,
    bank_data_path,
    model_dir,
    preds_format_path,
    preds_dest_path,
    m,
) -> None:

    train_data, bank_data = load_data(
        swift_data_path=swift_data_path, bank_data_path=bank_data_path
    )

    # Merging with bank details
    train_data = pd.merge(
        train_data,
        bank_data,
        left_on="OrderingAccount",
        right_on="Account",
        how="left",
    )
    train_data = pd.merge(
        train_data,
        bank_data,
        left_on="BeneficiaryAccount",
        right_on="Account",
        how="left",
        suffixes=["_order", "_ben"],
    )

    # Creating predictions format file
    create_pred_format(
        preds_format_path=preds_format_path, dt=train_data, suffix="test"
    )

    # Feature engineering

    train_data = create_features(
        df=train_data, model_dir=model_dir, map_from_train_set=True
    )

    # Keep below columns for training and testing
    cols_to_keep = [
        "SettlementAmount",
        "InstructedAmount",
        "Label",
        "hour",
        "Flags_ben",
        "MissingBenAccount",
        "MissingOrdAccount",
        "Sender_hour_frequency",
        # 'sender_currency_amount_average',
        "Sender_Receiver_frequency",
        "Sender_InstructedCurrency_frequency",
        "seq",
        # 'receiver_transactions',
        "Receiver_SettlementCurrency_frequency",
        "Receiver_hour_frequency",
        "DifferentOrderNum",
        "DifferentBenNum",
        "DifferentOrderName",
        "DifferentBenName",
        "DifferentOrderStreet",
        "DifferentBenStreet",
        "DifferentOrderZip",
        "DifferentBenZip",
    ]

    train_data_2 = train_data[cols_to_keep]
    #     print(train_data_2.head(3))

    # Separate DV - IDV series

    Y_train_data = train_data_2["Label"].values
    X_train_data = train_data_2.drop(["Label"], axis=1).values

    # load scaler from disk
    scaler = pickle.load(
        open(model_dir + "/finalized_scaler_" + m + ".sav", "rb")
    )
    X_train_data = scaler.transform(X_train_data)

    #     # load the model from disk
    #     model = pickle.load(open(model_dir + '/finalized_model_' + m + '.sav', 'rb'))

    #     # score on train data
    #     pred, pred_proba = score_data(train_data = train_data_2, X_train_data = X_train_data, Y_train_data = Y_train_data, model = model, m=m)

    #     # format predictions file
    #     preds = pd.read_csv(preds_format_path + '/test_pred_format.csv')#,index_col='MessageId')
    #     preds.loc[:,'Score'] = pred_proba

    #     # save below
    #     preds.to_csv(preds_dest_path + '/centralized_test_predictions_' + m + '.csv')
    return X_train_data, Y_train_data
