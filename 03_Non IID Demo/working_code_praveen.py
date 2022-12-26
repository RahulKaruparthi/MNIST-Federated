# for the operating system operations e.g., creating a folder.
import os

# Tensorflow and Keras are two packages for creating neural network models.
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

# Print tensorfow (TF) version. Make sure you have at least tensorflow 2.1.0
print(f"Tensorflow version: {tf.version.VERSION}")
from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common import Metrics
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")
# import the dataset.
# from tensorflow.keras.datasets import boston_housing

# import NN layers and other componenets.
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt # for plotting data and creating different charts.
import numpy as np # for math and arrays
import pandas as pd # data from for the data.
import seaborn as sns # for plotting
tf.random.set_seed(13) # to make sure the experiment is reproducible.
tf.debugging.set_log_device_placement(False)


# uncomment the following line to use a GPU (Graphical Processing Unit) 
#if you have it available at your machine. This unit will make your code run faster.
# tf.config.experimental.list_physical_devices('GPU')  

bank_df=pd.read_csv("/mnt/d/Federated Learning/bank_dataset/bank_dataset.csv")

# bank_df.head()
swift_df=pd.read_csv("/mnt/d/Federated Learning/swift_transaction_train_dataset/swift_transaction_train_dataset.csv",index_col="MessageId")
swift_df["Timestamp"] = swift_df["Timestamp"].astype("datetime64[ns]")
# swift_df.head()
swift_bank_df= pd.merge(
        swift_df,
        bank_df,
        left_on="OrderingAccount",
        right_on="Account",
        how="left",
    )
NUM_CLIENTS = 10
swift_bank_df=swift_bank_df.head(4000000)
BATCH_SIZE = 32

def load_datasets():
    # Download and transform CIFAR-10 (train and test)
    trainset, testset =  train_test_split(swift_bank_df, test_size=0.4)
    trainloadset,valloadset=train_test_split(trainset,test_size=0.5)
#     print(len(trainloadset))
#     print(len(valloadset))
    le1=len(trainloadset)//10
    le2=len(valloadset)//10
    
    trainloaders=[]
    valloaders=[]
    for i in range(1,NUM_CLIENTS+1):
        temp_df1=trainloadset.head(le1)
        trainloadset=trainloadset.tail(len(trainloadset)-le1)
        temp_df2=valloadset.head(le2)
        valloadset=valloadset.tail(len(valloadset)-le2)
        trainloaders.append(temp_df1)
        valloaders.append(temp_df2)
    
        
    
    return trainloaders, valloaders, testset

trainloaders, valloaders, testloader = load_datasets()

def data_prep(swift_bank_df):
    swift_bank_df = swift_bank_df.dropna()
    swift_bank_df['hour'] = swift_bank_df['Timestamp'].dt.hour
    cols_to_keep=['SettlementAmount',
        'InstructedAmount',
        'Label',
        'hour']
    swift_bank_df2 = swift_bank_df[cols_to_keep]
    return swift_bank_df2
 
def stats(train_dataset):
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()
    return train_stats

def norm(x):
    train_stats=stats(x)
    return (x - train_stats['mean']) / train_stats['std']
  
def build_model1_two_hidden_layers(normed_train_data):
    # Keras model object created from Sequential class. This will be the container that contains all layers.
    model = Sequential()

    # The model so far is empty. It can be constructed by adding layers and compilation.
    # This Keras model with multiple hidden layers.
    
    # Input Layer with 10 Neurons
    model.add(Dense(32, input_shape = (normed_train_data.shape[1],)))    # Input layer => input_shape must be explicitly designated
#     model.add(Activation('relu')) # relu or sigmoid.
    
#     model.add(Dense(128,Activation('relu')))                         # Hidden layer 1 => only output dimension should be designated (output dimension = # of Neurons = 50)
    
    
    
    
    model.add(Dense(1))                          # Output layer => output dimension = 1 since it is a regression problem
    
    # Activation: sigmoid, softmax, tanh, relu, LeakyReLU. 
    #Optimizer: SGD, Adam, RMSProp, etc. # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    learning_rate = 0.0001
    optimizer = optimizers.SGD(learning_rate)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=optimizer,
                metrics=['accuracy']) # for regression problems, mean squared error (MSE) is often employed
    return model
 
def train(model, trainloader,valloader,train_labels,valid_labels, epochs: int, verbose=False):
    """Train the network on the training set."""
    batch_size=32
    EPOCHS=2
    normed_train_data=trainloader
    normed_valid_dataset=valloader
    history = model.fit(
        normed_train_data, 
        train_labels,
        batch_size = batch_size,
        epochs=EPOCHS, 
        verbose=1,
        shuffle=True,
        steps_per_epoch = int(normed_train_data.shape[0] / batch_size) ,
        validation_data = (normed_valid_dataset, valid_labels),   
    )
    


def test(model, testloader,test_labels):
    normed_test_data=testloader
    """Evaluate the network on the entire test set."""
    print('Test Split: ')
    loss, accuracy =  model.evaluate(normed_test_data, test_labels, verbose=2)

    print("Accuracy   : {:5.2f} ".format(accuracy))
    
  
tl=data_prep(trainloaders[0])
vl=data_prep(valloaders[0])
testl=data_prep(testloader)
train_labels=tl.pop('Label')
val_labels=vl.pop('Label')
test_labels=testl.pop('Label')
trainloader=norm(tl)
valloader=norm(vl)
testloader=norm(testl)
model=build_model1_two_hidden_layers(trainloader)
model.save("my_model")

train(model, trainloader,valloader,train_labels,val_labels, 1)

test(model, testloader,test_labels)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        
    def get_parameters(self,config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        tl=data_prep(self.trainloader)
        vl=data_prep(self.valloader)
        train_labels=tl.pop('Label')
        val_labels=vl.pop('Label')
        test_labels=testl.pop('Label')
        trainloader=norm(tl)
        valloader=norm(vl)
        train(self.model,trainloader,valloader,train_labels,val_labels, 1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(testloader, test_labels, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}
      
def client_fn(cid: str) -> FlowerClient:
"""Create a Flower client representing a single organization."""

    loaded_model = keras.models.load_model("my_model")

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

      # Create a  single Flower client representing a single organization
    return FlowerClient(loaded_model, trainloader, valloader)
  
  
# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=10,  # Wait until all 10 clients are available
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
