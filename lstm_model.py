## import packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm, trange
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## hyperparameters
batch_size = 2
hidden_layers = 32
learning_rate = 0.001
epochs = 150

# MSE Loss
# Adam Optimizer

## create dataset
class DroughtDataset(Dataset):
    """ Drought dataset."""

    def __init__(self, np_array_x, np_array_y):
        """
        Args:
            np_array_x (string): Path to the npy file with annotations.
            np_array_y (string): Path to the npy file with annotations.
        """
        self.X = np.load(np_array_x)
        self.Y = np.load(np_array_y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx][0]

## loading training data
drought_dataset = DroughtDataset('x_train.npy', 'y_train.npy')

for i in range(len(drought_dataset)):
    sample = drought_dataset[i]
    # sample[0].shape = (6,4)

## call dataloader
dataloader = DataLoader(drought_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# batch,seq,input_var

## create LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=hidden_layers, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size,batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, hidden_cell = self.lstm(input_seq)#, self.hidden_cell.cuda())
        predictions = self.linear(lstm_out[:, -1])
        return predictions.squeeze(1)

## cross entropy loss and optimizer
model = LSTM()
model = model
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## training the model
for i in trange(epochs):
    sum_loss = 0
    for batch in dataloader:
        seq, labels = batch
        seq, labels = seq.float(), labels.float()
        optimizer.zero_grad()
        #model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
        #                torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        
        single_loss.backward()
        optimizer.step()
        sum_loss += single_loss
        
    if i%5 == 1:
        print(f'epoch: {i:3} loss: {sum_loss.cpu().item()/len(dataloader):10.8f}')

print(f'epoch: {i:3} loss: {single_loss.cpu().item():10.10f}')

## loading testing data
drought_dataset_test = DroughtDataset('x_test.npy', 'y_test.npy')

test_size = len(drought_dataset_test)

for i in range(len(drought_dataset_test)):
    sample = drought_dataset[i]
    # sample[0] is the x test
    # sample[1] is the y test

## making predictions


