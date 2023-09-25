#!/usr/bin/python3
# New and streamlined version of peatrain -- PyTorch Edition.
# Run in the working directory with 
#   python peatrain_pytorch.py
#
# Based on code developed by Lachlan Mitchell and Russell Edson,
# Biometry Hub, University of Adelaide.
# Date last modified: 16/07/2023

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, ToTensor


# Neural Network model definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution2d_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.3)
        self.output = nn.Sequential(
            nn.Linear(in_features=1024, out_features=2)
            # Note that PyTorch works on unnormalised logits with
            # cross entropy loss, so we don't need a softmax layer.
        )
    
    def forward(self, x):
        x = self.convolution2d_stack(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

model = NeuralNetwork()
print(model)

# Prepare training/validation data
batch_size = 64
train_validation_split = [0.8, 0.2]

class ImageDataset(Dataset):
    def __init__(self, annotations_file):
        self.image_labels = pd.read_csv(annotations_file)
    
    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        image_path = self.image_labels.iloc[idx, 0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = Compose([ToTensor()])  # Normalises to 0-1
        image = transform(image)
        label = self.image_labels.iloc[idx, 1]
        return image, label

images = ImageDataset("annotations.csv")
images = random_split(images, lengths=train_validation_split)
training_data, validation_data = images
train_dataloader = DataLoader(training_data, batch_size=batch_size)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

# Model training
epochs = 30
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.RMSprop(model.parameters(), lr=0.001)
acc_loss = pd.DataFrame({"epoch": [], "accuracy": [], "loss": []})

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")

    # Train
    size = len(train_dataloader.dataset)
    model.train()
    count = 0
    for batch, (X, y) in enumerate(train_dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        loss = loss.item()
        count += len(X)

        print(f"loss: {loss:>7f}  [{count:>4d}/{size:>4d}]")

    # Validate (and compute accuracy/loss estimates)
    size = len(validation_dataloader.dataset)
    num_batches = len(validation_dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in validation_dataloader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()
    loss /= num_batches
    accuracy = correct / size
        
    print(f"Validation:\nAccuracy:{accuracy:>7f}, loss:{loss:>7f}\n")
    acc_loss.loc[len(acc_loss)] = [epoch + 1, accuracy, loss]

# Save training accuracy/loss values
acc_loss_filename = "acc_loss.csv"
acc_loss.to_csv(acc_loss_filename, index=False)
print(f"Saved running accuracy/loss recording to {acc_loss_filename}")

# Save output model (as .pth for PyTorch models)
model_filename = "nnmodel.pth"
torch.save(model.state_dict(), model_filename)
print(f"Saved neural network model to {model_filename}")
