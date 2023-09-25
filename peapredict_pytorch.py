#!/usr/bin/python3
# Given an NN model and image(s), predict a plant. PyTorch Edition.
# Run with 
#   python peapredict_pytorch.py MODEL IMAGE...
#
# Based on code developed by Lachlan Mitchell and Russell Edson,
# Biometry Hub, University of Adelaide.
# Date last modified: 16/07/2023

import sys
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


model_filename = sys.argv[1]
image_filenames = sys.argv[2:]

# Neural Network model definition (exactly the same as the model
# declared in peatrain -- PyTorch doesn't save the model structure
# in the model file apparently)
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
            # cross entropy loss, so we don't have a softmax layer.
        )
    
    def forward(self, x):
        x = self.convolution2d_stack(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

# Load model
model = NeuralNetwork()
model.load_state_dict(torch.load(model_filename))
model.eval()

# Read in image(s) and predict
for image_filename in image_filenames:
    image = cv2.imread(image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = Compose([ToTensor()])  # Normalises to 0-1
    image = transform(image).unsqueeze(dim=0)

    pred = nn.functional.softmax(model(image)[0], dim=0)
    not_plant = pred[0].item()
    plant = pred[1].item()
    print(f"{image_filename}  Not plant: {not_plant:>7f}, Plant: {plant:>7f}")
