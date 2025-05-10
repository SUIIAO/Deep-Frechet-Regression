import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader 
import numpy as np
import pandas as pd
import os 
import random
from sklearn.preprocessing import StandardScaler 

# Define a normalization layer with learnable parameters
class Norm(nn.Module):
  def __init__(self, d, axis = -2, eps = 1e-6):
    super().__init__()
    self.d = d  # Dimensionality of the input
    self.axis = axis  # Axis along which normalization is performed
    self.eps = eps  # Small constant to prevent division by zero
    
    # Learnable parameters for scaling (alpha) and shifting (bias)
    if axis == -2: # If normalizing by columns (Batch Norm style)
      self.alpha = nn.Parameter(torch.ones(d)) # Scaling parameter
      self.bias = nn.Parameter(torch.zeros(d)) # Bias parameter
    else: # If normalizing by rows (Layer Norm style)
      self.alpha = nn.Parameter(torch.ones(d)) # Scaling parameter
      self.bias = nn. Parameter(torch.zeros(d)) # Bias parameter
    
  def forward(self,x):
    # Get the size of the dimension we are normalizing over
    dim_size = x.shape[self.axis]

    # Compute mean and standard deviation along the specified axis
    if dim_size > 1:
      avg = x.mean(axis=self.axis, keepdim=True)
      std = x.std(axis=self.axis, unbiased=False, keepdim=True)
      std = std + self.eps
    else:
      avg = x
      std = torch.full_like(x, self.eps)

    # Apply normalization
    norm = self.alpha * (x - avg) / std + self.bias

    return norm


# Define a Multilayer Perceptron (MLP) model
class MLP(nn.Module):
  def __init__(self, num_features, num_response, number_layer=4, hidden=64, dropout=0.3):
    super(MLP, self).__init__()
    self.hidden = hidden  # Number of neurons in hidden layers
    self.layer = number_layer - 1  # Number of hidden layers (excluding input and output)
    
    # Input layer
    self.linear_1 = nn.Linear(num_features, self.hidden)
    
    # Hidden layers (stored in a ModuleList for flexibility)
    self.linear_hidden = nn.ModuleList()
    for i in range(self.layer):
      self.linear_hidden.append(nn.Linear(self.hidden, self.hidden))
    
    # Output layer
    self.linear_out = nn.Linear(self.hidden, num_response)
    
    # Normalization layers
    self.norm1 = Norm(self.hidden, axis = -2) # Normalization after the first layer
    self.linear_bn = nn.ModuleList()
    for i in range(self.layer):
      self.linear_bn.append(Norm(self.hidden, axis = -2)) # Normalization after hidden layers

    
    # Dropout layers (to prevent overfitting)
    self.drop = nn.ModuleList()
    for i in range(self.layer):
      self.drop.append(nn.Dropout(dropout)) # Dropout applied before each hidden layer


  def forward(self, x):
    # Pass input through the input layer and apply ReLU activation
    x_ = F.relu(self.linear_1(x))
    x_ = self.norm1(x_) # Apply normalization
    
    # Pass through each hidden layer with residual connections
    for i in range(self.layer):
      # Residual connection: add input to the output of the layer
      x_ = x_ + F.relu(self.linear_hidden[i](self.drop[i](x_)))
      x_ = self.linear_bn[i](x_) # Apply normalization
    
    # Pass through the output layer
    x_ = self.linear_out(x_)

    
    return x_
