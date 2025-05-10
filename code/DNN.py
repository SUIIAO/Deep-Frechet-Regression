import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader 
import numpy as np
import pandas as pd
import os 
import random
from sklearn.preprocessing import StandardScaler 
import NN_class

# Define the deep neural network for regression
def DNN_regression(x, y, xout, layer=4, hidden=64, dropout=0.3, lr=0.0005, num_epochs=2000, seed=None):
  # --- Reproducibility Setup ---
  py_seed = None
  if seed is not None:
    try:
      py_seed = int(seed)
      random.seed(py_seed)
      torch.manual_seed(py_seed)
      np.random.seed(py_seed)
    except (ValueError, TypeError):
      print(f"Warning: Invalid seed value provided ({seed}). Proceeding without setting seed.")

  # Ensure input parameters are integers where needed
  layer = int(layer)
  hidden = int(hidden)
  num_epochs = int(num_epochs)
  
  # Convert input data to PyTorch tensors
  x = torch.tensor(np.array(x).astype(np.float32))
  y = torch.tensor(np.array(y).astype(np.float32))
  xout = torch.tensor(np.array(xout).astype(np.float32))
  
  # Get the dimensions of input and output
  n,p = x.shape # n: number of samples, p: number of features
  dim_res = y.shape[1] # Dimension of the response variable
  
  # Split data into training and validation sets
  n_train = int(n*0.8) # 80% for training
  x_train = x[:n_train,]
  x_valid = x[n_train:,]
  y_train = y[:n_train,]
  y_valid = y[n_train:,]

  # Set up the training dataset and data loader
  train_ds = TensorDataset(x_train,y_train)
  train_dl = DataLoader(train_ds, batch_size=30, shuffle=True)

  # Initialize the model, optimizer, and loss function
  model = NN_class.MLP(num_features=p, num_response=dim_res, number_layer=layer, hidden=hidden, dropout=dropout)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam optimizer
  loss = nn.MSELoss() # Mean Squared Error loss
  
  # Initialize variables to track the best model and loss
  min_valid_loss = float("inf")  # Minimum validation loss
  err_valid = []  # List to store validation losses
  min_epoch = 0  # Epoch with the minimum validation loss
  
  # Training loop
  for epoch in range(num_epochs):
    model = model.train() # Set model to training mode
    for batch_idx, (x_obs, y_obs) in enumerate(train_dl):
      # Forward pass
      y_pred = model(x_obs)
      cost = loss(y_pred, y_obs)
      
      # Backward pass and optimization
      optimizer.zero_grad()
      cost.backward()
      optimizer.step()

    # Validation step
    model = model.eval()           # Set model to evaluation mode
    with torch.no_grad():          # Disable gradient computation for validation
      y_pred = model(x_valid)      # Predict on validation set
      cost = loss(y_pred, y_valid) # Compute validation loss
      err_valid += [cost.item()]   # Store validation loss

      # Save the best model if the validation loss improves
      if min_valid_loss > err_valid[epoch]:
        min_valid_loss = err_valid[epoch]
        best_model = model.state_dict() # Save model state
        min_epoch = epoch # Update best epoch
    if min_epoch + 500 < epoch:
      break
    
  
  # Reload the best model
  model = NN_class.MLP(p,dim_res,layer,hidden)
  model.load_state_dict(best_model) # Load saved best model state
  model.eval() # Set model to evaluation mode
  
  # Predict on the new data (xout)
  with torch.no_grad():
    y_pred = model(xout) # Make predictions
  
  # Return predictions as a NumPy array
  return y_pred.detach().numpy()
