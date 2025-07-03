# File containing Loss modules and functions

import torch
import torch.nn as nn
import numpy as np
        



# Definition of MSE loss function
class MSEloss(nn.Module):

    def __init__(self, device):
        super(MSEloss, self).__init__()
        self.device = device

    def forward(self, predictions, actuals):
        diff = torch.sub(predictions, actuals).to(self.device)
        diff = torch.square(diff)
        final_loss = torch.mean(diff)
        return final_loss
        



#Definition of MAE loss function
class MAEloss(nn.Module):

    def __init__(self, device):
        super(MAEloss, self).__init__()
        self.device = device

    def forward(self, predictions, actuals):
        diff = torch.sub(predictions, actuals).to(self.device)
        diff = torch.abs(diff)
        final_loss = torch.mean(diff)
        return final_loss




# Definition of function to return the denormalized validation loss
def denorm_validation_loss(pred, act, lev):
    
    # Denormalize predicted and real load
    actuals = act * lev
    preds = pred * lev
    
    # Get the denormalized loss (MSE)
    loss = np.square(np.subtract(actuals,preds)).mean()
    
    return loss