# File defining the Tester class for TES-RNN Model

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from loss_modules import MSEloss



class TESRNNTester(nn.Module):
    def __init__(self, model, dataloader, run_id, config, test_run):
        super(TESRNNTester, self).__init__()
        self.model = model.to(config['device'])
        self.config = config
        self.dl = dataloader
        self.run_id = str(run_id)
        self.csv_save_path =  os.path.join('Results', self.run_id)
        self.criterion = MSEloss(self.config['device'])
        self.test_run = test_run
        
        
    # Actual testing
    def testing(self):
        
        # print("\nTesting model ... ")
        
        # Only one iteration for the single Base Station timeseries
        for clust_num, (train, val, test, idx) in enumerate(self.dl):
            
            # Evaluation mode (testing phase) with dropout layers in training mode
            self.model.eval()
            for m in self.model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()
                    
            with torch.no_grad():
                
                # Get normalized predictions and normalized traffic of testing set
                predictions, actuals = self.model(train, val, test, idx, self.test_run, False, True)
                
                # Get total overprovisioning and number of sla_violations from prediction
                test_loss = self.criterion(predictions, actuals)
                
                # Store outputs of the tester
                # np.save(os.path.join(self.csv_save_path, 'test_loss_' + str(self.test_run) + '.npy'), float(test_loss))
                # np.save(os.path.join(self.csv_save_path, 'test_predictions_' + str(self.test_run) + '.npy'), np.squeeze(predictions.cpu()))
                # np.save(os.path.join(self.csv_save_path, 'test_actuals_' + str(self.test_run) + '.npy'), np.squeeze(actuals.cpu()))
                # print("Test MSE Loss: ", float(test_loss))


                return predictions, actuals
                
