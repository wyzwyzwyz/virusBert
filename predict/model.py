import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN_model(nn.Module):
    def __init__(self,num_fature,dropout=0.5):
        super(DNN_model, self).__init__()
        self.num_fature=num_fature
        self.model=nn.Sequential(
            nn.Linear(num_fature,num_fature*4),
            nn.BatchNorm1d(num_fature*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_fature * 4, num_fature * 16),
            nn.BatchNorm1d(num_fature * 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_fature * 16, num_fature*4),
            nn.BatchNorm1d(num_fature * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_fature * 4, num_fature),
            nn.BatchNorm1d(num_fature),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_fature, 2)
        )

    def forward(self, x):
        result = self.model(x)
        return result


