import numpy as np
from algo import ValueFunctionWithApproximation
import torch
from torch import nn

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method
        self.net = nn.Sequential(
            nn.Linear(in_features=state_dims, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )
        self.net.double()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.loss = torch.nn.MSELoss()

    def __call__(self,s):
        # TODO: implement this method
        self.net.eval()
        s = torch.tensor(s).to(torch.double)
        v = self.net(s)
        return v.item()

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        self.net.train()
        s_tau = torch.tensor(s_tau, dtype=torch.double)
        G = torch.tensor([G], dtype=torch.double)
        val = self.net(s_tau)
        loss = self.loss(val, G)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return None
