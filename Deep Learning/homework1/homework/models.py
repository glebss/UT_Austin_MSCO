import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        return torch.mean(-torch.log(F.softmax(input, dim=1).gather(1, target.unsqueeze(1))))


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        self.linear = torch.nn.Linear(in_features=3 * 64 * 64, out_features=6)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.linear(x.flatten(start_dim=1))


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(in_features=3 * 64 * 64, out_features=128)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        self.linear_out = torch.nn.Linear(in_features=128, out_features=6)
        torch.nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x = x.flatten(start_dim=1)
        out = self.relu(self.linear1(x))
        return self.linear_out(out)


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
