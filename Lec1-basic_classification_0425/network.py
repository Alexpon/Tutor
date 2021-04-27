import torch

class NN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(NN, self).__init__()
        self.hidden     = torch.nn.Linear(n_feature, n_hidden)
        self.activation = torch.nn.ReLU()
        self.out        = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.out(x)
        return x

class NN2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(NN2, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.net(x)
