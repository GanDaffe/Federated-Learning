from torch import nn
import torch


class MLP_header(nn.Module): 
    def __init__(self) -> None:
        super(MLP_header, self).__init__()
        self.layer1 = nn.Linear(28*28, 200)
        self.layer2 = nn.Linear(200, 200)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)
        
        return x
    
class MLP(nn.Module): 
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.encode = MLP_header() 
        self.classification = nn.Linear(200, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x) 
        x = self.classification(x)
        return x
