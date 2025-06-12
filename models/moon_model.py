from torch import nn 
import torch 
import torch.nn.functional as F
from models.mlp import MLP_header 
from models.lstm import LSTM_header 
from models.resnet import ResNet34, ResNet50

class MoonTypeModel(nn.Module): 
    
    def __init__(self, dataset_name):

        self.lstm = False
        if dataset_name == 'fmnist':
            self.features = MLP_header() 
            num_ftrs = 200

        elif dataset_name in ['cifar10', 'cifar100']:
            model = ResNet34() if dataset_name ==  'cifar10' else ResNet50()
            self.features = nn.Sequential(*list(model.resnet.children())[:-1])
            num_ftrs = model.resnet.fc.in_features

        elif dataset_name == 'sent140': 
            self.lstm = True
            self.features = LSTM_header()
            num_ftrs = 256

        if dataset_name in ['fmnist', 'cifar10']:
            n_classes = 10
        elif dataset_name == 'cifar100': 
            n_classes = 100

        self.l3 = nn.Linear(num_ftrs, n_classes) 

    def forward(self, x): 
        h = self.features(x)
        h = h.view(h.size(0), -1)

        y = self.l3(h) 
        if self.base_model == 'lstm': 
            y = F.sigmoid(y)
        return h, h, y
    
def get_moon_model(dataset_name: str): 
    return MoonTypeModel(dataset_name) 
