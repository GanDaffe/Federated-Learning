from torch import nn 
import torch 
import torch.nn.functional as F
from models.mlp import MLP_header 
from models.textcnn import CNN_Text_Header
from models.resnet import ResNet50, ResNet18
from models.lstm import LSTM_Header

class MoonTypeModel(nn.Module): 
    
    def __init__(self, dataset_name):
        
        super().__init__()
        self.use_lstm = False
        if dataset_name == 'fmnist':
            self.features = MLP_header() 
            num_ftrs = 200

        elif dataset_name in ['cifar10', 'cifar100']:
            model = ResNet50() if dataset_name ==  'cifar100' else ResNet18()
            self.features = nn.Sequential(*list(model.resnet.children())[:-1])
            num_ftrs = model.resnet.fc.in_features

        elif dataset_name == 'agnews': 
            self.use_lstm = True
            self.features = LSTM_Header()
            num_ftrs = 256

        if dataset_name in ['fmnist', 'cifar10']:
            n_classes = 10
        elif dataset_name == 'cifar100': 
            n_classes = 100
        elif dataset_name == 'agnews': 
            n_classes = 4
            
        self.l3 = nn.Linear(num_ftrs, n_classes) 

    def forward(self, x): 
        h = self.features(x)
        if not self.use_lstm:
            h = h.view(h.size(0), -1)

        y = self.l3(h)
        return h, h, y
    
def get_moon_model(dataset_name: str): 
    return MoonTypeModel(dataset_name) 
