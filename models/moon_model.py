from torch import nn 
import torch 
import torch.nn.functional as F
from models.mlp import MLP_header 
from models.textcnn import CNN_Text_Header
from models.resnet import ResNet34, ResNet18

class MoonTypeModel(nn.Module): 
    
    def __init__(self, dataset_name):
        
        super().__init__()
        if dataset_name == 'fmnist':
            self.features = MLP_header() 
            num_ftrs = 200

        elif dataset_name in ['cifar10', 'cifar100']:
            model = ResNet34() if dataset_name ==  'cifar100' else ResNet18()
            self.features = nn.Sequential(*list(model.resnet.children())[:-1])
            num_ftrs = model.resnet.fc.in_features

        elif dataset_name == 'agnews': 
            embed_num = 2000
            embed_dim = 32
            kernel_sizes = [3, 4, 5]
            kernel_num = 32
            dropout = 0.5
            class_num = 4
            self.features = CNN_Text_Header(
                embed_num=embed_num,
                embed_dim=embed_dim,
                class_num=class_num,
                kernel_sizes=kernel_sizes,
                kernel_num=kernel_num,
                dropout=dropout
            )
            num_ftrs = len(kernel_sizes) * kernel_num

        if dataset_name in ['fmnist', 'cifar10']:
            n_classes = 10
        elif dataset_name == 'cifar100': 
            n_classes = 100
        elif dataset_name == 'agnews': 
            n_classes = 4
            
        self.l3 = nn.Linear(num_ftrs, n_classes) 

    def forward(self, x): 
        h = self.features(x)
        h = h.view(h.size(0), -1)

        y = self.l3(h)
        return h, h, y
    
def get_moon_model(dataset_name: str): 
    return MoonTypeModel(dataset_name) 
