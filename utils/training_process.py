import numpy as np
from collections import OrderedDict
from typing import List, Dict
import random
import math
import copy
import gc
import torch
import torch.nn as nn
from models import *

def get_model(dataset_name, moon_type=False):

    if moon_type: 
        return get_moon_model(dataset_name=dataset_name)
    
    if dataset_name == 'cifar100': 
        model = ResNet50()
    elif dataset_name == 'cifar10': 
        model = ResNet18()
    elif dataset_name == 'agnews': 
        model = CNN_Text() 
    elif dataset_name == 'fmnist':
        model = MLP()
    return model


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict)

def train(
    net,
    trainloader,
    criterion,
    optimizer,
    device,
    num_epochs: int = 1,
    get_grad_norm = False, 
    proximal_mu: float = None):

    net.to(device)
    net.train()
    loss_, acc = 0.0, 0

    if proximal_mu is not None:
        global_params = copy.deepcopy(net).parameters()

    if get_grad_norm: 
        full_grad = [torch.zeros_like(p) for p in net.parameters() if p.requires_grad]
        total_grad_samples = 0

    for e in range(num_epochs): 
        running_loss, running_corrects, tot = 0.0, 0, 0

        for images, labels in trainloader:

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, labels)

            if proximal_mu is not None:
                proximal_term = sum((local - global_).norm(2)
                                    for local, global_ in zip(net.parameters(), global_params))
                loss += (proximal_mu / 2) * proximal_term

            loss.backward()

            if get_grad_norm and e == 0: 
                with torch.no_grad(): 
                    for i, p in enumerate(net.parameters()):
                        if p.grad is not None:
                            full_grad[i] += p.grad.detach() * images.size(0) 
                total_grad_samples += images.size(0)

            optimizer.step()

            preds = torch.argmax(outputs, dim=1)

            tot += images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            running_loss += loss.item() * images.size(0)

        running_loss /= tot
        accuracy = running_corrects / tot

        loss_ += running_loss
        acc += accuracy 
    
    loss_ /= num_epochs
    acc /= num_epochs

    if get_grad_norm:
        full_grad_ = [full_grad[i] / total_grad_samples for i in range(len(full_grad))]
        norm_grad = sum(g.norm()**2 for g in full_grad_).item()

        return loss_, acc, norm_grad

    return running_loss, accuracy


def test(net, testloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    corrects, total_loss, tot = 0, 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            if isinstance(net, MoonTypeModel): 
                _, _, outputs = net(images)
            else:
                outputs = net(images)

            loss = criterion(outputs, labels)

            if isinstance(net, MoonTypeModel):
                _, preds = torch.max(outputs.data, 1)
            else:  
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            corrects += torch.sum(preds == labels).item()
            total_loss += loss.item() * images.size(0)
            tot += images.size(0)

    total_loss /= tot
    accuracy = corrects / tot

    return total_loss, accuracy

def compute_entropy(counts: Dict):
    entropy = 0.0
    counts = list(counts.values())
    counts = [0 if value is None else value for value in counts]
    for value in counts:
        entropy += -value/sum(counts) * math.log(value/sum(counts), len(counts)) if value != 0 else 0
    return entropy