import torch.nn.functional as F
from global_import import * 
from torch import nn
from client.client import BaseClient

class FedNTD_Client(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        last_layer = list(self.net.modules())[-1]
        self.num_classes = last_layer.out_features

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        tau = config['tau']
        beta = config['beta']

        optimizer = torch.optim.SGD(params=self.net.parameters(), lr=config['learning_rate'])

        loss, accuracy = train_fedntd(
            self.net,
            self.trainloader, 
            optimizer,
            self.device,
            tau,
            beta,
            self.num_classes,
            self.local_train_epcs
        )

        return self.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid}

def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits
        
class NTD_Loss(nn.Module):

    def __init__(self, num_classes=10, tau=3, beta=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss

def train_fedntd(
    net,
    trainloader,
    optimizer,
    device,
    tau, 
    beta,
    num_classes,
    num_epochs: int = 1):

    global_net = copy.deepcopy(net)

    global_net.to(device)
    net.to(device)

    net.train()
    for param in global_net.parameters():
        param.requires_grad = False

    loss_, acc = 0.0, 0
    criterion = NTD_Loss(num_classes=num_classes, tau=tau, beta=beta) 

    for e in range(num_epochs):
        running_loss, running_corrects, tot = 0.0, 0, 0

        for images, labels in trainloader:

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = net(images)
            dg_outputs = global_net(images)

            loss = criterion(outputs, labels, dg_outputs)

            loss.backward()
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
    return running_loss, accuracy