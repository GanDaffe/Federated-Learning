from client.client import BaseClient
from global_import import *

class FedCLS_Client(BaseClient):
    def __init__(self, *args, num_classes, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.num_classes = num_classes

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        optimizer = torch.optim.SGD(params=self.net.parameters(), lr=config['learning_rate'])
        loss, accuracy = train(self.net, self.trainloader, self.criterion, optimizer, device=self.device, num_epochs=self.local_train_epcs)

        return self.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid, "num_classes": self.num_classes}
