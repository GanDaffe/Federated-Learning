from client.client import BaseClient
from global_import import *

class FedAAW_Client(BaseClient):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        optimizer = torch.optim.SGD(params=self.net.parameters(), lr=config['learning_rate'])
        loss, accuracy, f_norm = train(self.net, self.trainloader, self.criterion, optimizer, device=self.device, num_epochs=self.local_train_epcs, get_grad_norm=True)

        return self.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid, "f_norm": f_norm}
