from global_import import * 

class BaseClient(fl.client.NumPyClient):
    def __init__(self, 
                 cid, 
                 net, 
                 local_train_epcs, 
                 trainloader, 
                 criterion,
                 device):
         
        self.cid = cid
        self.net = net
        self.local_train_epcs = local_train_epcs
        self.trainloader = trainloader
        self.criterion = criterion
        self.device = device

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        optimizer = torch.optim.SGD(params=self.net.parameters(), lr=config['learning_rate'])
        loss, accuracy = train(self.net, self.trainloader, self.criterion, optimizer, self.device, self.local_train_epcs)
        return self.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid}
    
    def evaluate(self, parameters, config):
        return None