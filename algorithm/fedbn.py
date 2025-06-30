from global_import import *
from algorithm.fedavg import FedAvg

class FedBN(FedAvg): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __repr__(self):
        return "FedBN"

    def configure_fit(self, server_round, parameters, client_manager):
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(sample_size, min_num_clients)
        config = {"learning_rate": self.learning_rate, "round": server_round}
        self.learning_rate = self.learning_rate * self.decay_rate 

        return [(client, FitIns(parameters, config)) for client in clients]
    
