from algorithm.fedavg import FedAvg 
from global_import import *

class MOON(FedAvg):

    def __init__(self, *args, temperature: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def __repr__(self):
        return "MOON"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config = {"learning_rate": self.learning_rate, "temperature": self.temperature, "device": self.device}

        fit_ins = FitIns(parameters, config)

        fit_configs = [(client, fit_ins) for client in clients]
        return fit_configs
