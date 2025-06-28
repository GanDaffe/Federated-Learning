from global_import import *
from algorithm.fedavg import FedAvg 

class FedNTD(FedAvg):
    def __init__(self, *args, tau=3, beta=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau 
        self.beta = beta

    def __repr__(self) -> str:
        return "FedNTD"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config = {"tau": self.tau, "beta": self.beta, "learning_rate": self.learning_rate}

        return [(client, FitIns(parameters, config)) for client in clients]  