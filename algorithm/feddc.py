from global_import import *
from algorithm.fedavg import FedAvg

class FedDC(FedAvg):
    def __init__(self, *args, alpha: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        n_par = sum([p.numel() for p in self.net.parameters()])
        self.state_grad_diff = np.zeros(n_par)

    def __repr__(self) -> str:
        return "FedDC"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        num_clients, min_req = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(num_clients, min_req)

        config = {
            'global_update_last': self.state_grad_diff.astype(np.float64).tobytes(),
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
        }
        self.learning_rate = self.learning_rate * self.decay_rate 

        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        delta_gs = [np.frombuffer(fit_res.metrics['delta_g'], dtype=np.float64) for _, fit_res in results if 'delta_g' in fit_res.metrics]
        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        weights_results = []
        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            drift = np.frombuffer(fit_res.metrics['drift'], dtype=np.float64)

            shapes = [w.shape for w in weights]
            sizes = [w.size for w in weights]

            splits = np.split(drift, np.cumsum(sizes)[:-1])
            drift_reshaped = [d.reshape(shape) for d, shape in zip(splits, shapes)]

            updated_weights = [w + d for w, d in zip(weights, drift_reshaped)]

            weights_results.append((updated_weights, fit_res.num_examples))
            
        aggregated = aggregate(weights_results)
    
        self.current_parameters = ndarrays_to_parameters(aggregated)
        delta_g_mean = np.sum(delta_gs, axis=0) / len(delta_gs)
        self.state_grad_diff = delta_g_mean.copy()

        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        loss = sum(losses) / total_examples
        accuracy = sum(corrects) / total_examples
        print(f"Round {server_round} - train_loss: {loss:.4f} - train_acc: {accuracy:.4f}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return self.current_parameters, metrics_aggregated