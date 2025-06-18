from global_import import *
from algorithm.fedavg import FedAvg

class FedDC(FedAvg):
    def __init__(self, *args, alpha: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        param_size = sum(p.size for p in get_parameters(self.net))
        self.g = np.zeros(param_size, dtype=np.float64)

    def __repr__(self) -> str:
        return "FedDC"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config = {
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "server_round": server_round,
            "g": self.g.tobytes()  # Gửi global update g^{(t-1)}
        }
        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        weights_results = []
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        for _, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            drift = np.frombuffer(fit_res.metrics['drift'], dtype=np.float64)

            param_shapes = [p.shape for p in params]
            param_sizes = [p.size for p in params]
            if drift.size != sum(param_sizes):
                raise ValueError(f"Drift size {drift.size} không khớp param size {sum(param_sizes)}")

            # split drift thành list arrays tương ứng từng layer
            splits = np.split(drift, np.cumsum(param_sizes)[:-1])
            drift_reshaped = [arr.reshape(shape) for arr, shape in zip(splits, param_shapes)]
            # corrected weight = θ_i^{(t)} + h_i^{(t)}
            weights = [p + d for p, d in zip(params, drift_reshaped)]
            weights_results.append((weights, fit_res.num_examples))

        self.current_parameters = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        loss = sum(losses) / total_examples
        accuracy = sum(corrects) / total_examples
        print(f"Round {server_round} - train_loss: {loss:.4f} - train_acc: {accuracy:.4f}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        # global update g^{(t)} = average of g_i^{(t)}
        g = np.zeros_like(self.g)
        for _, fit_res in results:
            local_update = np.frombuffer(fit_res.metrics['local_update'], dtype=np.float64)
            if local_update.size != g.size:
                raise ValueError(f"local_update size {local_update.size} không khớp global g size {g.size}")
            g += local_update * (fit_res.num_examples / total_examples)
        self.g = g

        return self.current_parameters, metrics_aggregated