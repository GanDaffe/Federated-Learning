from global_import import *
from algorithm.fedavg import FedAvg 

class FedAAW(FedAvg):
    def __init__(self, *args, beta: float = 0.01, gamma: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta 
        self.gamma = gamma
        self.R_t = {} 

    def __repr__(self) -> str:
        return "FedAAW base FedAvg"


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        if server_round <= 1: 
            for _, fit_res in results: 
                self.R_t[fit_res.metrics['id']] = fit_res.metrics['f_norm'] 
        else: 
            for _, fit_res in results: 
                prev_round_R = self.R_t[fit_res.metrics['id']] 
                self.R_t[fit_res.metrics['id']] = (prev_round_R * (server_round - 1) + fit_res.metrics['f_norm']) / server_round
        
        examples = [fit_res.num_examples for _, fit_res in results]
        all_samples = sum(examples)
        q, p, q_sum = {}, {}, 0.0

        for _, fit_res in results: 
            cid = fit_res.metrics['id']
            q[cid] = (fit_res.num_examples / all_samples) + self.beta / self.R_t[cid] - self.gamma
            q_sum += np.exp(q[cid])

        for cid in q: 
            p[cid] = np.exp(q[cid]) / q_sum
             
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), p[fit_res.metrics['id']])for _, fit_res in results]
        print(f"Round {server_round} | p = {p}")
        
        self.current_parameters = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}
        
        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        examples = [fit_res.num_examples for _, fit_res in results]
        loss = sum(losses) / all_samples
        accuracy = sum(corrects) / all_samples
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return self.current_parameters, metrics_aggregated