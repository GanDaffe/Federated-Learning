from algorithm.fedavg import FedAvg
from global_import import *

class FedHCW(FedAvg): 

    def __init__(self, 
                 *args, 
                 entropies: List[float], 
                 temperature: float = 0.2, 
                 alpha: float = 5, 
                 **kwargs
    ): 
        super().__init__(*args, **kwargs) 

        self.entropies = entropies
        self.temperature = temperature 
        self.alpha = alpha  
        self.current_angles = {}

    
    def __repr__(self): 
        return 'FedHCW'
    
    
    def aggregate_cluster(self, cluster_id, cluster_clients: List[FitRes]):
        weight_results = [(parameters_to_ndarrays(fit_res.parameters),
                            fit_res.num_examples * np.exp(self.entropies[int(fit_res.metrics["id"])]/self.temperature))
                            for fit_res in cluster_clients]
        losses = [fit_res.num_examples * fit_res.metrics["loss"] for fit_res in cluster_clients]
        correct = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for fit_res in cluster_clients]
        examples = [fit_res.num_examples for fit_res in cluster_clients]
        loss = sum(losses) / sum(examples)
        accuracy = sum(correct) / sum(examples)

        aggregated_params = ndarrays_to_parameters(aggregate(weight_results))

        total_examples = sum(fit_res.num_examples for fit_res in cluster_clients)

        representative_metrics = dict(cluster_clients[0].metrics)
        representative_metrics["cluster_id"] = cluster_id
        representative_metrics["loss"] = loss
        representative_metrics["accuracy"] = accuracy

        # print([fit_res.metrics["id"] for fit_res in cluster_clients])
        return FitRes(parameters=aggregated_params,
                      num_examples=total_examples,
                      metrics=representative_metrics,
                      status=Status(code=0, message="Aggregated successfully")
                    )


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        cluster_data = {}

        for client_res in results:
            client, fit_res = client_res
            cluster_id = fit_res.metrics["cluster_id"]
            if cluster_id not in cluster_data:
                cluster_data[cluster_id] = []

            cluster_data[cluster_id].append(fit_res)

        cluster_results = {}
        weights_results = []
        num_examples = []
        ids = []
        cluster_results = {}

        for cluster_id, fit_res_list in cluster_data.items():
            if len(fit_res_list) > 1:
                fit_res = self.aggregate_cluster(cluster_id, fit_res_list)
            else:
                fit_res = fit_res_list[0]

            cluster_results[cluster_id] = fit_res
            weights_results.append(parameters_to_ndarrays(fit_res.parameters))
            num_examples.append(fit_res.num_examples)
            ids.append(int(fit_res.metrics['cluster_id']))

        local_updates = np.array(weights_results, dtype=object) - np.array(parameters_to_ndarrays(self.current_parameters), dtype=object)

        local_gradients = -local_updates/self.learning_rate

        global_gradient = np.sum(np.array(num_examples).reshape(len(num_examples), 1) * local_gradients, axis=0) / sum(num_examples)

        local_grad_vectors = [np.concatenate([arr for arr in local_gradient], axis = None)
                              for local_gradient in local_gradients]

        global_grad_vector = np.concatenate([arr for arr in global_gradient], axis = None)

        instant_angles = np.arccos([np.dot(local_grad_vector, global_grad_vector) / (np.linalg.norm(local_grad_vector) * np.linalg.norm(global_grad_vector))
                          for local_grad_vector in local_grad_vectors])
            
        id_to_instant_angle = dict(zip(ids, instant_angles))
        smoothed_angles = []
        for cluster_id in ids:
            prev_angle = self.current_angles.get(cluster_id, None)
            curr_angle = id_to_instant_angle[cluster_id]
            if prev_angle is None:
                smoothed = curr_angle
            else:
                smoothed = (server_round - 1) / server_round * prev_angle + 1 / server_round * curr_angle
            smoothed_angles.append(smoothed)
            self.current_angles[cluster_id] = smoothed

        maps = self.alpha*(1-np.exp(-np.exp(-self.alpha*(np.array(smoothed_angles)-1))))

        weights = num_examples * np.exp(maps) / sum(num_examples * np.exp(maps))

        parameters_aggregated = np.sum(weights.reshape(len(weights), 1) * np.array(weights_results, dtype=object), axis=0)

        self.current_parameters = ndarrays_to_parameters(parameters_aggregated)
        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in cluster_results.items()]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in cluster_results.items()]
        
        for _, v in cluster_data.items():
            if len(v) == 1:
                fit_res = v[0]
                losses.append(fit_res.num_examples * fit_res.metrics["loss"])
                corrects.append(round(fit_res.num_examples * fit_res.metrics["accuracy"]))
                loss = sum(losses) / sum(num_examples)

        loss = sum(losses) / sum(num_examples)
        accuracy = sum(corrects) / sum(num_examples)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return self.current_parameters, metrics_aggregated