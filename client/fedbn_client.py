from client.client import BaseClient
from global_import import *
import pickle

class FedBNClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bn_state_dir = os.path.join('bn_state_dir', f"client_{self.cid}")
        os.makedirs(bn_state_dir, exist_ok=True)
        self.bn_state_pkl = os.path.join(bn_state_dir, f"client_{self.cid}.pkl")

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        self._save_bn_statedict(self.net)
        return [
            val.cpu().numpy()
            for name, val in self.net.state_dict().items()
            if "bn" not in name
        ]

    def set_parameters(self, parameters):
        keys = [k for k in self.net.state_dict().keys() if "bn" not in k]
        bn_state_dict = self._load_bn_statedict()

        state_dict = OrderedDict()
        params_dict = zip(keys, parameters)

        for k, v in params_dict:
          state_dict[k] = torch.tensor(v, dtype=torch.float32)

        state_dict.update(bn_state_dict)
        self.net.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        if config["round"] == 1:
            set_parameters(self.net, parameters)
        else:
            self.set_parameters(parameters)

        optimizer = torch.optim.SGD(self.net.parameters(), lr=config["learning_rate"])
        loss, acc = train(self.net, self.trainloader, self.criterion, optimizer, device=self.device, num_epochs=self.local_train_epcs)
            
        return self.get_parameters(config), len(self.trainloader.sampler), {
            "loss": loss,
            "accuracy": acc,
            "id": self.cid
        }
    
    def _save_bn_statedict(self, net) -> None:
        bn_state = {
            name: val.cpu().numpy()
            for name, val in net.state_dict().items()
            if "bn" in name
        }

        with open(self.bn_state_pkl, "wb") as handle:
            pickle.dump(bn_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_bn_statedict(self) -> Dict[str, torch.tensor]:
        with open(self.bn_state_pkl, "rb") as handle:
            data = pickle.load(handle)
        bn_state_dict = {k: torch.tensor(v) for k, v in data.items()}
        return bn_state_dict