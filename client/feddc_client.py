from client.client import BaseClient
from global_import import *

class FedDC_client(BaseClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = [p.numel() for p in self.net.parameters()]
        self.n_par = sum(self.shapes)

        self.state_grad_diff = np.zeros(self.n_par, dtype=np.float64)
        self.params_drift = np.zeros(self.n_par, dtype=np.float64)

    def _get_flat(self) -> np.ndarray:
        return np.concatenate([
            p.data.cpu().numpy().reshape(-1) for p in self.net.parameters()
        ])

    def fit(self, parameters: List[np.ndarray], config: Dict):
        set_parameters(self.net, parameters)
        param_vec = self._get_flat()  # 1D numpy array size = self.n_par

        global_update_last = np.frombuffer(config['global_update_last'], dtype=np.float64).copy()
        alpha = config['alpha']
        lr = config['learning_rate']
        num_epochs = self.local_train_epcs

        loss, acc, theta_i_current = train_feddc(
            net=self.net,
            local_update_last=torch.from_numpy(self.state_grad_diff).to(self.device),
            global_update_last=torch.from_numpy(global_update_last).to(self.device),
            global_model_param=torch.tensor(param_vec, device=self.device),
            h_i=torch.tensor(self.params_drift, device=self.device),
            lr=lr,
            trainloader=self.trainloader,
            criterion=self.criterion,
            device=self.device,
            alpha=alpha,
            num_epochs=num_epochs
        )

        theta_vec = theta_i_current.cpu().numpy()

        new_local_update = theta_vec - param_vec   
        
        # h_i is parameters drift
        self.params_drift += new_local_update

        # g_i is the local update value of i-th client’s local parameters in last round
        self.state_grad_diff = new_local_update

        updated_params = get_parameters(self.net)
        return updated_params, len(self.trainloader.dataset), {
            'drift':   self.params_drift.astype(np.float64).tobytes(),
            'delta_g': new_local_update.astype(np.float64).tobytes(),    
            'loss':    float(loss),
            'accuracy': float(acc),
        }

def train_feddc(
    net,
    local_update_last: torch.Tensor,
    global_update_last: torch.Tensor,
    global_model_param: torch.Tensor,
    h_i: torch.Tensor,
    lr: float,
    trainloader,
    criterion,
    device,
    alpha: float,
    num_epochs: int = 1
) -> Tuple[float, float]:

    # update_diff: g_i^{(t-1)} - g^{(t-1)}
    update_diff = local_update_last - global_update_last

    net.to(device)
    net.train()
    optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)

    total_loss, total_acc = 0.0, 0.0

    for _ in range(num_epochs):
        running_loss, running_corrects, total_samples = 0.0, 0, 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss_f = criterion(outputs, labels)  # Loss dữ liệu

            local_parameter = torch.cat([param.flatten() for param in net.parameters()])

            # loss_r: (alpha/2) * ||theta_i - theta^{(t)} + h_i||^2
            r_coef = alpha / 2
            loss_r = r_coef * torch.sum((local_parameter - (global_model_param - h_i)) ** 2)

            # loss_g: (1/(eta * num_batches * num_epoch)) * <theta_i, g_i^{(t-1)} - g^{(t-1)}>
            g_coef = 1 / (lr * len(trainloader) * num_epochs)
            loss_g = g_coef * torch.sum(local_parameter * update_diff)

            # Tổng loss
            loss = loss_f + loss_r + loss_g
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            total_samples += images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            running_loss += loss.item() * images.size(0)

        total_loss += running_loss / total_samples
        total_acc += running_corrects / total_samples

    total_loss /= num_epochs
    total_acc /= num_epochs
    theta_i_current = torch.cat([param.flatten() for param in net.parameters()]).detach()

    return total_loss, total_acc, theta_i_current