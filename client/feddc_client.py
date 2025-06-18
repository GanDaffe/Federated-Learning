from client.client import BaseClient
from global_import import *

class FedDC_client(BaseClient):

    def __init__(self, *args, prev_model_save_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_model_save_dir = os.path.join(prev_model_save_dir, str(self.cid))

        if not os.path.exists(self.prev_model_save_dir):
            os.makedirs(self.prev_model_save_dir)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        set_parameters(self.net, parameters)

        global_model_param = torch.cat([
            torch.tensor(p, device=self.device, dtype=torch.float32).flatten()
            for p in parameters
        ])

        param_size = global_model_param.numel()
        path_local_update = os.path.join(self.prev_model_save_dir, 'local_update_last.pt')
        path_h_i = os.path.join(self.prev_model_save_dir, 'h_i.pt')

        # local_update_last = g_i^{(t-1)}
        # h_i 
        if config['server_round'] <= 1:
            local_update_last = torch.zeros(param_size, device=self.device, dtype=torch.float32)
            h_i = torch.zeros(param_size, device=self.device, dtype=torch.float32)
        else:
            local_update_last = load_(path_local_update, device=self.device, shape=param_size)
            h_i = load_(path_h_i, device=self.device, shape=param_size)
        
        # g
        buf = config['g']
        g_np = np.frombuffer(buf, dtype=np.float64)
        global_update_last = torch.tensor(g_np, device=self.device, dtype=torch.float32)

        loss, accuracy, theta_i_current = train_feddc(
            net=self.net,
            local_update_last=local_update_last,
            global_update_last=global_update_last,
            global_model_param=global_model_param,
            h_i=h_i,
            lr=config['learning_rate'],
            trainloader=self.trainloader,
            criterion=self.criterion,
            device=self.device,
            alpha=config['alpha'],
            num_epochs=self.local_train_epcs
        )

        # g_i^{(t)} = theta_i_current - theta^{(t-1)}
        new_local_update = theta_i_current - global_model_param

        # h_i^{(t)} = h_i^{(t-1)} + (theta_i^{(t)} - theta^{(t-1)})
        new_h_i = h_i + (theta_i_current - global_model_param)

        save_(path_local_update, new_local_update)
        save_(path_h_i, new_h_i)

        model_param = get_parameters(self.net)
        num_examples = len(self.trainloader.sampler)
        drift_np64 = new_h_i.detach().cpu().numpy().astype(np.float64)
        local_update_np64 = new_local_update.detach().cpu().numpy().astype(np.float64)

        return model_param, len(self.trainloader.sampler), {
            "drift": drift_np64.tobytes(),
            "local_update": local_update_np64.tobytes(),
            "loss": float(loss),
            "accuracy": float(accuracy),
            "id": self.cid
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
            loss_r = r_coef * torch.sum((h_i + local_parameter - global_model_param) ** 2)

            # loss_g: (1/(eta * num_batches)) * <theta_i, g_i^{(t-1)} - g^{(t-1)}>
            g_coef = 1 / (lr * len(trainloader))
            loss_g = g_coef * torch.sum(update_diff * local_parameter)

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

def save_(file_path: str, data):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(data, file_path)

def load_(file_path: str, device, shape):
    if os.path.exists(file_path):
        t = torch.load(file_path, map_location=device)
        t = t.to(device)
        return t
    else:
        return torch.zeros(shape, device=device)