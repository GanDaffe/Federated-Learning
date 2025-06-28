import flwr as fl
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Context

from utils import get_model, get_train_data, clustering, set_seed, compute_entropy, get_parameters, normalize_distribution, compute_uniform_distribution, kl_divergence
from client import *     
from algorithm import *
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

def get_num_clients(dataset_name): 
    if dataset_name in ['fmnist', 'agnews']:
        return 120
    elif dataset_name in ['cifar10', 'cifar100']:
        return 80
    
def get_num_rounds(dataset_name):
    if dataset_name in ['fmnist', 'agnews']:
        return 100
    elif dataset_name == 'cifar10':
        return 300
    elif dataset_name == 'cifar100': 
        return 500
# -------------------------------- HYPER PARAMETERS -------------------------------------

RANDOM_SEED = 42
ALGO = 'fedhcw'
BATCH_SIZE = 32
PARTITION_FRACTION = [0.3, 0.3, 0.3, 0.1]
ALPHA = [0.01, 0.05, 0.1, 100]
LR = 0.01
LOCAL_TRAINING = 10
DATASET_NAME = 'fmnist' 
DISTANCE = 'hellinger'
NUM_CLIENTS = get_num_clients(DATASET_NAME) 
NUM_ROUNDS = get_num_rounds(DATASET_NAME)

EXP_NAME = f'{ALGO}_{DATASET_NAME}-dataset_{LR}-lr_{NUM_CLIENTS}-clients_{LOCAL_TRAINING}-epochs'
set_seed(RANDOM_SEED)

    
# ------------------------------------ PREPROCESS -----------------------------------------
ids, dist, trainloaders, testloader, client_dataset_ratio = get_train_data(
    dataset_name=DATASET_NAME,
    num_clients=NUM_CLIENTS,
    batch_size=BATCH_SIZE, 
    fractions=PARTITION_FRACTION,
    alphas=ALPHA
)

if ALGO in ['fedhcw']:
    client_cluster_index, distrib_ = clustering(
        dist, 
        algo='kmeans'
    )

    num_cluster = len(list(set(client_cluster_index.values()))) - 1
    print(f'Number of Clusters: {num_cluster}')
    
    inc = 1
    for k, v in client_cluster_index.items():
        if v == -1:
            client_cluster_index[k] = num_cluster + inc
            inc += 1
            
    for k, v in client_cluster_index.items():
        print(f'Client {k + 1}: Cluster: {v}')

    for i in range(NUM_CLIENTS):
        print(f"Client {i+1}: {dist[i]}")

if ALGO == 'feddisco': 
    dk = {}
    num_classes = len(dist[0])  
    uniform_dist = compute_uniform_distribution(num_classes)

    for client_id in range(NUM_CLIENTS):
        client_distribution = normalize_distribution(dist[client_id])
        dk[client_id] = kl_divergence(client_distribution, uniform_dist)
        
entropies = [compute_entropy(dist[i]) for i in range(NUM_CLIENTS)]
    
# ---------------------------- CLIENT_FN -------------------------------------

def client_fn(context: Context): 
    cid = int(context.node_config["partition-id"])
    is_moon_type = True if ALGO == 'moon' else False 

    net = get_model(dataset_name=DATASET_NAME, moon_type=is_moon_type) 
    criterion = nn.CrossEntropyLoss()
    if ALGO in ['fedadp', 'fedavg', 'fedimp', 'fedprox', 'feddisco']:
        return BaseClient(cid, net, LOCAL_TRAINING, trainloaders[cid], criterion, DEVICE).to_client()
    elif ALGO in ['fedhcw']:
        return ClusterFedClient(cid, net, LOCAL_TRAINING, trainloaders[cid], criterion, DEVICE, cluster_id=client_cluster_index[cid]).to_client()
    elif ALGO == 'fedbn':
        return FedBNClient(cid, net, LOCAL_TRAINING, trainloaders[cid], criterion, DEVICE).to_client()
    elif ALGO == 'moon':
        return MoonClient(cid, net, LOCAL_TRAINING, trainloaders[cid], criterion, DEVICE, dir='/moon_save_point/').to_client()
    elif ALGO == 'fedaaw':
        return FedAAW_Client(cid, net, LOCAL_TRAINING, trainloaders[cid], criterion, DEVICE).to_client()
    elif ALGO == 'feddc':
        return FedDC_client(cid, net, LOCAL_TRAINING, trainloaders[cid], criterion, DEVICE).to_client()
    elif ALGO == 'fedntd':
        return FedNTD_Client(cid, net, LOCAL_TRAINING, trainloaders[cid], DEVICE, criterion=None).to_client()
# ---------------------------- STRATEGY ----------------------------------------

is_moon_type = True if ALGO == 'moon' else False 
net_ = get_model(DATASET_NAME, is_moon_type)
current_parameters = ndarrays_to_parameters(get_parameters(net_)) 

def get_algorithm():
    if ALGO == 'fedavg':
        return FedAvg
    elif ALGO == 'fedadp':
        return FedAdp
    elif ALGO == 'fedbn':
        return FedBN 
    elif ALGO == 'moon':
        return MOON 
    elif ALGO == 'fedprox':
        return FedProx
    elif ALGO == 'fedhcw':
        return FedHCW
    elif ALGO == 'fedimp':
        return FedImp
    elif ALGO == 'fedaaw':
        return FedAAW
    elif ALGO == 'feddisco': 
        return FedDisco
    elif ALGO == 'feddc':
        return FedDC
    elif ALGO == 'fedntd':
        return FedNTD
    
def get_strategy(): 
    algo = get_algorithm() 
    if ALGO in ['fedavg', 'feddc', 'fedprox', 'fedadp', 'fedbn', 'moon', 'fedaaw', 'fedntd']:
        return algo(
            exp_name            = EXP_NAME,
            net                 = net_,
            num_rounds          = NUM_ROUNDS,
            num_clients         = NUM_CLIENTS, 
            testloader          = testloader,
            learning_rate       = LR,
            current_parameters  = current_parameters
        )
    elif ALGO in ['fedhcw', 'fedimp']:
        return algo(
            exp_name            = EXP_NAME, 
            net                 = net_, 
            num_rounds          = NUM_ROUNDS, 
            num_clients         = NUM_CLIENTS, 
            testloader          = testloader, 
            learning_rate       = LR,
            current_parameters  = current_parameters,
            entropies           = entropies

        )
    elif ALGO == 'feddisco':
        return algo(
            exp_name            = EXP_NAME,
            net                 = net_,
            num_rounds          = NUM_ROUNDS,
            num_clients         = NUM_CLIENTS,
            testloader          = testloader,
            learning_rate       = LR,
            current_parameters  = current_parameters,
            dk                  = dk,
        )

# ---------------------------- RUN SIMULATION -------------------------------------

client_resources = {"num_cpus": 2, "num_gpus": 0.2} if DEVICE == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}
fl.simulation.start_simulation(
            client_fn           = client_fn,
            num_clients         = NUM_CLIENTS,
            config              = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy            = get_strategy(),
            client_manager      = SimpleClientManager(),
            client_resources     = client_resources
        )