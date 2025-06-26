import numpy as np
from typing import List
import random
import torch
from collections import Counter
from torch.distributions.dirichlet import Dirichlet
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, FashionMNIST
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.cluster import OPTICS
from models import CustomDataset
from tqdm import tqdm 
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from utils.distance import hellinger, jensen_shannon_divergence_distance
from datasets import load_dataset 
import re

# ------------------------------ Preprocess ----------------------------------------
def clean_text(tweet):
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    tweet = tweet.lower()
    tweet = re.sub(urlPattern, '', tweet)
    tweet = re.sub(userPattern, '', tweet)
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
    tweet = tweet.replace('\r', '').replace('\n', ' ').lower()
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
    tweet = re.sub(r'[^\x00-\x7f]', r'', tweet)

    banned_list = string.punctuation + 'Ã' + '±' + 'ã' + '¼' + 'â' + '»' + '§'
    table = str.maketrans('', '', banned_list)
    tweet = tweet.translate(table)

    tweet = " ".join(word.strip() for word in re.split('#|_', tweet))
    tweet = ' '.join([word if ('$' not in word) and ('&' not in word) else '' for word in tweet.split(' ')])
    tweet = re.sub("\s\s+", " ", tweet)
    return tweet.strip()

def preprocess_text_agnews(root):
    keep = ['text', 'label']
    data = {key: root[key] for key in keep}

    data['text'] = [clean_text(text) for text in data['text']]
    data['label'] = [int(label) for label in data['label']] 

    return data

def get_transform(dataset_name):
    if dataset_name in ['cifar10', 'cifar100']:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name in ['emnist', 'fmnist']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
    else:
        return None

# ------------------------------------------- Load dataset --------------------------------------
def load_data(dataset: str): 
    datasets = {
        'cifar10': (CIFAR10, 'image'),
        'emnist': (EMNIST, 'image'),
        'fmnist': (FashionMNIST, 'image'),
        'cifar100': (CIFAR100, 'image'),
        'agnews': ('text', 'text')  
    }

    if dataset in datasets:
        if dataset == 'agnews':
            return load_agnews()

        dataset_class, datatype = datasets[dataset]
        transform = get_transform(dataset)

        if dataset in ['cifar10', 'cifar100']:
            trainset = dataset_class("data", train=True, download=True, transform=transform)
            testset = dataset_class("data", train=False, download=True, transform=transform)
        else:
            trainset = dataset_class("data", train=True, download=True, transform=transform)
            testset = dataset_class("data", train=False, download=True, transform=transform)

        return trainset, testset


def load_agnews():
    dataset = load_dataset("ag_news")

    train_raw = dataset['train']
    test_raw = dataset['test']

    train_data = preprocess_text_agnews(train_raw)
    test_data = preprocess_text_agnews(test_raw)

    max_words = 2000
    max_len = 500
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_data['text'])
    
    train_seq = tokenizer.texts_to_sequences(train_data['text'])
    test_seq = tokenizer.texts_to_sequences(test_data['text'])

    train_pad = torch.tensor(pad_sequences(train_seq, maxlen=max_len, padding='post', truncating='post'), dtype=torch.long)
    test_pad = torch.tensor(pad_sequences(test_seq, maxlen=max_len, padding='post', truncating='post'), dtype=torch.long)

    train_labels = torch.tensor(train_data['label'], dtype=torch.long)
    test_labels = torch.tensor(test_data['label'], dtype=torch.long)

    trainset = CustomDataset(train_pad, train_labels)
    testset = CustomDataset(test_pad, test_labels)

    return trainset, testset

def partition_data(dataset,
                   num_clients,
                   alpha,
                   classes_name):

    num_classes = len(classes_name)

    client_size = len(dataset) // num_clients
    label_size = len(dataset) // num_classes

    indices_class = [[] for _ in range(num_classes)]

    for i, lab in enumerate(dataset.targets):
        indices_class[lab].append(i)

    labels = list(range(num_classes))
    ids = []
    label_dist = []

    for i in tqdm(range(num_clients)):
        concentration = torch.ones(len(labels)) * alpha
        dist = Dirichlet(concentration).sample()

        client_indices = []
        for _ in range(client_size):
            if not labels:
                break

            label = random.choices(labels, dist)[0]
            if indices_class[label]:
                id_sample = random.choice(indices_class[label])
                client_indices.append(id_sample)
                indices_class[label].remove(id_sample)

                if not indices_class[label]:
                    dist = renormalize(dist, labels, label)
                    labels.remove(label)

        ids.append(client_indices)
        counter = Counter(list(map(lambda x: dataset[x][1], ids[i])))
        label_dist.append({classes_name[j]: counter.get(j, 0) for j in range(num_classes)})

    return ids, label_dist

def partition_data_special_case(trainset, num_clients: int, num_iids: int):
    classes = trainset.classes
    client_size = int(len(trainset)/num_clients)
    label_size = int(len(trainset)/len(classes))
    data = list(map(lambda x: (trainset[x][1], x), range(len(trainset))))
    data.sort()
    data = list(map(lambda x: data[x][1], range(len(data))))
    
    grouped_data = [data[i*label_size:(i+1)*label_size] for i in range(len(classes))]
    non_iid_labels = random.sample(range(len(classes)), 2) if len(classes) == 10 else list(range(10))
    non_iid_data = []
    for label in non_iid_labels:
        non_iid_data += grouped_data[label]

    ids = []
    label_dist = []
    for i in range(num_clients):
        temp_data = data if i < num_iids else non_iid_data
        id = random.sample(temp_data, client_size)
        ids.append(id)
        
        counter = Counter(list(map(lambda x: trainset[x][1], ids[i])))
        label_dist.append({classes[i]: counter.get(i, 0) for i in range(len(classes))})

    return ids, label_dist


def get_train_data(dataset_name,
                   num_clients,
                   batch_size, 
                   alphas: list = [0.5, 0.7, 0.9, 1],
                   special_case = False):

    num_folds = len(alphas)
    trainset, testset = load_data(dataset_name)

    base = len(trainset) // num_folds
    extra = len(trainset) % num_folds
    classes = trainset.classes

    fold_len = [base + 1 if i < extra else base for i in range(num_folds)]
    partition_fold = random_split(trainset, fold_len)

    base_clients = num_clients // num_folds
    extra_clients = num_clients % num_folds
    clients_per_fold = [base_clients + 1 if i < extra_clients else base_clients for i in range(num_folds)]

    ids, labels_dist = [], []

    for i in range(num_folds):
        sub_set = partition_fold[i]
        if dataset_name in ['cifar10', 'cifar100','agnews']:
            data = [trainset.data[idx] for idx in sub_set.indices]
            targets = [trainset.targets[idx] for idx in sub_set.indices]
        else:
            data = trainset.data[sub_set.indices]
            targets = trainset.targets[sub_set.indices].tolist()

        sub_dataset = CustomDataset(data, targets)

        if special_case:
            id, dist = partition_data_special_case(sub_dataset, clients_per_fold[i])
        else:
            id, dist = partition_data(
                sub_dataset,
                clients_per_fold[i],
                alphas[i],
                classes,
            )

        ids.extend(id)
        labels_dist.extend(dist)

    trainloaders = []

    for i in range(num_clients):
        trainloaders.append(DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(ids[i])))
    testloader = DataLoader(testset, batch_size=batch_size)
    
    client_dataset_ratio: float = int(len(trainset) / num_clients) / len(trainset)
    
    return ids, labels_dist, trainloaders, testloader, client_dataset_ratio

# ------------------------------------------ Support function --------------------------------------- 
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"Seeds set to {seed_value}")

def normalize_distribution(distribution):
    total = sum(distribution.values())
    return [count / total for count in distribution.values()]

def compute_uniform_distribution(num_classes):
    return [1 / num_classes] * num_classes


def renormalize(dist: torch.tensor, labels: List[int], label: int):
    idx = labels.index(label)
    dist[idx] = 0
    dist /= sum(dist)
    dist = torch.concat((dist[:idx], dist[idx+1:]))
    return dist

def build_distribution(dist, noise_level=0.05):
    distrib_ = [
        np.array(list(d.values())) / sum(d.values()) if sum(d.values()) > 0 else np.zeros(len(d))
        for d in dist
    ]
    distrib_ = np.array(distrib_)
    noise = np.random.lognormal(mean=0.0, sigma=noise_level, size=distrib_.shape)
    distrib_ += noise
    distrib_ = distrib_ / distrib_.sum(axis=1, keepdims=True)
    return distrib_

def get_optics_instance(distance, min_smp, xi):
    if distance == 'hellinger':
        return OPTICS(min_samples=min_smp, xi=xi, metric=hellinger, min_cluster_size=5)
    elif distance == 'jensenshannon':
        return OPTICS(min_samples=min_smp, xi=xi, metric=jensen_shannon_divergence_distance, min_cluster_size=5)
    else:
        return OPTICS(min_samples=min_smp, xi=xi, metric=distance, min_cluster_size=5)

def clustering(dist, min_smp=2, xi=0.15, algo='kmeans', distance='manhattan', noise_level=0.05, num_clusters=8):
    distrib_ = build_distribution(dist, noise_level=noise_level)
    
    if algo == 'optics':
        optics = get_optics_instance(distance, min_smp, xi)
        optics.fit(distrib_)
        labels = optics.labels_
    elif algo == 'kmeans': 
        labels, centroid = kmeans(X=distrib_, num_clusters=num_clusters, distance_func=hellinger, verbose=False) 

    client_cluster_index = {i: int(lab) for i, lab in enumerate(labels)}

    return client_cluster_index, distrib_

def kmeans(X, num_clusters=4, distance_func=None, max_iter=100, tol=1e-4, verbose=False):
    n_samples = len(X)
    X = np.array(X)

    if distance_func is None:
        distance_func = lambda x, y: np.linalg.norm(x - y)

    random_indices = random.sample(range(n_samples), num_clusters)
    centroids = X[random_indices]

    for iteration in range(max_iter):
        labels = []
        for x in X:
            distances = [distance_func(x, centroid) for centroid in centroids]
            label = np.argmin(distances)
            labels.append(label)
        labels = np.array(labels)

        new_centroids = []
        for k in range(num_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                new_centroids.append(X[random.randint(0, n_samples - 1)])
            else:
                new_centroids.append(np.mean(cluster_points, axis=0))
        new_centroids = np.array(new_centroids)

        shift = sum(distance_func(c, nc) for c, nc in zip(centroids, new_centroids))
        if verbose:
            print(f"Iteration {iteration + 1}: total centroid shift = {shift:.6f}")
        if shift < tol:
            break

        centroids = new_centroids

    return labels, centroids