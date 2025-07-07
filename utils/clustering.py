from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import numpy as np
import random
from sklearn.cluster import OPTICS
from scipy.optimize import linear_sum_assignment

from utils.distance import hellinger, jensen_shannon_divergence_distance

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
        if distance == 'hellinger':
            labels, centroid = kmeans(X=distrib_, num_clusters=num_clusters, distance_func=hellinger, verbose=False) 
        elif distance == 'jensenshannon':
            labels, centroid = kmeans(X=distrib_, num_clusters=num_clusters, distance_func=jensen_shannon_divergence_distance, verbose=False)
    elif algo == 'agglomerative':
        if distance == 'hellinger':
            dists = pairwise_distances(distrib_, metric=hellinger)
            model = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
            labels = model.fit_predict(dists)
        elif distance == 'jensenshannon':
            dists = pairwise_distances(distrib_, metric=jensen_shannon_divergence_distance)
            model = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
            labels = model.fit_predict(dists)
        else:
            model = AgglomerativeClustering(n_clusters=num_clusters, affinity=distance, linkage='average')
            labels = model.fit_predict(distrib_)
    elif algo == 'bkmeans':
        if distance == 'hellinger':
            labels, centroid = balanced_kmeans(X=distrib_, num_clusters=num_clusters, cluster_size=5, distance_func=hellinger, verbose=False)
        elif distance == 'jensenshannon':
            labels, centroid = balanced_kmeans(X=distrib_, num_clusters=num_clusters, cluster_size=5, distance_func=jensen_shannon_divergence_distance, verbose=False)
        else:
            labels, centroid = balanced_kmeans(X=distrib_, num_clusters=num_clusters, cluster_size=5, verbose=False)

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

def balanced_kmeans(X, num_clusters, cluster_size, distance_func=None, max_iter=100, tol=1e-4, verbose=False):
    """
    Balanced k-Means clustering based on the fixed-size k-Means algorithm.

    Parameters:
        X (np.ndarray): Input data, shape (n_samples, n_features)
        num_clusters (int): Number of clusters k
        cluster_size (int): Number of points per cluster (assumed equal for all)
        distance_func (callable): Function to compute distance between two points
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance to stop the iteration
        verbose (bool): Whether to print iteration info

    Returns:
        labels (np.ndarray): Cluster assignment for each point
        centroids (np.ndarray): Final centroid positions
    """
    n_samples = len(X)
    X = np.array(X)

    if distance_func is None:
        distance_func = lambda x, y: np.linalg.norm(x - y)

    # --- Khởi tạo tâm cụm ban đầu (Initial centroids C^0)
    initial_indices = random.sample(range(n_samples), num_clusters)
    centroids = X[initial_indices]
    t = 0

    for iteration in range(max_iter):
        # Tính tổng cộng dồn số điểm cần trong mỗi cụm: c(j) = sum_{l=1}^j n_l  (3)
        cluster_sizes = [cluster_size] * num_clusters
        cumulative_cluster_sizes = np.cumsum(cluster_sizes)

        # Gán mẫu vào cụm: solve assignment bằng Hungarian algorithm (5)
        cost_matrix = np.zeros((n_samples, n_samples))
        for a in range(n_samples):
            for i in range(n_samples):
                # W(a, i) = dist(X_i, C_{argmin_j c(j) >= a})^2
                cluster_index = np.argmax(cumulative_cluster_sizes >= a + 1)
                cost_matrix[a, i] = distance_func(X[i], centroids[cluster_index]) ** 2

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Tạo nhãn dựa trên assignment (6)
        labels = np.zeros(n_samples, dtype=int)
        for a, i in zip(row_ind, col_ind):
            cluster_index = np.argmax(cumulative_cluster_sizes >= a + 1)
            labels[i] = cluster_index

        # Cập nhật tâm cụm mới (2)
        new_centroids = []
        for k in range(num_clusters):
            points_in_cluster = X[labels == k]
            if len(points_in_cluster) == 0:
                new_centroids.append(X[random.randint(0, n_samples - 1)])
            else:
                new_centroids.append(np.mean(points_in_cluster, axis=0))
        new_centroids = np.array(new_centroids)

        shift = sum(distance_func(c, nc) for c, nc in zip(centroids, new_centroids))
        if verbose:
            print(f"[Iter {iteration}] centroid shift: {shift:.6f}")
        if shift < tol:
            break

        centroids = new_centroids
        t += 1

    return labels, centroids