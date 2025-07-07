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

def clustering(dist, min_smp=2, xi=0.15, algo='kmeans', distance='manhattan', noise_level=0.05, num_clusters=8, cluster_size=None):
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
            labels, centroid = balanced_kmeans(X=distrib_, num_clusters=num_clusters, cluster_sizes=cluster_size, distance_func=hellinger, verbose=False)
        elif distance == 'jensenshannon':
            labels, centroid = balanced_kmeans(X=distrib_, num_clusters=num_clusters, cluster_sizes=cluster_size, distance_func=jensen_shannon_divergence_distance, verbose=False)
        else:
            labels, centroid = balanced_kmeans(X=distrib_, num_clusters=num_clusters, cluster_sizes=cluster_size, verbose=False)

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

def balanced_kmeans(X, num_clusters, cluster_sizes, distance_func=None, max_iter=100, tol=1e-4, verbose=False):
    X = np.asarray(X)
    n_samples, _ = X.shape

    # kiểm tra
    assert len(cluster_sizes) == num_clusters
    assert sum(cluster_sizes) == n_samples

    if distance_func is None:
        distance_func = lambda u, v: np.linalg.norm(u - v)

    init_idx = random.sample(range(n_samples), num_clusters)
    centroids = X[init_idx].copy()

    # tính cumsum và slot→cluster mapping
    cum_sizes = np.cumsum(cluster_sizes)
    # slots[a] = cluster index cho slot thứ a (a từ 0..n-1)
    slots = np.searchsorted(cum_sizes, np.arange(1, n_samples+1))

    for it in range(max_iter):
        # vector hóa: với mỗi cluster j, ta tính dist từ X tới centroids[j]
        # rồi gán cho tất cả a mà slots[a]==j
        cost = np.empty((n_samples, n_samples), dtype=float)
        for j in range(num_clusters):
            # indices của các slot a trỏ vào cluster j
            mask = (slots == j)
            # tính distance từ centroid j đến mọi X
            dists = np.array([distance_func(x, centroids[j]) for x in X])
            # bình phương và gán vào các slot
            cost[mask, :] = dists[None, :]**2

        # assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        # X[i] gán cluster slots[a] nếu (a,i) match
        labels = np.empty(n_samples, dtype=int)
        for a, i in zip(row_ind, col_ind):
            labels[i] = slots[a]

        # update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(num_clusters):
            pts = X[labels == j]
            if len(pts) == 0:
                new_centroids[j] = X[random.randrange(n_samples)]
            else:
                new_centroids[j] = pts.mean(axis=0)

        # check hội tụ
        shift = sum(distance_func(c0, c1) for c0, c1 in zip(centroids, new_centroids))
        if verbose:
            print(f"[FixedKMeans] it={it+1}, shift={shift:.6f}")
        centroids = new_centroids
        if shift < tol:
            break

    return labels, centroids