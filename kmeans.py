import sys
import numpy as np
import pandas as pd
import math
from evaluation import read_csv


def select_centroids(df: pd.DataFrame, k:int) -> np.ndarray:
    return df.sample(k, axis=0).to_numpy()


def calc_sse(centroids:np.ndarray, clusters:np.ndarray) -> float:
    new_sse = 0
    for i in range(centroids.shape[0]):
        new_sse += np.sum([euc_dist(centroids[i], clusters[j]) ** 2 for j in range(clusters.shape[0])])
    return new_sse


def euc_dist(x:pd.Series, centroid:pd.Series) -> float:
    return math.sqrt(np.sum(np.square(x - centroid)))


def main(csv_file:str, k:int, threshold:float):
    data = read_csv(csv_file)
    df = data[0]
    #step 1: select initial k centroids
    centroids = select_centroids(df, k)
    clusters = np.array([])
    # while not converged
    prev_sse = -1
    current_sse = 0
    while abs(prev_sse - current_sse) > threshold:
        # step 2: set up cluster sum and num_pts arrays. used for finding means to recalculate centroids
        cluster_sums = np.zeros(shape=(k, df.columns.size))
        num_pts = np.zeros(shape=k)
        clusters = np.full(shape=(k, df.shape[0], df.shape[1]), fill_value=0)
        # step 3: compute distances for all points and add to appropriate cluster
        for i in range(df.shape[0]):
            dists = np.array([euc_dist(df.iloc[i], centroids[l]) for l in range(k)])
            cluster = np.argmin(dists)
            cluster_sums[cluster] = cluster_sums[cluster] + df.iloc[i]
            clusters[cluster][i] = df.iloc[i].to_numpy(dtype="float32")
            num_pts[cluster] += 1
        centroids = np.array([cluster_sums[l] / num_pts[l] for l in range(k)])
        prev_sse = current_sse
        current_sse = calc_sse(centroids, clusters)
    final_clusters = {}
    zero_arr = np.zeros(shape=(df.shape[1]))
    shape = clusters[0].shape[0]
    for i in range(clusters.shape[0]):
        mask = [np.array_equal(clusters[i][j], zero_arr) for j in range(shape)]
        new_mask = np.logical_not(mask)
        cluster = clusters[i][new_mask]
        final_clusters[f"#{i+1}"] = cluster
    return final_clusters


if __name__ == "__main__":
    print(main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3])))
    print("Usage: python3 kmeans.py <csv_file> <k> <threshold>]")