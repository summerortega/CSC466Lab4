import numpy as np
import pandas as pd
import numpy
from sklearn.metrics import pairwise_distances
from kmeans import read_csv
def main(filename, threshold):
    df = read_csv(filename)
    data_points = [[x] for x in df.to_numpy()]
    clusters = [{"type": "leaf", "data": x} for x in df.to_numpy()]
    matrix = pairwise_distances(data_points, Y=None, metric='euclidean')

    while len(clusters) > 1:
        # for j in range(len(clusters)):
        #     for k in range(j+1, len(clusters)):
        #         matrix[j][k] = distance_calculation(clusters[j], clusters[k])

        # calulcating minium distanc eand signle loink distance
        matrix[matrix == 0] = np.inf
        a = np.argmin(matrix) % len(matrix)
        b = np.argmin(matrix) // len(matrix)

        matrix[matrix == np.inf] = 0
        df_matrix = pd.DataFrame(matrix)

        a = df_matrix.min(axis=0).idxmin()
        b = df_matrix.iloc[a].idxmin()

        row_a = df_matrix[a]
        row_a[row_a == np.inf] = 0

        row_b = df_matrix[b]
        row_b[row_b == np.inf] = 0

        min_arr = np.minimum(row_a, row_b)

        df_matrix = df_matrix.iloc[a] = min_arr
        df_matrix.iloc[:, a] = min_arr

        df_matrix = df_matrix.drop(b, axis=0)
        df_matrix = df_matrix.drop(b, axis=1)

        data_points[a] = data_points[a].extend(data_points[b])
        data_points.pop(b)

        clusters[a] = {"type": "node", "height": 21, "nodes": [clusters[a], clusters[b]]}
        clusters.pop(b)










