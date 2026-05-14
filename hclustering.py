import numpy as np
import pandas as pd
import numpy
from sklearn.metrics import pairwise_distances
from kmeans import read_csv
def main(filename, threshold):
    df = read_csv(filename)
    arr = df.to_numpy()
    data_points = [x for x in arr]
    clusters = [{"type": "leaf", "data": x} for x in arr]
    matrix = pd.DataFrame(pairwise_distances(data_points, Y=None, metric='euclidean'))

    while len(matrix) > 1:
        matrix[matrix == 0] = np.inf

        a = matrix.min(axis=0).idxmin()
        b = matrix.loc[a].idxmin()

        matrix[matrix == np.inf] = 0

        row_a = matrix[a]
        row_b = matrix[b]

        min_arr = np.minimum(row_a, row_b)

        matrix.loc[b] = min_arr
        matrix.loc[:, b] = min_arr

        matrix = matrix.drop(b, axis=0).reset_index(drop=True)
        matrix = matrix.drop(b, axis=1)
        matrix.columns = range(matrix.columns.size)

        clusters[a] = {"type": "node", "height": 21, "nodes": [clusters[a], clusters[b]]}
        print(clusters[b])

        clusters.pop(b)

    dentrogram = {"type": "root", "height": 21, "nodes": clusters[0]}











