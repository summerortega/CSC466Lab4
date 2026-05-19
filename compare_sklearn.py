import sys

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from evaluation import read_csv, print_cluster_report
def main(filename, ground_truth_col=None):
    df, ground_truth = read_csv(filename, ground_truth_col)
    data = df.to_numpy()

    print("\nSklearn KMeans")
    model = KMeans(n_clusters=4, random_state=0)
    labels = model.fit_predict(data)
    print_cluster_report(data, labels, ground_truth)

    # print("\nSklearn Agglomerative (Hierarchical)")
    # model = AgglomerativeClustering(n_clusters=2, linkage="single")
    # labels = model.fit_predict(data)
    # print_cluster_report(data, labels, ground_truth)
    #
    # print("\nSklearn DBSCAN")
    # model = DBSCAN(eps=0.4, min_samples=3)
    # labels = model.fit_predict(data)
    # print_cluster_report(data, labels, ground_truth)


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 1:
        main(args[0])
    elif len(args) == 2:
        main(args[0], int(args[1]))
    else:
        print("Usage: python3 compare_sklearn.py <csv_path> [<ground_truth_col>]")