# CSC466_Lab4

Summer Mariana Ortega
sorteg16@calpoly.edu

Diego Melgoza
drmelgoz@calpoly.edu

kmeans.py: The k-means clustering algorithm. The program outputs basic statistics for the clustering using functions from evaluation.py.
To properly call the program, input a CSV file path, the # of clusters desired, and a threshold value (optional; by default set to 0.025). Call "python3 kmeans.py" for more specifics.

With hclustering.py if you provide the threshold, the program will store the cut clusters into cut_clusters.json. 
The dendrogram is printed out as stored into dendrogram.json

The optional third argument is the column index of the ground-truth class label (Refer to header).
This allows for the Rand Index to be computed. 

Evaluation.py is used to calculate evaluation metrics for the 3 clustering algorithms. 
