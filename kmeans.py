import sys
from evaluation import read_csv


def main(csv_file, k):
    df = read_csv(csv_file)



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    print("Usage: python3 kmeans.py <csv_file> <k>]")