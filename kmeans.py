import sys
import pandas as pd

def main(csv_file, k):
    df = pd.read_csv(csv_file)


def read_csv(csv_path:str) -> tuple[pd.DataFrame, pd.Series]:
    # read entire csv
    df = pd.read_csv(csv_path, header=None)

    restrictions = df.iloc[0].astype(int)
    # drop metadata
    df = df.drop(index=0).reset_index(drop=True)

    clean_df = restrictions[restrictions == 1].index

    x = df.loc[:, clean_df].astype(float)

    return x

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    print("Usage: python3 kmeans.py <csv_file> <k>]")