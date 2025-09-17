# src/data.py
from sklearn.datasets import load_iris
import pandas as pd

def load_dataset():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y

if __name__ == "__main__":
    X, y = load_dataset()
    df = pd.concat([X, y.rename("target")], axis=1)
    print(df.head())
