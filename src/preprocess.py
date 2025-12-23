import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(path):
    df = pd.read_csv(path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

if __name__ == "__main__":
    preprocess("data/raw/creditcard.csv")
