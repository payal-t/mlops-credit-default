"""
Split data.csv into train and test sets for local training
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Ensure data folder exists
    os.makedirs("data", exist_ok=True)

    # Load CSV
    data_path = "data.csv"  # Make sure data.csv is in the same folder as this script
    df = pd.read_csv(data_path, sep=",", quotechar='"')
    print("Columns in dataset:", df.columns)

    # Rename target for simplicity
    df = df.rename(columns={"default.payment.next.month": "default"})

    # Split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["default"]
    )

    # Save split datasets
    train_df.to_csv("data/train_set.csv", index=False)
    test_df.to_csv("data/test_set.csv", index=False)

    print("Train and test CSV files saved in 'data/' folder.")

if __name__ == "__main__":
    main()
