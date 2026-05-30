import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_prep_data():
    """
    Loads and prepares the heart disease dataset for model training.
    """
    # Construct the absolute path to the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "data", "Heart_Disease_Prediction.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    # Load and prepare the data
    df = pd.read_csv(file_path)
    df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

    X = df.drop(columns=['Heart Disease'])
    y = df['Heart Disease']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Running data preparation script...")
    try:
        X_train, X_test, y_train, y_test = load_and_prep_data()
        print("Data loaded and prepared successfully.")
        print(f"  Training features shape: {X_train.shape}")
        print(f"  Testing features shape:  {X_test.shape}")
        print(f"  Training labels shape:   {y_train.shape}")
        print(f"  Testing labels shape:    {y_test.shape}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
