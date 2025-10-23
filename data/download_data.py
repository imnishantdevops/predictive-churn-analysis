# data/download_data.py
import kagglehub
import pandas as pd
import os

def download_dataset():
    print("Downloading Telco Customer Churn Dataset...")

    try:
        # Download dataset from Kaggle
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        print(f"Dataset downloaded to: {path}")

        # List files in the downloaded directory
        print("Files in downloaded directory:")
        for file in os.listdir(path):
            print(f"  - {file}")

        # Try to find the dataset file
        possible_filenames = [
            "WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "Telco-Customer-Churn.csv",
            "customer_churn.csv"
        ]

        csv_file_path = None
        for filename in possible_filenames:
            potential_path = os.path.join(path, filename)
            if os.path.exists(potential_path):
                csv_file_path = potential_path
                break

        # If not found, pick the first .csv file
        if csv_file_path is None:
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    csv_file_path = os.path.join(path, file)
                    break

        # Save locally in /data directory
        if csv_file_path and os.path.exists(csv_file_path):
            print(f"Found dataset file: {csv_file_path}")

            df = pd.read_csv(csv_file_path)
            os.makedirs("data", exist_ok=True)
            output_path = "data/customer_churn.csv"
            df.to_csv(output_path, index=False)

            print(f"Dataset saved as: {output_path}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            return True
        else:
            print("Could not find CSV file in downloaded dataset")
            return False

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


if __name__ == "__main__":
    download_dataset()
