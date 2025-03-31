import pandas as pd

def load_data(train_file_path:str, test_file_path:str):
    # Load datasets
    df_train = pd.read_csv(train_file_path)
    df_test = pd.read_csv(test_file_path)
    
    # Create copies to avoid modifying original data
    df_train_copy = df_train.copy()
    df_test_copy = df_test.copy()
    
    return df_train_copy, df_test_copy

if __name__ == "__main__":
    df_train, df_test = load_data()
    print("Data Loaded Successfully")
