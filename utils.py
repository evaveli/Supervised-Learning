import os
import pandas as pd
import time
from scipy.io import arff


def load_datasets(directory, dataset_name):
    """
    Loads training and testing datasets from ARFF files across multiple folds.

    Args:
        directory (str): Directory containing the ARFF files.
        dataset_name (str): Base name of the dataset.
        n_folds (int, optional): Number of folds to load. Defaults to 10.

    Returns:
        Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
            - List of training DataFrames.
            - List of testing DataFrames.
            - List of combined training and testing DataFrames per fold.
    """
    train_dfs = []
    test_dfs = []
    train_and_test_dfs = []
    for i in range(10):
        train_file = os.path.join(directory, f"{dataset_name}.fold.{i:06d}.train.arff")
        test_file = os.path.join(directory, f"{dataset_name}.fold.{i:06d}.test.arff")

        data_train, _ = arff.loadarff(train_file)
        data_test, _ = arff.loadarff(test_file)

        df_train = pd.DataFrame(data_train)
        df_test = pd.DataFrame(data_test)

        df_train.attrs["fold"] = i
        df_test.attrs["fold"] = i
        df_train.attrs["origin"] = "train"
        df_test.attrs["origin"] = "test"

        combined_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

        train_dfs.append(df_train)
        test_dfs.append(df_test)
        train_and_test_dfs.append(combined_df)
    return train_dfs, test_dfs, train_and_test_dfs


def get_process_time(process, start_time):
    """
    Calculates and logs the elapsed time for a given process.

    Args:
        process (str): Description of the process.
        start_time (float): Timestamp when the process started.

    Returns:
        float: Current timestamp after the process completion.

    """
    current_time = time.time()
    process_time = current_time - start_time
    print(f"Finished {process} in {process_time:.3f} seconds.")
    return current_time


def create_directory(directory, filename):
    """
    Creates a directory if it does not exist.

    Args:
        directory (str): Directory path.
        filename (str): File name.

    Returns:
        Tuple[bool, str]:
            - bool: True if the file exists, False otherwise.
            - str: Full path to the output file.

    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    output_file = os.path.join(directory, filename)
    file_exists = os.path.isfile(output_file)

    return file_exists, output_file
