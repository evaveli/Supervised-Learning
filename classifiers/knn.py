import csv
import os
import multiprocessing
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from concurrent.futures import ProcessPoolExecutor
from metrics.feature_weighting import determine_feature_weights
from metrics.voting_schemes import (
    majority_vote,
    inverse_distance_weighted_vote,
    sheppard_vote,
)

from utils import create_directory


def process_single_fold(args):
    """
    Processes a single cross-validation fold using batched KNN.

    Parameters:
        args (tuple): Contains (train_fold, test_fold, k, distance_method, p, voting_method, w_method).

    Returns:
        dict: Evaluation metrics including accuracy, precision, recall, f1, and processing time.
    """
    train_fold, test_fold, k, distance_method, p, voting_method, w_method = args

    # Prepare data
    X_train = train_fold[:, :-1].astype(float)
    y_train = train_fold[:, -1]
    y_test = test_fold[:, -1]

    # Calculate feature weights
    feature_weights = determine_feature_weights(X_train, y_train, method=w_method)

    start_time = time.time()

    # Use the batched KNN implementation
    predictions = kNNAlgorithm_batched(
        train_fold,
        test_fold,
        k=k,
        distance_metric=distance_method,
        p=p,
        voting_method=voting_method,
        feature_weights=feature_weights,
        batch_size=100,  # Process 100 test instances at a time
    )

    fold_time = time.time() - start_time

    # Calculate metrics
    metrics = {
        "accuracy": np.mean(predictions == y_test),
        "precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "time": fold_time,
    }

    return metrics


def cross_validate_kNN_weighted_parallel(
    train_folds,
    test_folds,
    distance_methods,
    k_values,
    p_values,
    voting_methods,
    weighting_methods,
    output_dir,
):
    """
    Performs parallel cross-validation for weighted KNN across multiple parameter combinations.

    Parameters:
        train_folds (list of np.ndarray): List of training data folds.
        test_folds (list of np.ndarray): List of testing data folds.
        distance_methods (list of str): Distance metrics to evaluate (e.g., 'minkowski', 'clark').
        k_values (list of int): K values (number of neighbors) to evaluate.
        p_values (list of int): P values for Minkowski distance (e.g., 1, 2, 3).
        voting_methods (list of str): Voting strategies to evaluate (e.g., 'uniform', 'distance').
        weighting_methods (list of str): Feature weighting methods (e.g., 'equal', 'information_gain', 'reliefF').

    Returns:
        list of dict: Aggregated metrics for each parameter combination.
    """
    results = []

    # Use all CPU cores except one
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {n_workers} worker processes")

    file_exists, output_file = create_directory(output_dir, "knn_metrics.csv")

    # Create a process pool
    with ProcessPoolExecutor(max_workers=n_workers) as executor, open(
        output_file, mode="a", newline=""
    ) as csv_file:

        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow(
                [
                    "w_method",
                    "k",
                    "p",
                    "voting_method",
                    "distance_method",
                    "mean_accuracy",
                    "std_accuracy",
                    "mean_precision",
                    "mean_recall",
                    "mean_f1",
                    "mean_time",
                ]
            )
        # Iterate through all parameter combinations
        for w_method in weighting_methods:
            for k in k_values:
                for p in p_values:
                    for voting_method in voting_methods:
                        for distance_method in distance_methods:
                            print(
                                f"\nEvaluating: w={w_method}, k={k}, p={p}, "
                                f"voting={voting_method}, distance={distance_method}"
                            )

                            # Prepare arguments for all folds
                            fold_args = [
                                (
                                    train_fold,
                                    test_fold,
                                    k,
                                    distance_method,
                                    p,
                                    voting_method,
                                    w_method,
                                )
                                for train_fold, test_fold in zip(
                                    train_folds, test_folds
                                )
                            ]

                            # Process all folds in parallel
                            fold_results = list(
                                executor.map(process_single_fold, fold_args)
                            )

                            # Calculate mean metrics across folds
                            mean_accuracy = np.mean(
                                [r["accuracy"] for r in fold_results]
                            )
                            std_accuracy = np.std([r["accuracy"] for r in fold_results])
                            mean_precision = np.mean(
                                [r["precision"] for r in fold_results]
                            )
                            mean_recall = np.mean([r["recall"] for r in fold_results])
                            mean_f1 = np.mean([r["f1"] for r in fold_results])
                            mean_time = np.mean([r["time"] for r in fold_results])

                            # Print results
                            print(
                                f"Results: Acc={mean_accuracy:.4f} (±{std_accuracy:.4f}), "
                                f"Prec={mean_precision:.4f}, "
                                f"Rec={mean_recall:.4f}, "
                                f"F1={mean_f1:.4f}, "
                                f"Time={mean_time:.4f}s"
                            )

                            csv_writer.writerow(
                                [
                                    w_method,
                                    k,
                                    p,
                                    voting_method,
                                    distance_method,
                                    mean_accuracy,
                                    std_accuracy,
                                    mean_precision,
                                    mean_recall,
                                    mean_f1,
                                    mean_time,
                                ]
                            )

                            # Store results
                            results.append(
                                {
                                    "w_method": w_method,
                                    "k": k,
                                    "p": p,
                                    "voting_method": voting_method,
                                    "distance_method": distance_method,
                                    "mean_accuracy": mean_accuracy,
                                    "std_accuracy": std_accuracy,
                                    "mean_time": mean_time,
                                    "mean_precision": mean_precision,
                                    "mean_recall": mean_recall,
                                    "mean_f1": mean_f1,
                                }
                            )

    return results


def kNNAlgorithm_batched(
    train_matrix,
    test_matrix,
    k,
    distance_metric="minkowski",
    p=2,
    voting_method="majority",
    feature_weights=None,
    batch_size=100,
):
    """
    Performs K-Nearest Neighbors classification in batches.

    Parameters:
        train_matrix (np.ndarray): Training data with features and labels (last column).
        test_matrix (np.ndarray): Testing data with features and labels (last column).
        k (int): Number of nearest neighbors to consider.
        distance_metric (str): Distance metric to use ('minkowski', 'clark'). Defaults to 'minkowski'.
        p (int): Order for Minkowski distance (1, 2, 3). Relevant if distance_metric is 'minkowski'. Defaults to 2.
        voting_method (str): Voting strategy ('majority', 'inverse_distance', 'sheppard'). Defaults to 'majority'.
        feature_weights (np.ndarray): Weights for each feature. Defaults to None (equal weighting).
        batch_size (int): Number of test instances to process per batch. Defaults to 100.

    Returns:
        np.ndarray: Array of predicted labels for the test data.
    """
    train_features = train_matrix[:, :-1].astype(float)
    train_labels = train_matrix[:, -1]
    test_features = test_matrix[:, :-1].astype(float)
    predictions = []

    actual_k = min(k, len(train_features) - 1)
    if actual_k < 1:
        print(
            f"Warning: Not enough instances ({len(train_features)}) for k={k}. Using k=1"
        )
        actual_k = 1

    if feature_weights is not None:
        train_features = train_features * feature_weights
        test_features = test_features * feature_weights

    # Process test instances in batches
    for i in range(0, len(test_features), batch_size):
        batch_end = min(i + batch_size, len(test_features))
        test_batch = test_features[i:batch_end]

        # Compute distances for the whole batch at once
        if distance_metric == "minkowski":
            diff = train_features[:, np.newaxis] - test_batch
            distances = np.sum(np.abs(diff) ** p, axis=2) ** (1 / p)
        elif distance_metric == "clark":
            diff = np.abs(train_features[:, np.newaxis] - test_batch)
            sum_values = train_features[:, np.newaxis] + test_batch
            distances = np.sum(diff / (sum_values + 1e-10), axis=2)

        # Find k nearest neighbors for each test instance in batch
        for j in range(len(test_batch)):
            nearest_indices = np.argpartition(distances[:, j], actual_k)[:actual_k]
            # Convert numpy arrays to lists for the voting functions
            k_nearest_labels = train_labels[nearest_indices].tolist()
            k_nearest_distances = distances[:, j][nearest_indices].tolist()

            if voting_method == "majority":
                prediction = majority_vote(k_nearest_labels)
            elif voting_method == "inverse_distance":
                prediction = inverse_distance_weighted_vote(
                    k_nearest_labels, k_nearest_distances, p
                )
            elif voting_method == "sheppard":
                prediction = sheppard_vote(k_nearest_labels, k_nearest_distances)

            predictions.append(prediction)

    return np.array(predictions)
