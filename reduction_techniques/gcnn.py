import csv
import os
import numpy as np
import time
from classifiers.knn import kNNAlgorithm_batched
from metrics.feature_weighting import determine_feature_weights
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import create_directory


def initialize_subset(train_data):
    """
    Initializes a subset by selecting the first instance of each binary class.

    Parameters:
        train_data (np.ndarray): Training data array where the last column represents class labels (0 or 1).

    Returns:
        list: A list containing the indices of the first occurrence of class 0 and class 1.
    """
    # Get labels
    labels = train_data[:, -1]

    # Find first instance of class 0
    class0_idx = np.where(labels == 0)[0][0]
    # Find first instance of class 1
    class1_idx = np.where(labels == 1)[0][0]

    initial_subset = [class0_idx, class1_idx]

    return initial_subset


def get_remaining_indices(train_data, initial_subset):
    """
    Retrieves indices from the training data excluding those in the initial subset.

    Parameters:
        train_data (np.ndarray): The training data array.
        initial_subset (list): List of indices to exclude.

    Returns:
        np.ndarray: Array of remaining indices not in the initial subset.
    """

    # Get all possible indices
    all_indices = np.arange(len(train_data))

    # Remove the indices that are in initial subset
    remaining_indices = np.setdiff1d(all_indices, initial_subset)

    return remaining_indices


#####################################################################################


def should_add_instance_vectorized(
    candidate_idx, train_data, selected_indices, alpha=0.95
):
    """
    Vectorized version of should_add_instance
    """
    # Get candidate data
    candidate_features = train_data[candidate_idx, :-1].astype(float)
    candidate_label = train_data[candidate_idx, -1]

    if len(selected_indices) == 0:
        return True

    # Get selected subset data - vectorized
    S_features = train_data[selected_indices, :-1].astype(float)
    S_labels = train_data[selected_indices, -1]

    # Compute all distances at once
    distances = np.sqrt(np.sum((S_features - candidate_features) ** 2, axis=1))

    # Split distances by friend/enemy
    friend_mask = S_labels == candidate_label
    enemy_mask = ~friend_mask

    if not enemy_mask.any() or not friend_mask.any():
        return True

    nearest_friend_dist = np.min(distances[friend_mask])
    nearest_enemy_dist = np.min(distances[enemy_mask])

    should_add = nearest_enemy_dist < (alpha * nearest_friend_dist)

    if should_add:
        print(
            f"Adding instance {candidate_idx}: enemy_dist={nearest_enemy_dist:.3f}, friend_dist={nearest_friend_dist:.3f}"
        )

    return should_add


def process_fold(args):
    """
    Separate function to process a single fold for parallel execution
    """
    fold, fold_idx, alpha = args
    print(f"\nProcessing fold {fold_idx+1}")
    print(f"Original size: {len(fold)}")
    reduced_fold = gcnn_reduction_optimized(fold, alpha)
    reduction_ratio = (len(fold) - len(reduced_fold)) / len(fold) * 100
    print(f"Reduction: {reduction_ratio:.2f}%")
    return reduced_fold


def gcnn_reduction_optimized(train_data, alpha=0.95, batch_size=100):
    """
    Optimized GCNN reduction process with batch processing
    """
    # Initialize with one instance per class
    selected_indices = initialize_subset(train_data)

    # Get remaining instances to process
    remaining = get_remaining_indices(train_data, selected_indices)

    # Process instances in batches
    for i in range(0, len(remaining), batch_size):
        batch_indices = remaining[i : i + batch_size]

        # Process each instance in the batch
        for idx in batch_indices:
            if should_add_instance_vectorized(idx, train_data, selected_indices, alpha):
                selected_indices.append(idx)

    print(f"Final reduced set size: {len(selected_indices)}")
    return train_data[selected_indices]


def apply_gcnn_to_folds_parallel(train_folds, alpha=0.95):
    """
    Apply GCNN to folds in parallel
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    n_workers = max(1, multiprocessing.cpu_count() - 1)

    # Prepare arguments for parallel processing
    fold_args = [(fold, idx, alpha) for idx, fold in enumerate(train_folds)]

    # Process folds in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        reduced_folds = list(executor.map(process_fold, fold_args))  # Removed lambda

    return reduced_folds


# Modify the evaluation function to use parallel processing


def evaluate_reduced_gcnn_knn_parallel(
    train_folds,
    test_folds,
    distance_methods,
    k_values,
    p_values,
    voting_methods,
    weighting_methods,
    output_dir,
    alpha_values=[0.85, 0.90, 0.95],
):
    """
    Parallelized version of GCNN evaluation with batched KNN
    """
    results = []
    reduced_folds = {}
    reduction_times = {}

    print("Reducing datasets with GCNN...")
    for alpha in alpha_values:
        key = alpha
        start_time = time.time()
        reduced_folds[key] = apply_gcnn_to_folds_parallel(train_folds, alpha)
        end_time = time.time()
        reduction_times[key] = [end_time - start_time] * len(train_folds)

        avg_reduction_time = np.mean(reduction_times[key])
        avg_reduction_ratio = np.mean(
            [
                len(reduced) / len(train)
                for reduced, train in zip(reduced_folds[key], train_folds)
            ]
        )
        print(
            f"alpha={alpha}: Avg reduction time = {avg_reduction_time:.4f} sec, "
            f"Avg reduction ratio = {avg_reduction_ratio:.2%}"
        )

    print("\nPerforming grid search on reduced datasets...")
    file_exists, output_file = create_directory(output_dir, "gcnn_metrics.csv")

    with open(output_file, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write headers if the file is newly created
        if not file_exists:
            csv_writer.writerow(
                [
                    "Feature_Weighting",
                    "Alpha",
                    "KNN_k",
                    "p",
                    "Distance_Method",
                    "Voting_Method",
                    "Mean_Accuracy",
                    "Std_Accuracy",
                    "Mean_Precision",
                    "Mean_Recall",
                    "Mean_F1",
                    "Mean_Time_s",
                ]
            )
        # Use batched KNN for evaluation
        for w_method in weighting_methods:
            for alpha in alpha_values:
                key = alpha
                for k in k_values:
                    for p in p_values:
                        for voting_method in voting_methods:
                            for distance_method in distance_methods:
                                if distance_method == "clark" and p != 2:
                                    continue

                                fold_accuracies = []
                                fold_times = []
                                fold_precisions = []
                                fold_recalls = []
                                fold_f1s = []

                                for reduced_train, test_fold in zip(
                                    reduced_folds[key], test_folds
                                ):
                                    X_train = reduced_train[:, :-1].astype(float)
                                    y_train = reduced_train[:, -1]
                                    y_test = test_fold[:, -1]

                                    feature_weights = determine_feature_weights(
                                        X_train, y_train, method=w_method
                                    )

                                    start_time = time.time()
                                    # Use batched KNN instead of regular KNN
                                    predictions = kNNAlgorithm_batched(
                                        reduced_train,
                                        test_fold,
                                        k=k,
                                        distance_metric=distance_method,
                                        p=p,
                                        voting_method=voting_method,
                                        feature_weights=feature_weights,
                                        batch_size=100,  # Adjust batch size as needed
                                    )
                                    end_time = time.time()

                                    fold_time = end_time - start_time
                                    accuracy = np.mean(predictions == y_test)
                                    precision = precision_score(
                                        y_test,
                                        predictions,
                                        average="weighted",
                                        zero_division=0,
                                    )
                                    recall = recall_score(
                                        y_test,
                                        predictions,
                                        average="weighted",
                                        zero_division=0,
                                    )
                                    f1 = f1_score(
                                        y_test,
                                        predictions,
                                        average="weighted",
                                        zero_division=0,
                                    )

                                    fold_accuracies.append(accuracy)
                                    fold_times.append(fold_time)
                                    fold_precisions.append(precision)
                                    fold_recalls.append(recall)
                                    fold_f1s.append(f1)

                                # Calculate means and store results
                                mean_accuracy = np.mean(fold_accuracies)
                                std_accuracy = np.std(fold_accuracies)
                                mean_time = np.mean(fold_times)
                                mean_precision = np.mean(fold_precisions)
                                mean_recall = np.mean(fold_recalls)
                                mean_f1 = np.mean(fold_f1s)

                                print(
                                    f"Feature Weighting: {w_method}, "
                                    f"GCNN(alpha={alpha}) + "
                                    f"KNN(k={k}, p={p}, dis={distance_method}, v={voting_method}): "
                                    f"Mean Acc={mean_accuracy:.4f} (±{std_accuracy:.4f}), "
                                    f"Mean Precision={mean_precision:.4f}, "
                                    f"Mean Recall={mean_recall:.4f}, "
                                    f"Mean F1={mean_f1:.4f}, "
                                    f"Time={mean_time:.4f}s"
                                )

                                csv_writer.writerow(
                                    [
                                        w_method,
                                        alpha,
                                        k,
                                        p,
                                        distance_method,
                                        voting_method,
                                        mean_accuracy,
                                        std_accuracy,
                                        mean_precision,
                                        mean_recall,
                                        mean_f1,
                                        mean_time,
                                    ]
                                )

                                results.append(
                                    {
                                        "w_method": w_method,
                                        "alpha": alpha,
                                        "k": k,
                                        "p": p,
                                        "voting_method": voting_method,
                                        "distance_method": distance_method,
                                        "mean_accuracy": mean_accuracy,
                                        "std_accuracy": std_accuracy,
                                        "classifier_time": mean_time,
                                        "reduction_time": np.mean(reduction_times[key]),
                                        "mean_time": mean_time,
                                        "mean_precision": mean_precision,
                                        "mean_recall": mean_recall,
                                        "mean_f1": mean_f1,
                                    }
                                )

        return results, reduced_folds
