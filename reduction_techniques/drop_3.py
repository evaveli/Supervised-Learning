import csv
import os
import numpy as np
import time
from classifiers.knn import kNNAlgorithm_batched
from metrics.feature_weighting import determine_feature_weights
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import create_directory


def get_majority_class(classes):
    """
    Helper function to get majority class from a list of classes
    """
    # Convert to integers, rounding first to avoid float precision issues
    classes = np.round(classes).astype(int)
    unique, counts = np.unique(classes, return_counts=True)
    return unique[np.argmax(counts)]


def find_k_nnb_vectorized(instanceIDX, data, k):
    """Vectorized version of k-nearest neighbor finding with safety checks"""
    train_features = data[:, :-1].astype(float)
    instance = train_features[instanceIDX]

    # Calculate actual k based on dataset size
    actual_k = min(k, len(data) - 1)  # -1 to exclude the instance itself

    if actual_k < 1:
        return np.array([], dtype=int)

    # Vectorized distance calculation
    distances = np.sqrt(np.sum((train_features - instance) ** 2, axis=1))

    # Mask out the instance itself
    distances[instanceIDX] = np.inf

    # Get k nearest indices
    nearest_indices = np.argpartition(distances, actual_k)[:actual_k]
    return nearest_indices


def noise_filtering_phase_vectorized(data, k):
    """Vectorized noise filtering"""
    keep_instances = []

    # Pre-calculate all k-nearest neighbors
    all_neighbors = []
    train_features = data[:, :-1].astype(float)

    for i in range(len(data)):
        # Vectorized distance calculation
        distances = np.sqrt(np.sum((train_features - train_features[i]) ** 2, axis=1))
        distances[i] = np.inf  # Exclude self
        nearest_indices = np.argpartition(distances, k)[:k]
        all_neighbors.append(nearest_indices)

    # Vectorized majority class check
    for i in range(len(data)):
        neighbors = all_neighbors[i]
        neighbor_classes = data[neighbors, -1]
        majority_class = np.bincount(neighbor_classes.astype(int)).argmax()
        current_class = int(round(data[i, -1]))

        if current_class == majority_class:
            keep_instances.append(i)

    return data[keep_instances]


def instance_reduction_phase_vectorized(data, k):
    """Vectorized instance reduction phase with safety checks"""
    if len(data) <= 1:
        return data

    # Pre-calculate all distances
    train_features = data[:, :-1].astype(float)
    train_labels = data[:, -1]

    # Calculate all pairwise distances
    n_samples = len(data)
    distances = np.zeros((n_samples, n_samples))

    # Process in batches to manage memory
    batch_size = 100
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        diff = train_features[i:end_idx, np.newaxis] - train_features
        distances[i:end_idx] = np.sqrt(np.sum(diff**2, axis=2))

    # Calculate nearest enemy distances
    enemy_distances = []
    for i in range(n_samples):
        enemy_mask = train_labels != train_labels[i]
        if np.any(enemy_mask):
            min_enemy_dist = np.min(distances[i][enemy_mask])
        else:
            min_enemy_dist = np.inf
        enemy_distances.append((i, min_enemy_dist))

    # Sort by enemy distance
    enemy_distances.sort(key=lambda x: x[1], reverse=True)
    keep_instances = list(range(len(data)))

    # Adjust k if necessary
    actual_k = min(k, len(data) - 1)
    if actual_k < 1:
        return data

    # Process instances
    for idx, _ in enemy_distances:
        if idx in keep_instances:
            if len(keep_instances) <= actual_k:
                break  # Stop if we can't reduce further

            current_data = data[keep_instances]
            current_idx = keep_instances.index(idx)

            # Find associates using vectorized operations
            current_distances = distances[idx][keep_instances]
            associates = np.where(
                current_distances <= np.partition(current_distances, actual_k)[actual_k]
            )[0]
            associates = associates[associates != current_idx]

            # Check classification without current instance
            can_remove = True
            temp_keep = keep_instances.copy()
            temp_keep.remove(idx)

            if len(temp_keep) < actual_k:
                can_remove = False
            else:
                temp_data = data[temp_keep]

                for associate in associates:
                    if associate >= len(keep_instances):
                        continue

                    orig_associate = keep_instances[associate]

                    # Get classifications with and without current instance
                    with_neighbors = find_k_nnb_vectorized(
                        associate, current_data, actual_k
                    )
                    if len(with_neighbors) == 0:
                        can_remove = False
                        break

                    with_class = np.bincount(
                        current_data[with_neighbors, -1].astype(int)
                    ).argmax()

                    new_associate_idx = temp_keep.index(orig_associate)
                    without_neighbors = find_k_nnb_vectorized(
                        new_associate_idx, temp_data, actual_k
                    )
                    if len(without_neighbors) == 0:
                        can_remove = False
                        break

                    without_class = np.bincount(
                        temp_data[without_neighbors, -1].astype(int)
                    ).argmax()

                    if with_class != without_class:
                        can_remove = False
                        break

            if can_remove:
                keep_instances.remove(idx)

    return data[keep_instances]


def DROP3_optimized(train_data, k=3):
    """DROP3 implementation"""
    print("Starting Phase 1: Noise Filtering...")
    filtered_data = noise_filtering_phase_vectorized(train_data, k)

    print("Starting Phase 2: Instance Reduction...")
    reduced_data = instance_reduction_phase_vectorized(filtered_data, k)

    return reduced_data


def process_fold(args):
    """
    Process a single fold with DROP3 reduction.
    Must be at module level for parallel processing.
    """
    fold, k = args
    start_time = time.time()
    reduced = DROP3_optimized(fold, k=k)
    end_time = time.time()
    return reduced, end_time - start_time


def evaluate_reduced_drop3_knn_parallel(
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
    Parallel evaluation with optimized DROP3 and batched KNN.

    Parameters:
        train_folds: list of training folds
        test_folds: list of test folds
        distance_methods: list of distance metrics to use
        k_values: list of k values to try
        p_values: list of p values for Minkowski distance
        voting_methods: list of voting methods
        weighting_methods: list of feature weighting methods
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    results = []
    reduced_folds = {}
    reduction_times = {}
    n_workers = max(1, multiprocessing.cpu_count() - 1)

    print("Reducing datasets with DROP3...")
    for k in k_values:
        key = k
        fold_args = [(fold, k) for fold in train_folds]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            fold_results = list(executor.map(process_fold, fold_args))

        reduced_folds[key] = [result[0] for result in fold_results]
        reduction_times[key] = [result[1] for result in fold_results]

        avg_reduction_time = np.mean(reduction_times[key])
        avg_reduction_ratio = np.mean(
            [
                len(reduced) / len(train)
                for reduced, train in zip(reduced_folds[key], train_folds)
            ]
        )
        print(
            f"k={k}: Avg reduction time = {avg_reduction_time:.4f} sec, "
            f"Avg reduction ratio = {avg_reduction_ratio:.2%}"
        )

    file_exists, output_file = create_directory(output_dir, "drop3_metrics.csv")

    print("\nPerforming grid search on reduced datasets...")

    with open(output_file, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write headers if the file is newly created
        if not file_exists:
            csv_writer.writerow(
                [
                    "Feature_Weighting",
                    "DROP3_k",
                    "KNN_k",
                    "p",
                    "Distance_Method",
                    "Voting_Method",
                    "Mean_Accuracy",
                    "Std_Accuracy",
                    "Mean_Precision",
                    "Mean_Recall",
                    "Mean_F1",
                    "Total_Time_s",
                ]
            )
        # Now perform grid search using pre-reduced datasets and batched KNN
        for w_method in weighting_methods:
            for k in k_values:
                key = k
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

                                # Determine feature weights
                                feature_weights = determine_feature_weights(
                                    X_train, y_train, method=w_method
                                )

                                start_time = time.time()
                                # Use batched KNN for predictions
                                predictions = kNNAlgorithm_batched(
                                    reduced_train,
                                    test_fold,
                                    k=k,
                                    distance_metric=distance_method,
                                    p=p,
                                    voting_method=voting_method,
                                    feature_weights=feature_weights,
                                    batch_size=100,  # Adjust batch size based on your memory
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

                            mean_accuracy = np.mean(fold_accuracies)
                            std_accuracy = np.std(fold_accuracies)
                            mean_time = np.mean(fold_times)
                            total_time = mean_time  # + np.mean(reduction_times[key])
                            mean_precision = np.mean(fold_precisions)
                            mean_recall = np.mean(fold_recalls)
                            mean_f1 = np.mean(fold_f1s)

                            print(
                                f"Feature Weighting: {w_method}, "
                                f"DROP3(k={k}) + "
                                f"KNN(k={k}, p={p}, dis={distance_method}, v={voting_method}): "
                                f"Mean Acc={mean_accuracy:.4f} (±{std_accuracy:.4f}), "
                                f"Mean Precision={mean_precision:.4f}, "
                                f"Mean Recall={mean_recall:.4f}, "
                                f"Mean F1={mean_f1:.4f}, "
                                f"Time={total_time:.4f}s"
                            )
                            csv_writer.writerow(
                                [
                                    w_method,
                                    k,
                                    k,
                                    p,
                                    distance_method,
                                    voting_method,
                                    mean_accuracy,
                                    std_accuracy,
                                    mean_precision,
                                    mean_recall,
                                    mean_f1,
                                    total_time,
                                ]
                            )

                            results.append(
                                {
                                    "w_method": w_method,
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
