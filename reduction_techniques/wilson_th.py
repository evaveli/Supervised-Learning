import csv
import os
import numpy as np
import time
from classifiers.knn import kNNAlgorithm_batched
from metrics.feature_weighting import determine_feature_weights
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import create_directory


def calculate_class_probabilities(instanceIDX, data, k):
    """
    Vectorized version of class probability calculation
    """
    train_labels = data[:, -1]

    # Vectorized distance calculation
    instance = data[instanceIDX, :-1].astype(float)
    diff = data[:, :-1].astype(float) - instance
    distances = np.sqrt(np.sum(diff * diff, axis=1))

    # Create mask for self-distance
    mask = np.ones(len(data), dtype=bool)
    mask[instanceIDX] = False

    # Get k nearest neighbors indices
    k_nearest_idx = np.argpartition(distances[mask], k)[:k]
    k_distances = distances[mask][k_nearest_idx]
    k_labels = train_labels[mask][k_nearest_idx]

    # Calculate probabilities
    unique_classes = np.unique(train_labels)
    class_probs = {}

    contributions = 1 / (1 + k_distances)
    for class_label in unique_classes:
        class_mask = k_labels == class_label
        class_probs[class_label] = np.sum(contributions[class_mask])

    # Normalize
    total_prob = sum(class_probs.values())
    if total_prob > 0:
        for class_label in class_probs:
            class_probs[class_label] /= total_prob

    return class_probs


def wilsonth_reduction(data, k, threshold):
    """
    Optimized WilsonTh implementation with pre-computed distances
    """
    # Pre-compute all pairwise distances
    print("Pre-computing distances...")
    features = data[:, :-1].astype(float)
    distances = np.zeros((len(data), len(data)))
    batch_size = 100  # Process in batches to manage memory

    for i in range(0, len(data), batch_size):
        batch_end = min(i + batch_size, len(data))
        diff = features[i:batch_end, np.newaxis, :] - features
        distances[i:batch_end] = np.sqrt(np.sum(diff * diff, axis=2))

    keep_indices = []
    removed_count = 0

    print("Processing instances...")
    for idx in range(len(data)):
        class_probs = calculate_class_probabilities(idx, data, k)
        instance_class = data[idx, -1]
        instance_class_prob = class_probs.get(instance_class, 0)

        if instance_class_prob > threshold:
            keep_indices.append(idx)
        else:
            removed_count += 1

        if idx % 100 == 0:  # Changed to report less frequently
            print(f"Processed {idx}/{len(data)} instances, removed: {removed_count}")

    reduced_data = data[keep_indices]
    print(f"Original size: {len(data)}, Reduced size: {len(reduced_data)}")
    print(f"Removed {removed_count} instances ({(removed_count/len(data))*100:.2f}%)")

    return reduced_data


def process_fold(args):
    train_fold, k, threshold = args
    return wilsonth_reduction(train_fold, k, threshold)


def evaluate_reduced_wilsonth_knn(
    train_folds,
    test_folds,
    distance_methods,
    k_values,
    p_values,
    voting_methods,
    weighting_methods,
    output_dir,
    thresholds=[0.3, 0.4, 0.5, 0.6],
):
    """
    Evaluate WilsonTh reduction with KNN classifier using all parameter combinations.
    First reduces all datasets using WilsonTh, then performs grid search on reduced datasets.
    """
    results = []
    reduced_folds = (
        {}
    )  # Dictionary to store reduced datasets for each k,threshold combo
    reduction_times = {}  # Dictionary to store reduction times

    file_exists, output_file = create_directory(output_dir, "wilson_th_metrics.csv")

    print("Reducing datasets with WilsonTh...")
    # Store reductions for each k and threshold combination
    for k in k_values:
        for threshold in thresholds:
            key = (k, threshold)  # Tuple key for the dictionaries
            reduced_folds[key] = []
            reduction_times[key] = []

            print(f"\nProcessing k={k}, threshold={threshold}")
            for train_fold in train_folds:
                start_time = time.time()
                reduced_train = wilsonth_reduction(train_fold, k=k, threshold=threshold)
                end_time = time.time()

                reduced_folds[key].append(reduced_train)
                reduction_times[key].append(end_time - start_time)

            avg_reduction_time = np.mean(reduction_times[key])
            avg_reduction_ratio = np.mean(
                [
                    len(reduced) / len(train)
                    for reduced, train in zip(reduced_folds[key], train_folds)
                ]
            )
            print(
                f"k={k}, th={threshold}: Avg reduction time = {avg_reduction_time:.4f} sec, "
                f"Avg reduction ratio = {avg_reduction_ratio:.2%}"
            )

    with open(output_file, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write headers if the file is newly created
        if not file_exists:
            csv_writer.writerow(
                [
                    "Feature_Weighting",
                    "WilsonTh_k",
                    "Threshold",
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
        print("\nPerforming grid search on reduced datasets...")
        # Now perform grid search using pre-reduced datasets
        for w_method in weighting_methods:
            for k in k_values:
                for threshold in thresholds:
                    key = (k, threshold)
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

                                    # Determine feature weights based on the chosen method
                                    feature_weights = determine_feature_weights(
                                        X_train, y_train, method=w_method
                                    )

                                    start_time = time.time()
                                    predictions = kNNAlgorithm_batched(
                                        reduced_train,
                                        test_fold,
                                        k=k,
                                        distance_metric=distance_method,
                                        p=p,
                                        voting_method=voting_method,
                                        feature_weights=feature_weights,
                                        batch_size=100,  # You can adjust batch size based on your memory
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
                                total_time = (
                                    mean_time  # + np.mean(reduction_times[key])
                                )
                                mean_precision = np.mean(fold_precisions)
                                mean_recall = np.mean(fold_recalls)
                                mean_f1 = np.mean(fold_f1s)

                                print(
                                    f"Feature Weighting: {w_method},+"
                                    f"WilsonTh(k={k},th={threshold}) + "
                                    f"KNN(k={k}, p={p}, dis={distance_method}, v={voting_method}): "
                                    f"Mean Acc. = {mean_accuracy:.4f}, "
                                    f"Mean Acc={mean_accuracy:.4f} (±{std_accuracy:.4f}), "
                                    f"Mean Precision={mean_precision:.4f}, "
                                    f"Mean Recall={mean_recall:.4f}, "
                                    f"Total Time = {total_time:.4f} sec"
                                )

                                csv_writer.writerow(
                                    [
                                        w_method,
                                        k,
                                        threshold,
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
                                        "threshold": threshold,
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
