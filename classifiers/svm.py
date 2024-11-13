import csv
import os
import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import create_directory


def cross_validate_svm(
    train_folds,
    test_folds,
    output_dir,
    c_values=[0.1, 1, 10, 100, 1000],
    gamma_values=[1, 0.1, 0.01, 0.001, 0.0001],
    kernels=["linear", "poly", "rbf", "sigmoid"],
):
    """
    Performs cross-validation for SVM models across different hyperparameters and logs results to a CSV.

    Args:
        train_folds (List[np.ndarray]): Training data folds with features and labels.
        test_folds (List[np.ndarray]): Testing data folds with features and labels.
        c_values (List[float]): List of C values for SVM. Defaults to [0.1, 1, 10, 100, 1000].
        gamma_values (List[float]): List of gamma values for SVM. Defaults to [1, 0.1, 0.01, 0.001, 0.0001].
        kernels (List[str]): List of kernel types for SVM. Defaults to ["linear", "poly", "rbf", "sigmoid"].
        output_csv (str): CSV file to append results. Defaults to "svm_evaluation_results.csv".

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing hyperparameters and performance metrics.
    """
    results = []
    file_exists, output_file = create_directory(
        output_dir, "svm_evaluation_results.csv"
    )

    with open(output_file, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow(
                [
                    "C",
                    "Gamma",
                    "Kernel",
                    "Mean_Accuracy",
                    "Std_Accuracy",
                    "Mean_Precision",
                    "Mean_Recall",
                    "Mean_F1",
                    "Mean_Time_s",
                ]
            )
        for c in c_values:
            for gamma in gamma_values:
                for kernel in kernels:
                    fold_accuracies = []
                    fold_times = []
                    fold_precisions = []
                    fold_recalls = []
                    fold_f1s = []

                    for train_fold, test_fold in zip(train_folds, test_folds):
                        X_train = train_fold[:, :-1].astype(float)
                        y_train = train_fold[:, -1]
                        X_test = test_fold[:, :-1].astype(float)
                        y_test = test_fold[:, -1]

                        sv_classifier = svm.SVC(C=c, gamma=gamma, kernel=kernel)

                        start_time = time.time()
                        sv_classifier.fit(X_train, y_train)
                        predictions = sv_classifier.predict(X_test)
                        end_time = time.time()

                        fold_time = end_time - start_time
                        accuracy = np.mean(predictions == y_test)
                        precision = precision_score(
                            y_test, predictions, average="weighted", zero_division=0
                        )
                        recall = recall_score(
                            y_test, predictions, average="weighted", zero_division=0
                        )
                        f1 = f1_score(
                            y_test, predictions, average="weighted", zero_division=0
                        )

                        fold_accuracies.append(accuracy)
                        fold_times.append(fold_time)
                        fold_precisions.append(precision)
                        fold_recalls.append(recall)
                        fold_f1s.append(f1)

                    mean_accuracy = np.mean(fold_accuracies)
                    std_accuracy = np.std(fold_accuracies)
                    mean_time = np.mean(fold_times)
                    mean_precision = np.mean(fold_precisions)
                    mean_recall = np.mean(fold_recalls)
                    mean_f1 = np.mean(fold_f1s)

                    results.append(
                        {
                            "C": c,
                            "gamma": gamma,
                            "kernel": kernel,
                            "mean_accuracy": mean_accuracy,
                            "std_accuracy": std_accuracy,
                            "mean_time": mean_time,
                            "mean_precision": mean_precision,
                            "mean_recall": mean_recall,
                            "mean_f1": mean_f1,
                        }
                    )

                    print(
                        f"C={c}, gamma={gamma}, kernel={kernel}, "
                        f"Mean Acc={mean_accuracy:.4f} (±{std_accuracy:.4f}), "
                        f"Mean Precision={mean_precision:.4f}, "
                        f"Mean Recall={mean_recall:.4f}, "
                        f"Mean F1={mean_f1:.4f}, "
                        f"Time={mean_time:.4f}s"
                    )

                    csv_writer.writerow(
                        [
                            c,
                            gamma,
                            kernel,
                            mean_accuracy,
                            std_accuracy,
                            mean_precision,
                            mean_recall,
                            mean_f1,
                            mean_time,
                        ]
                    )

        return results


def get_average_results(results):
    """
    Computes average performance metrics for each unique hyperparameter configuration.

    Args:
        results (List[pd.DataFrame]):
            A list of DataFrames, each containing a 'params' column (dict) and numeric performance metrics.

    Returns:
        pd.DataFrame:
            A DataFrame indexed by 'params_str' with mean values of numeric metrics.
    """
    df_concat = pd.concat(results, ignore_index=True)
    df_concat["params_str"] = df_concat["params"].astype(str)

    # Exclude non-numeric columns from the aggregation
    numeric_cols = df_concat.select_dtypes(include=[float, int]).columns
    df_means = df_concat.groupby("params_str")[numeric_cols].mean()

    return df_means
