import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

from reporting.report import create_summary_statistics_report


def create_method_comparison_visualization(
    labels, accuracy_arrays, critical_difference, title, filename, report_dir
):
    """
    Create a compact heatmap showing pairwise comparisons between methods
    """
    """
    Create a heatmap or table showing pairwise comparisons between methods
    """

    # Create matrix for storing mean differences
    n_methods = len(labels)
    mean_diff_matrix = np.zeros((n_methods, n_methods))

    # Track if any significant differences exist
    significant_pairs = []
    statistic, p_value = stats.friedmanchisquare(*accuracy_arrays)

    # Fill matrix with pairwise comparisons and check significance
    for i in range(n_methods):
        for j in range(n_methods):
            if i != j:
                diff = np.abs(accuracy_arrays[i] - accuracy_arrays[j])
                mean_diff_matrix[i, j] = np.mean(diff)
                if np.any(diff > critical_difference):
                    significant_pairs.append(f"{labels[i]} vs {labels[j]}")

    # Create DataFrame
    mean_df = pd.DataFrame(mean_diff_matrix, index=labels, columns=labels)

    # Create plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(mean_df, annot=True, cmap="YlOrRd", fmt=".4f")
    plt.title(f"Mean Differences between {title}")

    # Add significance information as text below the heatmap
    if significant_pairs:
        plt.figtext(
            0.1,
            0.02,
            f"Significant differences found between:\n{', '.join(significant_pairs)}",
            fontsize=8,
            wrap=True,
        )

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

    # Create a summary table
    summary_table = pd.DataFrame(
        {
            "Method": labels,
            "Mean Accuracy": [np.mean(arr) for arr in accuracy_arrays],
            "Std Accuracy": [np.std(arr) for arr in accuracy_arrays],
            "Min Accuracy": [np.min(arr) for arr in accuracy_arrays],
            "Max Accuracy": [np.max(arr) for arr in accuracy_arrays],
            "Configurations": [len(arr) for arr in accuracy_arrays],
        }
    )

    # Create summary statistics report with all required parameters
    stats_filename = f"{title.lower().replace(' ', '_')}_summary_statistics.pdf"
    create_summary_statistics_report(
        summary_table=summary_table,
        title=f"Summary Statistics for {title}",
        p_value=p_value,
        critical_difference=critical_difference,
        filename=os.path.join(report_dir, stats_filename),
    )


def analyze_knn_method(knn_results, method_key, methods_list, method_name, report_dir):
    """
    Generic analysis function for KNN methods
    """
    # Group results by method into numpy arrays
    grouped_results = {method: [] for method in methods_list}

    # Group accuracies
    for result in knn_results:
        method = str(result[method_key]) if method_key == "k" else result[method_key]
        if method in methods_list:
            grouped_results[method].append(result["mean_accuracy"])

    # Convert to numpy arrays
    accuracy_arrays = []
    labels = []

    for method in methods_list:
        if grouped_results[method]:
            accuracy_arrays.append(np.array(grouped_results[method]))
            labels.append(method)

    accuracy_arrays = np.array(accuracy_arrays)

    # Perform Friedman test
    statistic, p_value = stats.friedmanchisquare(*accuracy_arrays)

    print(f"\nFriedman Test Results for {method_name}:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"\nSignificant differences found between {method_name}!")

        k = len(labels)
        n = len(accuracy_arrays[0])
        q_alpha = 2.89
        critical_difference = q_alpha * np.sqrt((k * (k + 1)) / (6 * n))

        print(f"\nNemenyi Test Results:")
        print(f"Critical difference: {critical_difference:.4f}")

        # Pass method_name and create specific filename
        create_method_comparison_visualization(
            labels,
            accuracy_arrays,
            critical_difference,
            title=method_name,
            filename=os.path.join(
                report_dir, f"{method_name.lower().replace(' ', '_')}_comparison.pdf"
            ),
            report_dir=report_dir,
        )

    return grouped_results


# Simplified main analysis functions
def analyze_voting_schemes_knn(knn_results, report_dir):
    return analyze_knn_method(
        knn_results,
        method_key="voting_method",
        methods_list=["majority", "inverse_distance", "sheppard"],
        method_name="Voting Schemes",
        report_dir=report_dir,
    )


def analyze_svm_kernels(svm_results, report_dir):
    """
    Analyze performance differences between SVM kernel types
    """
    return analyze_knn_method(
        svm_results,
        method_key="kernel",
        methods_list=["linear", "poly", "rbf", "sigmoid"],
        method_name="SVM Kernels",
        report_dir=report_dir,
    )


def analyze_svm_c_values(svm_results, report_dir):
    """
    Analyze performance differences between SVM C values
    """
    return analyze_knn_method(
        svm_results,
        method_key="C",
        methods_list=[0.1, 1, 10, 100, 1000],  # C values as strings
        method_name="C Values",
        report_dir=report_dir,
    )


def analyze_svm_gamma_values(svm_results, report_dir):
    """
    Analyze performance differences between SVM gamma values
    """
    return analyze_knn_method(
        svm_results,
        method_key="gamma",
        methods_list=[1, 0.1, 0.01, 0.001, 0.0001],  # gamma values as strings
        method_name="Gamma Values",
        report_dir=report_dir,
    )


def analyze_k_values_knn(knn_results, report_dir):
    return analyze_knn_method(
        knn_results,
        method_key="k",
        methods_list=["1", "3", "5", "7"],  # k values as strings directly
        method_name="K Values",
        report_dir=report_dir,
    )


def analyze_reduction_techniques(
    wilson_results, gcnn_results, drop3_results, report_dir
):
    """
    Analyze the performance differences between reduction techniques
    """
    min_size = min(len(wilson_results), len(gcnn_results), len(drop3_results))

    # Take random samples of size min_size from each result set
    np.random.seed(42)  # for reproducibility
    wilson_sample = np.random.choice(wilson_results, size=min_size, replace=False)
    gcnn_sample = np.random.choice(gcnn_results, size=min_size, replace=False)
    drop3_sample = np.random.choice(drop3_results, size=min_size, replace=False)

    # Combine all results into one list with method identifier
    combined_results = []
    # Combine all results into one list with a method identifier
    combined_results = []

    # Add method identifier to each result
    for result in wilson_sample:
        result["reduction_method"] = "Wilson"
        combined_results.append(result)

    for result in gcnn_sample:
        result["reduction_method"] = "GCNN"
        combined_results.append(result)

    for result in drop3_sample:
        result["reduction_method"] = "DROP3"
        combined_results.append(result)

    return analyze_knn_method(
        combined_results,
        method_key="reduction_method",
        methods_list=["Wilson", "GCNN", "DROP3"],
        method_name="Reduction Techniques",
        report_dir=report_dir,
    )


def analyze_weighting_methods_knn(knn_results, report_dir):
    return analyze_knn_method(
        knn_results,
        method_key="w_method",
        methods_list=["equal", "information_gain", "reliefF"],
        method_name="Feature Weighting Methods",
        report_dir=report_dir,
    )


def get_top_configurations(results, algorithm, n_top=3):
    """
    Retrieves the top N configurations for a specified algorithm based on performance metrics.

    Parameters:
        results (List[Dict[str, Any]]): Model performance results.
        algorithm (str): The algorithm name (e.g., "SVM", "GCNN").
        n_top (int, optional): Number of top configurations to retrieve. Defaults to 3.

    Returns:
        List[Dict[str, Any]]: Top N configurations with standardized keys and metrics.
    """
    top_performers = sorted(
        results,
        key=lambda x: (
            x["mean_precision"],
            x["mean_recall"],
            x["mean_accuracy"],
            -x["std_accuracy"],
            x["mean_f1"],
            -x["mean_time"],
        ),
        reverse=True,
    )[:n_top]

    standardized_results = []
    for idx, result in enumerate(top_performers):
        if algorithm == "SVM":
            config = {
                "C": result["C"],
                "gamma": result["gamma"],
                "kernel": result["kernel"],
            }
            prefix = "SVM_top_"
        elif algorithm == "SVM_Wilson":
            config = {
                "C": result["C"],
                "gamma": result["gamma"],
                "kernel": result["kernel"],
                "wilson_k": result["wilson_k"],
                "wilson_threshold": result["wilson_threshold"],
            }
            prefix = "SVM_Wilson_top_"
        elif algorithm == "GCNN":
            config = {
                "alpha": result["alpha"],
                "k": result["k"],
                "p": result["p"],
                "V": result["voting_method"],
                "D": result["distance_method"],
                "W": result["w_method"],
            }
            prefix = "Reduction_GCNN_KNN_top_"

        elif algorithm == "DROP3":
            config = {
                "k": result["k"],
                "p": result["p"],
                "V": result["voting_method"],
                "D": result["distance_method"],
                "W": result["w_method"],
            }
            prefix = "Reduction_DROP3_KNN_top_"
        elif algorithm == "WILSON":
            config = {
                "k": result["k"],
                "threshold": result["threshold"],  # Added threshold
                "p": result["p"],
                "V": result["voting_method"],
                "D": result["distance_method"],
                "W": result["w_method"],
            }
            prefix = "Reduction_WilsonTh_KNN_top_"
        else:  # KNN
            config = {
                "k": result["k"],
                "p": result["p"],
                "V": result["voting_method"],
                "D": result["distance_method"],
                "W": result["w_method"],
            }
            prefix = "KNN_top_"

        standardized_results.append(
            {
                "method": f"{prefix}{idx+1}",
                "configuration": config,
                "accuracy": result["mean_accuracy"],
                "precision": result["mean_precision"],
                "recall": result["mean_recall"],
                "f1": result["mean_f1"],
                "std_accuracy": result["std_accuracy"],
                "mean_time": result["mean_time"],
            }
        )

    return standardized_results


def run_friedman_test(methods_results):
    """
    Performs a Friedman test on multiple model results to evaluate performance differences.

    Parameters:
        methods_results (List[Dict[str, Any]]):
            A list of dictionaries, each containing performance metrics for a model.
            Expected keys per dictionary:
                - 'method' (str)
                - 'precision' (float)
                - 'recall' (float)
                - 'accuracy' (float)
                - 'std_accuracy' (float)
                - 'f1' (float)
                - 'mean_time' (float)

    Returns:
        Dict[str, Any]:
            A dictionary with the following keys:
                - 'ranks' (Dict[str, float]): Average ranks of each method.
                - 'friedman_statistic' (float): Calculated Friedman test statistic.
                - 'critical_difference' (float): Critical difference threshold.
                - 'sorted_results' (List[Dict[str, Any]]): Models sorted by performance.
    """
    methods = [result["method"] for result in methods_results]
    metrics = [
        {
            "precision": result["precision"],
            "recall": result["recall"],
            "accuracy": result["accuracy"],
            "std_accuracy": result["std_accuracy"],  # changed back to 'std'
            "f1": result["f1"],
            "mean_time": result["mean_time"],  # changed back to 'time'
        }
        for result in methods_results
    ]

    # Create sorted pairs but keep reference to original results
    method_to_result = {result["method"]: result for result in methods_results}
    sorted_pairs = sorted(
        zip(methods, metrics),
        key=lambda x: (
            x[1]["precision"],
            x[1]["recall"],
            x[1]["accuracy"],
            -x[1]["std_accuracy"],
            x[1]["f1"],
            -x[1]["mean_time"],
        ),
        reverse=True,
    )

    # Create sorted results list using original full result dictionaries
    sorted_results = [method_to_result[method] for method, _ in sorted_pairs]

    ranks = {method: rank for rank, (method, _) in enumerate(sorted_pairs, 1)}

    k = len(methods)
    rank_sum_squared = sum(rank**2 for rank in ranks.values())
    chi_squared = (12 * 1) / (k * (k + 1)) * (rank_sum_squared - k * (k + 1) ** 2 / 4)
    ff = chi_squared / (k - 1)

    return {
        "ranks": ranks,
        "friedman_statistic": ff,
        "critical_difference": 2.89 * np.sqrt((k * (k + 1)) / (6 * 1)),
        "sorted_results": sorted_results,
    }
