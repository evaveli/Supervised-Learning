import pandas as pd
import time
import os

from utils import load_datasets, get_process_time, create_directory

from preprocessing.preprocessing import (
    process_folds,
    changecoding,
    rename_class_column,
    identify_binary_variables,
    identify_categorical_variables,
    scale_folds,
)
from reporting.stats_friedman import (
    run_friedman_test,
    get_top_configurations,
    analyze_voting_schemes_knn,
    analyze_weighting_methods_knn,
    analyze_k_values_knn,
    analyze_svm_kernels,
    analyze_svm_c_values,
    analyze_svm_gamma_values,
    analyze_reduction_techniques,
)
from classifiers.knn import cross_validate_kNN_weighted_parallel
from classifiers.svm import cross_validate_svm
from reduction_techniques.wilson_th import evaluate_reduced_wilsonth_knn
from reduction_techniques.gcnn import evaluate_reduced_gcnn_knn_parallel
from reduction_techniques.drop_3 import evaluate_reduced_drop3_knn_parallel
from reporting.report import (
    aggregate_results_by_x,
    create_performance_report,
    create_nemenyi_report,
    knn_create_reduction_comparison_report,
    svm_create_reduction_comparison_report,
    create_class_distribution_report,
    create_class_distribution_report,
    create_hyperparameter_cd_diagram,
)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "csv_results")
    report_dir = os.path.join(script_dir, "reports")

    dataset_name = (
        input("Enter the name of the dataset ['hepatitis', 'pen-based']: ")
        or "hepatitis"
    )

    directory_path = os.path.join(script_dir, "datasetsCBR", dataset_name)

    # track time
    start_time = time.time()

    # Load datasets
    train_dfs, test_dfs, train_and_test_dfs = load_datasets(
        directory_path, dataset_name
    )
    print(
        f"Loaded {len(train_dfs)} training datasets and {len(test_dfs)} test datasets"
    )

    # Change coding of datasets
    train_dfs = [changecoding(df) for df in train_dfs]
    test_dfs = [changecoding(df) for df in test_dfs]

    # Rename the last variable to class if it iis not called class/Class

    train_dfs = [rename_class_column(df) for df in train_dfs]
    test_dfs = [rename_class_column(df) for df in test_dfs]

    train_and_test_dfs = [changecoding(df) for df in train_and_test_dfs]
    train_and_test_dfs = [rename_class_column(df) for df in train_and_test_dfs]
    print("Changed coding of datasets")

    # Combine all training data to identify variables
    all_train = pd.concat(train_dfs, ignore_index=True)
    binary_vars = identify_binary_variables(all_train)
    categorical_variables = identify_categorical_variables(all_train, binary_vars)
    print("These are the categorical variables:", categorical_variables)
    continuous_vars = list(
        set(all_train.columns) - set(binary_vars) - set(categorical_variables)
    )
    print(
        f"Identified {len(binary_vars)} binary variables and {len(continuous_vars)} continuous variables and {len(categorical_variables)} "
    )

    # Process each fold
    train_folds, test_folds = process_folds(
        train_dfs, test_dfs, binary_vars, continuous_vars, categorical_variables
    )

    # After imputation, let us check the remaining missing values
    for i in range(10):
        missing_after = train_folds[i].isnull().sum() + test_folds[i].isnull().sum()
        value = missing_after[missing_after > 0]
        if value.empty:
            print(f"All fine! No missing values found in train and test for fold {i}")
        else:
            print(f"Missing values found in fold {i}:\n{value}")

    create_directory(report_dir, "")

    create_class_distribution_report(
        train_and_test_dfs[0], os.path.join(report_dir, "class_distribution.pdf")
    )

    print("Processed folds")
    print("Scaling folds...")
    train_folds, test_folds = scale_folds(
        train_folds,
        test_folds,
    )
    print("Scaled folds")

    train_folds = [df.to_numpy() for df in train_folds]
    test_folds = [df.to_numpy() for df in test_folds]

    # preprocessing time
    preprocessing_time = get_process_time("Preprocessing", start_time)

    # Creating the API for the kNN
    # add potential methods for voting and distance

    voting_methods = ["majority", "inverse_distance", "sheppard"]
    distance_methods = ["minkowski", "clark"]

    # dictionaries to abbreviate the results
    voting_method_abbr = {
        "majority": "MAJR",
        "inverse_distance": "INVD",
        "sheppard": "SHEP",
    }

    distance_method_abbr = {"minkowski": "MIN", "clark": "CLK"}

    k_values = [1, 3, 5, 7]  # Neighbors
    p_values = [
        1,
        2,
        3,
    ]  # 1 for Manhattan, 2 for Euclidean, 3 for cubic distance (for the Minkowski- Clark is fixed 2 )
    feature_weighting_methods = ["equal", "information_gain", "reliefF"]

    knn_results = cross_validate_kNN_weighted_parallel(
        train_folds,
        test_folds,
        distance_methods,
        k_values,
        p_values,
        voting_methods,
        feature_weighting_methods,
        output_dir=output_dir,
    )

    aggregate_results_by_x(knn_results, "voting_method", output_dir)
    aggregate_results_by_x(knn_results, "w_method", output_dir)

    # Create voting_sceme related summary tables figures.
    voting_analysis = analyze_voting_schemes_knn(knn_results, report_dir)

    # Create weighting_method related summary tables figures.
    weighting_method_analysis = analyze_weighting_methods_knn(knn_results, report_dir)

    # Create k value related summary tables figures.
    K_values_analysis = analyze_k_values_knn(knn_results, report_dir)

    knn_time = get_process_time("KNN took", preprocessing_time)
    # Create CD plots for Weighting methods
    create_hyperparameter_cd_diagram(
        knn_results,
        method_key="w_method",
        methods_list=["equal", "information_gain", "reliefF"],
        method_name="Feature Weighting Methods",
        report_dir=report_dir,
    )

    # Create CD plots for K values
    create_hyperparameter_cd_diagram(
        knn_results,
        method_key="k",  # This should match the key in your results dictionary
        methods_list=["1", "3", "5", "7"],  # Convert to strings
        method_name="K Values",
        report_dir=report_dir,
    )

    # Create CD plots for voting schemes
    create_hyperparameter_cd_diagram(
        knn_results,
        method_key="voting_method",  # This matches the key in your results dictionary
        methods_list=["majority", "inverse_distance", "sheppard"],
        method_name="Voting Schemes",
        report_dir=report_dir,
    )

    # Get standardized/top 3 KNN results using the get_top_configurations_knn function:
    best_knn = get_top_configurations(knn_results, algorithm="KNN", n_top=3)

    # Wilson Threshold reduction :
    wilson_start_time = time.time()
    wilsonth_results, wilsonth_reduced_folds_wilson = evaluate_reduced_wilsonth_knn(
        train_folds,
        test_folds,
        distance_methods=["minkowski", "clark"],
        k_values=[1, 3, 5, 7],
        p_values=[1, 2, 3],
        voting_methods=["majority", "inverse_distance", "sheppard"],
        weighting_methods=feature_weighting_methods,
        output_dir=output_dir,
        thresholds=[0.4, 0.5, 0.6],
    )

    best_wilson = get_top_configurations(wilsonth_results, algorithm="WILSON", n_top=3)
    wilson_total_time = time.time() - wilson_start_time
    # GCNN reduction

    gcnn_start_time = time.time()
    GCNN_results, GCNN_reduced_folds = evaluate_reduced_gcnn_knn_parallel(
        train_folds,
        test_folds,
        distance_methods=["minkowski", "clark"],
        k_values=[1, 3, 5, 7],
        p_values=[1, 2, 3],
        voting_methods=["majority", "inverse_distance", "sheppard"],
        weighting_methods=feature_weighting_methods,
        output_dir=output_dir,
        alpha_values=[0.85, 0.90, 0.95],
    )
    best_GCNN = get_top_configurations(GCNN_results, algorithm="GCNN", n_top=3)
    gcnn_total_time = time.time() - gcnn_start_time

    #     # Drop 3 reduction
    drop3_start_time = time.time()
    DROP3_results, DROP3_reduced_folds = evaluate_reduced_drop3_knn_parallel(
        train_folds,
        test_folds,
        distance_methods=["minkowski", "clark"],
        k_values=[1, 3, 5, 7],
        p_values=[1, 2, 3],
        voting_methods=["majority", "inverse_distance", "sheppard"],
        weighting_methods=feature_weighting_methods,
        output_dir=output_dir,
    )
    best_DROP3 = get_top_configurations(DROP3_results, algorithm="DROP3", n_top=3)
    drop3_total_time = time.time() - drop3_start_time

    # Compare Reduction algorithms
    reduction_analysis = analyze_reduction_techniques(
        wilsonth_results, GCNN_results, DROP3_results, report_dir
    )
    # SVM part

    SVM_result = cross_validate_svm(
        train_folds,
        test_folds,
        output_dir,
        c_values=[0.1, 1, 10, 100, 1000],
        gamma_values=[1, 0.1, 0.01, 0.001, 0.0001],
        kernels=["linear", "poly", "rbf", "sigmoid"],
    )

    # create CD plot for:
    # For SVM kernels
    create_hyperparameter_cd_diagram(
        SVM_result,
        method_key="kernel",
        methods_list=["linear", "poly", "rbf", "sigmoid"],
        method_name="SVM Kernels",
        report_dir=report_dir,
    )

    # For C values
    create_hyperparameter_cd_diagram(
        SVM_result,
        method_key="C",
        methods_list=["0.1", "1", "10", "100", "1000"],  # as strings
        method_name="SVM C Values",
        report_dir=report_dir,
    )

    # For gamma values
    create_hyperparameter_cd_diagram(
        SVM_result,
        method_key="gamma",
        methods_list=["1", "0.1", "0.01", "0.001", "0.0001"],  # as strings
        method_name="SVM Gamma Values",
        report_dir=report_dir,
    )

    aggregate_results_by_x(SVM_result, "C", output_dir)
    aggregate_results_by_x(SVM_result, "gamma", output_dir)
    aggregate_results_by_x(SVM_result, "kernel", output_dir)

    best_SVM = get_top_configurations(SVM_result, algorithm="SVM", n_top=3)

    # Compare kernels using Friedman test
    kernel_analysis = analyze_svm_kernels(SVM_result, report_dir)
    # Compare C values using Friedman test
    c_analysis = analyze_svm_c_values(SVM_result, report_dir)
    # Compare gamma parameters using Friedman test
    gamma_analysis = analyze_svm_gamma_values(SVM_result, report_dir)

    # Utilize the dataset which is provided by Wilson TH algorithm

    best_wilson_config = best_wilson[0][
        "configuration"
    ]  # First configuration from top 3
    k_best = best_wilson_config["k"]
    threshold_best = best_wilson_config["threshold"]
    best_reduced_folds = wilsonth_reduced_folds_wilson[(k_best, threshold_best)]

    SVM_result_Wilson = cross_validate_svm(
        best_reduced_folds,
        test_folds,
        output_dir,
        c_values=[0.1, 1, 10, 100, 1000],
        gamma_values=[1, 0.1, 0.01, 0.001, 0.0001],
        kernels=["linear", "poly", "rbf", "sigmoid"],
    )

    best_SVM_Wilson = get_top_configurations(
        SVM_result_Wilson, algorithm="SVM", n_top=3
    )

    # Update each result with Wilson information using shorter parameter names
    for result in best_SVM_Wilson:
        result["configuration"].update(
            {
                "reduction_method": "Wilson",
                "wil_k": k_best,
                "wil_th": threshold_best,
                "wil_p": best_wilson_config["p"],
                "wil_v": best_wilson_config["V"],
                "wil_d": best_wilson_config["D"],
                "wil_w": best_wilson_config["W"],
            }
        )
        # Update method name to reflect it's using Wilson reduction
        result["method"] = "Wilson_" + result["method"]

        # Get best GCNN configuration
    best_GCNN_config = best_GCNN[0]["configuration"]  # First configuration from top 3
    k_best = best_GCNN_config["k"]
    alpha_best = best_GCNN_config["alpha"]
    best_reduced_folds = GCNN_reduced_folds[alpha_best]  # Use alpha as the key

    # Run SVM on GCNN-reduced data
    SVM_result_GCNN = cross_validate_svm(
        best_reduced_folds,
        test_folds,
        output_dir,
        c_values=[0.1, 1, 10, 100, 1000],
        gamma_values=[1, 0.1, 0.01, 0.001, 0.0001],
        kernels=["linear", "poly", "rbf", "sigmoid"],
    )

    # Get best SVM configurations
    best_SVM_GCNN = get_top_configurations(SVM_result_GCNN, algorithm="SVM", n_top=3)

    # Extract GCNN parameters from best configuration
    gcnn_k_best = best_GCNN_config["k"]
    gcnn_p_best = best_GCNN_config["p"]
    gcnn_voting_method_best = best_GCNN_config["V"]
    gcnn_weighting_method_best = best_GCNN_config["W"]
    gcnn_distance_method_best = best_GCNN_config["D"]
    gcnn_alpha_best = best_GCNN_config["alpha"]

    # # Update each SVM result with GCNN information
    for result in best_SVM_GCNN:  # Note: changed from best_gcnn_config to best_SVM_GCNN
        result["configuration"].update(
            {
                "reduction_method": "GCNN",
                "gcnn_k": gcnn_k_best,
                "gcnn_p": gcnn_p_best,
                "gcnn_v": gcnn_voting_method_best,
                "gcnn_d": gcnn_distance_method_best,
                "gcnn_w": gcnn_weighting_method_best,
                "gcnn_a": gcnn_alpha_best,
            }
        )
        # Update method name to reflect it's using GCNN reduction
        result["method"] = "GCNN_" + result["method"]

    best_DROP3_config = best_DROP3[0]["configuration"]
    k_best = best_DROP3_config["k"]
    best_reduced_folds = DROP3_reduced_folds[k_best]
    # Run SVM on DROP3-reduced data
    SVM_result_DROP3 = cross_validate_svm(
        best_reduced_folds,
        test_folds,
        output_dir,
        c_values=[0.1, 1, 10, 100, 1000],
        gamma_values=[1, 0.1, 0.01, 0.001, 0.0001],
        kernels=["linear", "poly", "rbf", "sigmoid"],
    )

    # Get best SVM configurations
    best_SVM_DROP3 = get_top_configurations(SVM_result_DROP3, algorithm="SVM", n_top=3)

    # Extract DROP3 parameters from best configuration
    drop3_k_best = best_DROP3_config["k"]
    drop3_p_best = best_DROP3_config["p"]
    drop3_voting_method_best = best_DROP3_config["V"]
    drop3_weighting_method_best = best_DROP3_config["W"]
    drop3_distance_method_best = best_DROP3_config["D"]

    # Update each SVM result with DROP3 information
    for result in best_SVM_DROP3:
        result["configuration"].update(
            {
                "reduction_method": "DROP3",
                "drop3_k": drop3_k_best,
                "drop3_p": drop3_p_best,
                "drop3_v": drop3_voting_method_best,
                "drop3_d": drop3_distance_method_best,
                "drop3_w": drop3_weighting_method_best,
            }
        )
        # Update method name to reflect it's using DROP3 reduction
        result["method"] = "DROP3_" + result["method"]

    # Combine results

    all_results = (
        best_knn
        + best_wilson
        + best_SVM
        + best_GCNN
        + best_DROP3
        + best_SVM_Wilson
        + best_SVM_GCNN
        + best_SVM_DROP3
    )

    # # Run Friedman test
    friedman_results = run_friedman_test(all_results)

    knn_create_reduction_comparison_report(
        best_knn,
        best_wilson,
        best_GCNN,
        best_DROP3,
        train_folds,
        wilsonth_reduced_folds_wilson,
        GCNN_reduced_folds,
        DROP3_reduced_folds,
        wilson_total_time,
        gcnn_total_time,
        drop3_total_time,
        filename=os.path.join(report_dir, "reduction_comparison_report.pdf"),
    )

    svm_create_reduction_comparison_report(
        best_SVM,
        best_SVM_Wilson,
        best_SVM_GCNN,
        best_SVM_DROP3,
        train_folds,
        wilsonth_reduced_folds_wilson,
        GCNN_reduced_folds,
        DROP3_reduced_folds,
        wilson_total_time,
        gcnn_total_time,
        drop3_total_time,
        filename=os.path.join(report_dir, "reduction_comparison_report_svm.pdf"),
    )

    # # Create report with our pdf generator function.
    create_performance_report(
        friedman_results["sorted_results"],
        filename=os.path.join(report_dir, "model_performance_report.pdf"),
    )
    # # Create Nemenyi summary table with our function.
    create_nemenyi_report(
        friedman_results,
        filename=os.path.join(report_dir, "nemenyi_comparison_report.pdf"),
    )
