import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer


def logistic_regression_imputation(train_df, test_df, top_correlations, binary_vars):
    """
    Imputes missing values for binary variables in train and test datasets using logistic regression.

    Parameters:
        train_df (pd.DataFrame): Training data with missing values.
        test_df (pd.DataFrame): Test data with missing values.
        top_correlations (dict): Dictionary of top correlated predictors for each variable.
        binary_vars (list): List of binary variables to impute.

    Returns:
        tuple: Updated train and test DataFrames with imputed values.
    """
    models = {}
    train_df = train_df.copy()
    test_df = test_df.copy()

    for var in binary_vars:
        if var not in top_correlations:
            continue

        print(f"Processing variable: {var}")

        # Missing masks
        missing_mask_train = train_df[var] == -1
        missing_mask_test = test_df[var] == -1

        # Check if there are missing values
        missing_count_train = missing_mask_train.sum()
        missing_count_test = missing_mask_test.sum()

        if missing_count_train == 0 and missing_count_test == 0:
            print(f"No missing values in {var} in both datasets. Skipping.")
            continue
        else:
            print(
                f"Missing values in {var} - Train: {missing_count_train}, Test: {missing_count_test}"
            )

        predictors = top_correlations[var]
        # Training data
        X_train = train_df[predictors]
        y_train = train_df[var]

        # Valid rows (non-missing in target and all predictors)
        valid_mask_train = ~missing_mask_train & X_train.notnull().all(axis=1)
        X_train_valid = X_train[valid_mask_train]
        y_train_valid = y_train[valid_mask_train]

        # Impute missing values in predictors
        imputer = SimpleImputer(strategy="mean")
        X_train_imputed = imputer.fit_transform(X_train_valid)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)

        # Train the model
        model = LogisticRegression(random_state=11)
        try:
            model.fit(X_train_scaled, y_train_valid)
            print(f"Model trained successfully for {var}.")
        except Exception as e:
            print(f"Error fitting model for {var}: {e}")
            continue

        # Function to impute missing values
        def impute_missing_values(df, missing_mask):
            X_missing = df.loc[missing_mask, predictors]
            X_missing_imputed = imputer.transform(X_missing)
            X_missing_scaled = scaler.transform(X_missing_imputed)
            y_pred = model.predict(X_missing_scaled)
            df.loc[missing_mask, var] = y_pred
            print(f"Imputed {len(y_pred)} values for {var}.")
            return df

        # Impute missing values in training set
        if missing_count_train > 0:
            train_df = impute_missing_values(train_df, missing_mask_train)

        # Impute missing values in test set
        if missing_count_test > 0:
            test_df = impute_missing_values(test_df, missing_mask_test)

        # Store the model and preprocessing objects
        models[var] = {
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "predictors": predictors,
        }

    return train_df, test_df


def multiple_imputation(train_df, test_df, top_correlations, continuous_vars):
    """
    Impute missing values in continuous variables using multiple imputation.
    Trains on train_df and applies the imputation on both train_df and test_df.

    Parameters:
    - train_df: Training DataFrame.
    - test_df: Testing DataFrame.
    - top_correlations: Dictionary of top correlated predictors for each variable.
    - continuous_vars: List of continuous variables to impute.

    Returns:
    - train_df: Training DataFrame with imputed values.
    - test_df: Testing DataFrame with imputed values.
    """
    imputers = {}
    train_df = train_df.copy()
    test_df = test_df.copy()

    for var in continuous_vars:
        if var not in top_correlations:
            continue

        print(f"Processing variable: {var}")

        # Missing masks
        missing_mask_train = train_df[var].isnull()
        missing_mask_test = test_df[var].isnull()

        # Check if there are missing values
        missing_count_train = missing_mask_train.sum()
        missing_count_test = missing_mask_test.sum()

        if missing_count_train == 0 and missing_count_test == 0:
            print(f"No missing values in {var} in both datasets. Skipping.")
            continue
        else:
            print(
                f"Missing values in {var} - Train: {missing_count_train}, Test: {missing_count_test}"
            )

        predictors = top_correlations[var]
        all_columns = predictors + [var]

        # Combine train and test data for imputation
        combined_df = pd.concat(
            [train_df[all_columns], test_df[all_columns]], ignore_index=True
        )

        # Initialize IterativeImputer
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=11,
            sample_posterior=True,
        )

        # Fit imputer on combined data
        imputer.fit(combined_df)

        # Impute missing values in combined data
        combined_imputed = imputer.transform(combined_df)

        # Split the combined data back into train and test
        imputed_combined_df = pd.DataFrame(combined_imputed, columns=all_columns)
        train_imputed = imputed_combined_df.iloc[: len(train_df)]
        test_imputed = imputed_combined_df.iloc[len(train_df) :]

        # Update the train and test dataframes
        train_df[all_columns] = train_imputed.values
        test_df[all_columns] = test_imputed.values

        # Store the imputer and predictors
        imputers[var] = {"imputer": imputer, "predictors": predictors}

        print(f"Imputed missing values for {var} in both datasets.")

    return train_df, test_df


def multiple_imputation_categorical(
    train_df, test_df, top_correlations, categorical_vars
):
    """
    Impute missing values in continuous variables using multiple imputation.
    Trains on train_df and applies the imputation on both train_df and test_df.

    Parameters:
    - train_df: Training DataFrame.
    - test_df: Testing DataFrame.
    - top_correlations: Dictionary of top correlated predictors for each variable.
    - categorical_vars: List of continuous variables to impute.

    Returns:
    - train_df: Training DataFrame with imputed values.
    - test_df: Testing DataFrame with imputed values.
    """
    imputers = {}
    train_df = train_df.copy()
    test_df = test_df.copy()

    for var in categorical_vars:
        if var not in top_correlations:
            continue

        print(f"Processing variable: {var}")

        # Missing masks
        missing_mask_train = train_df[var].isnull()
        missing_mask_test = test_df[var].isnull()

        # Check if there are missing values
        missing_count_train = missing_mask_train.sum()
        missing_count_test = missing_mask_test.sum()

        if missing_count_train == 0 and missing_count_test == 0:
            print(f"No missing values in {var} in both datasets. Skipping.")
            continue
        else:
            print(
                f"Missing values in {var} - Train: {missing_count_train}, Test: {missing_count_test}"
            )

        predictors = top_correlations[var]
        all_columns = predictors + [var]

        # Keep original dtypes, in-order to make sure after rounding the type is original
        original_dtypes = train_df[all_columns].dtypes.to_dict()

        # Combine train and test data for imputation
        combined_df = pd.concat(
            [train_df[all_columns], test_df[all_columns]], ignore_index=True
        )

        # Initialize IterativeImputer
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=11,
            sample_posterior=True,
        )

        # Fit imputer on combined data
        imputer.fit(combined_df)

        # Impute missing values in combined data
        combined_imputed = imputer.transform(combined_df)
        # round values
        combined_imputed[:, -1] = np.round(combined_imputed[:, -1])

        # Split the combined data back into train and test
        imputed_combined_df = pd.DataFrame(combined_imputed, columns=all_columns)
        train_imputed = imputed_combined_df.iloc[: len(train_df)]
        test_imputed = imputed_combined_df.iloc[len(train_df) :]

        for col, dtype in original_dtypes.items():
            train_imputed[col] = train_imputed[col].astype(dtype)
            test_imputed[col] = test_imputed[col].astype(dtype)

        # Update the train and test dataframes
        train_df[all_columns] = train_imputed.values
        test_df[all_columns] = test_imputed.values

        # Store the imputer and predictors
        imputers[var] = {"imputer": imputer, "predictors": predictors}

        print(f"Imputed missing values for {var} in both datasets.")

    return train_df, test_df


def simple_imputation(train_df, test_df, continuous_vars):
    """
    Performs mean imputation for specified continuous variables.

    Parameters:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Test dataset.
        continuous_vars (list): List of continuous variable names to impute.

    Returns:
        tuple: Updated training and test DataFrames with imputed values.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    for var in continuous_vars:
        print(f"Processing variable: {var}")

        # Missing masks
        missing_mask_train = train_df[var].isnull()
        missing_mask_test = test_df[var].isnull()

        missing_count_train = missing_mask_train.sum()
        missing_count_test = missing_mask_test.sum()

        if missing_count_train == 0 and missing_count_test == 0:
            print(f"No missing values in {var} in both datasets. Skipping.")
            continue
        else:
            print(
                f"Missing values in {var} - Train: {missing_count_train}, Test: {missing_count_test}"
            )

        # Simple mean imputation for continuous variables
        train_mean = train_df[var].mean()

        # Impute missing values
        train_df.loc[missing_mask_train, var] = train_mean
        test_df.loc[missing_mask_test, var] = train_mean

        print(f"Imputed missing values for {var} using mean value: {train_mean:.4f}")

    return train_df, test_df


def simple_imputation_categorical(train_df, test_df, categorical_vars):
    """
    Performs mode imputation for specified categorical variables.

    Parameters:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Test dataset.
        categorical_vars (list): List of categorical variable names to impute.

    Returns:
        tuple: Updated training and test DataFrames with imputed values.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    for var in categorical_vars:
        print(f"Processing variable: {var}")

        # Missing masks
        missing_mask_train = train_df[var].isnull()
        missing_mask_test = test_df[var].isnull()

        missing_count_train = missing_mask_train.sum()
        missing_count_test = missing_mask_test.sum()

        if missing_count_train == 0 and missing_count_test == 0:
            print(f"No missing values in {var} in both datasets. Skipping.")
            continue
        else:
            print(
                f"Missing values in {var} - Train: {missing_count_train}, Test: {missing_count_test}"
            )

        # Use mode (most frequent value) for categorical variables
        mode_value = train_df[var].mode()[0]  # Take first mode if there are multiple

        # Impute missing values
        train_df.loc[missing_mask_train, var] = mode_value
        test_df.loc[missing_mask_test, var] = mode_value

        print(f"Imputed missing values for {var} using mode value: {mode_value}")

    return train_df, test_df
