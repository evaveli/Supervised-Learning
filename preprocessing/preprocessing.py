import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from preprocessing.data_analysis import frequency, get_top_correlations
from preprocessing.data_imputation import (
    logistic_regression_imputation,
    simple_imputation,
    simple_imputation_categorical,
)


def process_folds(train_dfs, test_dfs, binary_vars, continuous_vars, categorical_vars):
    """
    Processes multiple cross-validation folds by recoding, imputing, and encoding variables.

    Parameters:
        train_dfs (list of pd.DataFrame): List of training DataFrames for each fold.
        test_dfs (list of pd.DataFrame): List of testing DataFrames for each fold.
        binary_vars (list of str): Names of binary variables to process.
        continuous_vars (list of str): Names of continuous variables to impute.
        categorical_vars (list of str): Names of categorical variables to impute and encode.

    Returns:
        tuple:
            - recoded_train_folds (list of pd.DataFrame): Processed training DataFrames.
            - recoded_test_folds (list of pd.DataFrame): Processed testing DataFrames.
    """
    recoded_train_folds = []
    recoded_test_folds = []
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    for i in range(10):
        # Recode datasets
        combined_df = pd.concat((train_dfs[i], test_dfs[i]), axis=0, ignore_index=True)
        freq_df = frequency(combined_df)
        train_df_recoded = recode_dataset(train_dfs[i], freq_df)
        test_df_recoded = recode_dataset(test_dfs[i], freq_df)

        # Compute correlation matrix
        corr_matrix = train_df_recoded.corr()
        top_correlations = get_top_correlations(
            corr_matrix,
            exclude_vars=["Class"],
            n=5,
        )

        print("Imputation for fold: " + str(i))

        # Impute missing binary values using logistic regression
        train_df_imputed, test_df_imputed = logistic_regression_imputation(
            train_df_recoded, test_df_recoded, top_correlations, binary_vars
        )
        if len(continuous_vars) > 0:
            # Impute missing continuous values using multiple imputation
            train_df_imputed, test_df_imputed = simple_imputation(
                train_df_imputed, test_df_imputed, continuous_vars
            )
        if len(categorical_vars) > 0:
            train_df_imputed, test_df_imputed = simple_imputation_categorical(
                train_df_imputed, test_df_imputed, categorical_vars
            )
        if categorical_vars:
            print(f"Fold {i}: Applying One-Hot Encoding to categorical variables.")
            if not train_df_imputed[categorical_vars].empty:
                # Fit OHE on training data
                ohe.fit(train_df_imputed[categorical_vars])

                # Transform training data
                ohe_train = ohe.transform(train_df_imputed[categorical_vars])
                ohe_train_df = pd.DataFrame(
                    ohe_train,
                    columns=ohe.get_feature_names_out(categorical_vars),
                    index=train_df_imputed.index,
                )
                train_df_imputed = train_df_imputed.drop(columns=categorical_vars)
                train_df_imputed = pd.concat([train_df_imputed, ohe_train_df], axis=1)

                # Transform test data
                ohe_test = ohe.transform(test_df_imputed[categorical_vars])
                ohe_test_df = pd.DataFrame(
                    ohe_test,
                    columns=ohe.get_feature_names_out(categorical_vars),
                    index=test_df_imputed.index,
                )
                test_df_imputed = test_df_imputed.drop(columns=categorical_vars)
                test_df_imputed = pd.concat([test_df_imputed, ohe_test_df], axis=1)
            else:
                print(f"Fold {i}: No categorical variables to encode after imputation.")
        else:
            print(
                f"Fold {i}: No categorical variables provided. Skipping One-Hot Encoding."
            )

        # Add class variable back as the last column

        # Store class variable separately
        train_class = train_df_imputed["Class"].copy()
        test_class = test_df_imputed["Class"].copy()

        # Remove class temporarily for processing we add back as a last column.

        train_df_imputed = train_df_imputed.drop("Class", axis=1)
        test_df_imputed = test_df_imputed.drop("Class", axis=1)
        train_df_imputed["Class"] = train_class.values
        test_df_imputed["Class"] = test_class.values

        # Append the processed folds to lists
        recoded_train_folds.append(train_df_imputed)
        recoded_test_folds.append(test_df_imputed)

    return recoded_train_folds, recoded_test_folds


def scale_folds(recoded_train_folds, recoded_test_folds):
    """
    Scales continuous features using MinMaxScaler and ensures all features are numeric.

    Parameters:
        recoded_train_folds (list of pd.DataFrame): List of processed training DataFrames for each fold.
        recoded_test_folds (list of pd.DataFrame): List of processed testing DataFrames for each fold.

    Returns:
        tuple:
            - scaled_train_folds (list of pd.DataFrame): List of scaled training DataFrames.
            - scaled_test_folds (list of pd.DataFrame): List of scaled testing DataFrames.
    """
    scaled_train_folds = []
    scaled_test_folds = []
    for train_df, test_df in zip(recoded_train_folds, recoded_test_folds):

        features_to_scale = [col for col in train_df.columns if col != "Class"]
        scaler = MinMaxScaler()
        scaler.fit(train_df[features_to_scale])

        train_df[features_to_scale] = scaler.transform(train_df[features_to_scale])
        test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])

        for column in train_df.columns:
            # Try to convert to numeric, set errors='coerce' to handle non-numeric strings
            # Create a numpy array at the end, for faster knn - calculation
            train_df[column] = pd.to_numeric(train_df[column], errors="coerce")
            test_df[column] = pd.to_numeric(test_df[column], errors="coerce")

        # train_df = train_df.to_numpy()
        # test_df = test_df.to_numpy()
        scaled_train_folds.append(train_df)
        scaled_test_folds.append(test_df)

    return scaled_train_folds, scaled_test_folds


def identify_binary_variables(df):
    """
    Identifies binary variables in a DataFrame based on unique non-missing values.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        list: A list of column names that are binary variables.
    """
    binary_vars = []
    for col in df.columns:
        unique_values = (
            df[col].dropna().astype(str).str.replace("^b", "", regex=True).unique()
        )
        # Exclude missing value indicators
        unique_values = [v for v in unique_values if v not in ["?", "nan", "-1"]]
        if len(unique_values) == 2:
            binary_vars.append(col)
    return binary_vars


def identify_categorical_variables(df, binary_vars):
    """
    Identifies categorical variables in a DataFrame, excluding class variable and binary vars.

    Parameters:
        df (pd.DataFrame): DataFrame to analyze for categorical variables.
        binary_vars (list of str): List of binary variable names to exclude.

    Returns:
        list of str: List of identified categorical variable names.
    """
    categorical_vars = []

    for col in df.columns:
        # Skip the class column
        if col.lower() == "class":
            continue

        unique_values = df[col].nunique()
        print(
            f"Column: {col}, Type: {df[col].dtype}, Unique Values: {unique_values}, "
            f"Binary: {col in binary_vars}, Class: {col.lower() == 'class'}"
        )

        # Check if it's 'object' dtype or has a low number of unique values (e.g., <10)
        if (df[col].dtype == "object" or unique_values < 10) and col not in binary_vars:
            categorical_vars.append(col)

    return categorical_vars


def changecoding(df):
    """
    Converts object and categorical columns to string, decoding bytes if necessary.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The processed DataFrame with updated column types.
    """
    for col in df.columns:
        if df[col].dtype == object:
            # Check if it's bytes
            if df[col].apply(lambda x: isinstance(x, bytes)).any():
                df[col] = df[col].str.decode("utf-8", errors="ignore")
            # If it's not bytes, it might be a former categorical column
            else:
                df[col] = df[col].astype(str)
        elif "category" in str(df[col].dtype):
            df[col] = df[col].astype(str)
    return df


def rename_class_column(data):
    """
    Renames the last column of a DataFrame to 'Class' for consistency.

    This function ensures that the target variable in the dataset is consistently named 'Class',
    facilitating downstream processing and analysis.

    Parameters:
        data (pd.DataFrame): The DataFrame whose last column will be renamed to 'Class'.

    Returns:
        pd.DataFrame: The DataFrame with the last column renamed to 'Class'.
    """
    if isinstance(data, pd.DataFrame):
        columns = list(data.columns)
        if columns[-1].lower() != "class":
            columns[-1] = "Class"  # Always use 'Class' with capital C
            data.columns = columns
        else:
            # Even if it's 'class', standardize to 'Class'
            columns[-1] = "Class"
            data.columns = columns
        return data


def recode_dataset(df, freq_df):
    """
    Recodes categorical variables to numerical codes based on frequency.

    Missing values are encoded as -1.

    Parameters:
        df (pd.DataFrame): The DataFrame to recode.
        freq_df (pd.DataFrame): Frequency DataFrame with columns ['Variable', 'Value', 'Frequency'].

    Returns:
        pd.DataFrame: The recoded DataFrame with categorical variables transformed to numerical codes.
    """
    columns_to_process = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for column in columns_to_process:
        column_freq = freq_df[freq_df["Variable"] == column].sort_values(
            "Frequency", ascending=False
        )
        recode_map = {}
        code = 0
        for _, row in column_freq.iterrows():
            if row["Value"] == "?":
                recode_map[row["Value"]] = -1
            else:
                recode_map[row["Value"]] = code
                code += 1
        df[column] = df[column].map(recode_map)
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df
