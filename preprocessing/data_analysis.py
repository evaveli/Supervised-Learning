import pandas as pd


def get_top_correlations(corr_matrix, exclude_vars, n=5):
    """
    Returns the top n most correlated variables for each variable in the correlation matrix.

    Parameters:
        corr_matrix (pd.DataFrame): DataFrame containing correlation values.
        exclude_vars (list): List of variables to exclude from correlation analysis.
        n (int): Number of top correlations to return per variable. Default is 5.

    Returns:
        dict: Dictionary mapping each variable to a list of its top n correlated variables.
    """
    top_correlations = {}
    for column in corr_matrix.columns:
        if column not in exclude_vars:
            correlations = corr_matrix[column].abs().sort_values(ascending=False)
            correlations = correlations[
                (correlations.index != column)
                & (~correlations.index.isin(exclude_vars))  
            ]
            top_n = correlations.nlargest(n).index.tolist()
            top_correlations[column] = top_n
    return top_correlations


def frequency(df):
    """
    Returns a DataFrame with frequency counts of categorical variables in the input DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the categorical data.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Variable', 'Value', and 'Frequency',
        showing the count of each unique value for each categorical variable.
    """
    columns_to_process = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    if "origin" not in columns_to_process and "origin" in df.columns:
        columns_to_process.append("origin")

    frames = []
    for column in columns_to_process:
        series = df[column].astype(str)
        value_counts = series.value_counts(dropna=False).reset_index()
        value_counts.columns = ["Value", "Frequency"]
        value_counts["Variable"] = column
        frames.append(value_counts[["Variable", "Value", "Frequency"]])
    return pd.concat(frames, ignore_index=True)
