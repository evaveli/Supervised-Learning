import numpy as np
from skrebate import ReliefF
from sklearn.feature_selection import mutual_info_classif


def determine_feature_weights_information_gain(X, y):
    """
    Determines feature weights based on mutual information with the target variable.

    Parameters:
        X (array): Feature matrix.
        y (array): Target vector.

    Returns:
        np.ndarray: Array of feature weights normalized to sum to 1.
    """
    if len(np.unique(y)) < 2:
        print(
            "Warning: Single class detected in information gain. Using equal weights."
        )
        return np.ones(X.shape[1]) / X.shape[1]

    mi = mutual_info_classif(X, y, random_state=11)
    mi_sum = mi.sum()
    if mi_sum == 0:
        # All features have zero mutual information with the target
        # Assign equal weights to all features
        feature_weights = np.ones_like(mi) / len(mi)
    else:
        # Normalize the weights to sum to 1
        feature_weights = mi / mi_sum
    return feature_weights


def determine_feature_weights_reliefF(X, y):
    """
    Determines feature weights using the ReliefF algorithm based on feature importance.

    Parameters:
        X (array): Feature matrix.
        y (array): Target vector.

    Returns:
        np.ndarray: Array of normalized feature weights that sum to 1.
    """
    # Check for single class case first
    if len(np.unique(y)) < 2:
        print("Warning: Single class detected in ReliefF. Using equal weights.")
        return np.ones(X.shape[1]) / X.shape[1]

    relief = ReliefF(n_neighbors=100, n_features_to_select=X.shape[1])
    relief.fit(X, y)
    feature_weights = relief.feature_importances_
    # Ensure all weights are non-negative
    feature_weights = np.clip(feature_weights, a_min=0, a_max=None)
    feature_weights_sum = feature_weights.sum()
    # Handle division by 0 possibility
    if feature_weights_sum == 0:
        feature_weights = np.ones_like(feature_weights) / len(feature_weights)
    else:
        feature_weights = feature_weights / feature_weights_sum
    return feature_weights


def determine_feature_weights(X, y, method="equal"):
    """
    Determines feature weights based on the specified method.

    Parameters:
        X (array): Feature matrix.
        y (array): Target vector.
        method (str): Method to determine feature weights.
                      Options:
                        - "equal": Assigns equal weights to all features.
                        - "information_gain": Uses mutual information with the target.
                        - "reliefF": Uses the ReliefF algorithm for feature importance.

    Returns:
        np.ndarray: Normalized array of feature weights that sum to 1.
    """
    if method == "equal":
        feature_weights = np.ones(X.shape[1])
    elif method == "information_gain":
        feature_weights = determine_feature_weights_information_gain(X, y)
    elif method == "reliefF":
        feature_weights = determine_feature_weights_reliefF(X, y)
    else:
        raise ValueError("Unsupported feature weighting method.")
    feature_weights = feature_weights / feature_weights.sum()
    return feature_weights
