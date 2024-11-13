import numpy as np


# Function for calc minkowsie distance
def minkowski_distance(point1, point2, p, feature_weights):
    """
    Calculates the Minkowski distance between two points with optional feature weights.

    Parameters:
        point1 (array): First data point.
        point2 (array): Second data point.
        p (int or float): Order of the distance metric.
            - p=1: Manhattan distance
            - p=2: Euclidean distance
            - p=3: Cubic distance
        feature_weights (array): Weights for each feature.

    Returns:
        float: The Minkowski distance between point1 and point2.
    """
    if feature_weights is None:
        feature_weights = np.ones_like(point1)
    diff = np.abs(point1 - point2)
    diff_p = diff**p
    weighted_diff_p = feature_weights * diff_p
    return np.sum(weighted_diff_p) ** (1 / p)


def clark_distance(point1, point2, feature_weights=None):
    """
    Calculates the Clark distance between two points with optional feature weights.

    Parameters:
        point1 (array-like): First data point.
        point2 (array-like): Second data point.
        feature_weights (array-like): Weights for each feature.

    Returns:
        float: The Clark distance between point1 and point2.
    """
    if feature_weights is None:
        feature_weights = np.ones_like(point1)

    numerator = np.abs(point1 - point2)
    denominator = np.abs(point1) + np.abs(point2)

    # Handle division by zero to do this we replace zeros in denominator with small epsilon
    epsilon = np.finfo(float).eps
    denominator = np.where(denominator == 0, epsilon, denominator)

    fraction = numerator / denominator
    squared_fraction = fraction**2

    # Apply feature weights
    weighted_squared_fraction = feature_weights * squared_fraction

    # Sum and square root
    sum_weighted_squared_fraction = np.sum(weighted_squared_fraction)
    distance = np.sqrt(sum_weighted_squared_fraction)

    return distance
