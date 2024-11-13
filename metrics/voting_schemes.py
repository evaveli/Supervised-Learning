import numpy as np


def majority_vote(labels):
    """
    Determines the most common label in a list of labels.

    Args:
        labels (List[Any]): A list of labels .

    Returns:
        Any: The label with the highest frequency. In case of a tie, one of the top labels is returned.
    """
    unique_labels = set(labels)
    max_count = 0
    max_label = None
    for label in unique_labels:
        count = labels.count(label)
        if count > max_count:
            max_count = count
            max_label = label
    return max_label


# Inverse distance - for this we use the distance to derive weights - these weights will have an impact in the vote.
def inverse_distance_weighted_vote(labels, distances, p=1):
    """
    Returns the label with the highest inverse-distance-weighted vote.

    Args:
        labels (List[Any]): List of labels.
        distances (List[float]): List of non-negative distances.
        p (float): Power parameter for weighting (default is 1).

    Returns:
        Optional[Any]: The label with the highest weighted vote, or None if inputs are empty.
    """
    unique_labels = set(labels)
    label_weights = {label: 0 for label in unique_labels}
    for label, distance in zip(labels, distances):
        if distance == 0:
            # Handle zero distance
            weight = float("inf")
        else:
            weight = 1 / (distance**p)
        label_weights[label] += weight
    weighted_vote_label = max(label_weights, key=label_weights.get)
    return weighted_vote_label


# Sheppards'- Work - like for the inverse method we use the distances to create weights, but at this time we use the exp (-X) function.
def sheppard_vote(labels, distances):
    """
    Determines the majority label using Sheppard's vote with exponential weighting.

    Args:
        labels (List[Any]): List of labels.
        distances (List[float]): List of distances for each instance.

    Returns:
        Any: The label with the highest weighted vote.
    """
    unique_labels = set(labels)
    label_weights = {label: 0 for label in unique_labels}

    # Normalize distances by subtracting the minimum distance
    min_dist = min(distances)
    normalized_distances = [d - min_dist for d in distances]

    for label, distance in zip(labels, normalized_distances):
        # Clip large negative exponents to prevent overflow
        safe_dist = min(distance, 700)  # exp(700) is near the maximum safe value
        label_weights[label] += np.exp(-safe_dist)

    return max(label_weights, key=label_weights.get)
