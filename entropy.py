
import math
from collections import Counter

import numpy as np


def entropy(labels):
    """Calculate Shannon entropy for a collection of class labels."""
    labels = np.asarray(labels)

    if labels.size == 0:
        return 0.0

    counts = Counter(labels.tolist())
    total = labels.size

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count > 0
    )


def information_gain(labels, feature_values):
    """Calculate information gain from splitting on a categorical feature."""
    labels = np.asarray(labels)
    feature_values = np.asarray(feature_values)

    if labels.size != feature_values.size:
        raise ValueError("Labels and feature values must have equal lengths.")

    if labels.size == 0:
        return 0.0

    parent_entropy = entropy(labels)
    weighted_child_entropy = 0.0

    for value in np.unique(feature_values):
        # Boolean mask identifies rows belonging to this feature value.
        mask = feature_values == value
        child_labels = labels[mask]

        weighted_child_entropy += (
            child_labels.size / labels.size
        ) * entropy(child_labels)

    return parent_entropy - weighted_child_entropy
