
import numpy as np


def create_bitmask(feature_values, target_value):
    """Create a Boolean mask for rows matching a target feature value."""
    values = np.asarray(feature_values)
    return values == target_value


def apply_bitmask(data, mask):
    """Select rows from a NumPy array using a Boolean mask."""
    data = np.asarray(data)
    mask = np.asarray(mask, dtype=bool)

    if data.shape[0] != mask.shape[0]:
        raise ValueError("Mask length must match the number of rows.")

    return data[mask]


def feature_masks(X, feature_index):
    """Generate masks for each unique value in a feature column."""
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    if not 0 <= feature_index < X.shape[1]:
        raise IndexError("Feature index is out of range.")

    values = X[:, feature_index]

    return {
        value: values == value
        for value in np.unique(values)
    }


def split_by_mask(X, y, mask):
    """Split feature rows and target labels using the same Boolean mask."""
    X = np.asarray(X)
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if X.shape[0] != y.shape[0] or X.shape[0] != mask.shape[0]:
        raise ValueError("X, y, and mask must contain the same row count.")

    return X[mask], y[mask]


if __name__ == "__main__":
    X = np.array([
        ["Sunny", "Hot"],
        ["Sunny", "Cool"],
        ["Rainy", "Hot"],
        ["Rainy", "Cool"],
        ["Sunny", "Hot"],
    ])

    y = np.array(["No", "No", "Yes", "Yes", "No"])

    mask = create_bitmask(X[:, 0], "Sunny")
    selected_X, selected_y = split_by_mask(X, y, mask)

    print("=" * 45)
    print("NUMPY BITMASK ENGINE TEST")
    print("=" * 45)
    print("Mask:", mask)
    print("Selected rows:")
    print(selected_X)
    print("Selected labels:", selected_y.tolist())

    expected = np.array([True, True, False, False, True])
    print(
        "\nBitmask verification:",
        "PASS" if np.array_equal(mask, expected) else "FAIL",
    )
    print("=" * 45)
