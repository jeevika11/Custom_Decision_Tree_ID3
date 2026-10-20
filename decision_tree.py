
import numpy as np

from entropy import information_gain


class TreeNode:
    """Represents a decision node or a leaf in the tree."""

    def __init__(
        self,
        prediction=None,
        feature_index=None,
        feature_name=None,
        parent=None,
        depth=0,
    ):
        self.prediction = prediction
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.parent = parent
        self.depth = depth
        self.children = {}

    @property
    def is_leaf(self):
        return len(self.children) == 0


class DecisionTreeID3:
    """Categorical decision tree using recursive information gain."""

    def __init__(self, max_depth=5, min_samples_split=2):
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = []

    @staticmethod
    def _majority_class(y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, y, feature_names=None):
        """Train the tree on categorical features and target labels."""
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if X.shape[0] == 0:
            raise ValueError("Training data cannot be empty.")

        if feature_names is None:
            self.feature_names = [
                f"feature_{i}" for i in range(X.shape[1])
            ]
        else:
            if len(feature_names) != X.shape[1]:
                raise ValueError("Feature names must match X columns.")
            self.feature_names = list(feature_names)

        self.root = self._build_tree(
            X, y, list(range(X.shape[1])), parent=None, depth=0
        )
        return self

    def _build_tree(self, X, y, available_features, parent, depth):
        majority = self._majority_class(y)

        node = TreeNode(
            prediction=majority,
            parent=parent,
            depth=depth,
        )

        # Stop when the node is pure or the depth/sample limit is reached.
        if (
            np.all(y == y[0])
            or depth >= self.max_depth
            or X.shape[0] < self.min_samples_split
            or not available_features
        ):
            return node

        best_feature = None
        best_gain = -1.0

        for feature_index in available_features:
            gain = information_gain(y, X[:, feature_index])

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index

        # No useful split remains.
        if best_feature is None or best_gain <= 1e-12:
            return node

        node.feature_index = best_feature
        node.feature_name = self.feature_names[best_feature]

        remaining_features = [
            index for index in available_features
            if index != best_feature
        ]

        feature_values = X[:, best_feature]

        for value in np.unique(feature_values):
            mask = feature_values == value

            child_X = X[mask]
            child_y = y[mask]

            if child_y.size == 0:
                continue

            child = self._build_tree(
                child_X,
                child_y,
                remaining_features,
                parent=node,
                depth=depth + 1,
            )

            node.children[value] = child

        return node

    def _predict_one(self, row):
        node = self.root

        while node is not None and not node.is_leaf:
            value = row[node.feature_index]
            child = node.children.get(value)

            # Unseen category: fall back to this node's majority class.
            if child is None:
                return node.prediction

            node = child

        return node.prediction if node is not None else None

    def predict(self, X):
        """Predict class labels for one or more rows."""
        if self.root is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X, dtype=object)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("Prediction data has an invalid shape.")

        return np.asarray(
            [self._predict_one(row) for row in X]
        )

    def print_tree(self, node=None, indent=""):
        """Print the learned tree in a readable hierarchy."""
        if self.root is None:
            print("Tree has not been trained.")
            return

        if node is None:
            node = self.root

        if node.is_leaf:
            print(f"{indent}Predict: {node.prediction}")
            return

        print(f"{indent}Feature: {node.feature_name}")

        for value, child in node.children.items():
            print(f"{indent}  [{value}]")
            self.print_tree(child, indent + "    ")
