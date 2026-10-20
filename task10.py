
import numpy as np

from decision_tree import DecisionTreeID3


def main():
    X = np.array([
        ["Sunny", "Hot"],
        ["Sunny", "Cool"],
        ["Rainy", "Hot"],
        ["Rainy", "Cool"],
        ["Sunny", "Hot"],
        ["Rainy", "Cool"],
    ], dtype=object)

    y = np.array(["No", "No", "Yes", "Yes", "No", "Yes"])

    tree = DecisionTreeID3(max_depth=4)
    tree.fit(X, y, feature_names=["Weather", "Temperature"])

    predictions = tree.predict(X)
    accuracy = np.mean(predictions == y)

    print("=" * 50)
    print("CUSTOM DECISION TREE TEST")
    print("=" * 50)

    print("\nDecision tree:")
    tree.print_tree()

    print("\nActual labels     :", y.tolist())
    print("Predicted labels :", predictions.tolist())
    print(f"Training accuracy: {accuracy:.2%}")
    print("Depth limit      :", tree.max_depth)

    print("\nDecision tree verification:",
          "PASS" if accuracy == 1.0 else "CHECK RESULTS")

    print("=" * 50)


if __name__ == "__main__":
    main()
