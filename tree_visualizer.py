
from graphviz import Digraph

from decision_tree import DecisionTreeID3, TreeNode


def visualize_tree(tree, output_filename="decision_tree"):
    """Render a trained decision tree as a Graphviz diagram."""
    if not isinstance(tree, DecisionTreeID3) or tree.root is None:
        raise ValueError("Provide a trained DecisionTreeID3 instance.")

    dot = Digraph(
        name="DecisionTree",
        format="png",
        comment="Custom ID3 Decision Tree",
    )

    dot.attr(
        "graph",
        rankdir="TB",
        bgcolor="white",
        label="Custom Decision Tree (ID3)",
        labelloc="t",
    )
    dot.attr("node", fontname="Arial", fontsize="10")
    dot.attr("edge", fontname="Arial", fontsize="9")

    node_counter = 0

    def add_node(node: TreeNode):
        nonlocal node_counter

        current_id = f"node_{node_counter}"
        node_counter += 1

        if node.is_leaf:
            dot.node(
                current_id,
                label=f"Predict: {node.prediction}",
                shape="box",
                style="rounded,filled",
                fillcolor="lightgreen",
            )
        else:
            dot.node(
                current_id,
                label=f"Feature: {node.feature_name}",
                shape="ellipse",
                style="filled",
                fillcolor="lightblue",
            )

        for value, child in node.children.items():
            child_id = add_node(child)
            dot.edge(current_id, child_id, label=str(value))

        return current_id

    add_node(tree.root)

    rendered_path = dot.render(
        filename=output_filename,
        cleanup=True,
        view=False,
    )

    print("Decision tree visualization saved to:", rendered_path)
    return rendered_path


if __name__ == "__main__":
    import numpy as np

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

    visualize_tree(tree)
