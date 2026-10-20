# Task 10: Custom Decision Tree Induction (ID3)

## Overview
This project implements a custom decision tree classifier using the ID3 algorithm. It calculates entropy and information gain, uses NumPy Boolean bitmasks to split data, and supports recursive tree construction with a configurable maximum depth.

## Features
- Entropy calculation using information theory
- Information gain for feature selection
- NumPy Boolean bitmask operations
- Recursive decision tree construction
- Parent-child node relationships
- Maximum depth and minimum split-size controls
- Prediction for new samples
- Graphviz decision-tree visualization

## Project Structure
- `entropy.py` — Entropy and information gain
- `bitmask_engine.py` — Boolean masks and data splitting
- `decision_tree.py` — ID3 classifier and tree nodes
- `tree_visualizer.py` — Graphviz visualization
- `task10.py` — Main program and verification
- `requirements.txt` — Python dependencies
- `decision_tree.png` — Generated tree visualization

## Requirements
- Python 3
- NumPy
- Python Graphviz package
- Graphviz system application

## Installation
Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install Graphviz separately and ensure its `bin` directory is added to your system PATH.

## Run the Project

```bash
python task10.py
python tree_visualizer.py
```

## Verification
The sample dataset is classified with 100% training accuracy. The main program prints the learned tree, actual labels, predicted labels, and verification result.

## Limitations
This implementation is intended for categorical features. Continuous numerical features require additional threshold-based splitting logic.
