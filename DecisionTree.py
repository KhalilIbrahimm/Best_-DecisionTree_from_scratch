import numpy as np
import pandas as pd
from collections import Counter
from Node import Node
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=5, impurity_measure=None, random_state=None):
        '''
        Initializes the decision tree with hyperparameters.

        Arguments:
        - min_samples_split: Minimum number of data points required to perform a split.
        - max_depth: Maximum depth of the decision tree.
        - impurity_measure: The impurity measure to use, either 'entropy' or 'gini'.
        - random_state: Seed for reproducibility of random numbers.
        '''

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.random_state = random_state
        self.root = None
        
    def calculate_entropy(self, y):
        '''
        Calculates entropy for a target variable y.

        Args:
        - y: The target variable for the dataset.

        Returns:
        - Entropy for the target variable y.
        '''

        # Calculate the entropy of a set of labels y
        unique_labels, label_counts = np.unique(y, return_counts=True)
        entropy = 0
        total_samples = len(y)

        for count in label_counts:
            probability = count / total_samples
            entropy += probability * np.log2(probability)

        return -entropy
    
    def information_gain(self, parent, left_child, right_child):
        '''
        Calculates information gain for a given split based on either entropy or Gini index.

        Args:
        - parent: The target variable for the parent node.
        - left_child: The target variable for the left child node.
        - right_child: The target variable for the right child node.

        Returns:
        - Information gain for the split point.
        '''

        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        if len(left_child) == 0 or len(right_child) == 0:
            return 0
        
        # Calculate the information gain for the current threshold
        if self.impurity_measure == "entropy":
            return self.calculate_entropy(parent) - (num_left * self.calculate_entropy(left_child) + num_right * self.calculate_entropy(right_child))
        else:
            return self.gini(parent) - (num_left * self.gini(left_child) + num_right * self.gini(right_child))

    def equil_feature(self, X):
        '''
        Here, I'm implementing a method to check this condition:
        If all data points have identical feature values: – return a leaf with the most common label.
        '''
        for column in X.columns:
            if (X[column].nunique() == 1):
                pass
            else:
                return False
        return True
    
    def _split_dataset(self, X, y):
        '''
        Splits the dataset into two based on the best split point for either entropy or Gini index.

        Args:
        - X: Feature values for the dataset.
        - y: The target variable for the dataset.

        Returns:
        - The dictionary containing information about the best split used in the _learn function to build the tree.
        '''

        best_info_gain = -1
        best_split = None

        num_features = X.shape[1]

        for feature_index in range(num_features):
            feature_values = X.iloc[:, feature_index]  # Get the values of the current feature
            # Find unique values in the feature
            unique_values = np.unique(feature_values)

            # Iterate through each unique threshold value in the feature
            for pivot_value in unique_values:
                
                left_dataset = X[feature_values <= pivot_value]
                right_dataset = X[feature_values > pivot_value]
               
                y_left = y.loc[left_dataset.index]
                y_right = y.loc[right_dataset.index]
                
                left_df =  pd.concat([left_dataset, y_left], axis=1)
                right_df = pd.concat([right_dataset, y_right], axis=1)

                # Calculate the information gain for the current attribute
                gain_value = self.information_gain(y, y_left, y_right)

                # Update the best gain and threshold if the current gain is better
                if gain_value > best_info_gain:
                    best_split = {
                        'best_feature': X.columns[feature_index],
                        'threshold': pivot_value,
                        'df_left': left_df, 
                        'df_right': right_df,
                        'gain': gain_value
                    }
                    best_info_gain = gain_value
        
        return best_split
        
    def _learn(self, X, y, depth=0):
        '''
        Builds the decision tree recursively.

        Args:
        - X: Feature values for the dataset.
        - y: The target variable for the dataset.
        - depth: Current depth in the decision tree.

        Returns:
        - The root node of the constructed decision tree.
    
        '''
        # Implementing the conditions as requested:

        # If all data points have the same label: – return a leaf with that label.
        if len(np.unique(y)) == 1:
            return Node(value=np.unique(y)[0])

        # If all data points have identical feature values: – return a leaf with the most common label. 
        if self.equil_feature(X):
            return Node(value=(np.unique(y)).max())
        
        # Split the dataset and continue recursively
        best_split = self._split_dataset(X, y)
        
        # Build the tree recursively
        left_subset = self._learn(best_split["df_left"].iloc[:, :-1], best_split["df_left"].iloc[:, -1], depth=depth+1)
        right_subset = self._learn(best_split["df_right"].iloc[:, :-1], best_split["df_right"].iloc[:, -1], depth=depth+1)
        
        return Node(attribute=best_split["best_feature"], 
                    threshold=best_split["threshold"],  
                    left_child=left_subset, 
                    right_child=right_subset,
                    gain=best_split["gain"],
                    majority_label=np.unique(y).max())

    def learn(self, X, y, impurity_measure="entropy", prune=False):
        '''
        The main function for training the decision tree with optional pruning.

        Args:
        - X: Feature values for the dataset.
        - y: The target variable for the dataset.
        - impurity_measure: The impurity measure to use, either 'entropy' or 'gini'.
        - prune: Specifies whether the decision tree should be pruned.

        Returns:
        - No return value, the model is trained and stored in the object.
        '''

        self.impurity_measure = impurity_measure
        if prune:
            X_train, X_prune, y_train, y_prune = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        else:
            X_train = X
            y_train = y
        
        self.root = self._learn(X_train, y_train)
        if prune:
            model_acc = accuracy_score(y_prune, self.predict(X_prune))
            self.prune(model_acc, X_prune, y_prune)

    def print_tree(self, node=None, depth=0):
        # Check if the node parameter is None and use the root node if it's the case
        if node is None:
            node = self.root

        # If the node has a left child node, recursively print with the left child
        if node.left_child:
            self.print_tree(node.left_child, depth + 1)
        else:
            # If there is no left child, print a representation of a leaf node
            print("  " * depth, f"Class: {node.value}")

        # If the node has a right child node, recursively print with the right child
        if node.right_child:
            self.print_tree(node.right_child, depth + 1)
        else:
            # If there is no right child, print a representation of a leaf node
            print("  " * depth, f"Class: {node.value}")

    def _predict(self, node, x):
        '''
        Predicts the target variable for a single input using the decision tree.
        '''
        # Check if the node is a leaf node
        if node.is_leaf():
            return node.value  # Return the value (prediction) from the leaf node
        # Check if the value for the current node's attribute in the input (X) is less than the node's threshold
        if x[node.attribute] < node.threshold:
            # If the value is less, go to the left child node and make a recursive prediction
            return self._predict(node.left_child, x)
        # If the value is greater or equal, go to the right child node and make a recursive prediction
        return self._predict(node.right_child, x)

    def predict(self, X):
        '''
        Predicts the target variable for a set of inputs using the decision tree.
        '''
        # Check if the decision tree has been trained (has a root node)
        if self.root is None:
            raise ValueError("Decision tree is empty.")

        # Start the prediction by calling the _predict method with the root node and the inputs (X)
        return [self._predict(self.root, row) for index, row in X.iterrows()]

    def gini(self, y):
        '''
        Calculates the Gini impurity for a set of target labels y.

        Args:
        - y: The target variable for a dataset.

        Returns:
        - Gini impurity for the target labels y.
        '''

        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        gini = 1.0

        for count in class_counts:
            p = count / total_samples
            gini -= p ** 2

        return gini

    def _prune(self, visited_node, node: Node, prev_node, model_acc, X_val, y_val):
        '''
        Performs pruning of the decision tree by temporarily removing nodes and evaluating performance before potentially proceeding or restoring the node.

        Args:
        - visited_node: A set that stores visited nodes to avoid double pruning.
        - node: The current node to be considered.
        - prev_node: The parent node of the current node.
        - model_acc: The model's accuracy before pruning.
        - X_val: Feature values for the validation set.
        - y_val: The target variable for the validation set.

        Returns:
        - No specific return value. The model is pruned to improve performance.
        '''

        if node is None:
            pass
        elif node not in visited_node:
            visited_node.add(node)

            # Traverse down the tree
            if node.is_leaf():
                # If the node is a leaf node, create a temporary copy of the parent node and remove the node
                left_copy = prev_node.left_child
                right_copy = prev_node.right_child
                prev_node.left_child = None
                prev_node.right_child = None
                prev_node.value = prev_node.majority_label

                # Calculate accuracy after pruning
                pruned_accuracy = accuracy_score(y_val, self.predict(X_val))

                if pruned_accuracy >= model_acc:
                    # If the accuracy after pruning is better or equal to model_acc, keep the pruning
                    model_acc = pruned_accuracy
                else:
                    # If accuracy worsens, restore the node
                    prev_node.left_child = left_copy
                    prev_node.right_child = right_copy
                    prev_node.value = None
            else:
                # Recursively go down to the bottom of the tree
                self._prune(visited_node, node.left_child, node, model_acc, X_val, y_val)
                self._prune(visited_node, node.right_child, node, model_acc, X_val, y_val)

    def prune(self, model_acc, x, y):
        '''
        Calls the _prune() method.

        Args:
        - model_acc: The model's accuracy before pruning.
        - x: Feature values for the validation set.
        - y: The target variable for the validation set.

        Returns:
        - No specific return value.
        '''
        visited_node = set()
        self._prune(visited_node, self.root, None, model_acc, x, y)

