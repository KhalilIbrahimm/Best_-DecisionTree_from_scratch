
# A class to create nodes
class Node:
    def __init__(self, attribute=None, threshold=None, left_child=None, right_child=None, gain=None, value=None, majority_label=None):
        self.attribute = attribute  # The attribute used to split the node
        self.threshold = threshold  # The threshold value for splitting (for continuous attributes)
        self.left_child = left_child  # Reference to the left child (subtree)
        self.right_child = right_child  # Reference to the right child (subtree)
        self.majority_label = majority_label 
        # Leaf Node:
        self.gain = gain
        self.value = value  # The value that the node represents (for leaf nodes)
  
    def is_leaf(self):
        return self.left_child is None and self.right_child is None
