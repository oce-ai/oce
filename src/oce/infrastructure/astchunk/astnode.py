import tree_sitter as ts

from .preprocessing import ByteRange


class ASTNode:
    """
    A wrapper class for tree-sitter node.

    This class provides additional information for each node, including:
        - node_size: size of the node (in non-whitespace characters)
        - ancestors: ancestors of the node (list of tree-sitter nodes)

    Attributes:
        - node: tree-sitter node
        - node_size: size of the node (in non-whitespace characters)
        - ancestors: ancestors of the node (list of tree-sitter nodes)
    """
    def __init__(self, ts_node: ts.Node, node_size: int, ancestors: list[ts.Node] = []):
        self.node = ts_node
        self.node_size = node_size
        self.ancestors = ancestors

    @property
    def bcode(self):
        return self.node.text
    
    @property
    def strcode(self):
        return self.bcode.decode("utf8")
    
    @property
    def brange(self):
        return ByteRange(self.node.start_byte, self.node.end_byte)
    
    @property
    def start_line(self):
        return self.node.start_point.row
    
    @property
    def end_line(self):
        return self.node.end_point.row
    
    @property
    def start_col(self):
        return self.node.start_point.column
    
    @property
    def end_col(self):
        return self.node.end_point.column
    
    @property
    def size(self):
        """
        Define size as the number of non-whitespace characters
        """
        return self.node_size

    @property
    def length(self):
        """
        Define length as the number of lines covered by the node
        """
        return self.end_line - self.start_line + 1
