import string
import numpy as np
from dataclasses import dataclass

import tree_sitter as ts


@dataclass(frozen=True, order=True)
class IntRange:
    """
    A continuous range of integers from [start, stop).

    For example [0, 2) would include the integers 0 and 1. This range could be
    used to represent the first two characters of a document.
    """

    start: int
    """The start of the range."""
    stop: int
    """The exclusive end of the range."""

    def __post_init__(self):
        if self.stop < self.start:
            raise ValueError(f"A valid range must have {self.start=} <= {self.stop=}.")

    def contains(self, other: "IntRange") -> bool:
        """Check if this range fully contains another range."""
        return self.start <= other.start and self.stop >= other.stop

    def overlaps(self, other: "IntRange") -> bool:
        """Check if the two ranges have a non-zero intersection."""
        return max(self.start, other.start) < min(self.stop, other.stop)


# Commonly used alias for IntRange
ByteRange = IntRange
"""References a range of bytes in file."""


def get_nodes_in_brange(root_node: ts.Node, brange: ByteRange) -> list[ts.Node]:
    """
    Find and return all valid tree-sitter nodes fully contained within the specified byte range.

    This function traverses the syntax tree starting from the given root node and collects
    all nodes whose byte ranges are fully contained within the specified byte range.
    Nodes with type "ERROR" and their descendants are excluded from the results.
    """
    results = list[ts.Node]()
    worklist = [root_node]

    while worklist:
        n = worklist.pop()
        if n.type == "ERROR" or n.type == "module":
            if n.type == "module":
                for c in n.children:
                    worklist.append(c)
            continue
        n_range = ByteRange(n.start_byte, n.end_byte)
        if brange.contains(n_range):
            results.append(n)
        if brange.overlaps(n_range):
            for c in n.children:
                worklist.append(c)

    return results

def get_largest_node_in_brange(ts_node: ts.Node, brange: ByteRange, size_option: str = "non-ws") -> int:
    """
    Return the size of the largest node (in bytes or non-whitespace char) in the given byte range.
    """
    nodes = get_nodes_in_brange(ts_node, brange)
    if not nodes:
        return 0
    if size_option == "byte":
        node_sizes = [n.end_byte - n.start_byte for n in nodes]
    elif size_option == "non-ws":
        nws_cumsum = preprocess_nws_count(ts_node.text)
        node_sizes = [get_nws_count(nws_cumsum, ByteRange(n.start_byte, n.end_byte)) for n in nodes]
    else:
        raise ValueError(f"Unrecognized size option: {size_option}")

    return max(node_sizes)

def preprocess_nws_count(bstring: bytes) -> np.ndarray:
    """
    Given a byte string, construct a cumulative sum array that keeps track of non-whitespace char count at each index.

    This function performs a O(n) pre-computation and enables O(1) lookup of byte substring.
    """
    whitespace_bytes = tuple(ord(x) for x in string.whitespace)
    is_nws = np.array([x not in whitespace_bytes for x in bstring])
    is_nws_cumsum = np.cumsum(is_nws)
    nws_cumsum = np.concatenate([[0], is_nws_cumsum])
    return nws_cumsum

def get_nws_count(nws_cumsum: np.ndarray, brange: ByteRange) -> int:
    """
    Look up the non-whitespace char count within the given byte range.

    Notes:
        - need to convert int64 to int for json dump
    """
    return int(nws_cumsum[brange.stop] - nws_cumsum[brange.start])

def get_nws_count_direct(code: str) -> int:
    """
    O(n) computation of nonwhitespace count.

    This function can be used as a verifier.
    """
    return sum([1 for x in code if x not in string.whitespace])
