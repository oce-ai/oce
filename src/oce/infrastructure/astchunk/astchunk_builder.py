import re
from typing import Generator

import numpy as np
import pyrsistent
import tree_sitter as ts
from loguru import logger
from tree_sitter_language_pack import get_parser

from .astnode import ASTNode
from .astchunk import ASTChunk
from .compat import compat_parse
from .preprocessing import (
    ByteRange,
    preprocess_nws_count,
    get_nws_count
)

# Language name mapping for tree-sitter-language-pack
# Maps user-friendly names to tree-sitter-language-pack language identifiers
LANGUAGE_MAP = {
    # Original supported languages
    "python": "python",
    "java": "java",
    "csharp": "c_sharp",
    "c_sharp": "c_sharp",
    "typescript": "tsx",
    "tsx": "tsx",
    # Additional languages supported by tree-sitter-language-pack
    "javascript": "javascript",
    "jsx": "javascript",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "go": "go",
    "golang": "go",
    "rust": "rust",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "scala": "scala",
    "bash": "bash",
    "shell": "bash",
    "sql": "sql",
    "lua": "lua",
    "r": "r",
    "julia": "julia",
    "haskell": "haskell",
    "elixir": "elixir",
    "erlang": "erlang",
    "clojure": "clojure",
    "ocaml": "ocaml",
    "zig": "zig",
    "nim": "nim",
    "dart": "dart",
    "perl": "perl",
    "dockerfile": "dockerfile",
    "make": "make",
    "cmake": "cmake",
}

# Field names tree-sitter grammars use for the payload of a declaration. A node
# carrying one reads as a unit: whoever lands on it sees a subject and the code
# that belongs to it. Splitting one yields halves that open on `return {` or
# trail off mid-statement, which is what the intact-node budget below prevents.
#
# Matching on fields rather than node type names keeps this working per language.
# A type list has to be extended for every grammar and fails silently otherwise:
# measured on OpenClaw, Kotlin kept only 47.8% of its oversized declarations
# intact and Swift 53.8%, because their builder-DSL nodes (`annotated_lambda`,
# `call_suffix`) never appear in a TypeScript-derived list.
_DECLARATION_BODY_FIELDS = ("body", "block", "declaration_list", "field_declaration_list")

# Some grammars express structure through child node types instead of fields —
# Kotlin names none of its children, so a `class_declaration` there is only
# recognisable by the `class_body` hanging under it. Suffix matching covers the
# `*_body` / `*_block` convention these grammars follow.
_BODY_NODE_SUFFIXES = ("_body", "_block", "_statements", "_declaration_list")
_BODY_NODE_TYPES = frozenset({"block", "statements", "statement_block"})

# Node types that wrap a declaration without being one. Their own body belongs to
# the inner declaration, so they qualify only if that declaration does.
_DECLARATION_WRAPPER_TYPES = frozenset(
    {
        "decorated_definition",
        "export_statement",
        "expression_statement",
        "lexical_declaration",
        "variable_declaration",
        "variable_declarator",
        "public_field_definition",
        "property_declaration",
        "call_expression",
        "arguments",
        # Kotlin/Swift builder DSL: `android { ... }` parses as a call whose
        # trailing lambda holds the block.
        "call_suffix",
        "annotated_lambda",
        "lambda_literal",
        "function_body",
        "assignment",
    }
)

# How far past ``max_chunk_size`` a declaration may run before it is split
# anyway. Measured on the OpenClaw suite: test cases and handlers cluster
# between one and three times the window, so recursing into them was what broke
# two thirds of the chunks, while a hard ceiling still keeps a runaway
# ``describe`` block (150k characters and up) from becoming one chunk.
_INTACT_NODE_SIZE_FACTOR = 3

# Deepest wrapper chain followed when looking for the declaration inside a
# statement. `export default foo(() => {...})` needs a few hops; beyond that the
# node is an expression tree rather than a declaration.
_WRAPPER_SEARCH_DEPTH = 4

_REACT_COMPONENT_LANGUAGES = frozenset({"jsx", "tsx"})
_REACT_DECLARATION_TYPES = frozenset(
    {
        "class_declaration",
        "function_declaration",
        "lexical_declaration",
        "variable_declaration",
    }
)
_JSX_NODE_TYPES = frozenset(
    {"jsx_element", "jsx_fragment", "jsx_self_closing_element"}
)
_PASCAL_CASE_DECLARATION = re.compile(
    r"^(?:(?:async\s+)?function|class|const|let|var)\s+"
    r"([A-Z][A-Za-z0-9_$]*)\b"
)


def get_supported_languages() -> list[str]:
    """
    Get a list of all supported programming languages.

    Returns:
        A sorted list of supported language names.
    """
    return sorted(set(LANGUAGE_MAP.keys()))


class ASTChunkBuilder:
    """
    AST-based code chunker with fallback to character-based splitting.

    Attributes:
        - max_chunk_size: Maximum size for each AST chunk, using non-whitespace character count by default.
        - language: Programming language. Supports 40+ languages via tree-sitter-language-pack.
                   If language is not supported, automatically falls back to RecursiveCharacterTextSplitter.
        - metadata_template: Type of metadata to store (e.g., start/end line number, path to file, etc).
        - use_fallback: Whether fallback mode is active (set automatically).
    """
    def __init__(self, **configs):
        self.max_chunk_size: int = configs['max_chunk_size']
        self.language: str = configs['language']
        self.metadata_template: str = configs['metadata_template']
        self.use_fallback: bool = False
        self.parser = None
        # Above this an atomic declaration is split despite the cost, so one
        # oversized wrapper cannot swallow a whole file.
        self.intact_node_size: int = configs.get(
            'intact_node_size',
            self.max_chunk_size * _INTACT_NODE_SIZE_FACTOR,
        )

        # Try to get tree-sitter parser
        lang_key = self.language.lower()
        if lang_key in LANGUAGE_MAP:
            # Supported language: use AST-based chunking
            ts_language = LANGUAGE_MAP[lang_key]
            try:
                self.parser = get_parser(ts_language)
            except Exception as e:
                logger.warning(
                    "Failed to load {} parser; using text fallback: {}",
                    self.language,
                    e,
                )
                self.use_fallback = True
        else:
            logger.info(
                "Language {} has no tree-sitter parser; using text fallback",
                self.language,
            )
            self.use_fallback = True

    # ------------------------------ #
    #            Step #1             #
    # ------------------------------ #
    def assign_tree_to_windows(self, code: str, root_node: ts.Node) -> Generator[list[ASTNode], None, None]:
        """
        Assign AST tree to windows. A window is a tentative chunk consists of ASTNode before being converted into ASTChunk.

        This function serves as a wrapper function for self.assign_nodes_to_windows(). 
        Additionally, it also
            1. performs preprocessing for efficient AST node size computation.
            2. handles the edge case where the entire AST tree can fit in one window.

        Args:
            code: code to be chunked
            root_node: root node of the AST tree

        Yields:
            Lists (windows) of ASTNode
        """
        # Preprocessing non-whitespace character count
        nws_cumsum = preprocess_nws_count(bytes(code, "utf8"))
        if self.language.lower() in _REACT_COMPONENT_LANGUAGES:
            component_windows = list(
                self._assign_react_top_level_windows(
                    root_node,
                    nws_cumsum,
                )
            )
            if component_windows:
                yield from component_windows
                return

        tree_range = ByteRange(root_node.start_byte, root_node.end_byte)
        tree_size = get_nws_count(nws_cumsum, tree_range)

        # If the entire tree can fit in one window, assign tree to window
        if tree_size <= self.max_chunk_size:
            yield [ASTNode(root_node, tree_size)]
        # Otherwise, recursively assign children to windows
        else:
            ancestors = pyrsistent.v(root_node)
            yield from self.assign_nodes_to_windows(root_node.children, nws_cumsum, ancestors)

    def _assign_react_top_level_windows(
        self,
        root_node: ts.Node,
        nws_cumsum: np.ndarray,
    ) -> Generator[list[ASTNode], None, None]:
        """Keep each top-level JSX/TSX component on its own AST boundary."""
        nodes = root_node.children
        if not any(self._is_react_component(node) for node in nodes):
            return

        ancestors = pyrsistent.v(root_node)
        ordinary_nodes: list[ts.Node] = []
        for node in nodes:
            if not self._is_react_component(node):
                ordinary_nodes.append(node)
                continue
            if ordinary_nodes:
                yield from self.assign_nodes_to_windows(
                    ordinary_nodes,
                    nws_cumsum,
                    ancestors,
                )
                ordinary_nodes = []
            yield from self.assign_nodes_to_windows(
                [node],
                nws_cumsum,
                ancestors,
            )
        if ordinary_nodes:
            yield from self.assign_nodes_to_windows(
                ordinary_nodes,
                nws_cumsum,
                ancestors,
            )

    def _is_react_component(self, node: ts.Node) -> bool:
        declaration = node
        if node.type == "export_statement":
            declaration = next(
                (
                    child
                    for child in node.children
                    if child.type in _REACT_DECLARATION_TYPES
                ),
                node,
            )
        if declaration.type not in _REACT_DECLARATION_TYPES:
            return False
        if not self._has_pascal_case_name(declaration):
            return False
        return self._contains_jsx(declaration)

    @staticmethod
    def _has_pascal_case_name(node: ts.Node) -> bool:
        first_line = node.text.decode("utf8", errors="replace").splitlines()[0]
        return _PASCAL_CASE_DECLARATION.match(first_line.strip()) is not None

    @staticmethod
    def _contains_jsx(node: ts.Node) -> bool:
        pending = list(node.children)
        while pending:
            current = pending.pop()
            if current.type in _JSX_NODE_TYPES:
                return True
            pending.extend(current.children)
        return False
    
    def assign_nodes_to_windows(self, nodes: list[ts.Node], nws_cumsum: np.ndarray, ancestors: pyrsistent.pvector) -> Generator[list[ASTNode], None, None]:
        """
        Assign AST nodes to windows. A window is a tentative chunk consists of ASTNode before being converted into ASTChunk.

        This function:
            1. greedily assigns AST nodes to windows based on their non-whitespace character count.
            2. recursively processes child nodes if the current node exceeds the max chunk size.
            3. keeps track of the ancestors of each node for path construction.

        Args:
            nodes: list of AST nodes to be assigned to windows
            nws_cumsum: cumulative sum of non-whitespace characters
            ancestors: ancestors of the current node

        Yields:
            Lists (windows) of ASTNode
        """
        # Base case: no nodes to assign
        if not nodes:
            yield from []
            return

        # Initialize the current window
        current_window = []
        current_window_size = 0

        for node in nodes:
            node_range = ByteRange(node.start_byte, node.end_byte)
            node_size = get_nws_count(nws_cumsum, node_range)
            
            # Check if node needs recursive processing (i.e., too large to fit in a window)
            node_exceeds_limit = node_size > self.max_chunk_size
            
            # Handle the cases where we cannot add the current node to the current window
            # Case 1: current window is empty and node exceeds limit
            # Case 2: current window is not empty and adding the node exceeds limit
            if (len(current_window) == 0 and node_exceeds_limit) or \
            (current_window_size + node_size > self.max_chunk_size):
                
                # Clear current window if not empty
                if len(current_window) > 0:
                    yield current_window
                    current_window = []
                    current_window_size = 0
                
                # If node still exceeds limit, recursively process the node's children
                if node_exceeds_limit:
                    # A declaration that only modestly overshoots the window is
                    # kept whole. Recursing into it hands back its statements,
                    # and greedy packing then cuts between them, so the chunk
                    # that carries the name loses the body and the next one
                    # opens on a fragment like `return {`.
                    if self._is_intact_declaration(node, node_size):
                        yield [ASTNode(node, node_size, ancestors)]
                        continue
                    childs_ancestors = ancestors.append(node)
                    child_windows = list(self.assign_nodes_to_windows(node.children, nws_cumsum, childs_ancestors))
                    if child_windows:
                        # (optional) Greedily merge adjacent windows from the beginning if merged window does not exceed self.max_chunk_size
                        yield from self.merge_adjacent_windows(child_windows)
                    else:
                        # P1 FIX: Leaf node exceeds limit (no children to recurse into).
                        # Yield it as a standalone window instead of silently dropping it.
                        # Downstream _CapChunker will handle hard splitting at 6000 chars.
                        yield [ASTNode(node, node_size, ancestors)]
                else:
                    # Node fits in an empty window
                    current_window.append(ASTNode(node, node_size, ancestors))
                    current_window_size += node_size
                    
            # Case 3: node fits in current window
            else:
                current_window.append(ASTNode(node, node_size, ancestors))
                current_window_size += node_size

        # Add the last window if it's not empty
        if len(current_window) > 0:
            yield current_window

    def _is_intact_declaration(self, node: ts.Node, node_size: int) -> bool:
        """Whether a node should stay whole even though it overshoots the window.

        Size is what separates the two shapes a wrapper can take: one
        ``expression_statement`` is a single ``it(...)`` case, another is a
        ``describe`` block spanning a whole file. Past ``intact_node_size`` the
        node is split as before.

        Wrappers like ``export_statement`` are not declarations themselves, so
        the body check stops at depth 0 — the wrapper qualifies only if it
        directly owns a body, not if one hangs off something it wraps.
        """
        return node_size <= self.intact_node_size and self._carries_body(node, max_depth=0)

    def _carries_body(self, node: ts.Node, depth: int = 0, max_depth: int = _WRAPPER_SEARCH_DEPTH) -> bool:
        """Whether the node owns a body, following wrappers that delegate theirs.

        Two probes, because grammars disagree on how to express containment:
        field names where they exist, child node types otherwise. Together they
        cover Swift's ``call_suffix`` and Kotlin's ``class_body`` without either
        being enumerated by name.

        Wrappers are followed one level at a time because the body of
        ``export const handler = () => {...}`` hangs off the arrow function, not
        off the statement that declares it.

        ``max_depth`` caps how deep wrappers are followed. Passing 0 checks only
        the node itself, which keeps ``export_statement`` from being treated as
        a declaration when deciding whether to preserve it whole.
        """
        for field in _DECLARATION_BODY_FIELDS:
            if node.child_by_field_name(field) is not None:
                return True
        children = node.named_children
        if any(self._looks_like_body(child) for child in children):
            return True
        if depth >= max_depth or node.type not in _DECLARATION_WRAPPER_TYPES:
            return False
        return any(self._carries_body(child, depth + 1, max_depth) for child in children)

    @staticmethod
    def _looks_like_body(node: ts.Node) -> bool:
        """Whether a child node is the body of its parent, judged by its type."""
        return node.type in _BODY_NODE_TYPES or node.type.endswith(_BODY_NODE_SUFFIXES)


    def merge_adjacent_windows(self, ast_windows: list[list[ASTNode]]) -> Generator[list[ASTNode], None, None]:
        """
        Greedily merge adjacent windows of ASTNode if the merged window's total non whitespace character count
        does not exceed max_char_count.

        We choose to merge child windows in this function instead of self.assign_nodes_to_windows() because
        we want to maintain the structure of the original AST as much as possible. Therefore, we should only
        merge windows if all ASTNodes in the window are siblings.
        
        Args:
            ast_windows: A list of list (windows) of ASTNode
            
        Yields:
            Lists (windows) of ASTNode with adjacent windows merged where possible
        """
        assert ast_windows, "Expect non-empty ast_windows"
        
        # Start with a copy of the first list
        merged_windows = [ast_windows[0][:]]  
        
        for window in ast_windows[1:]:
            current_extending_window = merged_windows[-1]
            
            # Calculate the total character count if we merge
            merged_window_size = sum(n.size for n in current_extending_window) + sum(n.size for n in window)
            
            # If merging won't exceed the limit, merge the lists
            if merged_window_size <= self.max_chunk_size:
                current_extending_window.extend(window)
            else:
                # Otherwise, add the current list as a new entry
                merged_windows.append(window[:])
        
        yield from merged_windows
    
    # ------------------------------ #
    #            Step #2             #
    # ------------------------------ #
    def add_window_overlapping(self, ast_windows: list[list[ASTNode]], chunk_overlap: int) -> list[list[ASTNode]]:
        """
        Extend each window by adding overlapping ASTNodes from the previous and next window.

        Similar to regular document chunking, we add overlapping ASTNodes from the previous and next window
        to each window to provide context. However, we make this step optional since (1) AST Chunking naturally
        avoids breaking the struture of code, hence overlapping is less necessary for maintaining the completeness of
        code blocks (though the additional context may still be useful for downstream tasks); (2) overlapping
        ASTNodes from adjacent windows may cause high variance in chunk size, which makes it difficult to
        control each chunk's token count (especially when the downstream model has a strict limit on context length).

        Args:
            ast_windows: A list of list (windows) of ASTNode
            chunk_overlap: Number of ASTNodes to overlap between adjacent windows

        Returns:
            A list of list (windows) of ASTNode with overlapping ASTNodes added
        """
        assert chunk_overlap >= 0, f"Expect non-negative chunk_overlap, got {chunk_overlap}"

        if chunk_overlap == 0:
            return ast_windows

        new_code_windows = list[list[ASTNode]]()

        for i in range(len(ast_windows)):
            # Create a copy of the current window
            current_node_list = ast_windows[i].copy()
            
            # If there is a previous window, prepend its last chunk_overlap elements
            if i > 0:
                assert len(ast_windows[i-1]) > 0, f"Attempting to take elements from an empty window at {i-1}!"
                prev_window = ast_windows[i-1]
                last_k_nodes = prev_window[-min(chunk_overlap, len(prev_window)):]
                # Insert at the beginning (prepending all elements)
                current_node_list = last_k_nodes + current_node_list
            
            # If there is a next window, append its first chunk_overlap elements
            if i < len(ast_windows) - 1:
                assert len(ast_windows[i+1]) > 0, f"Attempting to take elements from an empty window at {i+1}!"
                next_window = ast_windows[i+1]
                first_k_nodes = next_window[:min(chunk_overlap, len(next_window))]
                # Append all elements
                current_node_list = current_node_list + first_k_nodes
                
            new_code_windows.append(current_node_list)
            
        return new_code_windows
    
    # ------------------------------ #
    #            Step #3             #
    # ------------------------------ #
    def convert_windows_to_chunks(self, ast_windows: list[list[ASTNode]], 
                                  repo_level_metadata: dict, chunk_expansion: bool) -> list[ASTChunk]:
        """
        Convert each tentative window of ASTNode into an ASTChunk object.

        This function finalizes the boundary of each chunk and build metadata for each chunk.
        Additionally, it also applies chunk expansion if specified. Chunk expansion is the process of
        adding chunk metadata (e.g., file path, class path) to the beginning of each chunk. It can consist of information
        (1) available in all chunking frameworks (e.g., file path, start line, end line, etc.) and
        (2) specific to AST Chunking (e.g., class path, function path, etc.).
        We found that chunk expansion can be helpful for downstream retrieval and sometimes generation tasks. 
        However, it is also worth noting that chunk expansion consumes additional tokens, thereby reducing the number of chunks that can fit in the context window.
        Hence, we make chunk expansion an optional step that can be turned on / off via the `chunk_expansion` flag.

        Args:
            ast_windows: A list of list (windows) of ASTNode
            repo_level_metadata: Repository-level metadata (e.g., repo name, file path)
            chunk_expansion: Whether to perform chunk expansion (i.e., add metadata headers to chunks)

        Returns:
            A list of ASTChunk objects
        """
        ast_chunks = list[ASTChunk]()

        for current_window in ast_windows:
            current_chunk = ASTChunk(
                ast_window=current_window,
                max_chunk_size=self.max_chunk_size,
                language=self.language,
                metadata_template=self.metadata_template
            )
            current_chunk.build_metadata(repo_level_metadata)
            
            # (optional) apply chunk expansion
            if chunk_expansion:
                current_chunk.apply_chunk_expansion()
            ast_chunks.append(current_chunk)

        return ast_chunks
    
    # ------------------------------ #
    #            Step #4             #
    # ------------------------------ #
    def convert_chunks_to_code_windows(self, ast_chunks: list[ASTChunk]) -> list[dict]:
        """
        Convert each ASTChunk object into a code window for downstream integration.

        Args:
            ast_chunks: A list of ASTChunk objects

        Returns:
            A list of code windows, where each code window is a dict with keys "content" and "metadata"
        """
        code_windows = []

        for current_chunk in ast_chunks:
            code_windows.append(current_chunk.to_code_window())
        
        return code_windows

    # ------------------------------ #
    #       AST Chunking Logic       #
    # ------------------------------ #
    def chunkify(self, code: str, **configs) -> list[dict]:
        """
        Parse a piece of code into structural-aware chunks using AST.
        Falls back to RecursiveCharacterTextSplitter if language is not supported.

        Args:
            code: code to be chunked
            **configs: additional arguments for building chunks and/or chunk metadata

        Returns:
            A list of code windows (dicts with "content" and "metadata" keys)
        """
        # Check if we should use fallback
        if self.use_fallback:
            return self._chunkify_fallback(code, **configs)

        # AST-based chunking (original logic)
        # step 1: greedily assign AST tree / AST nodes to windows
        #         see self.assign_tree_to_windows() and self.assign_nodes_to_windows() for details
        ast = compat_parse(self.parser, code)
        ast_windows = list(self.assign_tree_to_windows(
            code=code,
            root_node=ast.root_node
        ))
        # [after this step]: list[list[ASTNode]] where each sublist represents an AST window

        # step 2 (optional): add overlapping
        #                    for each window, take the last k ASTNodes from the previous window and the first k ASTNodes from the next window
        ast_windows = self.add_window_overlapping(
            ast_windows=ast_windows,
            chunk_overlap=configs.get("chunk_overlap", 0)
        )
        # [after this step]: list[list[ASTNode]] where each sublist represents an AST window

        # step 3: convert each AST window into an ASTChunk object
        ast_chunks = self.convert_windows_to_chunks(
            ast_windows=ast_windows,
            repo_level_metadata=configs.get("repo_level_metadata", {}),
            chunk_expansion=configs.get("chunk_expansion", False)
        )
        # [after this step]: list[ASTChunk]

        # step 4: convert each ASTChunk to a code window for downstream integration
        code_windows = self.convert_chunks_to_code_windows(
            ast_chunks=ast_chunks
        )
        # [after this step]: list[dict] where each dict represents a code window

        return code_windows

    def _chunkify_fallback(self, code: str, **configs) -> list[dict]:
        """
        Fallback chunking using RecursiveCharacterTextSplitter.
        Used when tree-sitter parser is not available for the language.

        Args:
            code: code to be chunked
            **configs: additional arguments (repo_level_metadata, chunk_expansion, etc.)

        Returns:
            A list of code windows compatible with AST-based output format
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_text_splitters.base import Language

        # Try to use language-specific separators
        try:
            # Convert string to Language enum
            lang_enum = Language(self.language.lower())
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang_enum,
                chunk_size=self.max_chunk_size,
                chunk_overlap=configs.get("chunk_overlap", 0) * 50,  # Approximate
            )
        except (ValueError, AttributeError, KeyError):
            # Language not supported by langchain, use default separators
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=configs.get("chunk_overlap", 0) * 50,
                separators=["\n\n", "\n", " ", ""],
            )

        chunks = splitter.split_text(code)

        # Convert to our format
        return self._format_chunks(chunks, code, configs)

    def _format_chunks(self, chunks: list[str], original_code: str, configs: dict) -> list[dict]:
        """Format chunks into code_windows format."""
        repo_level_metadata = configs.get("repo_level_metadata", {})
        chunk_expansion = configs.get("chunk_expansion", False)

        code_windows = []
        current_pos = 0

        for chunk_text in chunks:
            # Calculate line numbers
            lines_before = original_code[:current_pos].count('\n')
            lines_in_chunk = chunk_text.count('\n')
            start_line = lines_before
            end_line = lines_before + lines_in_chunk

            # Build metadata
            metadata = {
                "filepath": repo_level_metadata.get("filepath", ""),
                "chunk_size": len(chunk_text),
                "line_count": lines_in_chunk + 1,
                "start_line_no": start_line,
                "end_line_no": end_line,
                "node_count": 0,
                "chunking_method": "recursive_character",
            }

            # Apply chunk expansion
            content = chunk_text
            if chunk_expansion and repo_level_metadata.get("filepath"):
                expansion = f"'''\n{repo_level_metadata['filepath']}\n'''\n"
                content = expansion + content

            code_windows.append({
                "content": content,
                "metadata": metadata
            })

            current_pos += len(chunk_text)

        return code_windows
