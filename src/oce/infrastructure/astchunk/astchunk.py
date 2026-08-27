import tree_sitter as ts

from .astnode import ASTNode
from .preprocessing import ByteRange, get_nws_count_direct


class ASTChunk:
    """
    A chunk of code represented by a list of ASTNodes.

    This class provides additional information for each chunk, including:
        - chunk_text: rebuilt code text from the list of ASTNodes
        - chunk_size: size of the chunk (in non-whitespace characters)
        - chunk_ancestors: ancestors of the chunk (list of ancestor names)
        - metadata: additional metadata for the chunk (e.g., file path, class path, etc.)

    Attributes:
        - ast_window: list of ASTNode objects
        - max_chunk_size: maximum size for each AST chunk, using non-whitespace character count by default.
        - language: programming language
        - metadata_template: type of metadata to store (e.g., start/end line number, path to file, etc.)
    """
    def __init__(self, ast_window: list[ASTNode], max_chunk_size: int, language: str, metadata_template: str):
        self.ast_window = ast_window
        self.max_chunk_size = max_chunk_size
        self.language = language
        self.metadata_template = metadata_template
        assert len(self.ast_window) > 0, "Expect ASTChunk to be non-empty"

        self.chunk_text = self.rebuild_code(self.ast_window)
        self.chunk_size = get_nws_count_direct(self.chunk_text)

        # build chunk ancestors using the ancestors of the first ASTNode in the window
        self.chunk_ancestors = self.build_chunk_ancestors(
            self.ast_window[0].ancestors,
            self.ast_window[0].node,
        )

    @property
    def strcode(self):
        return self.chunk_text

    @property
    def brange(self):
        return ByteRange(self.ast_window[0].brange.start, self.ast_window[-1].brange.stop)

    @property
    def start_line(self):
        return self.ast_window[0].start_line

    @property
    def end_line(self):
        return self.ast_window[-1].end_line

    @property
    def end_col(self):
        return self.ast_window[-1].end_col

    @property
    def size(self):
        """
        Define size as the number of non-whitespace characters.
        """
        return self.chunk_size

    @property
    def length(self):
        """
        Define length as the number of lines covered by the chunk.
        """
        return self.end_line - self.start_line + 1

    def rebuild_code(self, ast_window: list[ASTNode]) -> str:
        """
        Rebuild source code from a list of ASTNodes.

        The code text stored in each ASTNode is inherited from the tree-sitter Node object, which omits 
        leading and trailing spaces and newlines between nodes. Therefore, this function restores the 
        original code by adding the necessary newlines and spaces.

        Args:
            ast_window: list of ASTNode objects

        Returns:
            Rebuilt source code string
        """
        if len(ast_window) == 0:
            return ""

        current_line, current_col = ast_window[0].start_line, ast_window[0].start_col
        code = " " * current_col

        for node in ast_window:
            # If we need to jump to a new line, add newline(s)
            if  node.start_line > current_line:
                # Add as many newlines as needed.
                code += "\n" * (node.start_line - current_line)
                current_line =  node.start_line
                # Reset the column since we are at a new line.
                current_col = 0
            # If we are on the correct line but need to add indentation spaces:
            if  node.start_col > current_col:
                code += " " * (node.start_col - current_col)
                current_col =  node.start_col
            # Append the node_text
            code += node.strcode
            # Update our cursor position to the given end coordinate.
            # (We trust that the given end coordinate is consistent with the node_text.)
            current_line, current_col =  node.end_line,  node.end_col

        return code

    # Declaration nodes whose own first line already reads as a signature.
    # Measured against tree-sitter-language-pack for the languages in use.
    CLASS_LIKE_NODES = {
        "class_definition",      # Python
        "class_declaration",     # Java, TypeScript, JavaScript, C#, PHP, Kotlin, Swift
        "class_specifier",       # C++
        "class",                 # Ruby
        "struct_item",           # Rust
        "struct_declaration",    # Swift
        "protocol_declaration",  # Swift
        "extension_declaration", # Swift
        "enum_declaration",      # Swift, Kotlin, Java, TypeScript
        "interface_declaration", # TypeScript, Java, Kotlin
        "object_declaration",    # Kotlin
        "type_declaration",      # Go
        "type_alias_declaration",# TypeScript
        "impl_item",             # Rust
        "trait_item",            # Rust
        "module",                # Ruby
        "namespace_definition",  # C++
    }

    FUNCTION_LIKE_NODES = {
        "function_definition",   # Python, C++, PHP, Bash
        "function_declaration",  # C, Go, Kotlin, Swift, JavaScript, TypeScript
        "method_declaration",    # Java, C#, Go, PHP
        "method_definition",     # TypeScript, JavaScript
        "method",                # Ruby
        "function_item",         # Rust
        "constructor_declaration",  # Java, C#
        "init_declaration",      # Swift
        "subscript_declaration", # Swift
        "computed_property",     # Swift
        "property_declaration",  # Swift, Kotlin
        "arrow_function",        # TypeScript, JavaScript
        "function_expression",   # TypeScript, JavaScript
        "generator_function_declaration",  # JavaScript
        "export_statement",      # TypeScript, JavaScript — wraps declarations, first line carries signature
    }

    # Nodes that carry no name of their own but whose parent binds one, e.g.
    # ``export const handle = async (req) => {...}``. The observed TypeScript
    # path is export_statement -> lexical_declaration -> variable_declarator ->
    # arrow_function -> statement_block, where only the declarator holds the
    # identifier. Treating these as signature carriers is what keeps the symbol
    # name attached to the body chunks.
    BINDING_NODES = {
        "variable_declarator",   # TypeScript, JavaScript
        "lexical_declaration",   # TypeScript, JavaScript (const/let)
        "variable_declaration",  # TypeScript, JavaScript (var)
        "public_field_definition",  # TypeScript class fields
        "pair",                  # object literal members
        "assignment",            # Python, Bash
        "short_var_declaration", # Go
        "const_declaration",     # Go
        "const_spec",            # Go
        "property_declaration",  # Kotlin, Swift
        "let_declaration",       # Rust
    }

    def build_chunk_ancestors(
        self,
        node_ancestors: list[ASTNode],
        current_node: ts.Node | None = None,
    ) -> list[str]:
        """Build the declaration path leading to this chunk.

        cAST splits an oversized declaration into a signature window and one or
        more body windows. A body window's own text starts at ``{``, so the
        enclosing symbol name is absent from the embedding input. Recording the
        ancestor signatures keeps that name searchable for those chunks.

        A whitelist of declaration node types alone is not enough. In the
        TypeScript sources measured here the dominant form binds a function to a
        name through a variable declarator, so the declaration node itself is
        anonymous. ``BINDING_NODES`` covers that case, and consecutive duplicate
        signatures are collapsed because a binding and the function it binds
        report the same first line.

        The ancestors list from astchunk_builder stops at the parent of the
        current window, so when a complete declaration fits in one window (e.g.,
        `export function Counter()` < 500 chars), its own signature is missing.
        This method now also checks the current_node to capture that signature.

        Args:
            node_ancestors: ancestors of the first ASTNode in the window
            current_node: the first node in the window itself (may carry signature)

        Returns:
            Signature lines ordered outermost first
        """
        signatures: list[str] = []

        # Collect signatures from ancestors
        for node in node_ancestors:
            if not self._is_signature_carrier(node.type):
                continue
            signature = self._signature_line(node)
            if signature and (not signatures or signatures[-1] != signature):
                signatures.append(signature)

        # Also check the current node itself (if it's a signature carrier)
        if current_node and self._is_signature_carrier(current_node.type):
            signature = self._signature_line(current_node)
            if signature and (not signatures or signatures[-1] != signature):
                signatures.append(signature)

        return signatures

    def _is_signature_carrier(self, node_type: str) -> bool:
        return (
            node_type in self.CLASS_LIKE_NODES
            or node_type in self.FUNCTION_LIKE_NODES
            or node_type in self.BINDING_NODES
        )

    @staticmethod
    def _signature_line(node) -> str:
        """Return the node's first non-empty line, trimmed of its body brace.

        Multi-line signatures are common (``function f(\\n  a,\\n  b,\\n) {``),
        so only the first line is kept to bound the prefix cost.
        """
        text = node.text.decode("utf8", errors="replace")
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped:
                return stripped.rstrip("{").rstrip()
        return ""

    def build_metadata(self, repo_level_metadata: dict):
        """
        Build metadata for the chunk.

        Args:
            repo_level_metadata: repository-level metadata (e.g., repo name, file path)
        """
        if self.metadata_template == "none":
            self.metadata = {}
        elif self.metadata_template == "default":
            filepath = repo_level_metadata.get("filepath", "")
            self.metadata = {
                "filepath": filepath,
                "chunk_size": self.chunk_size,
                "line_count": self.length,
                "start_line_no": self.start_line,
                "end_line_no": self.end_line,
                "end_column": self.end_col,
                "node_count": len(self.ast_window),
                # Exposed so adapters can attach the declaration path without
                # mutating chunk_text, which would shift reported line numbers.
                "ancestors": list(self.chunk_ancestors),
            }
        elif self.metadata_template == "coderagbench-repoeval":
            fpath_tuple = repo_level_metadata.get("fpath_tuple", [])
            repo = repo_level_metadata.get("repo", "")
            self.metadata = {
                "fpath_tuple": fpath_tuple,
                "repo": repo,
                "chunk_size": self.chunk_size,
                "line_count": self.length,
                "start_line_no": self.start_line,
                "end_line_no": self.end_line,
                "node_count": len(self.ast_window),
            }
        elif self.metadata_template == "coderagbench-swebench-lite":
            instance_id = repo_level_metadata.get("instance_id", "")
            filename = repo_level_metadata.get("filename", "")
            self.metadata = {
                "_id": f"{instance_id}_{self.start_line}-{self.end_line}",
                "title": filename,
            }
        else:
            raise ValueError(f"Unsupported Metadata Template Name: {self.metadata_template}!")

    def apply_chunk_expansion(self):
        """
        Apply chunk expansion to the chunk. Chunk expansion is the process of adding chunk expansion metadata 
        (e.g., file path, class path) to the beginning of each chunk.
        """
        self.chunk_expansion_metadata = {
            "filepath": "",
            "ancestors": "\n".join(["\t" * i + ancestor for i, ancestor in enumerate(self.chunk_ancestors)]),
        }
        if self.metadata_template == "default":
            self.chunk_expansion_metadata["filepath"] = self.metadata["filepath"]
        elif self.metadata_template == "coderagbench-repoeval":
            self.chunk_expansion_metadata["filepath"] = "/".join(self.metadata["fpath_tuple"]) 
        elif self.metadata_template == "coderagbench-swebench-lite":
            self.chunk_expansion_metadata["filepath"] = self.metadata["title"]

        chunk_expansion = "'''\n"
        chunk_expansion += f"{self.chunk_expansion_metadata['filepath']}\n" if self.chunk_expansion_metadata["filepath"] else ""
        chunk_expansion += f"{self.chunk_expansion_metadata['ancestors']}\n" if self.chunk_expansion_metadata["ancestors"] else ""
        chunk_expansion += "'''"

        self.chunk_text = f"{chunk_expansion}\n{self.chunk_text}"

    def to_code_window(self) -> dict:
        """
        Convert the ASTChunk object into a code window for downstream integration.
        """
        if self.metadata_template == "coderagbench-swebench-lite":
            code_window = {
                "_id": self.metadata["_id"],
                "title": self.metadata['title'],
                "text": self.chunk_text
            }
        else:
            code_window = {
                "content": self.chunk_text,
                "metadata": self.metadata
            }

        return code_window
