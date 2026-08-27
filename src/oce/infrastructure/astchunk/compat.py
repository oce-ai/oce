"""tree-sitter 0.25+ API 兼容层（修复 P0：access violation）。

新版 tree-sitter (通过 tree_sitter_language_pack) 的 Node/Tree API 全部变成了方法调用，
且 Node 不再有 .text / .children / .type 等属性。

**P0 根因（已修复）**：
原版 CompatNode 对所有属性（type/start_byte/start_point/text）都惰性访问原生 Node 对象，
但在 AST 遍历过程中原生 Node 会失效，导致读野内存 → Windows fatal exception: access violation。
Python 的 try/except 无法捕获此类原生层段错误。

**修复方案**：
构造时立即快照所有标量（type/start_byte/end_byte/start_point/end_point/child_count）为 Python 对象，
之后不再触碰原生 Node。同时持有 _tree 引用防 GC（虽然实验表明不持有也崩，但保险起见保留）。
text 属性通过快照的字节范围从 _source 切片，避免调用原生 .text 方法。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional


class _Point:
    """tree-sitter Point 的轻量快照（row, column）。"""

    __slots__ = ("row", "column")

    def __init__(self, row: int, column: int):
        self.row = row
        self.column = column

    def __repr__(self) -> str:
        return f"Point(row={self.row}, column={self.column})"


class CompatNode:
    """包装新版 tree-sitter Node，暴露旧版属性接口。
    
    **关键改动（P0 修复）**：
    - 构造时 eager 快照所有标量为 Python int/str，彻底脱离原生内存
    - 持有 _tree 引用防 GC（保险措施）
    - text 通过 _source 切片而不是调用原生 .text
    """

    __slots__ = (
        "_child_refs",  # 父节点有效时一次性取得的原生子节点
        "_source",      # bytes：完整源码
        "_tree",        # 持有 tree 引用防 GC
        # --- 以下为 eager 快照的标量 ---
        "type",         # str
        "start_byte",   # int
        "end_byte",     # int
        "start_point",  # _Point
        "end_point",    # _Point
        "is_named",     # bool
        "_field_names",  # tuple[str | None, ...]：与 _child_refs 同序
        "_children_cache",  # Optional[list[CompatNode]]：延迟构造+缓存
    )

    def __init__(self, node, source: bytes, tree):
        """构造时立即提取所有标量，子节点按需构造。

        Args:
            node: 原生 tree-sitter Node
            source: 完整源码 bytes（用于 text 切片）
            tree: 原生 Tree 对象（持有引用防 GC）
        """
        self._source = source
        self._tree = tree
        self._children_cache = None  # 延迟初始化

        # Eager 快照：立即读取原生对象的所有标量属性
        self.type = node.type
        self.start_byte = node.start_byte
        self.end_byte = node.end_byte
        self.is_named = node.is_named

        # Point 也要快照（原生 Point 对象同样可能失效）
        sp = node.start_point
        self.start_point = _Point(sp.row, sp.column)
        ep = node.end_point
        self.end_point = _Point(ep.row, ep.column)

        # tree-sitter 0.26 在 Windows 上反复调用 node.child(i) 可能返回失效
        # 的临时 Node。父节点仍有效时一次性取得 children，可稳定其生命周期。
        self._child_refs = tuple(node.children)
        # 字段名同样必须在此刻取：它只能通过仍然有效的父节点游标查询，
        # 快照之后原生节点不可再访问。与 _child_refs 保持同序。
        # field_name_for_child 在 0.25 之前不存在，缺失时降级为无字段信息，
        # 调用方会退回按节点类型判断。
        field_lookup = getattr(node, "field_name_for_child", None)
        if field_lookup is None:
            self._field_names = (None,) * len(self._child_refs)
        else:
            self._field_names = tuple(
                field_lookup(index) for index in range(len(self._child_refs))
            )

    @property
    def text(self) -> bytes:
        """通过快照的字节范围切片，不调用原生 .text。"""
        return self._source[self.start_byte:self.end_byte]

    @property
    def children(self) -> list[CompatNode]:
        """按需构造子节点列表（首次访问时构造，之后从缓存返回）。

        原生子节点已在父节点构造时取得；这里只创建 Python 快照。
        """
        if self._children_cache is None:
            self._children_cache = [
                CompatNode(child, self._source, self._tree)
                for child in self._child_refs
            ]
            self._child_refs = ()
        return self._children_cache

    @property
    def named_children(self) -> list[CompatNode]:
        """跳过标点等匿名节点，只保留语法上有意义的子节点。"""
        return [child for child in self.children if child.is_named]

    def child_by_field_name(self, field: str) -> CompatNode | None:
        """按字段名取子节点，语义对齐原生 tree-sitter 接口。

        字段名在构造时快照（原生节点此后不可访问），与 children 同序。
        """
        for child, name in zip(self.children, self._field_names):
            if name == field:
                return child
        return None


class CompatTree:
    """包装新版 tree-sitter Tree，暴露旧版属性接口。"""

    __slots__ = ("_tree", "_source")

    def __init__(self, tree, source: bytes):
        self._tree = tree
        self._source = source

    @property
    def root_node(self) -> CompatNode:
        """返回根节点的 CompatNode 包装（带 eager 快照）。"""
        return CompatNode(self._tree.root_node, self._source, self._tree)


def compat_parse(parser, code: str) -> CompatTree:
    """用新版 parser 解析代码，返回兼容旧版接口的 CompatTree。

    tree-sitter 0.25+ 的 parse() 只接受 bytes（0.25 曾接受 str，0.26 改回 bytes）。
    本函数统一接受 str，内部编码为 bytes 再解析。
    
    返回的 CompatTree.root_node 及其所有子孙节点都已 eager 快照标量，
    避免 P0 的 access violation。
    """
    source_bytes = code.encode("utf-8")
    tree = parser.parse(source_bytes)
    return CompatTree(tree, source_bytes)
