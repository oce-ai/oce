"""Diversity- and budget-aware retrieval result selection."""

from __future__ import annotations

from collections import Counter

from oce.domain.services.search import SearchHit, search_hit_key


class CoverageSelector:
    """Prefer repository coverage while suppressing overlapping source spans."""

    def __init__(
        self,
        *,
        max_per_path: int = 2,
        max_chars: int = 32_000,
        overlap_threshold: float = 0.6,
    ) -> None:
        if max_per_path < 1:
            raise ValueError("max_per_path must be positive")
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        if not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be between zero and one")
        self.max_per_path = max_per_path
        self.max_chars = max_chars
        self.overlap_threshold = overlap_threshold

    async def select(self, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if top_k <= 0 or not hits:
            return []

        selected: list[SearchHit] = []
        path_counts: Counter[str] = Counter()
        seen: set[tuple[str, str, int, int, str]] = set()
        used_chars = 0

        # 贪心填充策略：
        # - 优先保证仓库覆盖度（第一轮每个文件各选一个）
        # - 字符预算为硬限制，跳过放不下的大片段，继续尝试小片段
        # - top_k 为软上限，实际返回数量可能更少（受预算和文件数约束）
        for prefer_new_path in (True, False):
            for hit in hits:
                # 达到数量上限：继续尝试（可能有更小的片段能塞进预算）
                if len(selected) >= top_k:
                    continue

                if prefer_new_path != (path_counts[hit.path] == 0):
                    continue
                if path_counts[hit.path] >= self.max_per_path:
                    continue
                key = search_hit_key(hit)
                if key in seen or self._overlaps_selected(hit, selected):
                    continue

                # 字符预算检查：放不下就跳过，继续尝试后面的小片段
                hit_chars = len(hit.content)
                if selected and used_chars + hit_chars > self.max_chars:
                    continue

                selected.append(hit)
                seen.add(key)
                path_counts[hit.path] += 1
                used_chars += hit_chars
        return selected

    def _overlaps_selected(
        self,
        candidate: SearchHit,
        selected: list[SearchHit],
    ) -> bool:
        for hit in selected:
            if hit.path != candidate.path:
                continue
            overlap = max(
                0,
                min(hit.end_line, candidate.end_line)
                - max(hit.start_line, candidate.start_line)
                + 1,
            )
            shorter = min(
                hit.end_line - hit.start_line + 1,
                candidate.end_line - candidate.start_line + 1,
            )
            if overlap / shorter >= self.overlap_threshold:
                return True
        return False
