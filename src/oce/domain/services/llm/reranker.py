"""LLM-based reranker using flash models for semantic understanding.

使用轻量级 LLM 对检索结果重新排序，弥补 embedding 模型的语义理解不足，
特别是中文查询 vs 英文文件名的跨语言匹配问题。
"""
from __future__ import annotations

import re

from loguru import logger

from oce.domain.services.llm.client import LLMClient
from oce.domain.services.llm.prompts import RERANK_SYSTEM_PROMPT, RERANK_USER_TEMPLATE


class LLMReranker:
    """基于 LLM 的语义重排序器。"""

    def __init__(
        self,
        client: LLMClient,
        model: str = "deepseek-v4-flash",
        max_candidates: int = 50,
        output_top_k: int = 10,
        snippet_chars: int = 400,
    ):
        """
        Args:
            client: LLM 客户端
            model: 模型名称
            max_candidates: 最多重排序的候选数量（控制成本）
            output_top_k: 输出的 Top-K 结果数量
            snippet_chars: 每个候选送入 LLM 的代码字符上限。只给路径会让重排
                退化成文件名匹配，符号定义和调用链查询无从判断。
        """
        self.client = client
        self.model = model
        self.max_candidates = max_candidates
        self.output_top_k = output_top_k
        self.snippet_chars = snippet_chars

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        使用 LLM 重新排序候选结果。

        Args:
            query: 用户查询
            candidates: 候选结果列表，每个结果需包含 'path' 字段
            top_k: 返回的结果数量，默认使用 output_top_k

        Returns:
            重新排序后的候选结果（保留原始数据结构）
        """
        if not candidates:
            return []

        top_k = top_k or self.output_top_k

        logger.info(f"LLM rerank called: query='{query}', candidates={len(candidates)}, top_k={top_k}")

        # 限制候选数量（控制成本和 token 长度）
        candidates_subset = candidates[: self.max_candidates]

        try:
            # 按下标而非路径回收顺序：同一文件可能贡献多个片段，
            # 用路径做键会把它们折叠成一条，符号级查询正需要区分片段。
            order = await self._llm_rerank(query, candidates_subset, top_k)
            reranked_results = [candidates_subset[i] for i in order]

            # LLM 返回不足时按原始顺序补齐，保证下游拿到足够候选
            if len(reranked_results) < top_k:
                chosen = set(order)
                remaining = [
                    c for i, c in enumerate(candidates_subset) if i not in chosen
                ]
                reranked_results.extend(remaining[: top_k - len(reranked_results)])

            return reranked_results[:top_k]

        except Exception as e:
            logger.warning(f"LLM rerank failed: {e}, falling back to original order")
            return candidates[:top_k]

    def _format_candidate(self, index: int, candidate: dict) -> str:
        """把候选渲染成带路径、行号和代码的 <candidate> 元素。

        只给路径时 LLM 无法判断符号定义或调用关系，必须附带片段正文。
        用闭合标签而非 markdown 围栏：候选可能是 .md 文件，其正文自带 ```，
        围栏方案会让 30 个候选的边界互相撕裂。
        """
        path = str(candidate.get("path", "")).replace('"', "&quot;")
        start = candidate.get("start_line")
        end = candidate.get("end_line")
        lines_attr = f' lines="{start}-{end}"' if start and end else ""

        snippet = (candidate.get("content") or "").strip()
        if len(snippet) > self.snippet_chars:
            snippet = snippet[: self.snippet_chars] + "\n…"
        # 正文里出现闭合标签会提前终止候选，必须中和
        snippet = snippet.replace("</candidate", "<\\/candidate")

        open_tag = f'<candidate id="{index}" path="{path}"{lines_attr}>'
        if not snippet:
            return f"{open_tag}</candidate>"
        return f"{open_tag}\n{snippet}\n</candidate>"

    async def _llm_rerank(
        self, query: str, candidates: list[dict], top_k: int
    ) -> list[int]:
        """
        调用 LLM API 进行重排序。

        Returns:
            重排后的候选下标列表（0-based，已去重且落在候选范围内）
        """
        candidates_text = "\n".join(
            self._format_candidate(i + 1, c) for i, c in enumerate(candidates)
        )

        messages = [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": RERANK_USER_TEMPLATE.format(
                    query=query,
                    count=len(candidates),
                    candidates=candidates_text,
                    top_k=top_k,
                ),
            },
        ]

        response = await self.client.chat(messages, model=self.model, temperature=0.1)

        order: list[int] = []
        seen: set[int] = set()

        for line in response.strip().splitlines():
            # 容忍 "1"、"1."、"- 1"、"[1] path" 等多种回复变体，只取首个整数
            match = re.search(r"\d+", line)
            if match is None:
                continue
            index = int(match.group()) - 1
            if 0 <= index < len(candidates) and index not in seen:
                seen.add(index)
                order.append(index)

        order = order[:top_k]

        if not order:
            logger.warning(
                f"LLM returned no valid indices. Raw response: {response[:200]}"
            )
            return list(range(min(top_k, len(candidates))))

        return order
