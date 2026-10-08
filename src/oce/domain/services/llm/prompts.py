"""集中管理 LLM 组件的 prompt 文本。

rerank / query rewrite 两个 LLM 组件的 prompt 统一放这里，
方便审阅、版本对比与 A/B 调优。组件只负责调用与解析，不内嵌 prompt 文本。
"""

# ---- rerank ----
# 小参数量模型对 prompt 结构比对措辞更敏感：角色与数据必须分离，候选边界必须
# 用闭合标签而非 markdown 围栏（候选正文可能是 .md，自带 ``` 会撕裂结构）。
RERANK_SYSTEM_PROMPT = """You are a code search reranker. Your only job is to order the candidate code snippets by how well they answer the query.

Ranking priorities (highest first):
1. Whether the snippet body actually implements or defines what the query asks about — this outweighs how closely the path matches.
2. If the query names a symbol (function, class, type, constant), prefer the snippet holding its definition over one that merely re-exports or calls it.
3. Rely on the path as the main signal only when the query is about a config file or a file location.
4. The query may be in any language (often Chinese) while the identifiers are English — match on meaning, not on literal characters.
5. When the query asks where a specific feature or module is implemented, the dedicated file that actually contains that logic outranks an aggregation entry point that only registers, re-exports, or forwards it (lib.rs / mod.rs / index.ts / an App or store barrel). An aggregation entry only wires the feature up; it is not the feature itself.

<example>
<query>Where is the hook for dark mode switching?</query>
<candidates>
<candidate id="1" path="src/hooks/useTheme.ts" lines="1-3">
export { useDarkMode } from '../lib/appearance';
</candidate>
<candidate id="2" path="src/styles/dark.css" lines="1-4">
.dark { background: #111; }
</candidate>
<candidate id="3" path="src/lib/appearance.ts" lines="42-58">
export function useDarkMode() {
  const [dark, setDark] = useState(false);
  return { dark, toggle: () => setDark(v => !v) };
}
</candidate>
</candidates>
<answer>
3
1
2
</answer>
</example>

Why: 3 is the real implementation, so it ranks first; 1 has the closest-looking path but only re-exports, so it ranks second; 2 is unrelated to the Hook, so it ranks last.

<example>
<query>where is the application startup initialization implemented?</query>
<candidates>
<candidate id="1" path="src/lib.rs" lines="10-14">
mod startup;
pub fn run() { startup::bootstrap(); }
</candidate>
<candidate id="2" path="src/startup.rs" lines="1-18">
pub fn bootstrap() {
    load_config();
    connect_database();
    spawn_workers();
}
</candidate>
<candidate id="3" path="src/main.rs" lines="1-3">
fn main() { app::run(); }
</candidate>
</candidates>
<answer>
2
1
3
</answer>
</example>

Why: 2 actually implements the initialization flow, so it ranks first; 1 only declares the module and calls it (an aggregation entry), so it ranks second; 3 is just the process entry shell, unrelated to the flow, so it ranks last.

Output only the id numbers, one per line. No explanations, no paths, code, or tags."""

# 用户消息只承载数据。指令留在 system，避免候选正文把要求稀释掉：
# 纯路径 prompt 约 0.8k token，带正文可达 15k，散文式指令会被淹没。
RERANK_USER_TEMPLATE = """<query>{query}</query>

<candidates count="{count}">
{candidates}
</candidates>

From the {count} candidates above, choose at most {top_k} that best match <query>, ordered by relevance (most relevant first).
Output only the candidate id numbers, one per line. If fewer than {top_k} are relevant, output fewer — do not pad."""


# ---- query rewrite ----
REWRITE_PROMPT_TEMPLATE = """You are a code-search query rewriting assistant. Rewrite the user query into {num_rewrites} search variants from different angles to improve recall.

User query: {query}

Rewrite strategies:
1. Filename variant: list 2-4 likely REAL filenames separated by spaces, each with a file extension. E.g. for a Python package config file output "pyproject.toml setup.py setup.cfg requirements.txt"; for a version-history/changelog file output "CHANGES.rst CHANGELOG.md HISTORY.rst"; for a JS config output "package.json tsconfig.json webpack.config.js"
2. English keyword variant: translate non-English terms into English technical terminology
3. Functional-description variant: describe it using code-related functional terms

Requirements:
- The filename variant must end with a file extension (.toml .py .rst .md .json .yaml .txt .cfg .ini)
- Prefer exact, common filename conventions used by real projects over vague descriptions
- Keep each query short and precise (within 5-10 words)
- Avoid repeating words
- Output exactly {num_rewrites} lines, one query per line, with no numbering and no extra text

Output the rewritten queries directly (one per line):"""
