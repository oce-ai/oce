"""Line-span helpers shared by the chunkers.

A span is a 1-based inclusive line range paired with the exact text of those
lines. Keeping text and range in one value makes the invariant checkable: a
chunk must render at the line numbers it claims, because the formatter prints
``start_line + offset`` for each of its lines.

Capping keeps chunk text under a character budget. Without it the embedding
client re-splits long inputs and pools the pieces, and the vector store
truncates its text field, so a chunk would be indexed under content nobody can
reconstruct from the reported line range.

A line longer than the whole budget cannot be capped without breaking that
invariant, so it is dropped instead. Such lines are minified bundles or
generated single-line payloads; a mid-token slice of one is not something a
reader can act on, and emitting several pieces under the same line number makes
every one of them misreport its source range.
"""

from __future__ import annotations

# (start_line, end_line, text) with 1-based inclusive line numbers.
Span = tuple[int, int, str]


def slice_lines(lines: list[str], start_line: int, end_line: int) -> str:
    """Return the verbatim text of a 1-based inclusive line range."""
    return "\n".join(lines[start_line - 1 : end_line])


def trim_trailing_blank_lines(lines: list[str], start_line: int, end_line: int) -> int:
    """Pull ``end_line`` back over trailing blank lines.

    ``"\\n".join`` drops the final empty element, so a range ending on blank
    lines yields text with fewer lines than the range claims.
    """
    while end_line > start_line and not lines[end_line - 1].strip():
        end_line -= 1
    return end_line


def cap_span(
    lines: list[str],
    start_line: int,
    end_line: int,
    max_chars: int,
) -> list[Span]:
    """Split one line range into spans whose text fits ``max_chars``.

    Splits on line boundaries only. Lines that exceed the budget on their own
    are skipped, so every returned span's text equals the source lines it
    claims. Skipping one breaks the buffer, which is why the surrounding lines
    come back as separate spans rather than one range straddling the gap.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    spans: list[Span] = []
    buffer: list[str] = []
    current_start = start_line
    length = 0

    for line_no in range(start_line, end_line + 1):
        line = lines[line_no - 1]
        if len(line) > max_chars:
            if buffer:
                spans.append((current_start, line_no - 1, "\n".join(buffer)))
                buffer = []
                length = 0
            current_start = line_no + 1
            continue
        addition = len(line) + (1 if buffer else 0)
        if buffer and length + addition > max_chars:
            spans.append((current_start, line_no - 1, "\n".join(buffer)))
            buffer = []
            length = 0
            current_start = line_no
            addition = len(line)
        buffer.append(line)
        length += addition

    if buffer:
        spans.append((current_start, end_line, "\n".join(buffer)))
    return spans
