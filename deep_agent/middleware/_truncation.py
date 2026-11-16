"""大内容裁剪：字符预算、行内裁剪和头尾预览。

对齐 deepagents 的 ``middleware/_message_eviction.py`` 与
``backends/utils.py``：不静默截断，而是**明说**丢了什么 ——
丢整行时留一行 ``... [N lines truncated] ...`` 标记，行太长时说明
"每行只显示前 M 个字符"。

为什么要有这一层：模型上下文是有限的。一次 ``read_file`` 或一次
``grep`` 就可能吐出几十万字符，直接塞进请求会让整个请求被 provider
拒绝，或者把有用的历史挤掉。裁剪必须满足两条：

  1. **可恢复**：被裁掉的内容有地方存着，模型可以再读回来
     （deepagents 把它落到 backend 的 ``/large_tool_results/``）。
  2. **不撒谎**：裁剪后的文本要能看出少了什么，不能让人误以为
     看到的就是全部。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: 粗略的"每 token 多少字符"。保守取值（偏大），宁可少裁一点，
#: 也不要因为估算过于激进而丢掉本来还塞得下的内容。
NUM_CHARS_PER_TOKEN: Final = 4

#: 预览里单行的字符上限。挡的是"行数很少但每行极长"的形态
#: （minified 的 JS、一行一条记录的 jsonl）。这种内容按行裁剪没有
#: 意义，只能切字符。
PREVIEW_LINE_CHAR_LIMIT: Final = 1000

#: 预览保留的头部/尾部行数。
PREVIEW_HEAD_LINES: Final = 5
PREVIEW_TAIL_LINES: Final = 5

#: 中间被丢掉的行的标记模板。
TRUNCATION_MARKER_TEMPLATE: Final = "... [{omitted_lines} lines truncated] ..."


def char_budget(token_limit: int | None) -> int | None:
    """token 上限 -> 字符预算；``None``/非正数表示不限制。"""
    if not token_limit:
        return None
    return NUM_CHARS_PER_TOKEN * int(token_limit)


@dataclass(frozen=True, slots=True)
class ContentPreview:
    """预览文本 + 记录"丢了什么"的标记。

    两个标记是**独立的**：预览可能整行被丢（``lines_omitted``），
    也可能只是某一行太长被切（``lines_clipped``），两者可以同时成立。
    标记由构造它的代码显式给出，不从渲染后的文本反推 ——
    否则内容里恰好有一行长得像标记的正常文本就会造成误判。
    """

    text: str
    """渲染好的预览文本（带行号）。"""

    lines_omitted: bool
    """中间整行被丢掉了（预览里有截断标记）。"""

    lines_clipped: bool
    """至少有一行被切到了 ``PREVIEW_LINE_CHAR_LIMIT``。"""


def _clip_lines(lines: list[str]) -> tuple[list[str], bool]:
    """按行裁剪；返回 (裁剪后的行, 是否裁过)。"""
    clipped = [line[:PREVIEW_LINE_CHAR_LIMIT] for line in lines]
    return clipped, any(len(line) > PREVIEW_LINE_CHAR_LIMIT for line in lines)


def format_content_with_line_numbers(lines: list[str], start_line: int = 1) -> str:
    """给每行加行号 —— 模型据此知道自己看到的是文件的哪一段。"""
    return "\n".join(
        f"{number}. {line}" for number, line in enumerate(lines, start=start_line))


def create_content_preview(content: str, *, head_lines: int = PREVIEW_HEAD_LINES,
                           tail_lines: int = PREVIEW_TAIL_LINES,
                           max_chars: int | None = None) -> ContentPreview:
    """生成带行号的头尾预览。

    行数没超过 ``head_lines + tail_lines`` 时全给；否则给头尾两段，
    中间用一行标记隔开。

    ``max_chars`` 是预览文本的字符上限。给定时，头尾各分一半预算，
    行数按预算往下收 —— **不能先拼好再从尾部砍**，那样会把尾段
    （往往是最有信息量的最后几行）整个砍掉。

    Args:
        content: 完整内容。
        head_lines: 开头最多保留几行。
        tail_lines: 结尾最多保留几行。
        max_chars: 预览文本的字符上限。

    Returns:
        预览文本和"丢了什么"的标记。
    """
    lines = content.splitlines()

    if max_chars is not None:
        # 先按字符预算收窄头尾行数 —— 两边都要留，
        # 不能只顾头（尾段往往是结论所在）。
        head_lines, tail_lines = _fit_line_budget(
            lines, head_lines, tail_lines, max_chars)

    if len(lines) <= head_lines + tail_lines:
        shown, clipped = _clip_lines(lines)
        return ContentPreview(
            format_content_with_line_numbers(shown, start_line=1),
            lines_omitted=False,
            lines_clipped=clipped,
        )

    head, head_clipped = _clip_lines(lines[:head_lines])
    tail, tail_clipped = _clip_lines(lines[-tail_lines:])
    marker = TRUNCATION_MARKER_TEMPLATE.format(
        omitted_lines=len(lines) - head_lines - tail_lines)
    return ContentPreview(
        format_content_with_line_numbers(head, start_line=1)
        + f"\n{marker}\n"
        + format_content_with_line_numbers(tail, start_line=len(lines) - tail_lines + 1),
        lines_omitted=True,
        lines_clipped=head_clipped or tail_clipped,
    )


def _fit_line_budget(lines: list[str], head_lines: int, tail_lines: int,
                     max_chars: int) -> tuple[int, int]:
    """按字符预算收窄头尾的行数。

    头尾各拿一半预算；谁先装满谁停。返回 (头行数, 尾行数)。
    """
    head_room = max_chars // 2
    tail_room = max_chars - head_room
    marker_cost = len(TRUNCATION_MARKER_TEMPLATE.format(omitted_lines="0")) + 2

    def _take(pool: list[str], room: int) -> int:
        used, count = 0, 0
        for line in pool:
            cost = len(line) + len(str(count + 1)) + 3
            if used + cost > room:
                break
            used += cost
            count += 1
        return count

    head_count = min(head_lines, _take(lines, max(0, head_room - marker_cost)))
    tail_count = min(tail_lines, _take(lines[::-1], max(0, tail_room - marker_cost)))
    return head_count, tail_count


def preview_note(*, lines_omitted: bool, lines_clipped: bool = False,
                 subject: str = "result") -> str:
    """预览的引导句。

    只提**这次预览真的有**的损失 —— 不告诉模型去找一个其实没插进去的标记。
    """
    base = ("Here is a preview showing the head and tail of the {subject}"
            if lines_omitted else "Here is a preview of the {subject}")
    caveats = []
    if lines_omitted:
        caveats.append("lines of the form `... [N lines truncated] ...` "
                       "indicate omitted lines in the middle of the content")
    if lines_clipped:
        caveats.append(f"the output contains lines longer than "
                       f"{PREVIEW_LINE_CHAR_LIMIT} characters; this preview "
                       f"shows only their first {PREVIEW_LINE_CHAR_LIMIT} characters")
    note = base.format(subject=subject)
    if caveats:
        note += f" ({'; '.join(caveats)})"
    return f"{note}:"


def truncate_text(text: str, limit: int | None, *, marker: str = "...") -> str:
    """把文本硬裁到字符上限，并留一个可识别的尾标记。

    用于**没有恢复路径**的场景（比如 backend 写不进去时的兜底）。
    有恢复路径时应改用 :func:`create_content_preview`。
    """
    if limit is None or len(text) <= limit:
        return text
    return text[:max(0, limit - len(marker))] + marker


def truncate_lines(text: str, limit: int | None, *, marker: str = "...") -> str:
    """按行裁剪文本到字符上限：优先丢**尾部**的整行。

    按行裁而不是按字符裁，是为了不把一行劈成两半 ——
    半行的内容对模型基本没用，还会让"我看到的是第几行"变得不可信。
    """
    if limit is None or len(text) <= limit:
        return text
    lines = text.splitlines()
    kept: list[str] = []
    size = 0
    for line in lines:
        added = len(line) + 1
        if size + added > limit:
            break
        kept.append(line)
        size += added
    if not kept:
        return truncate_text(text, limit, marker=marker)
    omitted = len(lines) - len(kept)
    return "\n".join(kept) + f"\n{TRUNCATION_MARKER_TEMPLATE.format(omitted_lines=omitted)}"


__all__ = [
    "ContentPreview", "NUM_CHARS_PER_TOKEN", "PREVIEW_HEAD_LINES",
    "PREVIEW_LINE_CHAR_LIMIT", "PREVIEW_TAIL_LINES", "TRUNCATION_MARKER_TEMPLATE",
    "char_budget", "create_content_preview", "format_content_with_line_numbers",
    "preview_note", "truncate_lines", "truncate_text",
]
