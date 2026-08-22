"""Quarto .qmd parsing (spec §5.1).

Produces a flat list of Blocks per document: headings, paragraphs, lists, code blocks,
tables (with captions), equations, figure captions. Quarto directives are stripped:
div fences, shortcodes, hidden computation cells, LaTeX-only macros, index entries.
Cross-references are resolved in a second pass (`resolve_crossrefs`) once labels for the
whole corpus are known.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

BlockKind = Literal["heading", "para", "list", "code", "table", "equation", "figure"]

ATOMIC_KINDS = {"code", "table", "equation", "figure"}


@dataclass
class Block:
    kind: BlockKind
    text: str
    char_start: int
    char_end: int
    level: int = 0  # headings only
    anchor: str | None = None  # sec-/fig-/tbl-/eq- id if any


@dataclass
class Document:
    source_file: str
    title: str
    anchor: str | None
    blocks: list[Block] = field(default_factory=list)


# ---- chapter ordering -------------------------------------------------------------


def chapter_files_from_quarto_config(config_path: Path) -> list[str]:
    """Return the `book.chapters` entries (relative to the quarto root) in order, flattening parts."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    out: list[str] = []

    def walk(items: list) -> None:
        for it in items:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict):
                if "part" in it and isinstance(it["part"], str) and it["part"].endswith(".qmd"):
                    out.append(it["part"])
                if "chapters" in it:
                    walk(it["chapters"])
                if "file" in it:
                    out.append(it["file"])

    book = cfg.get("book", {})
    walk(book.get("chapters", []))
    walk(book.get("appendices", []))
    return out


# ---- inline cleanup ---------------------------------------------------------------

_RE_INDEX = re.compile(r"\\index\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
_RE_INLINE_PY = re.compile(r"`\{python\}[^`]*`")
_RE_FOOTNOTE_REF = re.compile(r"\[\^[^\]]+\]")
_RE_CITE = re.compile(r"\s?\[-?@[^\]]+\]")
_RE_SHORTCODE = re.compile(r"\{\{<.*?>\}\}")
_RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RE_LATEX_CMD_LINE = re.compile(
    r"^\s*\\(newpage|noindent|chapterminitoc|clearpage|vspace|mlsysstack|begin\{marginfigure\}|end\{marginfigure\}|begin\{figure\*?\}|end\{figure\*?\}|centering)\b.*$",
    re.M,
)
_RE_IMAGE = re.compile(r"!\[(?P<cap>[^\]]*)\]\([^)]*\)(?:\{(?P<attrs>[^}]*)\})?")
_RE_ATTR = re.compile(r'([\w-]+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))')
_RE_TRAILING_ATTRS = re.compile(r"\s*\{([^{}]*)\}\s*$")
_RE_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_RE_SPAN_ATTR = re.compile(r"\[([^\]]+)\]\{[^}]*\}")


def clean_inline(text: str) -> str:
    text = _RE_HTML_COMMENT.sub("", text)
    text = _RE_INDEX.sub("", text)
    text = _RE_SHORTCODE.sub("", text)
    text = _RE_INLINE_PY.sub("[value]", text)
    text = _RE_FOOTNOTE_REF.sub("", text)
    text = _RE_CITE.sub("", text)
    text = _RE_SPAN_ATTR.sub(r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_attrs(attr_str: str) -> tuple[str | None, list[str], dict[str, str]]:
    """'#id .cls key=\"v\"' -> (id, [cls], {key: v})."""
    ident = None
    classes: list[str] = []
    kv: dict[str, str] = {}
    for m in _RE_ATTR.finditer(attr_str):
        kv[m.group(1)] = next(g for g in m.groups()[1:] if g is not None)
    rest = _RE_ATTR.sub("", attr_str)
    for tok in rest.split():
        if tok.startswith("#"):
            ident = tok[1:]
        elif tok.startswith("."):
            classes.append(tok[1:])
    return ident, classes, kv


# ---- block parser -----------------------------------------------------------------

_RE_FENCE_OPEN = re.compile(r"^(`{3,}|~{3,})\s*(\{([^}]*)\}|(\S+))?\s*$")
_RE_DIV_OPEN = re.compile(r"^(:{3,})\s*(\{(.*)\}|\S+)?\s*$")
_RE_DIV_CLOSE = re.compile(r"^(:{3,})\s*$")
_RE_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_RE_TABLE_CAPTION = re.compile(r"^:\s+(.*)$")
_RE_EQ_ONE_LINE = re.compile(r"^\$\$(.+)\$\$\s*(\{[^}]*\})?\s*$")
_RE_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_RE_FOOTNOTE_DEF = re.compile(r"^\[\^[^\]]+\]:\s*(.*)$")


@dataclass
class _DivFrame:
    dropped: bool
    depth: int
    fig_id: str | None = None
    caption: str | None = None
    callout_title: str | None = None
    start: int = 0


class QmdParser:
    def __init__(
        self,
        drop_code_langs: list[str] | None = None,
        drop_div_classes: list[str] | None = None,
        drop_div_attrs: list[str] | None = None,
    ) -> None:
        self.drop_code_langs = set(drop_code_langs or [])
        self.drop_div_classes = set(drop_div_classes or [])
        self.drop_div_attrs = list(drop_div_attrs or [])

    # -- helpers ------------------------------------------------------------------
    def _div_dropped(self, attr_str: str) -> bool:
        _, classes, _kv = parse_attrs(attr_str)
        if any(c in self.drop_div_classes for c in classes):
            return True
        return any(a in attr_str for a in self.drop_div_attrs)

    # -- main ---------------------------------------------------------------------
    def parse(self, text: str, source_file: str = "<memory>") -> Document:
        # strip YAML front matter
        body_start = 0
        if text.startswith("---"):
            m = re.search(r"^---\s*$", text[3:], re.M)
            if m:
                body_start = 3 + m.end()
        text_wo_comments = _RE_HTML_COMMENT.sub(lambda m: " " * len(m.group(0)), text)
        lines = text_wo_comments.splitlines(keepends=True)
        offsets: list[int] = []
        pos = 0
        for ln in lines:
            offsets.append(pos)
            pos += len(ln)

        doc = Document(source_file=source_file, title="", anchor=None)
        divs: list[_DivFrame] = []
        i = 0
        n = len(lines)

        def dropped() -> bool:
            return any(d.dropped for d in divs)

        para: list[str] = []
        para_start = 0
        para_is_list = False

        def flush_para(end_line: int) -> None:
            nonlocal para, para_is_list
            if para:
                txt = clean_inline(
                    " ".join(s.strip() for s in para)
                    if not para_is_list
                    else "\n".join(s.rstrip() for s in para)
                )
                txt = _RE_LATEX_CMD_LINE.sub("", txt).strip()
                if txt:
                    doc.blocks.append(
                        Block(
                            kind="list" if para_is_list else "para",
                            text=txt,
                            char_start=offsets[para_start],
                            char_end=offsets[end_line - 1] + len(lines[end_line - 1])
                            if end_line > 0
                            else offsets[para_start],
                        )
                    )
            para = []
            para_is_list = False

        while i < n:
            raw = lines[i]
            if offsets[i] < body_start:
                i += 1
                continue
            line = raw.rstrip("\n")
            stripped = line.strip()

            # fenced code
            m = _RE_FENCE_OPEN.match(stripped)
            if m:
                flush_para(i)
                fence = m.group(1)
                braced = (
                    m.group(3) is not None
                )  # {python} cell / {.tikz} / {=html}: Quarto-executed or raw
                lang = (m.group(3) or m.group(4) or "").strip()
                lang_name = lang.lstrip(".").split()[0].lower() if lang else ""
                start = i
                i += 1
                body: list[str] = []
                while i < n and not lines[i].rstrip("\n").strip().startswith(fence[0] * len(fence)):
                    body.append(lines[i].rstrip("\n"))
                    i += 1
                end = min(i, n - 1)
                i += 1
                if dropped() or (
                    braced and (lang_name in self.drop_code_langs or lang_name.startswith("="))
                ):
                    continue  # plain ```lang fences are display code and always kept
                code = "\n".join(body).rstrip()
                if code:
                    hdr = f"```{lang_name}\n" if lang_name else "```\n"
                    doc.blocks.append(
                        Block(
                            "code",
                            hdr + code + "\n```",
                            offsets[start],
                            offsets[end] + len(lines[end]),
                        )
                    )
                continue

            # div open / close
            m = _RE_DIV_OPEN.match(stripped)
            if m and (m.group(2) or m.group(1)):
                colons = m.group(1)
                attr = (m.group(3) if m.group(3) is not None else (m.group(2) or "")) or ""
                if m.group(2) is None:  # bare ':::' => close
                    flush_para(i)
                    if divs:
                        divs.pop()
                    i += 1
                    continue
                flush_para(i)
                ident, classes, kv = parse_attrs(attr)
                frame = _DivFrame(dropped=self._div_dropped(attr), depth=len(colons), start=i)
                if ident and ident.startswith("fig-"):
                    frame.fig_id = ident
                    frame.caption = kv.get("fig-cap") or kv.get("fig-alt")
                    if frame.caption and not dropped():
                        doc.blocks.append(
                            Block(
                                "figure",
                                f"Figure ({ident}): {clean_inline(frame.caption)}",
                                offsets[i],
                                offsets[i] + len(raw),
                                anchor=ident,
                            )
                        )
                if (
                    any(c.startswith("callout-") for c in classes)
                    and kv.get("title")
                    and not frame.dropped
                    and not dropped()
                ):
                    doc.blocks.append(
                        Block(
                            "para",
                            f"**{clean_inline(kv['title'])}.**",
                            offsets[i],
                            offsets[i] + len(raw),
                        )
                    )
                if any(c.startswith("callout-") for c in classes) and ident and not frame.dropped:
                    frame.callout_title = kv.get("title")
                divs.append(frame)
                i += 1
                continue
            if _RE_DIV_CLOSE.match(stripped):
                flush_para(i)
                if divs:
                    divs.pop()
                i += 1
                continue

            if dropped():
                i += 1
                continue

            # heading
            m = _RE_HEADING.match(line)
            if m:
                flush_para(i)
                level = len(m.group(1))
                title = m.group(2)
                anchor = None
                am = _RE_TRAILING_ATTRS.search(title)
                if am:
                    anchor, _cls, _kv = parse_attrs(am.group(1))
                    title = title[: am.start()].strip()
                title = clean_inline(title)
                if level == 1 and not doc.title:
                    doc.title = title
                    doc.anchor = anchor
                doc.blocks.append(
                    Block(
                        "heading",
                        title,
                        offsets[i],
                        offsets[i] + len(raw),
                        level=level,
                        anchor=anchor,
                    )
                )
                i += 1
                continue

            # table
            if _RE_TABLE_LINE.match(line):
                flush_para(i)
                start = i
                rows: list[str] = []
                while i < n and _RE_TABLE_LINE.match(lines[i].rstrip("\n")):
                    rows.append(clean_inline(lines[i].rstrip("\n")))
                    i += 1
                # optional blank line(s) then caption
                j = i
                while j < n and not lines[j].strip():
                    j += 1
                cap = None
                anchor = None
                if j < n:
                    cm = _RE_TABLE_CAPTION.match(lines[j].rstrip("\n"))
                    if cm:
                        cap_txt = cm.group(1)
                        am = _RE_TRAILING_ATTRS.search(cap_txt)
                        if am:
                            anchor, _c, _k = parse_attrs(am.group(1))
                            cap_txt = cap_txt[: am.start()]
                        cap = clean_inline(cap_txt)
                        i = j + 1
                end = i - 1
                txt = "\n".join(rows)
                if cap:
                    txt = f"Table ({anchor}): {cap}\n{txt}" if anchor else f"Table: {cap}\n{txt}"
                doc.blocks.append(
                    Block(
                        "table", txt, offsets[start], offsets[end] + len(lines[end]), anchor=anchor
                    )
                )
                continue

            # display equation
            if stripped.startswith("$$"):
                flush_para(i)
                start = i
                m1 = _RE_EQ_ONE_LINE.match(stripped)
                if m1:
                    body_eq = m1.group(1).strip()
                    anchor = parse_attrs(m1.group(2)[1:-1])[0] if m1.group(2) else None
                    i += 1
                else:
                    i += 1
                    eq_lines: list[str] = []
                    anchor = None
                    while i < n:
                        s = lines[i].rstrip("\n").strip()
                        if s.startswith("$$"):
                            tail = s[2:].strip()
                            am = _RE_TRAILING_ATTRS.search(tail)
                            if am:
                                anchor = parse_attrs(am.group(1))[0]
                            i += 1
                            break
                        eq_lines.append(s)
                        i += 1
                    body_eq = "\n".join(eq_lines).strip()
                end = i - 1
                txt = f"$$ {body_eq} $$" + (f" ({anchor})" if anchor else "")
                doc.blocks.append(
                    Block(
                        "equation",
                        txt,
                        offsets[start],
                        offsets[end] + len(lines[end]),
                        anchor=anchor,
                    )
                )
                continue

            # standalone image with caption
            if stripped.startswith("!["):
                flush_para(i)
                im = _RE_IMAGE.search(stripped)
                if im:
                    cap = im.group("cap")
                    anchor = None
                    if im.group("attrs"):
                        anchor, _c, kv = parse_attrs(im.group("attrs"))
                        cap = cap or kv.get("fig-cap") or kv.get("fig-alt") or ""
                    cap = clean_inline(cap)
                    if cap and not any(d.fig_id for d in divs):
                        label = f"Figure ({anchor}): " if anchor else "Figure: "
                        doc.blocks.append(
                            Block(
                                "figure",
                                label + cap,
                                offsets[i],
                                offsets[i] + len(raw),
                                anchor=anchor,
                            )
                        )
                i += 1
                continue

            # latex-only lines
            if _RE_LATEX_CMD_LINE.match(line):
                i += 1
                continue

            # footnote definition -> its own paragraph
            fm = _RE_FOOTNOTE_DEF.match(line)
            if fm:
                flush_para(i)
                para = [fm.group(1)]
                para_start = i
                i += 1
                while (
                    i < n
                    and lines[i].strip()
                    and not _RE_HEADING.match(lines[i])
                    and lines[i].startswith("    ")
                ):
                    para.append(lines[i])
                    i += 1
                flush_para(i)
                continue

            # blank line => paragraph boundary
            if not stripped:
                flush_para(i)
                i += 1
                continue

            # list / paragraph accumulation
            is_item = bool(_RE_LIST_ITEM.match(line))
            if not para:
                para_start = i
                para_is_list = is_item
            elif is_item and not para_is_list:
                flush_para(i)
                para_start = i
                para_is_list = True
            para.append(line)
            i += 1

        flush_para(n)
        return doc


# ---- cross-reference resolution -----------------------------------------------------

_RE_XREF = re.compile(r"(?<![\w@])@([A-Za-z]+-[\w-]+)")


def collect_labels(docs: list[Document]) -> dict[str, str]:
    """anchor id -> human label, for @sec-/@fig-/@tbl-/@eq- resolution."""
    labels: dict[str, str] = {}
    for d in docs:
        for b in d.blocks:
            if not b.anchor:
                continue
            if b.kind == "heading":
                labels[b.anchor] = f"§ {b.text}"
            elif b.kind == "figure":
                labels[b.anchor] = f"Figure [{b.anchor}]"
            elif b.kind == "table":
                labels[b.anchor] = f"Table [{b.anchor}]"
            elif b.kind == "equation":
                labels[b.anchor] = f"Equation [{b.anchor}]"
    return labels


def resolve_crossrefs(docs: list[Document], labels: dict[str, str]) -> None:
    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        low = key[0].lower() + key[1:]
        return labels.get(key) or labels.get(low) or m.group(0)

    for d in docs:
        for b in d.blocks:
            if b.kind in ("para", "list", "figure", "table"):
                b.text = _RE_XREF.sub(sub, b.text)


def parse_file(path: Path, parser: QmdParser, source_file: str | None = None) -> Document:
    return parser.parse(path.read_text(encoding="utf-8"), source_file or str(path))
