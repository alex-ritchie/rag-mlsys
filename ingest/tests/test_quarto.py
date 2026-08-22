from pathlib import Path

from mlsys_ingest.quarto import (
    QmdParser,
    chapter_files_from_quarto_config,
    clean_inline,
    collect_labels,
    resolve_crossrefs,
)

FIX = Path(__file__).parent / "fixtures"


def _parser():
    return QmdParser(
        drop_code_langs=["python", "tikz", "=html"],
        drop_div_classes=["callout-learning-objectives"],
        drop_div_attrs=['when-format="pdf"'],
    )


def test_parse_widgets_structure():
    doc = _parser().parse((FIX / "widgets.qmd").read_text(), "widgets.qmd")
    assert doc.title == "Widget Systems"
    assert doc.anchor == "sec-widgets"
    kinds = [b.kind for b in doc.blocks]
    assert (
        "table" in kinds
        and "equation" in kinds
        and "figure" in kinds
        and "code" in kinds
        and "list" in kinds
    )
    headings = [b.text for b in doc.blocks if b.kind == "heading"]
    assert headings[:3] == ["Widget Systems", "Purpose", "Widget Sizing"]
    all_text = "\n".join(b.text for b in doc.blocks)
    # dropped things
    assert "HIDDEN_SECRET" not in all_text
    assert "must be dropped" not in all_text
    assert "PDF-only" not in all_text
    assert "Explain widget throughput" not in all_text  # learning objectives dropped
    assert "chapterminitoc" not in all_text and "\\newpage" not in all_text
    assert "\\index" not in all_text and "{{<" not in all_text
    assert "[@doe2020widgets" not in all_text
    assert "[^fn-pitch]" not in all_text
    # kept things
    assert "Tooth pitch" in all_text  # footnote definition kept as a paragraph
    assert "[value]" in all_text  # inline python replaced
    assert "Why pitch dominates" in all_text  # callout title kept
    assert "def widget_size" in all_text  # visible code kept
    table = next(b for b in doc.blocks if b.kind == "table")
    assert (
        table.anchor == "tbl-widget-classes"
        and "Widget Classes" in table.text
        and "| Micro" in table.text
    )
    eq = next(b for b in doc.blocks if b.kind == "equation")
    assert eq.anchor == "eq-sizing" and "S = G" in eq.text
    fig = next(b for b in doc.blocks if b.kind == "figure")
    assert fig.anchor == "fig-widget-flow" and "Widget Flow" in fig.text


def test_char_offsets_point_into_source():
    src = (FIX / "widgets.qmd").read_text()
    doc = _parser().parse(src, "widgets.qmd")
    for b in doc.blocks:
        assert 0 <= b.char_start < b.char_end <= len(src)
    h = next(b for b in doc.blocks if b.kind == "heading" and b.text == "Widget Sizing")
    assert "## Widget Sizing" in src[h.char_start : h.char_end]


def test_crossrefs_resolve_and_pass_through():
    p = _parser()
    docs = [p.parse((FIX / f).read_text(), f) for f in ("widgets.qmd", "gears.qmd")]
    resolve_crossrefs(docs, collect_labels(docs))
    gears = "\n".join(b.text for b in docs[1].blocks)
    assert "Table [tbl-widget-classes]" in gears
    assert "Equation [eq-sizing]" in gears
    assert "@sec-does-not-exist" in gears
    widgets = "\n".join(b.text for b in docs[0].blocks)
    assert "§ Widget Sizing" in widgets


def test_chapter_order_flattens_parts():
    files = chapter_files_from_quarto_config(FIX / "_quarto-html-vol9.yml")
    assert files == [
        "index.qmd",
        "contents/vol9/frontmatter/about.qmd",
        "contents/vol9/intro/intro.qmd",
        "contents/vol9/parts/part_one.qmd",
        "contents/vol9/widgets/widgets.qmd",
        "contents/vol9/gears/gears.qmd",
        "contents/vol9/backmatter/references.qmd",
        "contents/vol9/backmatter/appendix_units.qmd",
        "contents/vol9/backmatter/glossary/glossary.qmd",
    ]


def test_clean_inline_examples():
    assert clean_inline(r"a **b**\index{B!x} c") == "a **b** c"
    assert clean_inline("x [@a; @b] y") == "x y"
    assert clean_inline("[text]{.smallcaps}") == "text"
