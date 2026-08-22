from pathlib import Path

from mlsys_ingest.chunk import chunk_document, heading_path_string
from mlsys_ingest.config import load_config
from mlsys_ingest.pipeline import chunk_corpus, parse_corpus, select_chapters
from mlsys_ingest.quarto import QmdParser
from mlsys_ingest.tokens import ApproxTokenCounter

FIX = Path(__file__).parent / "fixtures"
CFG = FIX / "ingest-fixture.yaml"


def test_heading_path_format():
    assert (
        heading_path_string(1, 8, "Model Optimization", ["8.3 Quantization"])
        == "Vol 1 > Ch 8: Model Optimization > 8.3 Quantization"
    )
    assert heading_path_string(2, 101, "Foo", []) == "Vol 2 > Appendix B: Foo"
    assert heading_path_string(2, 200, "Glossary", []).startswith("Vol 2 > Glossary")


def test_chunk_sizes_and_atomicity():
    doc = QmdParser(drop_code_langs=["python", "tikz"]).parse(
        (FIX / "gears.qmd").read_text(), "gears.qmd"
    )
    counter = ApproxTokenCounter()
    chunks = chunk_document(
        doc,
        volume=9,
        chapter_num=2,
        chapter_title="Gear Trains",
        commit_sha="x",
        counter=counter,
        min_tokens=60,
        max_tokens=120,
    )
    assert len(chunks) > 3
    for c in chunks:
        assert c.token_count <= 120 or c.oversize
        assert c.heading_path.startswith("Vol 9 > Ch 2: Gear Trains")
        assert c.embed_text().startswith(c.heading_path + "\n\n")
        assert c.content_hash
    # tiny sibling subsections are packed together rather than emitted as 3-token chunks
    tiny = [c for c in chunks if "Tiny Subsection" in c.section_path[-1]] if chunks else []
    assert all("One short line." in c.text and "Another short line." in c.text for c in tiny)


def test_atomic_blocks_never_split():
    doc = QmdParser(drop_code_langs=["python", "tikz"]).parse(
        (FIX / "widgets.qmd").read_text(), "widgets.qmd"
    )
    chunks = chunk_document(
        doc,
        volume=9,
        chapter_num=1,
        chapter_title="W",
        commit_sha="x",
        counter=ApproxTokenCounter(),
        min_tokens=10,
        max_tokens=40,
    )
    tables = [c for c in chunks if "| Micro" in c.text]
    assert len(tables) == 1
    t = tables[0]
    assert "| Heavy" in t.text  # whole table in one chunk
    assert t.oversize  # the table exceeds 40 tokens so it is its own oversize chunk


def test_pipeline_numbering_and_selection():
    cfg = load_config(CFG)
    checkout = cfg.checkout_path
    refs = select_chapters(cfg, checkout, cfg.volumes[0])
    assert [(r.chapter_num, r.kind) for r in refs] == [
        (1, "chapter"),
        (2, "chapter"),
        (3, "chapter"),
        (100, "appendix"),
        (200, "glossary"),
    ]
    parsed = parse_corpus(cfg, checkout)
    chunks = chunk_corpus(cfg, parsed, ApproxTokenCounter())
    assert {c.chapter_title for c in chunks} >= {
        "Widget Systems",
        "Gear Trains",
        "Units Appendix",
        "Glossary",
    }
    assert all(c.commit_sha == cfg.source.commit_sha for c in chunks)
    assert all(c.source_file.startswith("book/quarto/contents/vol9/") for c in chunks)
    assert not any("Front matter" in c.text or "Part divider" in c.text for c in chunks)
