from scripts.utils.text_chunker import split_statute_sentences_into_articles, explode_article_to_hierarchy, chunk_paragraphs


def test_split_articles_basic():
    sents = [
        "??ì¡?ëª©ì )",
        "??ë²•ì? ...",
        "??ì¡??•ì˜)",
        "1. ?©ì–´???»ì? ...",
    ]
    blocks = split_statute_sentences_into_articles(sents)
    assert len(blocks) == 2
    assert blocks[0]["heading"].startswith("??ì¡?)
    assert "??ë²? in blocks[0]["text"]


def test_explode_article_hierarchy():
    heading = "??ì¡??•ì˜)"
    body = "?????´ë–¤ ?´ìš©\n????1???¸ë??´ìš© A\n2???¸ë??´ìš© B"
    rows = explode_article_to_hierarchy(heading, body)
    # expect clause-only row for 1?? and two item rows for 2??
    assert any(r["clause_no"] == "1" and r["item_no"] is None for r in rows)
    assert any(r["clause_no"] == "2" and r["item_no"] == "1" for r in rows)
    assert any(r["clause_no"] == "2" and r["item_no"] == "2" for r in rows)


def test_chunk_paragraphs_overlap():
    paras = [f"para {i}" for i in range(10)]
    chunks = chunk_paragraphs(paras, min_chars=1, max_chars=20, overlap_ratio=0.5)
    assert len(chunks) >= 3
    assert chunks[0]["start_para"] == 0
