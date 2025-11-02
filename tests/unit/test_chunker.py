from scripts.utils.text_chunker import split_statute_sentences_into_articles, explode_article_to_hierarchy, chunk_paragraphs


def test_split_articles_basic():
    sents = [
        "제1조(목적)",
        "이 법은 ...",
        "제2조(정의)",
        "1. 용어의 뜻은 ...",
    ]
    blocks = split_statute_sentences_into_articles(sents)
    assert len(blocks) == 2
    assert blocks[0]["heading"].startswith("제1조")
    assert "이 법" in blocks[0]["text"]


def test_explode_article_hierarchy():
    heading = "제2조(정의)"
    body = "제1항 어떤 내용\n제2항 1호 세부내용 A\n2호 세부내용 B"
    rows = explode_article_to_hierarchy(heading, body)
    # expect clause-only row for 1항, and two item rows for 2항
    assert any(r["clause_no"] == "1" and r["item_no"] is None for r in rows)
    assert any(r["clause_no"] == "2" and r["item_no"] == "1" for r in rows)
    assert any(r["clause_no"] == "2" and r["item_no"] == "2" for r in rows)


def test_chunk_paragraphs_overlap():
    paras = [f"para {i}" for i in range(10)]
    chunks = chunk_paragraphs(paras, min_chars=1, max_chars=20, overlap_ratio=0.5)
    assert len(chunks) >= 3
    assert chunks[0]["start_para"] == 0
