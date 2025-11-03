from source.services.query_router import route_query


def test_route_article_pattern():
    assert route_query("??ì¡????€???Œë ¤ì¤?) == "text2sql"


def test_route_court_keyword():
    assert route_query("?€ë²•ì› ?ë? ëª©ë¡") == "text2sql"


def test_route_default_vector():
    assert route_query("ë¶€?¹ì´??ë°˜í™˜ ?”ê±´?€?") == "vector"
