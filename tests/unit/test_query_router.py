from core.services.search.sql_router import route_query


def test_route_article_pattern():
    assert route_query("제3조 에 대해 알려줘") == "text2sql"


def test_route_court_keyword():
    assert route_query("대법원 판례 목록") == "text2sql"


def test_route_default_vector():
    assert route_query("부당이득 반환 요건은?") == "vector"
