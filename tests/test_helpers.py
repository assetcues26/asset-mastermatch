"""Unit tests for utils.helpers."""

from asset_master_match.utils.helpers import (
    build_embed_query,
    normalize_text,
    safe_str,
    sanitize_make_model,
    sanitize_prompt_field,
    texts_match,
    to_int_or_none,
)


def test_to_int_or_none_accepts_int_and_float_strings():
    assert to_int_or_none(12) == 12
    assert to_int_or_none(12.0) == 12
    assert to_int_or_none("12") == 12
    assert to_int_or_none("12.0") == 12


def test_to_int_or_none_rejects_bad_values():
    assert to_int_or_none(None) is None
    assert to_int_or_none(True) is None
    assert to_int_or_none("12.5") is None
    assert to_int_or_none("abc") is None


def test_normalize_and_match():
    assert normalize_text("  hdd   16 gb  ") == "hdd 16 gb"
    assert texts_match("8 GB RAM", "8 gb ram")
    assert not texts_match("8 gb", "16 gb")


def test_safe_str():
    assert safe_str("  x  ") == "x"
    assert safe_str("") is None
    assert safe_str(None) is None


def test_sanitize_prompt_field_strips_and_truncates():
    assert sanitize_prompt_field("  a\x00b  c  ", 10) == "ab c"
    assert sanitize_prompt_field("abcdefghij", 5) == "abcde"


def test_sanitize_make_model():
    assert sanitize_make_model("  Dell XPS  ", 20) == "Dell XPS"
    assert len(sanitize_make_model("x" * 200, 50)) == 50


def test_build_embed_query_name_only_when_no_description():
    assert build_embed_query("  hdd  16 gb  ") == "hdd 16 gb"
    assert build_embed_query("hdd", None) == "hdd"
    assert build_embed_query("hdd", "   ") == "hdd"


def test_build_embed_query_appends_description_name_first():
    assert build_embed_query("hdd 16 gb", "DDR4 server ram") == "hdd 16 gb\nDDR4 server ram"


def test_build_embed_query_truncates_description_then_combined():
    name = "pump"
    desc = "D" * 50
    query = build_embed_query(
        name,
        desc,
        max_description_length=10,
        max_embed_query_length=1000,
    )
    assert query == "pump\n" + ("D" * 10)

    long_name = "N" * 20
    long_desc = "D" * 100
    query2 = build_embed_query(
        long_name,
        long_desc,
        max_description_length=100,
        max_embed_query_length=30,
    )
    assert query2.startswith(long_name + "\n")
    assert len(query2) == 30
    assert query2 == long_name + "\n" + ("D" * 9)


def test_build_embed_query_preserves_name_when_over_budget():
    name = "N" * 50
    query = build_embed_query(
        name,
        "description ignored if name alone fills budget",
        max_description_length=800,
        max_embed_query_length=40,
    )
    assert query == "N" * 40
    assert "description" not in query
