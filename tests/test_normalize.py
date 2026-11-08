from provenance_rag.normalize import answers_match, citations_match, clean_answer, grade_norm


def test_grade_norm_matches_official_rule():
    assert grade_norm("7-ABC 123") == "7ABC123"
    assert grade_norm("q3 fy27") == "Q3FY27"
    assert grade_norm("4.3.2") == "432"
    assert grade_norm("THERM_ALERT#") == "THERMALERT#"
    assert grade_norm("A·1") == "A1"


def test_answers_match_with_aliases():
    assert answers_match("94", "94", [])
    assert answers_match("Meridian 4.3.2", "4.3.2", ["Meridian 4.3.2"])
    assert not answers_match("Q3", "Q3 FY27", [])


def test_clean_answer_strips_units_and_prose():
    assert clean_answer("94 C") == "94"
    assert clean_answer("94°C") == "94"
    assert clean_answer("180 seconds") == "180"
    assert clean_answer("$84.50") == "84.50"
    assert clean_answer("Answer: REV-C2") == "REV-C2"
    assert clean_answer('"Q3 FY27"') == "Q3 FY27"
    assert clean_answer("4.3.2") == "4.3.2"
    assert clean_answer("12V-2x6") == "12V-2x6"
    assert clean_answer(84.5) == "84.5"
    assert clean_answer(212.0) == "212"


def test_clean_answer_refusals_become_empty():
    for text in ("", None, "N/A", "unknown", "I don't know", "Not found", "The corpus does not say"):
        assert clean_answer(text) == ""


def test_citation_set_comparison():
    assert citations_match(["/app/corpus/a/b.pdf", "c.csv"], ["c.csv", "a/b.pdf"])
    assert not citations_match(["a.pdf", "b.csv"], ["a.pdf"])
