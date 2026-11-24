"""Tests for LLM choice → master row mapping rules."""

from asset_master_match.core.config import MAX_MAKE_MODEL_LENGTH
from asset_master_match.services.match_service import _apply_llm_choice, _master_from_candidate


def test_master_keeps_makemodel_id_when_not_generated(sample_candidate):
    master = _master_from_candidate(
        sample_candidate,
        make_model="8 gb ram",
        make_model_generated="N",
    )
    assert master["MakeModel"] == "8 gb ram"
    assert master["MakeModelId"] == 10338
    assert master["Model"] is None
    assert master["ModelId"] is None
    assert master["CategoryId"] == 1039
    assert master["SubCategoryId"] == 10067


def test_master_returns_model_from_candidate(sample_candidate):
    row = {**sample_candidate, "makemodel": "Asus", "makemodelid": 98655, "model": "test1", "modelid": 1151}
    master = _master_from_candidate(
        row,
        make_model="Asus",
        make_model_generated="N",
    )
    assert master["MakeModel"] == "Asus"
    assert master["MakeModelId"] == 98655
    assert master["Model"] == "test1"
    assert master["ModelId"] == 1151


def test_master_keeps_model_when_makemodel_generated(sample_candidate):
    row = {**sample_candidate, "model": "test1", "modelid": "1151"}
    master = _master_from_candidate(
        row,
        make_model="Asus test1",
        make_model_generated="Y",
    )
    assert master["MakeModelId"] is None
    assert master["Model"] == "test1"
    assert master["ModelId"] == 1151


def test_master_nulls_makemodel_id_when_generated(sample_candidate):
    master = _master_from_candidate(
        sample_candidate,
        make_model="16 gb ram",
        make_model_generated="Y",
    )
    assert master["MakeModel"] == "16 gb ram"
    assert master["MakeModelId"] is None
    # Classification unchanged from candidate
    assert master["AssetClassId"] == 1107
    assert master["Category"] == "Machinery"


def test_apply_llm_refine_sets_generated_y(sample_candidate):
    master, flag, message = _apply_llm_choice(
        "hdd 16 gb ram",
        [sample_candidate],
        {
            "candidate_index": 1,
            "make_model": "16 gb ram",
            "make_model_generated": "Y",
        },
    )
    assert flag == "Y"
    assert message is None
    assert master["MakeModel"] == "16 gb ram"
    assert master["MakeModelId"] is None
    assert master["CategoryId"] == sample_candidate["categoryid"]


def test_apply_llm_unchanged_makemodel_forces_n(sample_candidate):
    master, flag, message = _apply_llm_choice(
        "hdd 8 gb ram",
        [sample_candidate],
        {
            "candidate_index": 1,
            "make_model": "8 gb ram",
            "make_model_generated": "Y",  # model lied; text equals pinecone
        },
    )
    assert flag == "N"
    assert message is None
    assert master["MakeModelId"] == 10338


def test_apply_llm_text_changed_but_flag_n_becomes_y(sample_candidate):
    master, flag, message = _apply_llm_choice(
        "hdd 16 gb ram",
        [sample_candidate],
        {
            "candidate_index": 1,
            "make_model": "16 gb ram",
            "make_model_generated": "N",
        },
    )
    assert flag == "Y"
    assert message is None
    assert master["MakeModelId"] is None


def test_apply_llm_zero_index_is_no_match(sample_candidate):
    master, flag, message = _apply_llm_choice(
        "Project Salary",
        [sample_candidate],
        {"candidate_index": 0, "make_model": "", "make_model_generated": "N"},
    )
    assert master is None
    assert flag == "N"
    assert message == "No acceptable master match"


def test_apply_llm_keeps_existing_when_rewrite_only_adds_size(sample_candidate):
    row = {**sample_candidate, "makemodel": "LINEAR WORKSTATION", "makemodelid": 1752}
    master, flag, message = _apply_llm_choice(
        "Fixed LINEAR WORKSTATION - 1200x650",
        [row],
        {
            "candidate_index": 1,
            "make_model": "LINEAR WORKSTATION - 1200x650",
            "make_model_generated": "Y",
        },
    )
    assert message is None
    assert flag == "N"
    assert master["MakeModel"] == "LINEAR WORKSTATION"
    assert master["MakeModelId"] == 1752


def test_apply_llm_prefers_closer_model_code(sample_candidate):
    vague = {
        **sample_candidate,
        "makemodel": "Cisco Catalyst Edge C8300-1N1S-6T",
        "makemodelid": 1477,
    }
    exact = {
        **sample_candidate,
        "makemodel": "Cisco C8300-1N1S-6T",
        "makemodelid": 1128,
    }
    master, flag, message = _apply_llm_choice(
        "CISCO-C8300-1N1S-6T Router",
        [vague, exact],
        {
            "candidate_index": 1,
            "make_model": "Cisco Catalyst Edge C8300-1N1S",
            "make_model_generated": "Y",
        },
    )
    assert message is None
    assert flag == "N"
    assert master["MakeModel"] == "Cisco C8300-1N1S-6T"
    assert master["MakeModelId"] == 1128


def test_apply_llm_bad_index_falls_back_to_first(sample_candidate):
    other = {**sample_candidate, "makemodel": "other", "makemodelid": 1}
    master, flag, message = _apply_llm_choice(
        "x",
        [sample_candidate, other],
        {"candidate_index": 99, "make_model": "nope", "make_model_generated": "Y"},
    )
    assert message == "fallback_index"
    assert flag == "N"
    assert master["MakeModel"] == "8 gb ram"
    assert master["MakeModelId"] == 10338


def test_apply_llm_truncates_generated_makemodel(sample_candidate):
    huge = "A" * (MAX_MAKE_MODEL_LENGTH + 50)
    master, flag, message = _apply_llm_choice(
        "x",
        [sample_candidate],
        {
            "candidate_index": 1,
            "make_model": huge,
            "make_model_generated": "Y",
        },
    )
    assert flag == "Y"
    assert message is None
    assert len(master["MakeModel"]) == MAX_MAKE_MODEL_LENGTH
    assert master["MakeModelId"] is None
