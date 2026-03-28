"""Validation tests for the tracked RAG eval dataset."""

from pathlib import Path

import pytest

from vgm.rag import RagEvalCase, load_rag_eval_cases

FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "rag_eval"
    / "seti_rules_reference_v1.jsonl"
)
EXTRACTED_SOURCE_DIR = FIXTURE_PATH.parent / "source_documents" / "extracted"
LOCAL_DOCUMENTS = {
    "seti-rules-en": "seti-rules-en.txt",
    "seti-faq": "seti-faq.txt",
    "seti-player-aid-en": "seti-player-aid-en.txt",
    "seti-alien-species-en": "seti-alien-species-en.txt",
}


def load_cases() -> list[RagEvalCase]:
    """Load the tracked SETI eval suite."""

    return load_rag_eval_cases(FIXTURE_PATH)


def local_source_docs_available() -> bool:
    """Check whether ignored local source documents are present."""

    return all((EXTRACTED_SOURCE_DIR / filename).exists() for filename in LOCAL_DOCUMENTS.values())


def load_document_pages(document_id: str) -> list[str]:
    """Split one extracted source document into 1-based pages."""

    document_path = EXTRACTED_SOURCE_DIR / LOCAL_DOCUMENTS[document_id]
    return document_path.read_text(errors="ignore").split("\f")


def test_seti_rules_reference_dataset_shape():
    cases = load_cases()

    assert len(cases) == 30
    assert len({case.case_id for case in cases}) == len(cases)
    assert {case.suite_id for case in cases} == {"seti_rules_reference_v1"}
    assert {case.game_id for case in cases} == {"seti"}
    assert {case.mode for case in cases} == {"rules_reference"}

    abstaining_cases = [case for case in cases if case.rubric.expected_abstain]
    assert len(abstaining_cases) == 2

    multi_turn_cases = [case for case in cases if len(case.conversation) > 1]
    assert 3 <= len(multi_turn_cases) <= 5

    for case in cases:
        assert case.conversation[0].role == "user"
        if case.rubric.expected_abstain:
            assert all(ref.document_type == "player_aid" for ref in case.retrieval_refs)
            assert case.rubric.preferred_source_id is None
        else:
            preferred_ref = next(
                ref for ref in case.retrieval_refs if ref.source_id == case.rubric.preferred_source_id
            )
            assert preferred_ref.document_type != "player_aid"


@pytest.mark.skipif(
    not local_source_docs_available(),
    reason="Ignored local SETI source documents are not present",
)
def test_seti_rules_reference_locators_resolve_against_local_docs():
    pages_by_document: dict[str, list[str]] = {}

    for case in load_cases():
        for retrieval_ref in case.retrieval_refs:
            pages = pages_by_document.setdefault(
                retrieval_ref.document_id,
                load_document_pages(retrieval_ref.document_id),
            )
            assert retrieval_ref.page <= len(pages)
            assert retrieval_ref.locator in pages[retrieval_ref.page - 1], (
                f"{case.case_id} locator did not resolve in "
                f"{retrieval_ref.document_id} page {retrieval_ref.page}"
            )
