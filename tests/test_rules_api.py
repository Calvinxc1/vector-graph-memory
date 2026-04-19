"""Tests for the pilot rules API surfaces."""

from __future__ import annotations

import asyncio

import pytest

fastapi = pytest.importorskip("fastapi")

from vgm.api.server import pilot_rules_inspection, pilot_rules_ruling, state
from vgm.rules import RulesRulingRequest


HTTPException = fastapi.HTTPException


class FakeRulesEngine:
    def answer(self, request: RulesRulingRequest):
        return {
            "backend": "live-pilot",
            "question": request.question,
        }

    def inspect_request(self, request: RulesRulingRequest):
        return {
            "question": request.question,
            "normalized_question": request.question.lower(),
            "evidence": {
                "question": request.question,
                "seed_id": "seti_landing_orbiter_seed_v1",
                "subsystem": "landing_and_orbiter_interactions",
                "nodes": [],
                "edges": [],
            },
            "seed_inference": {
                "normalized_question": request.question.lower(),
                "selected_seed_id": "seti_landing_orbiter_seed_v1",
                "selected_score": 1.0,
                "candidates": [
                    {
                        "seed_id": "seti_landing_orbiter_seed_v1",
                        "score": 1.0,
                    }
                ],
            },
            "selected_seed_id": "seti_landing_orbiter_seed_v1",
            "selected_case": None,
            "candidate_cases": [],
        }


def test_pilot_rules_inspection_returns_engine_trace():
    original = state.rules_ruling_engine
    state.rules_ruling_engine = FakeRulesEngine()
    try:
        response = asyncio.run(
            pilot_rules_inspection(
                RulesRulingRequest(question="Can an orbiter later land on the same planet?")
            )
        )
    finally:
        state.rules_ruling_engine = original

    assert response["question"] == "Can an orbiter later land on the same planet?"
    assert response["selected_seed_id"] == "seti_landing_orbiter_seed_v1"
    assert response["seed_inference"]["selected_seed_id"] == "seti_landing_orbiter_seed_v1"


def test_pilot_rules_inspection_raises_503_without_engine():
    original = state.rules_ruling_engine
    state.rules_ruling_engine = None
    try:
        try:
            asyncio.run(
                pilot_rules_inspection(
                    RulesRulingRequest(question="Can an orbiter later land on the same planet?")
                )
            )
        except HTTPException as exc:
            assert exc.status_code == 503
            assert "not initialized" in exc.detail
        else:
            raise AssertionError("Expected HTTPException")
    finally:
        state.rules_ruling_engine = original


def test_pilot_rules_ruling_raises_503_without_engine():
    original = state.rules_ruling_engine
    state.rules_ruling_engine = None
    try:
        try:
            asyncio.run(
                pilot_rules_ruling(
                    RulesRulingRequest(question="Can an orbiter later land on the same planet?")
                )
            )
        except HTTPException as exc:
            assert exc.status_code == 503
            assert "not initialized" in exc.detail
        else:
            raise AssertionError("Expected HTTPException")
    finally:
        state.rules_ruling_engine = original
