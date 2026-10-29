"""Scripted demo mode: parsing, the tool flow, and the automatic fallback when the LLM is unreachable."""
import sys

import pytest

from kiln.config import ROOT
from kiln.scripted import parse_request, run_scripted


@pytest.mark.parametrize("text, expected", [
    ("400 kg stoneware, 60 pieces, 3% moisture, 30C ambient, defects under 5%",
     dict(load_kg=400, num_pieces=60, clay_body="stoneware", moisture_pct=3, ambient_temp=30, max_defect_pct=5)),
    ("I have 250kg of porcelain, 40 pots, moisture 2.5%, room temp 28 C, defect rate below 3%",
     dict(load_kg=250, num_pieces=40, clay_body="porcelain", moisture_pct=2.5, ambient_temp=28, max_defect_pct=3)),
    ("400 kg earthenware, 60 pieces, 3% moisture, 30 °C ambient",   # no target given: default 5
     dict(load_kg=400, num_pieces=60, clay_body="earthenware", moisture_pct=3, ambient_temp=30, max_defect_pct=5)),
])
def test_parse_request(text, expected):
    inputs, missing = parse_request(text)
    assert missing == [] and inputs == expected


def test_parse_reports_what_is_missing():
    _, missing = parse_request("fire 400 kg stoneware")
    assert len(missing) == 3


def test_scripted_run_uses_both_tools_and_only_tool_numbers():
    out = run_scripted("400 kg stoneware, 60 pieces, 3% moisture, 30C ambient, defects under 5%")
    assert out["mode"] == "scripted" and [t["tool"] for t in out["trace"]] == ["optimize", "validate"]
    assert str(out["result"]["predicted"]["energy_kwh"]) in out["reply"]
    assert "Scripted demo" in out["reply"] and "human operator" in out["reply"]


def test_scripted_asks_for_missing_inputs():
    out = run_scripted("fire something")
    assert out["result"] is None and "I still need" in out["reply"]


def test_api_falls_back_to_scripted_when_llm_is_unreachable(monkeypatch):
    pytest.importorskip("httpx")
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient

    import app.main as main

    def boom():
        raise ConnectionError("no ollama")

    monkeypatch.delenv("KILN_LLM", raising=False)
    monkeypatch.setattr(main, "agent", boom)
    out = TestClient(main.app).post("/api/agent", json={"message": "400 kg stoneware, 60 pieces, 3% moisture, 30C ambient"}).json()
    assert out["mode"] == "scripted" and "not available" in out["reply"] and out["result"]


PREV = dict(load_kg=400, num_pieces=60, clay_body="stoneware", moisture_pct=3, ambient_temp=30, max_defect_pct=5)


def test_follow_up_changes_only_what_is_mentioned():
    inputs, missing = parse_request("defects under 3%", PREV)
    assert missing == [] and inputs == {**PREV, "max_defect_pct": 3}
    inputs, _ = parse_request("same load but porcelain", PREV)
    assert inputs["clay_body"] == "porcelain" and inputs["load_kg"] == 400


def test_run_scripted_follow_up_uses_previous_inputs():
    out = run_scripted("make it 2% defects", PREV)
    assert out["result"] and out["inputs"]["max_defect_pct"] == 2 and out["inputs"]["load_kg"] == 400


def test_busy_agent_reports_busy_instead_of_falling_back(monkeypatch):
    pytest.importorskip("httpx")
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient

    import app.main as main

    monkeypatch.delenv("KILN_LLM", raising=False)
    assert main._busy.acquire(blocking=False)
    try:
        out = TestClient(main.app).post("/api/agent", json={"message": "400 kg stoneware"}).json()
    finally:
        main._busy.release()
    assert "still working" in out["error"] and "mode" not in out