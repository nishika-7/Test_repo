"""FastAPI server: UI + agent endpoint + manual optimizer + guarded export.
Run from the project root:  uv run uvicorn app.main:app --reload
"""
import os
import threading
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kiln.agent import build_agent
from kiln.config import FEATURES
from kiln.optimizer import run_optimize
from kiln.safety import validate_schedule
from kiln.scripted import run_scripted
from kiln.trace import CHART, TRACE

app = FastAPI(title="Kiln optimizer")
_agent = None
_busy = threading.Lock()  # a Strands Agent handles one request at a time


def agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


class Job(BaseModel):
    clay_body: str
    load_kg: float
    num_pieces: int
    moisture_pct: float
    ambient_temp: float
    max_defect_pct: float = 5.0


class Chat(BaseModel):
    message: str
    prev: dict | None = None  # inputs of the previous result, so scripted follow-ups keep context


class Plan(BaseModel):
    schedule: dict
    max_defect_pct: float = 5.0


@app.post("/api/optimize")  # manual mode: no LLM, works offline
def optimize(j: Job):
    res = run_optimize(j.load_kg, j.num_pieces, j.clay_body, j.moisture_pct, j.ambient_temp, j.max_defect_pct)
    if "error" in res:
        return {"error": res["error"]}
    return {"result": res, "chart": dict(CHART), "trace": []}


@app.post("/api/agent")
def chat(c: Chat):
    if os.getenv("KILN_LLM") == "scripted":
        return run_scripted(c.message, c.prev)
    if not _busy.acquire(blocking=False):
        return {"error": "The agent is still working on the previous request. Wait for it to finish, then try again."}
    try:
        return _chat_llm(c)
    finally:
        _busy.release()


def _chat_llm(c: Chat):
    TRACE.clear()
    try:
        reply = str(agent()(c.message))
    except Exception as e:  # LLM not reachable (for example Ollama is not running): fall back, clearly labelled
        out = run_scripted(c.message, c.prev)
        out["reply"] = f"The LLM is not available ({type(e).__name__}: {str(e)[:140]}), so this is the scripted demo instead.\n\n" + out["reply"]
        return out
    opt = next((t for t in reversed(TRACE) if t["tool"] == "optimize"), None)
    trace = [
        {"tool": t["tool"], "ok": bool(t["result"].get("approved", t["result"].get("target_met", True)))
         if isinstance(t["result"], dict) else True}
        for t in TRACE
    ]
    return {
        "mode": "llm", "reply": reply, "trace": trace,
        "result": opt["result"] if opt else None,
        "inputs": {"max_defect_pct": 5.0, **opt["args"]} if opt and isinstance(opt["args"], dict) else None,
        "chart": dict(CHART),
    }


@app.post("/api/export")  # human approval step; the server re-validates, never trusts the browser
def export(p: Plan):
    v = validate_schedule(p.schedule, p.max_defect_pct)
    if not v["approved"]:
        return JSONResponse(status_code=422, content=v)
    out = Path("exports")
    out.mkdir(exist_ok=True)
    text = "\n".join(f"{k}: {p.schedule[k]}" for k in FEATURES)
    (out / "approved_schedule.txt").write_text(text)
    return {"approved": True, "text": text, "saved_to": "exports/approved_schedule.txt"}


# --- Newly added function and class for multi-hunk / code changes ---

def generate_optimizer_summary_report(run_id: str, metrics: dict) -> str:
    """Generates a formatted text summary for an optimization run."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "SUCCESS" if metrics.get("target_met", False) else "REVIEW REQUIRED"
    
    lines = [
        f"=== Kiln Run Report [{run_id}] ===",
        f"Timestamp: {timestamp}",
        f"Status: {status}",
        f"Final Defect Rate: {metrics.get('defect_pct', 'N/A')}%",
        f"Energy Used: {metrics.get('energy_kwh', 'N/A')} kWh",
        "================================"
    ]
    return "\n".join(lines)


class KilnAuditLogger:
    """Manages audit trails and local log rotation for production runs."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_count = 0

    def log_event(self, event_type: str, details: str) -> None:
        self.session_count += 1
        log_file = self.log_dir / "audit.log"
        entry = f"[{datetime.now().isoformat()}] [{event_type.upper()}] {details}\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry)


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")