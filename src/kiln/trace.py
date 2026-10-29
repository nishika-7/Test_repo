"""Per-request tool-call trace and a chart side-channel (kept out of the LLM context)."""
import functools

TRACE: list = []   # tool calls of the current request (single-user demo); cleared by the API
CHART: dict = {}   # chart data kept out of the LLM's context


def traced(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        out = fn(*a, **k)
        TRACE.append({"tool": fn.__name__, "args": k or a, "result": out})
        return out
    return w
