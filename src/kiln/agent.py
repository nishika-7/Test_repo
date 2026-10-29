"""Agent assembly: system prompt, model choice, CLI.  Run: uv run python -m kiln.agent "your request" """
from __future__ import annotations

import os
import sys

from strands import Agent

from .models import MODEL_LABEL
from .tools import optimize, predict, validate, what_if


SYSTEM_PROMPT = f"""You are a kiln firing assistant for ceramic producers.

Rules you must follow:
1. Never invent or estimate numbers yourself. Energy, defect rates and savings must come from tool results.
2. To recommend a schedule: call optimize, then call validate on the recommended_schedule.
   Only call a schedule safe if validate returned approved=true. If it was rejected, say why and try again
   (for example with a looser defect target) or tell the user the target is not achievable.
3. If the user omits an input optimize needs (load_kg, num_pieces, clay_body, moisture_pct, ambient_temp), ask for it.
4. Report validation warnings. If a value is outside the training range, say the prediction is an extrapolation.
5. Be honest about uncertainty: the defect model is less accurate than the energy model, and the models were
   trained on simulated data (current models: {MODEL_LABEL}).
6. Present the final answer as: the schedule, predicted energy and defect rate, saving vs baseline, and the key trade-off.
   Tell the user a human operator must review the schedule before it is loaded into a real kiln controller.
"""


def build_model():
    """Local Ollama by default (no account needed).
    KILN_LLM=groq uses Groq's hosted API (needs GROQ_API_KEY); KILN_LLM=bedrock uses Amazon Bedrock.
    Settings can live in a .env file in the project root."""
    provider = os.getenv("KILN_LLM", "ollama")
    if provider == "groq":
        from strands.models.openai import OpenAIModel  # needs: uv add "strands-agents[openai]"

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("KILN_LLM=groq needs GROQ_API_KEY (create one at console.groq.com/keys)")
        model_id = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        print(f"[kiln] LLM backend: groq ({model_id})", file=sys.stderr)
        return OpenAIModel(client_args={"api_key": api_key, "base_url": "https://api.groq.com/openai/v1"}, model_id=model_id)
    if provider == "bedrock":
        from strands.models import BedrockModel

        model_id = os.getenv("BEDROCK_MODEL_ID")
        if not model_id:
            raise RuntimeError("KILN_LLM=bedrock needs BEDROCK_MODEL_ID (a model your AWS account can use)")
        print(f"[kiln] LLM backend: bedrock ({model_id})", file=sys.stderr)
        return BedrockModel(model_id=model_id, region_name=os.getenv("AWS_REGION", "us-east-1"))
    from strands.models.ollama import OllamaModel  # needs: uv add "strands-agents[ollama]"

    host, model_id = os.getenv("OLLAMA_HOST", "http://localhost:11434"), os.getenv("OLLAMA_MODEL", "llama3.1")
    print(f"[kiln] LLM backend: ollama ({model_id} at {host})", file=sys.stderr)
    return OllamaModel(host=host, model_id=model_id)


def build_agent() -> Agent:
    return Agent(model=build_model(), tools=[predict, optimize, validate, what_if], system_prompt=SYSTEM_PROMPT)


def main():
    if os.getenv("KILN_LLM") == "scripted":
        from .scripted import run_scripted

        def ask(text):
            print(run_scripted(text)["reply"])
    else:
        agent = build_agent()
        ask = agent
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        print("Kiln agent ready. Ctrl+C to quit.")
        while True:
            ask(input("\nYou: "))


if __name__ == "__main__":
    main()