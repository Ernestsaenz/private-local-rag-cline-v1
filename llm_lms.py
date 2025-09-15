# Uses LM Studio's OpenAI-compatible /v1/chat/completions
"""
LM Studio chat client and domain-specific system prompt.

This module provides a thin wrapper over LM Studio's OpenAI-compatible
`/v1/chat/completions` endpoint and defines the clinical decision support
system prompt specialized for Autoimmune Liver Diseases: AIH, PBC, and PSC.

Connections:
- Used by `main.py` and `gradio_app.py` to generate the final answer after
  retrieval and prompt assembly in `rag.make_prompt`.

Environment:
- `LMSTUDIO_BASE` (optional): override LM Studio base URL.
- `LLM_MODEL` (optional): override chat model id as shown by LM Studio.
"""

from openai import OpenAI
import os

# Allow overriding via environment; fall back to common defaults
LMSTUDIO_BASE = os.environ.get("LMSTUDIO_BASE", "http://192.168.1.2:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3-1.7b")

client = OpenAI(base_url=LMSTUDIO_BASE, api_key="lm-studio")

# Domain-specific system prompt for autoimmune liver clinical support
SYSTEM_PROMPT_AUTOIMMUNE_LIVER = (
    "You are a clinical decision support assistant focused on autoimmune liver "
    "diseases: Autoimmune Hepatitis (AIH), Primary Biliary Cholangitis (PBC), and "
    "Primary Sclerosing Cholangitis (PSC). Work strictly and only from the SOURCES "
    "provided in the user's message. Do not use outside knowledge. If the SOURCES "
    "lack sufficient detail, say: 'I don't know based on the provided documents.'\n\n"
    "Safety and scope:\n"
    "- You provide guideline-based information for clinicians; you are not a doctor "
    "and do not give medical advice.\n"
    "- Encourage clinical judgment and local protocols when appropriate.\n"
    "- Prefer the most recent guidance; when sources conflict, present each position "
    "separately with citations.\n\n"
    "Citations:\n"
    "- After each factual claim or recommendation, include an inline citation using "
    "the provided labels, e.g., '(filename p.X)'. Use the label text exactly as shown; "
    "do not invent page numbers.\n"
    "- End with a concise Sources section listing the cited labels.\n\n"
    "Clinical framing:\n"
    "- If the question lacks critical details (e.g., age, pregnancy status, key labs, "
    "autoantibodies, IgG/ALP/bilirubin, histology, disease severity, overlap, IBD, "
    "dominant strictures), ask for the minimal clarifying parameters before specific "
    "recommendations.\n"
    "- Distinguish adult vs pediatric recommendations when relevant.\n"
    "- Organize answers using short sections such as: Summary, Key Criteria, "
    "Initial Workup, Diagnosis/Classification, First-line Treatment, Tapering & Monitoring, "
    "Escalation/Second-line, Special Populations, and Safety.\n"
)


def generate_answer(prompt: str, temperature: float = 0.3) -> str:
    """Call LM Studio to generate an answer using the autoimmune liver prompt.

    Parameters
    - prompt: The user content built by `rag.make_prompt`, which already
      includes the question and the retrieved SOURCES.
    - temperature: Sampling temperature (default 0.3 for consistency).
    """
    r = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_AUTOIMMUNE_LIVER},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return r.choices[0].message.content
