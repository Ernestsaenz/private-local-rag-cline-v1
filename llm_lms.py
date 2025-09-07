# Uses LM Studio's OpenAI-compatible /v1/chat/completions
from openai import OpenAI

LMSTUDIO_BASE = "http://localhost:1234/v1"
LLM_MODEL = "qwen_qwen3-4b-instruct-2507"   # or the exact name shown in LM Studio

client = OpenAI(base_url=LMSTUDIO_BASE, api_key="lm-studio")

def generate_answer(prompt, temperature=0.3):
    r = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful assistant. Cite sources by filename in parentheses."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return r.choices[0].message.content