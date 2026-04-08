import os
import requests
from openai import OpenAI

# Cost tracking (Local models = 0 cost)
LLM_TO_COST = {
    "gpt-3.5-turbo": {"input": 0.0015 / 1e3, "output": 0.002 / 1e3},
    "gpt-4": {"input": 0.03 / 1e3, "output": 0.06 / 1e3},
    "deepseek-coder-v2:16b": {"input": 0.0, "output": 0.0},
    "claude-sonnet-4-5-20250929": {"input": 0.0, "output": 0.0},
}

def query_gpt(prompt: str, model: str = "claude-sonnet-4-5-20250929", backend: str = "camel"):
    """
    Unified LLM query function.
    """

    try:
        # =========================
        # ✅ 1. Ollama local Model
        # =========================
        if backend == "ollama":
            client = OpenAI(
                base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
                api_key=os.getenv("OPENAI_API_KEY", "ollama")
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant coding RL reward functions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
            )

            content = response.choices[0].message.content

            # cost
            total_cost = 0.0
            if model in LLM_TO_COST and LLM_TO_COST[model]["input"] > 0:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_cost = (
                    prompt_tokens * LLM_TO_COST[model]["input"] +
                    completion_tokens * LLM_TO_COST[model]["output"]
                )

        # =========================
        # ✅ 2. Camel API
        # =========================
        elif backend == "camel":
            url = "https://camel.kr777.top/v1/chat/completions"

            headers = {
                "Authorization": "Bearer " + os.getenv("CAMEL_API_KEY"),
                "Content-Type": "application/json"
            }

            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant coding RL reward functions."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7,
            }

            response = requests.post(url, headers=headers, json=data)
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            total_cost = 0.0

        else:
            raise ValueError(f"Unsupported backend: {backend}")

        print(f"\n--- LLM RESPONSE ({model}, {backend}) ---\n{content}\n")

        return total_cost, content

    except Exception as e:
        print(f"❌ LLM Query Failed: {e}")
        raise