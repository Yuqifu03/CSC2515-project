import os
from openai import OpenAI  # Standard library for both OpenAI and Ollama

# Cost tracking (Local models = 0 cost)
LLM_TO_COST = {
    "gpt-3.5-turbo": {"input": 0.0015 / 1e3, "output": 0.002 / 1e3},
    "gpt-4": {"input": 0.03 / 1e3, "output": 0.06 / 1e3},
    "deepseek-coder-v2:16b": {"input": 0.0, "output": 0.0},
}

def query_gpt(prompt: str, model: str = "deepseek-coder-v2:16b"):
    """
    Modernized query function for local LLMs via Ollama.
    """
    # 1. Initialize client pointing to your local Ollama server
    client = OpenAI(
        base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "ollama")
    )

    try:
        # 2. Modern Chat Completion call
        response = client.chat.completions.create(
            model=model, # This now accepts the argument from reward.py
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant coding RL reward functions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )

        content = response.choices[0].message.content
        
        # 3. Calculate cost (kept for compatibility with your existing logic)
        total_cost = 0.0
        if model in LLM_TO_COST and LLM_TO_COST[model]["input"] > 0:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_cost = (prompt_tokens * LLM_TO_COST[model]["input"] + 
                          completion_tokens * LLM_TO_COST[model]["output"])

        # Optional: Print to console so you can see the LLM's 'thought process'
        print(f"\n--- LLM RESPONSE ({model}) ---\n{content}\n")
        
        return total_cost, content

    except Exception as e:
        print(f"❌ LLM Query Failed: {e}")
        return 0.0, ""