# Handles the LLM Teacher logic using raw requests API compatible with Anthropic format.
import os
import re
import json
import requests

class LLMTeacher:
  def __init__(self, backend="remote", model_name="claude-sonnet-4-5-20250929"):
    self.model_name = model_name
    self.backend = backend

    if backend == "local":
      self.api_base = os.environ.get("LOCAL_API_BASE", "http://localhost:11434")
      self.api_key = os.environ.get("LOCAL_API_KEY", "ollama")
    else:
      api_base = os.environ.get("OPENAI_API_BASE", None)
      api_key = os.environ.get("OPENAI_API_KEY", None)

      keys_file = os.path.join(os.path.dirname(__file__), "keys.conf")
      if os.path.exists(keys_file):
        with open(keys_file, "r") as f:
          for line in f:
            line = line.strip()
            if "=" in line:
              k, v = line.split("=", 1)
              if k.strip() == "api_base_url" and not api_base:
                api_base = v.strip()
              elif k.strip() == "api_key" and not api_key:
                api_key = v.strip()

      if api_base is not None:
        api_base = str(api_base).rstrip("/")
        if api_base.endswith("/v1"):
            api_base = api_base[:-3]

      self.api_base = api_base
      self.api_key = api_key

  def _call_llm_api(self, prompt):
    url = f"{self.api_base}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
    }
    data = {
        "model": self.model_name,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        print(">>> Sending prompt to LLM...")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        answer = result["content"][0]["text"]
        print(f"<<< LLM Response received (length {len(answer)})")
        return answer
    except Exception as e:
        print("Call failed:", str(e))
        if 'response' in locals() and hasattr(response, 'text'):
            print("Response:", response.text)
        return None

  def get_direct_reward(self, trajectory_summary):
    sys_prompt = "You are an expert RL reward designer. Provide ONLY a scalar float value between -2.0 and 2.0 to be added as an intrinsic reward for the MountainCar agent's next phase based on its recent performance. Provide no explanation, just the number."
    user_prompt = f"Agent's recent performance summary:\n{trajectory_summary}\n\nWhat should be the constant additional reward for the next period?"

    prompt = f"{sys_prompt}\n\n{user_prompt}"

    content = self._call_llm_api(prompt)
    if content:
      try:
        val = float(re.findall(r'[-+]?\d*\.?\d+', content)[0])
        return val
      except Exception as e:
        print(f"Error parsing float: {e}")
        print(f"  Raw LLM response: {repr(content)}")
        return 0.0
    return 0.0

  def get_batch_rewards(self, segment):
    """
    Accepts a list of (binned_state, action) tuples representing a trajectory segment.
    Returns a list of float rewards, one per step, by querying the LLM once.
    """
    steps_text = "\n".join(
      f"  step {i}: pos_bin={s[0]}, vel_bin={s[1]}, action={'left' if a==0 else 'idle' if a==1 else 'right'}"
      for i, (s, a) in enumerate(segment)
    )
    n = len(segment)

    sys_prompt = (
      "You are an expert RL reward designer for MountainCar-v0.\n"
      "The state is discretized into a 20x20 grid: pos_bin (0=leftmost, 19=rightmost), "
      "vel_bin (0=max left velocity, 19=max right velocity).\n"
      "Actions: left, idle, right. Goal: reach pos_bin >= 18.\n\n"
      "Given a trajectory segment, return ONLY a JSON array of floats (one per step), "
      "each between -1.0 and 1.0. No explanation, no markdown, just the JSON array.\n\n"
      "### Examples ###\n"
      "Input (3 steps):\n"
      "  step 0: pos_bin=5, vel_bin=8, action=left\n"
      "  step 1: pos_bin=4, vel_bin=7, action=left\n"
      "  step 2: pos_bin=3, vel_bin=6, action=right\n"
      "Output: [-0.2, -0.3, 0.5]\n\n"
      "Input (4 steps):\n"
      "  step 0: pos_bin=12, vel_bin=14, action=right\n"
      "  step 1: pos_bin=13, vel_bin=15, action=right\n"
      "  step 2: pos_bin=14, vel_bin=13, action=idle\n"
      "  step 3: pos_bin=13, vel_bin=11, action=left\n"
      "Output: [0.6, 0.8, -0.5, 0.1]\n"
    )

    user_prompt = f"Score the following {n} steps:\n{steps_text}"
    prompt = f"{sys_prompt}\n\n{user_prompt}"

    content = self._call_llm_api(prompt)
    if content:
      try:
        content_stripped = content.strip()
        # Handle potential markdown wrapping
        if content_stripped.startswith("```"):
          content_stripped = content_stripped.split("\n", 1)[-1]
        if content_stripped.endswith("```"):
          content_stripped = content_stripped.rsplit("```", 1)[0]
        content_stripped = content_stripped.strip()

        rewards = json.loads(content_stripped)
        if isinstance(rewards, list) and len(rewards) == n:
          return [float(r) for r in rewards]
        else:
          print(f"  [WARN] Expected JSON array of length {n}, got length {len(rewards) if isinstance(rewards, list) else 'non-list'}")
          print(f"  Raw LLM response: {repr(content)}")
          if isinstance(rewards, list):
            rewards = [float(r) for r in rewards]
            if len(rewards) < n:
              rewards.extend([0.0] * (n - len(rewards)))
            return rewards[:n]
      except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"  [ERROR] Failed to parse batch rewards: {e}")
        print(f"  Raw LLM response: {repr(content)}")
        # Fallback: try regex to extract individual floats
        floats = re.findall(r'[-+]?\d*\.?\d+', content)
        if len(floats) >= n:
          return [float(f) for f in floats[:n]]
        print(f"  [ERROR] Regex fallback found only {len(floats)} floats, needed {n}")

    return [0.0] * n

  def get_reward_function(self, trajectory_summary, prev_code=None):
    sys_prompt = "You are an expert RL reward designer. Output ONLY valid Python code containing a function `def intrinsic_reward(state, action):` returning a float. The MountainCar state is [position, velocity] where position is in [-1.2, 0.6] and velocity is [-0.07, 0.07]. Action is 0 (left), 1 (idle), 2 (right). Return no markdown formatting, just the raw code."

    user_prompt = f"Agent's recent performance:\n{trajectory_summary}\n\n"
    if prev_code:
      user_prompt += f"Previous reward code was:\n```python\n{prev_code}\n```\nRefine it to help the agent explore better and reach the goal at position >= 0.5. Notice that you MUST return a valid python code."
    else:
      user_prompt += "Write the initial `intrinsic_reward(state, action)` function to encourage moving back and forth to gain momentum."

    prompt = f"{sys_prompt}\n\n{user_prompt}"

    code = self._call_llm_api(prompt)
    if code:
      if code.startswith("```python"):
        code = code[9:]
      if code.startswith("```"):
        code = code[3:]
      if code.endswith("```"):
        code = code[:-3]
      return code.strip()

    print("Fallback to zero intrinsic reward.")
    return "def intrinsic_reward(state, action):\n    return 0.0"
