import os
import requests
from openai import OpenAI

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────
def reset(task_id: str) -> dict:
    r = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    r.raise_for_status()
    return r.json()


def get_state(task_id: str) -> dict:
    r = requests.get(f"{BASE_URL}/state", params={"task_id": task_id})
    r.raise_for_status()
    return r.json()


def step(task_id: str, action_type: str, target: str) -> dict:
    r = requests.post(
        f"{BASE_URL}/step",
        params={"task_id": task_id},
        json={"action_type": action_type, "target": target},
    )
    r.raise_for_status()
    return r.json()


def grade(task_id: str) -> dict:
    r = requests.get(f"{BASE_URL}/grader", params={"task_id": task_id})
    r.raise_for_status()
    return r.json()


def get_tasks() -> dict:
    r = requests.get(f"{BASE_URL}/tasks")
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────
def ask_llm(observation: dict, task_info: dict) -> dict:
    """Ask OpenAI LLM to decide next action based on current observation."""

    system_prompt = """You are an AI delivery agent operating in Mumbai, India.
Your goal is to pick up and deliver all packages as efficiently as possible.

Rules:
- You can only carry up to max_carry_capacity packages at once
- You must be AT the destination to deliver a package
- Moving costs fuel — manage it carefully
- Faster deliveries earn higher scores

Always respond with ONLY a valid JSON object like this:
{"action_type": "move", "target": "Andheri_Station"}

action_type must be one of: move, pick_up, deliver
target must be a valid location from the task locations list."""

    user_prompt = f"""Current observation:
- Location: {observation['current_location']}
- Carrying: {observation['carrying_packages']}
- Undelivered: {observation['undelivered_packages']}
- Delivered: {observation['delivered_packages']}
- Fuel remaining: {observation['fuel_remaining_L']}L
- Time elapsed: {observation['time_elapsed_min']} min
- Capacity: {len(observation['carrying_packages'])}/{observation['max_carry_capacity']}

Available locations: {task_info['locations']}

What is your next action? Respond with JSON only."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=100,
    )

    import json
    text = response.choices[0].message.content.strip()
    # Clean up any markdown formatting
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ─────────────────────────────────────────
# Run one episode
# ─────────────────────────────────────────
def run_episode(task_id: str, use_llm: bool = False) -> dict:
    print(f"\n{'='*50}")
    print(f"Starting episode: {task_id.upper()}")
    print(f"{'='*50}")

    # Get task info
    tasks = get_tasks()
    task_info = next(t for t in tasks["tasks"] if t["task_id"] == task_id)

    # Reset environment
    result = reset(task_id)
    obs = result["observation"]
    done = False
    step_count = 0

    print(f"Start: {obs['current_location']}")
    print(f"Packages to deliver: {obs['undelivered_packages']}")
    print(f"Fuel: {obs['fuel_remaining_L']}L\n")

    while not done:
        step_count += 1

        if use_llm and OPENAI_API_KEY:
            # LLM decides action
            try:
                action = ask_llm(obs, task_info)
            except Exception as e:
                print(f"LLM error: {e}, falling back to greedy")
                action = greedy_action(obs)
        else:
            # Greedy baseline
            action = greedy_action(obs)

        print(f"Step {step_count}: {action['action_type']} → {action['target']}")

        result = step(task_id, action["action_type"], action["target"])
        obs = result["observation"]
        done = result["done"]

        print(f"  Reward: {result['reward']['value']} | {result['reward']['reason']}")
        print(f"  Location: {obs['current_location']} | Fuel: {round(obs['fuel_remaining_L'], 1)}L | Time: {round(obs['time_elapsed_min'], 1)}min")

    # Grade the episode
    score = grade(task_id)
    print(f"\n📊 SCORE: {score['score']}")
    print(f"✅ Deliveries: {score['deliveries_completed']}/{score['total_deliveries']}")
    print(f"⛽ Fuel used: {score['fuel_used_L']}L")
    print(f"⏱️  Time: {score['time_elapsed_min']} min")
    print(f"📝 {score['reason']}")

    return score


# ─────────────────────────────────────────
# Greedy baseline agent
# ─────────────────────────────────────────
def greedy_action(obs: dict) -> dict:
    """Simple greedy agent — no LLM needed."""

    # Deliver if at destination
    for pkg in obs["carrying_packages"]:
        if pkg == obs["current_location"]:
            return {"action_type": "deliver", "target": pkg}

    # Pick up if capacity available
    if len(obs["carrying_packages"]) < obs["max_carry_capacity"]:
        for pkg in obs["undelivered_packages"]:
            if pkg not in obs["carrying_packages"]:
                return {"action_type": "pick_up", "target": pkg}

    # Move toward carrying package destination
    if obs["carrying_packages"]:
        return {"action_type": "move", "target": obs["carrying_packages"][0]}

    # Move toward nearest undelivered
    if obs["undelivered_packages"]:
        return {"action_type": "move", "target": obs["undelivered_packages"][0]}

    return {"action_type": "move", "target": obs["current_location"]}


# ─────────────────────────────────────────
# Main — runs all 3 tasks
# ─────────────────────────────────────────
if __name__ == "__main__":
    use_llm = bool(OPENAI_API_KEY)

    if use_llm:
        print("🤖 Using LLM agent (OpenAI)")
    else:
        print("🤖 Using greedy baseline agent (no API key set)")

    all_scores = {}
    for task_id in ["easy", "medium", "hard"]:
        score = run_episode(task_id, use_llm=use_llm)
        all_scores[task_id] = score["score"]

    print(f"\n{'='*50}")
    print("FINAL SCORES")
    print(f"{'='*50}")
    for task_id, score in all_scores.items():
        print(f"{task_id.upper()}: {score}")
    print(f"Average: {round(sum(all_scores.values()) / len(all_scores), 4)}")