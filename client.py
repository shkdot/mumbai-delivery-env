import os
import requests
from typing import Optional

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

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
# Greedy baseline agent
# ─────────────────────────────────────────
def greedy_action(obs: dict) -> Optional[dict]:
    current = obs["current_location"]
    carrying = obs["carrying_packages"]
    undelivered = obs["undelivered_packages"]
    warehouses = obs["warehouses"]
    capacity = obs["max_carry_capacity"]

    # 1. Deliver if at destination
    for pkg in carrying:
        if pkg["destination"] == current:
            return {
                "action_type": "deliver",
                "target": pkg["package_id"]
            }

    # 2. Move toward carrying package destination
    if carrying:
        return {
            "action_type": "move",
            "target": carrying[0]["destination"]
        }

    # 3. Pick up if at warehouse and capacity available
    if undelivered and len(carrying) < capacity:
        next_pkg = undelivered[0]
        wh = next(
            (w for w in warehouses if w["name"] == next_pkg["warehouse"]),
            warehouses[0]
        )
        wh_loc = wh["location"]

        if current == wh_loc:
            return {
                "action_type": "pick_up",
                "target": next_pkg["package_id"]
            }
        else:
            return {
                "action_type": "move",
                "target": wh_loc
            }

    return None

# ─────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────
def ask_llm(obs: dict, task_info: dict) -> Optional[dict]:
    from openai import OpenAI
    import json

    client = OpenAI(api_key=OPENAI_API_KEY)

    carrying = obs["carrying_packages"]
    undelivered = obs["undelivered_packages"]
    warehouses = obs["warehouses"]

    system_prompt = """You are an AI delivery agent in Mumbai, India.
Your goal is to pick up and deliver all packages as efficiently as possible.

Rules:
- Use pick_up with package_id to pick up a package (must be at correct warehouse)
- Use deliver with package_id to deliver a package (must be at destination)
- Use move with location name to travel to a location
- Manage fuel carefully

Always respond with ONLY valid JSON:
{"action_type": "move", "target": "Andheri_Station"}
{"action_type": "pick_up", "target": "PKG_001"}
{"action_type": "deliver", "target": "PKG_001"}"""

    user_prompt = f"""Current State:
- Location: {obs['current_location']}
- Carrying: {[(p['package_id'], p['destination']) for p in carrying]}
- Undelivered: {[(p['package_id'], p['destination'], p['warehouse']) for p in undelivered]}
- Delivered: {len(obs['delivered_packages'])} packages
- Fuel: {obs['fuel_remaining_L']}L
- Time: {obs['time_elapsed_min']} min
- Traffic: {obs.get('time_of_day', 'non_peak')}
- Capacity: {len(carrying)}/{obs['max_carry_capacity']}
- Warehouses: {[(w['name'], w['location']) for w in warehouses]}

What is your next action? JSON only."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"  LLM error: {e}, falling back to greedy")
        return greedy_action(obs)

# ─────────────────────────────────────────
# Run one episode
# ─────────────────────────────────────────
def run_episode(task_id: str, use_llm: bool = False) -> dict:
    print(f"\n{'='*50}")
    print(f"Starting episode: {task_id.upper()}")
    print(f"{'='*50}")

    # Reset
    result = reset(task_id)
    obs = result["observation"]
    done = False
    step_count = 0

    print(f"Start: {obs['current_location']}")
    print(f"Packages: {len(obs['undelivered_packages'])}")
    print(f"Fuel: {obs['fuel_remaining_L']}L")
    print(f"Vehicle: {obs['mileage_km_per_L']} km/L\n")

    while not done:
        step_count += 1

        if use_llm and OPENAI_API_KEY:
            action = ask_llm(obs, {})
        else:
            action = greedy_action(obs)

        if action is None:
            print("No action available — ending episode")
            break

        print(f"Step {step_count}: "
              f"{action['action_type']} → {action['target']}")

        result = step(task_id, action["action_type"], action["target"])
        obs = result["observation"]
        done = result["done"]
        info = result.get("info", {})

        print(f"  Reward: {result['reward']['value']} "
              f"| {result['reward']['reason'][:60]}")
        print(f"  📍 {obs['current_location']} "
              f"| ⛽ {round(obs['fuel_remaining_L'], 2)}L "
              f"| ⏱ {round(obs['time_elapsed_min'], 1)}min "
              f"| 🚦 {obs.get('time_of_day', 'non_peak')}")

        if info.get("new_order_added"):
            print(f"  ⚡ NEW ORDER ARRIVED!")

    # Grade
    score = grade(task_id)
    print(f"\n{'─'*50}")
    print(f"📊 SCORE:       {score['score']}")
    print(f"✅ Deliveries:  "
          f"{score['deliveries_completed']}/{score['total_deliveries']}")
    print(f"⛽ Fuel used:   {score['fuel_used_L']}L")
    print(f"⏱  Time:        {score['time_elapsed_min']} min")
    print(f"🚦 Signal delay:{score['total_signal_delays_min']} min")
    print(f"📝 {score['reason']}")

    return score

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    use_llm = bool(OPENAI_API_KEY)

    if use_llm:
        print("🤖 Using LLM agent (OpenAI)")
    else:
        print("🤖 Using greedy baseline agent (no API key)")

    all_scores = {}
    for task_id in ["easy", "medium", "hard"]:
        score = run_episode(task_id, use_llm=use_llm)
        all_scores[task_id] = score["score"]

    print(f"\n{'='*50}")
    print("FINAL SCORES")
    print(f"{'='*50}")
    for task_id, score in all_scores.items():
        print(f"{task_id.upper():10} {score}")
    print(f"{'─'*50}")
    print(f"{'Average':10} "
          f"{round(sum(all_scores.values())/len(all_scores), 4)}")