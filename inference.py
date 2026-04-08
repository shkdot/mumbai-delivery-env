import os
import requests
import sys

BASE_URL = os.getenv("API_BASE_URL") or os.getenv("OPENENV_BASE_URL") or "http://localhost:7860"


def greedy_action(obs: dict):
    current = obs["current_location"]
    carrying = obs["carrying_packages"]
    undelivered = obs["undelivered_packages"]
    warehouses = obs["warehouses"]
    capacity = obs["max_carry_capacity"]

    for pkg in carrying:
        if pkg["destination"] == current:
            return {
                "action_type": "deliver",
                "target": pkg["package_id"]
            }

    if carrying:
        return {
            "action_type": "move",
            "target": carrying[0]["destination"]
        }

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
        return {"action_type": "move", "target": wh_loc}

    return None


def run_inference(task_id: str):
    r = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    r.raise_for_status()
    obs = r.json()["observation"]

    done = False
    step_count = 0

    print(f"[START] task={task_id}", flush=True)

    while not done:
        action = greedy_action(obs)
        if action is None:
            break

        r = requests.post(
            f"{BASE_URL}/step",
            params={"task_id": task_id},
            json=action
        )
        r.raise_for_status()

        data = r.json()
        obs = data["observation"]
        done = data["done"]
        reward = data["reward"]["value"]
        step_count += 1

        print(f"[STEP] step={step_count} reward={reward}", flush=True)

    r = requests.get(f"{BASE_URL}/grader", params={"task_id": task_id})
    r.raise_for_status()
    result = r.json()

    print(
        f"[END] task={task_id} score={result['score']} steps={step_count}",
        flush=True
    )


if __name__ == "__main__":
    for task_id in ["easy", "medium", "hard"]:
        run_inference(task_id)