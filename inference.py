import os
import requests

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")


def run_inference(task_id: str) -> dict:
    # Reset
    r = requests.post(
        f"{BASE_URL}/reset",
        params={"task_id": task_id}
    )
    obs = r.json()["observation"]
    done = False

    while not done:
        action = greedy_action(obs)
        if action is None:
            break
        r = requests.post(
            f"{BASE_URL}/step",
            params={"task_id": task_id},
            json=action
        )
        data = r.json()
        obs = data["observation"]
        done = data["done"]

    # Grade
    r = requests.get(
        f"{BASE_URL}/grader",
        params={"task_id": task_id}
    )
    return r.json()


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
            (w for w in warehouses
             if w["name"] == next_pkg["warehouse"]),
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


if __name__ == "__main__":
    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        result = run_inference(task_id)
        scores[task_id] = result["score"]
        print(f"{task_id}: {result['score']}")

    print(f"\nAverage: {sum(scores.values()) / len(scores):.4f}")