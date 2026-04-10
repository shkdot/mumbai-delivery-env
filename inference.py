import os
import requests

# ── Required environment variables ──
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://abubakarshk-mumbai-delivery-env.hf.space"
)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_URL = API_BASE_URL.rstrip("/")


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


def run_inference(task_id: str):
    try:
        # Reset environment
        r = requests.post(
            f"{BASE_URL}/reset",
            params={"task_id": task_id},
            timeout=30
        )
        r.raise_for_status()
        obs = r.json()["observation"]
        done = False
        step_count = 0
        total_reward = 0.0

        print(f"[START] task={task_id}", flush=True)

        while not done:
            action = greedy_action(obs)
            if action is None:
                break

            r = requests.post(
                f"{BASE_URL}/step",
                params={"task_id": task_id},
                json=action,
                timeout=30
            )
            r.raise_for_status()
            data = r.json()
            obs = data["observation"]
            done = data["done"]
            reward = data["reward"]["value"]
            total_reward += reward
            step_count += 1

            print(
                f"[STEP] step={step_count} "
                f"action={action['action_type']} "
                f"target={action['target']} "
                f"reward={reward}",
                flush=True
            )

        # Grade
        r = requests.get(
            f"{BASE_URL}/grader",
            params={"task_id": task_id},
            timeout=30
        )
        r.raise_for_status()
        result = r.json()

        print(
            f"[END] task={task_id} "
            f"score={result['score']} "
            f"steps={step_count} "
            f"deliveries={result['deliveries_completed']}/"
            f"{result['total_deliveries']}",
            flush=True
        )

        return result

    except Exception as e:
        print(f"[END] task={task_id} score=0.0 steps=0 error={str(e)}", flush=True)
        return {"score": 0.0}


if __name__ == "__main__":
    for task_id in ["easy", "medium", "hard"]:
        run_inference(task_id)