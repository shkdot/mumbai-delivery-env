import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from server.models import Action, Observation, StepResponse, TaskInfo, GraderResult
from server.my_hackathon_env_environment import DeliveryEnvironment, TASKS

# ─────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────
app = FastAPI(
    title="Mumbai Delivery RL Environment",
    description="A real-world delivery optimization environment for RL agents based on Mumbai road network.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Active environments store
# ─────────────────────────────────────────
_envs: Dict[str, DeliveryEnvironment] = {}


def _get_env(task_id: str) -> DeliveryEnvironment:
    if task_id not in _envs:
        raise HTTPException(
            status_code=404,
            detail=f"No active session for task '{task_id}'. Call /reset first."
        )
    return _envs[task_id]


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Mumbai Delivery RL Environment",
        "version": "1.0.0",
        "endpoints": ["/tasks", "/reset", "/step", "/state", "/grader", "/baseline"],
    }


# ── /tasks ──────────────────────────────
@app.get("/tasks")
def list_tasks():
    """Returns all available tasks and their action schema."""
    return {
        "tasks": [task.dict() for task in TASKS.values()],
        "action_schema": {
            "action_type": "string — one of: 'move', 'pick_up', 'deliver'",
            "target": "string — location name (see task.locations)",
        },
    }


# ── /reset ──────────────────────────────
@app.post("/reset")
def reset(task_id: str = "easy"):
    """Reset environment for a given task. Returns initial observation."""
    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id. Choose from: {list(TASKS.keys())}"
        )
    env = DeliveryEnvironment(task_id=task_id)
    _envs[task_id] = env
    obs = env.reset()
    return {"task_id": task_id, "observation": obs.dict()}


# ── /state ──────────────────────────────
@app.get("/state")
def get_state(task_id: str = "easy"):
    """Returns current state without taking any action."""
    env = _get_env(task_id)
    return {"task_id": task_id, "observation": env.state().dict()}


# ── /step ───────────────────────────────
@app.post("/step")
def step(action: Action, task_id: str = "easy"):
    """Take one action in the environment."""
    env = _get_env(task_id)
    result: StepResponse = env.step(action)
    return result.dict()


# ── /grader ─────────────────────────────
@app.get("/grader")
def grade(task_id: str = "easy"):
    """Returns grader score (0.0 to 1.0) for the current episode."""
    env = _get_env(task_id)
    result: GraderResult = env.grade()
    return result.dict()


# ── /baseline ───────────────────────────
@app.get("/baseline")
def baseline():
    """
    Runs a simple greedy baseline agent on all 3 tasks.
    Returns scores for easy, medium, hard.
    """
    results = {}

    for task_id in TASKS.keys():
        env = DeliveryEnvironment(task_id=task_id)
        obs = env.reset()
        done = False

        while not done:
            # Greedy strategy:
            # 1. If not at capacity, pick up nearest available package
            # 2. If carrying packages, deliver if at destination
            # 3. Otherwise move toward nearest undelivered destination

            action = None

            # Try to pick up packages if not at capacity
            if (len(obs.carrying_packages) < obs.max_carry_capacity
                    and obs.undelivered_packages):
                for pkg in obs.undelivered_packages:
                    if pkg not in obs.carrying_packages:
                        action = Action(action_type="pick_up", target=pkg)
                        break

            # Try to deliver if at destination
            if action is None and obs.carrying_packages:
                for pkg in obs.carrying_packages:
                    if pkg == obs.current_location:
                        action = Action(action_type="deliver", target=pkg)
                        break

            # Move toward nearest carrying package destination
            if action is None and obs.carrying_packages:
                from server.my_hackathon_env_environment import DISTANCE_KM
                nearest = min(
                    obs.carrying_packages,
                    key=lambda p: DISTANCE_KM[obs.current_location].get(p, 999)
                )
                action = Action(action_type="move", target=nearest)

            # Move toward nearest undelivered package
            elif action is None and obs.undelivered_packages:
                from server.my_hackathon_env_environment import DISTANCE_KM
                nearest = min(
                    obs.undelivered_packages,
                    key=lambda p: DISTANCE_KM[obs.current_location].get(p, 999)
                )
                action = Action(action_type="move", target=nearest)

            # Nothing to do
            if action is None:
                break

            result = env.step(action)
            obs = result.observation
            done = result.done

        grade = env.grade()
        results[task_id] = {
            "score": grade.score,
            "deliveries_completed": grade.deliveries_completed,
            "total_deliveries": grade.total_deliveries,
            "distance_travelled_km": grade.distance_travelled_km,
            "time_elapsed_min": grade.time_elapsed_min,
            "fuel_used_L": grade.fuel_used_L,
            "reason": grade.reason,
        }

    return {"baseline_results": results}