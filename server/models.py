from pydantic import BaseModel
from typing import List, Dict, Optional

# ─────────────────────────────────────────
# Observation — what the AI agent sees
# ─────────────────────────────────────────
class Observation(BaseModel):
    current_location: str
    undelivered_packages: List[str]  # list of destination names
    delivered_packages: List[str]
    distance_travelled_km: float
    time_elapsed_min: float
    fuel_remaining_L: float
    carrying_packages: List[str]     # packages currently on vehicle
    max_carry_capacity: int

# ─────────────────────────────────────────
# Action — what the AI agent can do
# ─────────────────────────────────────────
class Action(BaseModel):
    action_type: str   # "move" or "deliver"
    target: str        # location name to move to OR package destination to deliver

# ─────────────────────────────────────────
# Reward — feedback after each step
# ─────────────────────────────────────────
class Reward(BaseModel):
    value: float
    reason: str

# ─────────────────────────────────────────
# Step Response — returned after every step
# ─────────────────────────────────────────
class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict

# ─────────────────────────────────────────
# Task Definition
# ─────────────────────────────────────────
class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: str       # "easy", "medium", "hard"
    description: str
    locations: List[str]
    packages: List[str]   # delivery destinations
    max_steps: int
    max_carry_capacity: int
    fuel_capacity_L: float
    time_limit_min: float

# ─────────────────────────────────────────
# Grader Result
# ─────────────────────────────────────────
class GraderResult(BaseModel):
    task_id: str
    score: float          # 0.0 to 1.0
    deliveries_completed: int
    total_deliveries: int
    distance_travelled_km: float
    time_elapsed_min: float
    fuel_used_L: float
    reason: str