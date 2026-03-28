import json
import os
from typing import Dict, List, Optional, Tuple
from server.models import (
    Observation, Action, Reward, StepResponse, TaskInfo, GraderResult
)

# ─────────────────────────────────────────
# Load real Mumbai distance matrix
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_PATH = os.path.join(BASE_DIR, "distance_matrix.json")

with open(MATRIX_PATH, "r") as f:
    _matrix_data = json.load(f)

DISTANCE_KM: Dict[str, Dict[str, float]] = _matrix_data["distance_km"]
DURATION_MIN: Dict[str, Dict[str, float]] = _matrix_data["duration_min"]
ALL_LOCATIONS: List[str] = _matrix_data["locations"]

# ─────────────────────────────────────────
# Task Definitions
# ─────────────────────────────────────────
TASKS = {
    "easy": TaskInfo(
        task_id="easy",
        name="Local Andheri Delivery",
        difficulty="easy",
        description=(
            "Deliver 2 packages within the central Andheri area. "
            "Short distances, no time pressure, low complexity."
        ),
        locations=[
            "NESCO_Warehouse",
            "Andheri_Station",
            "Jogeshwari_West",
            "Reliance_Digital",
            "Andheri_East",
        ],
        packages=["Andheri_Station", "Reliance_Digital"],
        max_steps=20,
        max_carry_capacity=2,
        fuel_capacity_L=10.0,
        time_limit_min=60.0,
    ),
    "medium": TaskInfo(
        task_id="medium",
        name="Andheri-Jogeshwari Multi-Stop",
        difficulty="medium",
        description=(
            "Deliver 4 packages across Andheri and Jogeshwari. "
            "More stops, fuel management needed."
        ),
        locations=[
            "NESCO_Warehouse",
            "Andheri_Station",
            "Jogeshwari_West",
            "Reliance_Digital",
            "Andheri_East",
            "Mahakali_Caves",
            "Lokhandwala_Market",
            "Decathlon_Andheri",
        ],
        packages=[
            "Andheri_Station",
            "Mahakali_Caves",
            "Lokhandwala_Market",
            "Decathlon_Andheri",
        ],
        max_steps=40,
        max_carry_capacity=3,
        fuel_capacity_L=20.0,
        time_limit_min=120.0,
    ),
    "hard": TaskInfo(
        task_id="hard",
        name="Mumbai Full Zone Delivery",
        difficulty="hard",
        description=(
            "Deliver 6 packages across the full 15km Mumbai zone. "
            "Tight time windows, fuel constraints, maximum complexity."
        ),
        locations=ALL_LOCATIONS,
        packages=[
            "Versova_Beach",
            "Inorbit_Mall_Malad",
            "Powai_Lake",
            "IIT_Bombay",
            "Bangur_Nagar",
            "JVLR_Junction",
        ],
        max_steps=60,
        max_carry_capacity=3,
        fuel_capacity_L=30.0,
        time_limit_min=180.0,
    ),
}

# ─────────────────────────────────────────
# Fuel consumption rate
# ─────────────────────────────────────────
FUEL_PER_KM = 0.5  # litres per km


# ─────────────────────────────────────────
# Environment Class
# ─────────────────────────────────────────
class DeliveryEnvironment:

    def __init__(self, task_id: str = "easy"):
        if task_id not in TASKS:
            raise ValueError(f"task_id must be one of {list(TASKS.keys())}")
        self.task: TaskInfo = TASKS[task_id]
        self._state: Optional[Observation] = None
        self._steps: int = 0
        self._done: bool = False

    # ──────────────────────────────────────
    # reset()
    # ──────────────────────────────────────
    def reset(self) -> Observation:
        self._steps = 0
        self._done = False
        self._state = Observation(
            current_location="NESCO_Warehouse",
            undelivered_packages=list(self.task.packages),
            delivered_packages=[],
            distance_travelled_km=0.0,
            time_elapsed_min=0.0,
            fuel_remaining_L=self.task.fuel_capacity_L,
            carrying_packages=[],
            max_carry_capacity=self.task.max_carry_capacity,
        )
        return self._state

    # ──────────────────────────────────────
    # state()
    # ──────────────────────────────────────
    def state(self) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return self._state

    # ──────────────────────────────────────
    # step()
    # ──────────────────────────────────────
    def step(self, action: Action) -> StepResponse:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start again.")
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        self._steps += 1
        reward_value = 0.0
        reward_reason = ""

        # ── ACTION: move ──
        if action.action_type == "move":
            target = action.target

            # Invalid location
            if target not in self.task.locations:
                reward_value = -2.0
                reward_reason = f"Invalid location: {target}"

            # Already here
            elif target == self._state.current_location:
                reward_value = -1.0
                reward_reason = "Already at this location"

            else:
                dist = DISTANCE_KM[self._state.current_location][target]
                time = DURATION_MIN[self._state.current_location][target]
                fuel_used = dist * FUEL_PER_KM

                # Out of fuel
                if fuel_used > self._state.fuel_remaining_L:
                    reward_value = -10.0
                    reward_reason = "Out of fuel!"
                    self._done = True
                else:
                    # Valid move
                    self._state.current_location = target
                    self._state.distance_travelled_km += dist
                    self._state.time_elapsed_min += time
                    self._state.fuel_remaining_L -= fuel_used

                    # Small penalty per km to encourage efficiency
                    reward_value = -dist * 0.1
                    reward_reason = f"Moved to {target} ({dist}km, {time}min)"

                    # Bonus if carrying packages (productive move)
                    if self._state.carrying_packages:
                        reward_value += 0.5
                        reward_reason += " [carrying packages]"

        # ── ACTION: pick_up ──
        elif action.action_type == "pick_up":
            target = action.target

            # Package not available
            if target not in self._state.undelivered_packages:
                reward_value = -2.0
                reward_reason = f"Package for {target} not available"

            # Already carrying it
            elif target in self._state.carrying_packages:
                reward_value = -1.0
                reward_reason = f"Already carrying package for {target}"

            # Over capacity
            elif len(self._state.carrying_packages) >= self._state.max_carry_capacity:
                reward_value = -2.0
                reward_reason = "Vehicle at full capacity"

            else:
                self._state.carrying_packages.append(target)
                reward_value = 0.5
                reward_reason = f"Picked up package for {target}"

        # ── ACTION: deliver ──
        elif action.action_type == "deliver":
            target = action.target

            # Not carrying this package
            if target not in self._state.carrying_packages:
                reward_value = -2.0
                reward_reason = f"Not carrying package for {target}"

            # Wrong location
            elif self._state.current_location != target:
                reward_value = -3.0
                reward_reason = (
                    f"Wrong location. Must be at {target} "
                    f"but you are at {self._state.current_location}"
                )

            else:
                # Successful delivery!
                self._state.carrying_packages.remove(target)
                self._state.undelivered_packages.remove(target)
                self._state.delivered_packages.append(target)

                # Reward based on how fast delivery was
                time_bonus = max(
                    0,
                    (self.task.time_limit_min - self._state.time_elapsed_min)
                    / self.task.time_limit_min,
                )
                reward_value = 10.0 + (5.0 * time_bonus)
                reward_reason = f"Delivered to {target}! Time bonus: {round(time_bonus, 2)}"

        # ── Invalid action type ──
        else:
            reward_value = -2.0
            reward_reason = f"Unknown action type: {action.action_type}"

        # ── Check termination conditions ──
        # All delivered
        if not self._state.undelivered_packages and not self._state.carrying_packages:
            self._done = True
            reward_value += 20.0  # Big bonus for completing all deliveries
            reward_reason += " | ALL DELIVERIES COMPLETE! 🎉"

        # Time limit exceeded
        if self._state.time_elapsed_min >= self.task.time_limit_min:
            self._done = True
            reward_reason += " | TIME LIMIT REACHED"

        # Max steps exceeded
        if self._steps >= self.task.max_steps:
            self._done = True
            reward_reason += " | MAX STEPS REACHED"

        return StepResponse(
            observation=self._state,
            reward=Reward(value=round(reward_value, 3), reason=reward_reason),
            done=self._done,
            info={
                "steps": self._steps,
                "max_steps": self.task.max_steps,
                "deliveries_done": len(self._state.delivered_packages),
                "total_deliveries": len(self.task.packages),
            },
        )

    # ──────────────────────────────────────
    # grade() — returns score 0.0 to 1.0
    # ──────────────────────────────────────
    def grade(self) -> GraderResult:
        if self._state is None:
            raise RuntimeError("Call reset() and run episode before grading.")

        total = len(self.task.packages)
        delivered = len(self._state.delivered_packages)

        # Base score from deliveries
        delivery_score = delivered / total if total > 0 else 0.0

        # Efficiency bonus (fuel saved)
        fuel_used = self.task.fuel_capacity_L - self._state.fuel_remaining_L
        fuel_efficiency = max(
            0, 1.0 - (fuel_used / self.task.fuel_capacity_L)
        )

        # Time efficiency bonus
        time_efficiency = max(
            0,
            1.0 - (self._state.time_elapsed_min / self.task.time_limit_min),
        )

        # Final score weighted
        score = (
            delivery_score * 0.7
            + fuel_efficiency * 0.15
            + time_efficiency * 0.15
        )

        return GraderResult(
            task_id=self.task.task_id,
            score=round(min(score, 1.0), 4),
            deliveries_completed=delivered,
            total_deliveries=total,
            distance_travelled_km=round(self._state.distance_travelled_km, 2),
            time_elapsed_min=round(self._state.time_elapsed_min, 2),
            fuel_used_L=round(fuel_used, 2),
            reason=(
                f"{delivered}/{total} deliveries | "
                f"fuel efficiency: {round(fuel_efficiency, 2)} | "
                f"time efficiency: {round(time_efficiency, 2)}"
            ),
        )