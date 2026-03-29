import json
import os
import random
from typing import Dict, List, Optional
from server.models import (
    Observation, Action, Reward, StepResponse,
    TaskInfo, GraderResult, Package, Warehouse, Vehicle, SignalInfo
)

# ─────────────────────────────────────────
# Load real Mumbai data
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "distance_matrix.json"), "r") as f:
    _matrix_data = json.load(f)

with open(os.path.join(BASE_DIR, "signal_matrix.json"), "r") as f:
    _signal_data = json.load(f)

DISTANCE_KM: Dict[str, Dict[str, float]] = _matrix_data["distance_km"]
DURATION_MIN: Dict[str, Dict[str, float]] = _matrix_data["duration_min"]
ALL_LOCATIONS: List[str] = _matrix_data["locations"]
SIGNAL_COUNTS: Dict[str, Dict[str, int]] = _signal_data["signals"]

# ─────────────────────────────────────────
# Vehicles
# ─────────────────────────────────────────
SCOOTER = Vehicle(
    name="Scooter (Activa)",
    fuel_capacity_L=5.0,
    mileage_km_per_L=45.0,
)

BIKE = Vehicle(
    name="Bike (Splendor)",
    fuel_capacity_L=10.0,
    mileage_km_per_L=70.0,
)

# ─────────────────────────────────────────
# Warehouses
# ─────────────────────────────────────────
WAREHOUSE_NESCO = Warehouse(
    name="NESCO_Warehouse",
    location="NESCO_Warehouse",
    zone="north",
)

WAREHOUSE_MAHAKALI = Warehouse(
    name="Mahakali_Warehouse",
    location="Mahakali_Caves",
    zone="east",
)

WAREHOUSE_LOKHANDWALA = Warehouse(
    name="Lokhandwala_Warehouse",
    location="Lokhandwala_Market",
    zone="west",
)

# ─────────────────────────────────────────
# Peak hour logic
# ─────────────────────────────────────────
EPISODE_START_HOUR = 9  # Episodes start at 9am


def get_time_of_day(time_elapsed_min: float) -> str:
    hour = (EPISODE_START_HOUR + time_elapsed_min / 60) % 24
    if 8 <= hour <= 11:
        return "morning_peak"
    elif 17 <= hour <= 21:
        return "evening_peak"
    return "non_peak"


def is_peak(time_elapsed_min: float) -> bool:
    tod = get_time_of_day(time_elapsed_min)
    return tod in ("morning_peak", "evening_peak")


# ─────────────────────────────────────────
# Signal delay calculation
# ─────────────────────────────────────────
def calculate_signal_delay(
    origin: str,
    destination: str,
    time_elapsed_min: float,
    peak_enabled: bool,
) -> tuple:
    num_signals = SIGNAL_COUNTS.get(origin, {}).get(destination, 3)
    peak = is_peak(time_elapsed_min) if peak_enabled else False
    red_prob = 0.70 if peak else 0.35

    total_delay = 0.0
    red_count = 0

    for _ in range(num_signals):
        if random.random() < red_prob:
            wait_sec = random.uniform(30, 90)
            red_count += 1
        else:
            wait_sec = random.uniform(0, 8)
        total_delay += wait_sec / 60

    return round(total_delay, 2), red_count, num_signals


# ─────────────────────────────────────────
# Package ID generator
# ─────────────────────────────────────────
def make_package_id(index: int) -> str:
    return f"PKG_{index:03d}"


# ─────────────────────────────────────────
# Nearest warehouse finder
# ─────────────────────────────────────────
def nearest_warehouse(
    location: str,
    warehouses: List[Warehouse]
) -> Warehouse:
    return min(
        warehouses,
        key=lambda w: DISTANCE_KM.get(location, {}).get(w.location, 999)
    )


# ─────────────────────────────────────────
# Task Definitions
# ─────────────────────────────────────────
EASY_LOCATIONS = [
    "NESCO_Warehouse",
    "Andheri_Station",
    "Jogeshwari_West",
    "Reliance_Digital",
    "Andheri_East",
]

MEDIUM_LOCATIONS = [
    "NESCO_Warehouse",
    "Andheri_Station",
    "Jogeshwari_West",
    "Reliance_Digital",
    "Andheri_East",
    "Mahakali_Caves",
    "Lokhandwala_Market",
    "Decathlon_Andheri",
]

TASKS = {
    "easy": TaskInfo(
        task_id="easy",
        name="Local Andheri Delivery",
        difficulty="easy",
        description=(
            "Deliver 2 packages in the Andheri zone using a scooter. "
            "No traffic signals, fixed orders, simple routing."
        ),
        locations=EASY_LOCATIONS,
        warehouses=[WAREHOUSE_NESCO],
        initial_packages=2,
        max_steps=20,
        max_carry_capacity=2,
        vehicle=SCOOTER,
        time_limit_min=60.0,
        signals_enabled=False,
        peak_hours_enabled=False,
        dynamic_orders=False,
        new_order_probability=0.0,
    ),
    "medium": TaskInfo(
        task_id="medium",
        name="Andheri Multi-Stop Delivery",
        difficulty="medium",
        description=(
            "Deliver 4 packages across Andheri and Jogeshwari. "
            "Non-peak traffic signals add variable delays. "
            "Manage fuel carefully on a scooter."
        ),
        locations=MEDIUM_LOCATIONS,
        warehouses=[WAREHOUSE_NESCO],
        initial_packages=4,
        max_steps=40,
        max_carry_capacity=3,
        vehicle=SCOOTER,
        time_limit_min=120.0,
        signals_enabled=True,
        peak_hours_enabled=False,
        dynamic_orders=False,
        new_order_probability=0.0,
    ),
    "hard": TaskInfo(
        task_id="hard",
        name="Mumbai Full Zone Rush Delivery",
        difficulty="hard",
        description=(
            "Deliver 6 packages across the full 15km Mumbai zone "
            "using a bike. 3 dark stores, peak hour signals, "
            "dynamic new orders mid-episode. "
            "Agent must decide optimal warehouse and route."
        ),
        locations=ALL_LOCATIONS,
        warehouses=[
            WAREHOUSE_NESCO,
            WAREHOUSE_MAHAKALI,
            WAREHOUSE_LOKHANDWALA,
        ],
        initial_packages=6,
        max_steps=80,
        max_carry_capacity=3,
        vehicle=BIKE,
        time_limit_min=180.0,
        signals_enabled=True,
        peak_hours_enabled=True,
        dynamic_orders=True,
        new_order_probability=0.15,
    ),
}

# Possible destinations for dynamic new orders (hard task)
HARD_DESTINATIONS = [
    "Versova_Beach", "Inorbit_Mall_Malad", "Powai_Lake",
    "IIT_Bombay", "Bangur_Nagar", "JVLR_Junction",
    "Goregaon_East", "Decathlon_Andheri", "Andheri_East",
]


# ─────────────────────────────────────────
# Environment Class
# ─────────────────────────────────────────
class DeliveryEnvironment:

    def __init__(self, task_id: str = "easy"):
        if task_id not in TASKS:
            raise ValueError(f"task_id must be one of: {list(TASKS.keys())}")
        self.task: TaskInfo = TASKS[task_id]
        self._state: Optional[Observation] = None
        self._steps: int = 0
        self._done: bool = False
        self._pkg_counter: int = 0
        self._total_signal_delay: float = 0.0
        self._total_deliveries_including_dynamic: int = 0

    # ─────────────────────────────────────
    # Package factory
    # ─────────────────────────────────────
    def _make_package(
        self,
        destination: str,
        is_new: bool = False
    ) -> Package:
        self._pkg_counter += 1
        wh = nearest_warehouse(destination, self.task.warehouses)
        return Package(
            package_id=make_package_id(self._pkg_counter),
            destination=destination,
            warehouse=wh.name,
            is_new_order=is_new,
        )

    # ─────────────────────────────────────
    # reset()
    # ─────────────────────────────────────
    def reset(self) -> Observation:
        self._steps = 0
        self._done = False
        self._pkg_counter = 0
        self._total_signal_delay = 0.0

        # Build initial packages
        if self.task.task_id == "easy":
            destinations = ["Andheri_Station", "Reliance_Digital"]
        elif self.task.task_id == "medium":
            destinations = [
                "Andheri_Station", "Mahakali_Caves",
                "Lokhandwala_Market", "Decathlon_Andheri",
            ]
        else:  # hard
            destinations = [
                "Versova_Beach", "Inorbit_Mall_Malad",
                "Powai_Lake", "IIT_Bombay",
                "Bangur_Nagar", "JVLR_Junction",
            ]

        packages = [self._make_package(d) for d in destinations]
        self._total_deliveries_including_dynamic = len(packages)

        self._state = Observation(
            current_location="NESCO_Warehouse",
            undelivered_packages=packages,
            delivered_packages=[],
            carrying_packages=[],
            max_carry_capacity=self.task.max_carry_capacity,
            fuel_remaining_L=self.task.vehicle.fuel_capacity_L,
            mileage_km_per_L=self.task.vehicle.mileage_km_per_L,
            distance_travelled_km=0.0,
            time_elapsed_min=0.0,
            warehouses=self.task.warehouses,
            is_peak_hour=is_peak(0.0),
            time_of_day=get_time_of_day(0.0),
            new_order_available=False,
        )
        return self._state

    # ─────────────────────────────────────
    # state()
    # ─────────────────────────────────────
    def state(self) -> Observation:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ─────────────────────────────────────
    # _maybe_add_new_order()
    # ─────────────────────────────────────
    def _maybe_add_new_order(self) -> bool:
        """Randomly add a new order mid-episode for hard task."""
        if not self.task.dynamic_orders:
            return False
        if random.random() > self.task.new_order_probability:
            return False

        # Pick a random destination not already in undelivered
        current_dests = {
            p.destination for p in self._state.undelivered_packages
        }
        current_dests |= {
            p.destination for p in self._state.carrying_packages
        }

        available = [
            d for d in HARD_DESTINATIONS
            if d not in current_dests
        ]
        if not available:
            return False

        destination = random.choice(available)
        new_pkg = self._make_package(destination, is_new=True)
        self._state.undelivered_packages.append(new_pkg)
        self._total_deliveries_including_dynamic += 1
        return True

    # ─────────────────────────────────────
    # step()
    # ─────────────────────────────────────
    def step(self, action: Action) -> StepResponse:
        if self._done:
            raise RuntimeError("Episode done. Call reset().")
        if self._state is None:
            raise RuntimeError("Call reset() first.")

        self._steps += 1
        reward_value = 0.0
        reward_reason = ""
        signal_info = {}
        new_order_added = False

        # ── ACTION: move ──
        if action.action_type == "move":
            target = action.target

            if target not in self.task.locations:
                reward_value = -2.0
                reward_reason = f"Invalid location: {target}"

            elif target == self._state.current_location:
                reward_value = -1.0
                reward_reason = "Already at this location"

            else:
                dist = DISTANCE_KM[self._state.current_location][target]
                base_time = DURATION_MIN[self._state.current_location][target]
                fuel_used = dist / self.task.vehicle.mileage_km_per_L

                if fuel_used > self._state.fuel_remaining_L:
                    reward_value = -10.0
                    reward_reason = "Out of fuel! Episode ending."
                    self._done = True

                else:
                    # Signal delay
                    signal_delay = 0.0
                    red_signals = 0
                    total_signals = 0

                    if self.task.signals_enabled:
                        signal_delay, red_signals, total_signals = \
                            calculate_signal_delay(
                                self._state.current_location,
                                target,
                                self._state.time_elapsed_min,
                                self.task.peak_hours_enabled,
                            )
                        self._total_signal_delay += signal_delay

                    actual_time = base_time + signal_delay

                    # Update state
                    self._state.current_location = target
                    self._state.distance_travelled_km += dist
                    self._state.time_elapsed_min += actual_time
                    self._state.fuel_remaining_L = round(
                        self._state.fuel_remaining_L - fuel_used, 4
                    )
                    self._state.is_peak_hour = is_peak(
                        self._state.time_elapsed_min
                    )
                    self._state.time_of_day = get_time_of_day(
                        self._state.time_elapsed_min
                    )

                    # Reward
                    reward_value = -dist * 0.1
                    reward_reason = (
                        f"Moved to {target} "
                        f"({dist}km, {round(actual_time, 1)}min)"
                    )

                    if total_signals > 0:
                        reward_value -= signal_delay * 0.2
                        reward_reason += (
                            f" | {red_signals}/{total_signals} red signals"
                            f" +{round(signal_delay, 1)}min delay"
                        )
                        signal_info = {
                            "total_signals": total_signals,
                            "red_signals": red_signals,
                            "signal_delay_min": signal_delay,
                            "is_peak_hour": self._state.is_peak_hour,
                        }

                    if self._state.carrying_packages:
                        reward_value += 0.5
                        reward_reason += " [carrying packages +0.5]"

                    # Maybe add new order
                    new_order_added = self._maybe_add_new_order()
                    if new_order_added:
                        reward_reason += " | ⚡ NEW ORDER ARRIVED!"

        # ── ACTION: pick_up ──
        elif action.action_type == "pick_up":
            target = action.target

            # Find package by ID
            pkg = next(
                (p for p in self._state.undelivered_packages
                 if p.package_id == target),
                None
            )

            if pkg is None:
                reward_value = -2.0
                reward_reason = f"Package {target} not found"

            elif any(p.package_id == target
                     for p in self._state.carrying_packages):
                reward_value = -1.0
                reward_reason = f"Already carrying {target}"

            elif len(self._state.carrying_packages) >= \
                    self._state.max_carry_capacity:
                reward_value = -2.0
                reward_reason = "Vehicle at full capacity"

            else:
                # Must be at the warehouse for this package
                wh = next(
                    (w for w in self.task.warehouses
                     if w.name == pkg.warehouse),
                    None
                )
                if wh and self._state.current_location != wh.location:
                    reward_value = -3.0
                    reward_reason = (
                        f"Must be at {wh.location} to pick up {target}. "
                        f"You are at {self._state.current_location}"
                    )
                else:
                    self._state.carrying_packages.append(pkg)
                    self._state.undelivered_packages.remove(pkg)
                    reward_value = 1.0
                    reward_reason = (
                        f"Picked up {target} "
                        f"(deliver to {pkg.destination})"
                    )

        # ── ACTION: deliver ──
        elif action.action_type == "deliver":
            target = action.target

            pkg = next(
                (p for p in self._state.carrying_packages
                 if p.package_id == target),
                None
            )

            if pkg is None:
                reward_value = -2.0
                reward_reason = f"Not carrying package {target}"

            elif self._state.current_location != pkg.destination:
                reward_value = -3.0
                reward_reason = (
                    f"Wrong location. {target} goes to "
                    f"{pkg.destination}, you are at "
                    f"{self._state.current_location}"
                )

            else:
                self._state.carrying_packages.remove(pkg)
                self._state.delivered_packages.append(pkg)

                time_bonus = max(
                    0,
                    (self.task.time_limit_min
                     - self._state.time_elapsed_min)
                    / self.task.time_limit_min,
                )
                reward_value = 10.0 + (5.0 * time_bonus)
                reward_reason = (
                    f"✅ Delivered {target} to {pkg.destination}! "
                    f"Time bonus: {round(time_bonus, 2)}"
                )

        else:
            reward_value = -2.0
            reward_reason = f"Unknown action: {action.action_type}"

        # ── Update new_order_available flag ──
        self._state.new_order_available = new_order_added

        # ── Termination checks ──
        all_done = (
            not self._state.undelivered_packages
            and not self._state.carrying_packages
        )
        if all_done:
            self._done = True
            reward_value += 20.0
            reward_reason += " | 🎉 ALL DELIVERIES COMPLETE!"

        if self._state.time_elapsed_min >= self.task.time_limit_min:
            self._done = True
            reward_reason += " | ⏰ TIME LIMIT REACHED"

        if self._steps >= self.task.max_steps:
            self._done = True
            reward_reason += " | 🔚 MAX STEPS REACHED"

        return StepResponse(
            observation=self._state,
            reward=Reward(
                value=round(reward_value, 3),
                reason=reward_reason,
            ),
            done=self._done,
            info={
                "steps": self._steps,
                "max_steps": self.task.max_steps,
                "deliveries_done": len(self._state.delivered_packages),
                "total_deliveries": self._total_deliveries_including_dynamic,
                "signal_info": signal_info,
                "new_order_added": new_order_added,
                "total_signal_delay_min": round(
                    self._total_signal_delay, 2
                ),
            },
        )

    # ─────────────────────────────────────
    # grade()
    # ─────────────────────────────────────
    def grade(self) -> GraderResult:
        if self._state is None:
            raise RuntimeError("Call reset() first.")

        total = self._total_deliveries_including_dynamic
        delivered = len(self._state.delivered_packages)
        delivery_score = delivered / total if total > 0 else 0.0

        fuel_used = (
            self.task.vehicle.fuel_capacity_L
            - self._state.fuel_remaining_L
        )
        fuel_efficiency = max(
            0,
            1.0 - (fuel_used / self.task.vehicle.fuel_capacity_L)
        )
        time_efficiency = max(
            0,
            1.0 - (
                self._state.time_elapsed_min / self.task.time_limit_min
            ),
        )

        score = (
            delivery_score * 0.70
            + fuel_efficiency * 0.15
            + time_efficiency * 0.15
        )

        return GraderResult(
            task_id=self.task.task_id,
            score=round(min(score, 1.0), 4),
            deliveries_completed=delivered,
            total_deliveries=total,
            distance_travelled_km=round(
                self._state.distance_travelled_km, 2
            ),
            time_elapsed_min=round(self._state.time_elapsed_min, 2),
            fuel_used_L=round(fuel_used, 4),
            total_signal_delays_min=round(self._total_signal_delay, 2),
            reason=(
                f"{delivered}/{total} deliveries | "
                f"fuel efficiency: {round(fuel_efficiency, 2)} | "
                f"time efficiency: {round(time_efficiency, 2)} | "
                f"signal delays: {round(self._total_signal_delay, 2)}min"
            ),
        )