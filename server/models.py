from pydantic import BaseModel
from typing import List, Dict, Optional

# ─────────────────────────────────────────
# Vehicle types
# ─────────────────────────────────────────
class Vehicle(BaseModel):
    name: str                  # "Scooter" or "Bike"
    fuel_capacity_L: float     # 5.0 or 10.0
    mileage_km_per_L: float    # 45.0 or 70.0

    @property
    def max_range_km(self) -> float:
        return self.fuel_capacity_L * self.mileage_km_per_L


# ─────────────────────────────────────────
# Package
# ─────────────────────────────────────────
class Package(BaseModel):
    package_id: str            # unique ID e.g. "PKG_001"
    destination: str           # location name
    warehouse: str             # which warehouse to pick up from
    is_new_order: bool = False # True if appeared mid-episode


# ─────────────────────────────────────────
# Warehouse
# ─────────────────────────────────────────
class Warehouse(BaseModel):
    name: str                  # "NESCO_Warehouse"
    location: str              # same as location name on map
    zone: str                  # "north", "east", "west"


# ─────────────────────────────────────────
# Signal Info (returned in step info)
# ─────────────────────────────────────────
class SignalInfo(BaseModel):
    total_signals: int
    red_signals: int
    signal_delay_min: float
    is_peak_hour: bool


# ─────────────────────────────────────────
# Observation — what agent sees
# ─────────────────────────────────────────
class Observation(BaseModel):
    # Location
    current_location: str

    # Packages
    undelivered_packages: List[Package]
    delivered_packages: List[Package]
    carrying_packages: List[Package]
    max_carry_capacity: int

    # Vehicle
    fuel_remaining_L: float
    mileage_km_per_L: float
    distance_travelled_km: float
    time_elapsed_min: float

    # Warehouses (visible to agent)
    warehouses: List[Warehouse]

    # Time context
    is_peak_hour: bool
    time_of_day: str           # "morning_peak", "non_peak", "evening_peak"

    # Episode context
    new_order_available: bool  # hint that new order just appeared


# ─────────────────────────────────────────
# Action
# ─────────────────────────────────────────
class Action(BaseModel):
    action_type: str
    # action_type options:
    # "move"      → target = location name
    # "pick_up"   → target = package_id
    # "deliver"   → target = package_id
    target: str


# ─────────────────────────────────────────
# Reward
# ─────────────────────────────────────────
class Reward(BaseModel):
    value: float
    reason: str


# ─────────────────────────────────────────
# Step Response
# ─────────────────────────────────────────
class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict


# ─────────────────────────────────────────
# Task Info
# ─────────────────────────────────────────
class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: str
    description: str
    locations: List[str]
    warehouses: List[Warehouse]
    initial_packages: int
    max_steps: int
    max_carry_capacity: int
    vehicle: Vehicle
    time_limit_min: float
    signals_enabled: bool
    peak_hours_enabled: bool
    dynamic_orders: bool       # new orders mid-episode?
    new_order_probability: float  # chance per step


# ─────────────────────────────────────────
# Grader Result
# ─────────────────────────────────────────
class GraderResult(BaseModel):
    task_id: str
    score: float               # 0.0 to 1.0
    deliveries_completed: int
    total_deliveries: int
    distance_travelled_km: float
    time_elapsed_min: float
    fuel_used_L: float
    total_signal_delays_min: float
    reason: str