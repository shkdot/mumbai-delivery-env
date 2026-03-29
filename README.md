# 🚚 Mumbai Delivery RL Environment

---
title: Mumbai Delivery RL Environment
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

A real-world delivery optimization environment built for the OpenEnv Hackathon.
An RL agent must pick up and deliver packages across real Mumbai locations using
actual road distances and durations fetched from OSRM (Open Source Routing Machine).

---

## 🗺️ Environment Overview

The environment simulates a delivery agent operating in the **Andheri-Malad-Powai
zone of Mumbai** (15km radius). Road distances and travel times are based on
**real Mumbai road network data** — not simulated straight-line distances.

### Locations Covered
- Andheri West/East, Jogeshwari, Versova, Lokhandwala
- Goregaon East, Bangur Nagar, Malad (Inorbit Mall)
- Powai Lake, IIT Bombay, JVLR Junction
- Mahakali Caves, Reliance Digital, Decathlon, NESCO Warehouse

---

## 🎯 Tasks

| Task | Difficulty | Packages | Locations | Time Limit | Fuel |
|------|------------|----------|-----------|------------|------|
| easy | Easy | 2 | 5 | 60 min | 10L |
| medium | Medium | 4 | 8 | 120 min | 20L |
| hard | Hard | 6 | 15 | 180 min | 30L |

---

## 🔁 Action Space

| Action | Description |
|--------|-------------|
| `move` | Travel to a target location (costs fuel + time) |
| `pick_up` | Pick up a package at current location |
| `deliver` | Deliver a carried package at its destination |
```json
{"action_type": "move", "target": "Andheri_Station"}
{"action_type": "pick_up", "target": "Mahakali_Caves"}
{"action_type": "deliver", "target": "Powai_Lake"}
```

---

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `current_location` | string | Agent's current location |
| `undelivered_packages` | list[string] | Pending delivery destinations |
| `delivered_packages` | list[string] | Completed deliveries |
| `distance_travelled_km` | float | Total km covered |
| `time_elapsed_min` | float | Total time elapsed |
| `fuel_remaining_L` | float | Remaining fuel |
| `carrying_packages` | list[string] | Packages on vehicle |
| `max_carry_capacity` | int | Max packages vehicle can carry |

---

## 🏆 Reward System

| Event | Reward |
|-------|--------|
| Successful delivery | +10 to +15 (with time bonus) |
| All deliveries complete | +20 bonus |
| Moving (per km) | -0.1 |
| Carrying packages while moving | +0.5 |
| Invalid action | -2 |
| Wrong delivery location | -3 |
| Out of fuel | -10 (episode ends) |

---

## 📊 Grading

Scores range from **0.0 to 1.0**:

| Component | Weight |
|-----------|--------|
| Delivery completion | 70% |
| Fuel efficiency | 15% |
| Time efficiency | 15% |

---

## 🚀 Setup & Running Locally

### Prerequisites
- Python 3.10+
- Git
- Docker (optional)

### Install & Run
```bash
# Clone the repo
git clone https://huggingface.co/spaces/your-username/mumbai-delivery-env
cd mumbai-delivery-env

# Install dependencies
pip install -r server/requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run with Docker
```bash
docker build -t mumbai-delivery-env .
docker run -p 7860:7860 mumbai-delivery-env
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Environment info |
| `/tasks` | GET | List all tasks + action schema |
| `/reset?task_id=easy` | POST | Reset environment |
| `/state?task_id=easy` | GET | Get current state |
| `/step?task_id=easy` | POST | Take an action |
| `/grader?task_id=easy` | GET | Get episode score |
| `/baseline` | GET | Run greedy baseline on all tasks |

---

## 🤖 Running the Baseline Agent
```bash
# Greedy baseline (no API key needed)
python client.py

# LLM agent (requires OpenAI API key)
export OPENAI_API_KEY=your-key-here
python client.py
```

---

## 📁 Project Structure
```
mumbai-delivery-env/
├── server/
│   ├── __init__.py
│   ├── app.py                        # FastAPI server + endpoints
│   ├── models.py                     # Pydantic typed models
│   ├── my_hackathon_env_environment.py  # Core RL environment
│   └── requirements.txt
├── distance_matrix.json              # Real Mumbai road distances (OSRM)
├── client.py                         # Baseline inference script
├── Dockerfile
├── openenv.yaml                      # OpenEnv spec
├── pyproject.toml
└── README.md
```

---

## 🌍 Real-World Data

Road distances are sourced from **OSRM (Open Source Routing Machine)**
using OpenStreetMap data. All 225 location pairs (15×15) have been
pre-fetched and stored in `distance_matrix.json` for reproducible,
offline evaluation.

---

## 📍 Map Coverage

**Center:** Andheri, Mumbai
**Radius:** ~15km
**Boundaries:**
- North: Malad / Inorbit Mall
- South: Santacruz / Andheri East
- West: Versova Beach
- East: Powai Lake / IIT Bombay