import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import Dict

from server.models import Action, StepResponse, GraderResult
from server.my_hackathon_env_environment import (
    DeliveryEnvironment, TASKS
)

# ─────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────
app = FastAPI(
    title="Mumbai Delivery RL Environment",
    description=(
        "A real-world delivery optimization environment for RL agents "
        "based on the Mumbai road network (Andheri-Malad-Powai zone). "
        "Features real road distances, traffic signals, multiple "
        "warehouses and dynamic order generation."
    ),
    version="2.0.0",
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
            detail=(
                f"No active session for '{task_id}'. "
                f"Call POST /reset?task_id={task_id} first."
            )
        )
    return _envs[task_id]


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Mumbai Delivery RL Environment",
        "version": "2.0.0",
        "description": (
            "Real-world delivery RL environment based on "
            "Mumbai road network"
        ),
        "endpoints": [
            "/tasks",
            "/reset",
            "/step",
            "/state",
            "/grader",
            "/baseline",
            "/visualizer",
        ],
    }


# ── /tasks ──────────────────────────────
@app.get("/tasks")
def list_tasks():
    """Returns all tasks with full config and action schema."""
    return {
        "tasks": [task.dict() for task in TASKS.values()],
        "action_schema": {
            "action_type": "move | pick_up | deliver",
            "target": (
                "move → location name | "
                "pick_up → package_id | "
                "deliver → package_id"
            ),
        },
    }


# ── /reset ──────────────────────────────
@app.post("/reset")
def reset(task_id: str = "easy"):
    """Reset environment. Returns initial observation."""
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
    """Returns current observation without taking action."""
    env = _get_env(task_id)
    return {
        "task_id": task_id,
        "observation": env.state().dict()
    }


# ── /step ───────────────────────────────
@app.post("/step")
def step(action: Action, task_id: str = "easy"):
    """Take one action. Returns observation, reward, done, info."""
    env = _get_env(task_id)
    result: StepResponse = env.step(action)
    return result.dict()


# ── /grader ─────────────────────────────
@app.get("/grader")
def grade(task_id: str = "easy"):
    """Returns grader score 0.0 to 1.0 for current episode."""
    env = _get_env(task_id)
    result: GraderResult = env.grade()
    return result.dict()


# ── /baseline ───────────────────────────
@app.get("/baseline")
def baseline():
    """
    Runs greedy baseline agent on all 3 tasks.
    Returns scores for easy, medium, hard.
    """
    results = {}

    for task_id in TASKS.keys():
        env = DeliveryEnvironment(task_id=task_id)
        obs = env.reset()
        done = False

        while not done:
            action = _greedy_action(obs)
            if action is None:
                break
            result = env.step(action)
            obs = result.observation
            done = result.done

        grade_result = env.grade()
        results[task_id] = grade_result.dict()

    return {"baseline_results": results}


# ─────────────────────────────────────────
# Greedy baseline agent logic
# ─────────────────────────────────────────
def _greedy_action(obs):
    from server.my_hackathon_env_environment import DISTANCE_KM

    # 1. Deliver if at destination
    for pkg in obs.carrying_packages:
        if pkg.destination == obs.current_location:
            return Action(
                action_type="deliver",
                target=pkg.package_id
            )

    # 2. Move toward carrying package destination
    if obs.carrying_packages:
        nearest_pkg = min(
            obs.carrying_packages,
            key=lambda p: DISTANCE_KM.get(
                obs.current_location, {}
            ).get(p.destination, 999)
        )
        return Action(
            action_type="move",
            target=nearest_pkg.destination
        )

    # 3. Go to warehouse to pick up
    if obs.undelivered_packages:
        if len(obs.carrying_packages) < obs.max_carry_capacity:
            # Find nearest warehouse that has packages
            next_pkg = obs.undelivered_packages[0]
            wh = next(
                (w for w in obs.warehouses
                 if w.name == next_pkg.warehouse),
                obs.warehouses[0]
            )
            # If already at warehouse, pick up
            if obs.current_location == wh.location:
                return Action(
                    action_type="pick_up",
                    target=next_pkg.package_id
                )
            # Move to warehouse
            return Action(
                action_type="move",
                target=wh.location
            )

    return None


# ── /visualizer ─────────────────────────
@app.get("/visualizer", response_class=HTMLResponse)
def visualizer():
    """Live visual map of the delivery environment."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mumbai Delivery RL - Visualizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
        }
        header {
            background: #1e293b;
            padding: 16px 24px;
            border-bottom: 1px solid #334155;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        header h1 {
            font-size: 1.2rem;
            color: #f1f5f9;
        }
        header span {
            font-size: 0.8rem;
            background: #0ea5e9;
            padding: 2px 10px;
            border-radius: 999px;
            color: white;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 360px;
            gap: 16px;
            padding: 16px;
            height: calc(100vh - 60px);
        }
        .map-panel {
            background: #1e293b;
            border-radius: 12px;
            border: 1px solid #334155;
            position: relative;
            overflow: hidden;
        }
        canvas {
            width: 100%;
            height: 100%;
        }
        .side-panel {
            display: flex;
            flex-direction: column;
            gap: 12px;
            overflow-y: auto;
        }
        .card {
            background: #1e293b;
            border-radius: 12px;
            border: 1px solid #334155;
            padding: 16px;
        }
        .card h3 {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #94a3b8;
            margin-bottom: 12px;
        }
        .controls {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        select, button {
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid #334155;
            background: #0f172a;
            color: #e2e8f0;
            font-size: 0.85rem;
            cursor: pointer;
        }
        button {
            background: #0ea5e9;
            border-color: #0ea5e9;
            color: white;
            font-weight: 600;
        }
        button:hover { background: #0284c7; }
        button.danger {
            background: #ef4444;
            border-color: #ef4444;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .stat {
            background: #0f172a;
            border-radius: 8px;
            padding: 10px;
        }
        .stat-label {
            font-size: 0.7rem;
            color: #64748b;
            margin-bottom: 4px;
        }
        .stat-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: #f1f5f9;
        }
        .pkg-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
            max-height: 200px;
            overflow-y: auto;
        }
        .pkg {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #0f172a;
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 0.8rem;
        }
        .pkg-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .dot-pending { background: #f59e0b; }
        .dot-carrying { background: #0ea5e9; }
        .dot-delivered { background: #22c55e; }
        .log {
            background: #0f172a;
            border-radius: 8px;
            padding: 10px;
            font-size: 0.75rem;
            font-family: monospace;
            max-height: 150px;
            overflow-y: auto;
            color: #94a3b8;
        }
        .log-entry { margin-bottom: 4px; }
        .log-entry.good { color: #22c55e; }
        .log-entry.bad { color: #ef4444; }
        .log-entry.info { color: #0ea5e9; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .badge-peak { background: #ef444420; color: #ef4444; }
        .badge-nonpeak { background: #22c55e20; color: #22c55e; }
        .score-bar {
            height: 8px;
            background: #0f172a;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 6px;
        }
        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, #0ea5e9, #22c55e);
            border-radius: 999px;
            transition: width 0.3s;
        }
    </style>
</head>
<body>
<header>
    <h1>🚚 Mumbai Delivery RL Environment</h1>
    <span>v2.0.0</span>
</header>
<div class="container">
    <div class="map-panel">
        <canvas id="mapCanvas"></canvas>
    </div>
    <div class="side-panel">
        <!-- Controls -->
        <div class="card">
            <h3>Controls</h3>
            <div class="controls">
                <select id="taskSelect">
                    <option value="easy">Easy</option>
                    <option value="medium">Medium</option>
                    <option value="hard">Hard</option>
                </select>
                <button onclick="resetEnv()">Reset</button>
                <button onclick="stepGreedy()">Step</button>
                <button onclick="runAuto()">▶ Auto</button>
                <button class="danger" onclick="stopAuto()">■ Stop</button>
            </div>
        </div>

        <!-- Stats -->
        <div class="card">
            <h3>Status</h3>
            <div class="stat-grid">
                <div class="stat">
                    <div class="stat-label">Location</div>
                    <div class="stat-value" id="statLocation">—</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Time</div>
                    <div class="stat-value" id="statTime">0 min</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Fuel</div>
                    <div class="stat-value" id="statFuel">—</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Distance</div>
                    <div class="stat-value" id="statDist">0 km</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Deliveries</div>
                    <div class="stat-value" id="statDeliveries">0/0</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Traffic</div>
                    <div class="stat-value" id="statTraffic">—</div>
                </div>
            </div>
            <div style="margin-top:10px">
                <div class="stat-label">Score</div>
                <div class="stat-value" id="statScore">0.0000</div>
                <div class="score-bar">
                    <div class="score-fill" id="scoreBar" style="width:0%"></div>
                </div>
            </div>
        </div>

        <!-- Packages -->
        <div class="card">
            <h3>Packages</h3>
            <div class="pkg-list" id="pkgList"></div>
        </div>

        <!-- Log -->
        <div class="card">
            <h3>Action Log</h3>
            <div class="log" id="actionLog"></div>
        </div>
    </div>
</div>

<script>
// ── Location coordinates (normalized to canvas) ──
const LOCATIONS = {
    "NESCO_Warehouse":      { x: 0.48, y: 0.28, label: "NESCO" },
    "Inorbit_Mall_Malad":   { x: 0.28, y: 0.10, label: "Inorbit" },
    "Lokhandwala_Market":   { x: 0.22, y: 0.42, label: "Lokhandwala" },
    "Versova_Beach":        { x: 0.08, y: 0.44, label: "Versova" },
    "Andheri_Station":      { x: 0.36, y: 0.56, label: "Andheri Stn" },
    "Jogeshwari_West":      { x: 0.34, y: 0.58, label: "Jogeshwari" },
    "Reliance_Digital":     { x: 0.40, y: 0.50, label: "Reliance" },
    "Andheri_East":         { x: 0.52, y: 0.64, label: "Andheri East" },
    "Mahakali_Caves":       { x: 0.56, y: 0.62, label: "Mahakali" },
    "JVLR_Junction":        { x: 0.60, y: 0.70, label: "JVLR" },
    "Powai_Lake":           { x: 0.78, y: 0.62, label: "Powai" },
    "IIT_Bombay":           { x: 0.84, y: 0.46, label: "IIT" },
    "Goregaon_East":        { x: 0.52, y: 0.18, label: "Goregaon E" },
    "Bangur_Nagar":         { x: 0.28, y: 0.14, label: "Bangur" },
    "Decathlon_Andheri":    { x: 0.20, y: 0.58, label: "Decathlon" },
};

const WAREHOUSES = [
    "NESCO_Warehouse",
    "Mahakali_Caves",
    "Lokhandwala_Market"
];

const BASE = "http://localhost:7860";
let state = null;
let autoInterval = null;
let canvas, ctx;
let greedyState = null;

window.onload = () => {
    canvas = document.getElementById("mapCanvas");
    ctx = canvas.getContext("2d");
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
    drawMap(null);
};

function resizeCanvas() {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    drawMap(state);
}

// ── Draw map ──
function drawMap(obs) {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, W, H);

    // Draw connections (roads)
    ctx.strokeStyle = "#1e3a5f";
    ctx.lineWidth = 1.5;
    const locs = Object.keys(LOCATIONS);
    for (let i = 0; i < locs.length; i++) {
        for (let j = i + 1; j < locs.length; j++) {
            const a = LOCATIONS[locs[i]];
            const b = LOCATIONS[locs[j]];
            const dx = Math.abs(a.x - b.x);
            const dy = Math.abs(a.y - b.y);
            if (dx < 0.25 && dy < 0.25) {
                ctx.beginPath();
                ctx.moveTo(a.x * W, a.y * H);
                ctx.lineTo(b.x * W, b.y * H);
                ctx.stroke();
            }
        }
    }

    // Draw nodes
    for (const [name, pos] of Object.entries(LOCATIONS)) {
        const x = pos.x * W;
        const y = pos.y * H;
        const isWarehouse = WAREHOUSES.includes(name);
        const isCurrent = obs && obs.current_location === name;
        const isDelivered = obs && obs.delivered_packages &&
            obs.delivered_packages.some(p => p.destination === name);
        const isTarget = obs && obs.carrying_packages &&
            obs.carrying_packages.some(p => p.destination === name);
        const isPending = obs && obs.undelivered_packages &&
            obs.undelivered_packages.some(p => p.destination === name);

        // Node circle
        ctx.beginPath();
        ctx.arc(x, y, isWarehouse ? 10 : 7, 0, Math.PI * 2);

        if (isCurrent) {
            ctx.fillStyle = "#f59e0b";
        } else if (isWarehouse) {
            ctx.fillStyle = "#7c3aed";
        } else if (isDelivered) {
            ctx.fillStyle = "#22c55e";
        } else if (isTarget) {
            ctx.fillStyle = "#0ea5e9";
        } else if (isPending) {
            ctx.fillStyle = "#f97316";
        } else {
            ctx.fillStyle = "#334155";
        }
        ctx.fill();

        // Warehouse ring
        if (isWarehouse) {
            ctx.beginPath();
            ctx.arc(x, y, 14, 0, Math.PI * 2);
            ctx.strokeStyle = "#7c3aed";
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Current location pulse ring
        if (isCurrent) {
            ctx.beginPath();
            ctx.arc(x, y, 16, 0, Math.PI * 2);
            ctx.strokeStyle = "#f59e0b88";
            ctx.lineWidth = 3;
            ctx.stroke();
        }

        // Label
        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px Segoe UI";
        ctx.textAlign = "center";
        ctx.fillText(pos.label, x, y + 22);
    }

    // Legend
    const legend = [
        { color: "#f59e0b", label: "Agent" },
        { color: "#7c3aed", label: "Warehouse" },
        { color: "#f97316", label: "Pending" },
        { color: "#0ea5e9", label: "Carrying" },
        { color: "#22c55e", label: "Delivered" },
    ];
    legend.forEach((item, i) => {
        ctx.beginPath();
        ctx.arc(16, H - 90 + i * 18, 5, 0, Math.PI * 2);
        ctx.fillStyle = item.color;
        ctx.fill();
        ctx.fillStyle = "#94a3b8";
        ctx.font = "11px Segoe UI";
        ctx.textAlign = "left";
        ctx.fillText(item.label, 26, H - 86 + i * 18);
    });
}

// ── Update UI ──
function updateUI(obs, reward, done, info) {
    state = obs;
    drawMap(obs);

    // Stats
    document.getElementById("statLocation").textContent =
        obs.current_location.replace(/_/g, " ").substring(0, 12);
    document.getElementById("statTime").textContent =
        `${obs.time_elapsed_min.toFixed(1)} min`;
    document.getElementById("statFuel").textContent =
        `${obs.fuel_remaining_L.toFixed(2)}L`;
    document.getElementById("statDist").textContent =
        `${obs.distance_travelled_km.toFixed(1)} km`;
    document.getElementById("statDeliveries").textContent =
        `${obs.delivered_packages.length}/${
            obs.delivered_packages.length +
            obs.undelivered_packages.length +
            obs.carrying_packages.length
        }`;

    const tod = obs.time_of_day || "non_peak";
    const trafficEl = document.getElementById("statTraffic");
    if (tod.includes("peak")) {
        trafficEl.innerHTML =
            '<span class="badge badge-peak">🔴 Peak</span>';
    } else {
        trafficEl.innerHTML =
            '<span class="badge badge-nonpeak">🟢 Normal</span>';
    }

    // Packages
    const pkgList = document.getElementById("pkgList");
    pkgList.innerHTML = "";
    obs.carrying_packages.forEach(p => {
        pkgList.innerHTML += `
            <div class="pkg">
                <div class="pkg-dot dot-carrying"></div>
                <span>${p.package_id} → ${p.destination.replace(/_/g,' ')}</span>
                ${p.is_new_order ?
                    '<span class="badge badge-peak">NEW</span>' : ""}
            </div>`;
    });
    obs.undelivered_packages.forEach(p => {
        pkgList.innerHTML += `
            <div class="pkg">
                <div class="pkg-dot dot-pending"></div>
                <span>${p.package_id} → ${p.destination.replace(/_/g,' ')}</span>
                ${p.is_new_order ?
                    '<span class="badge badge-peak">NEW</span>' : ""}
            </div>`;
    });
    obs.delivered_packages.forEach(p => {
        pkgList.innerHTML += `
            <div class="pkg">
                <div class="pkg-dot dot-delivered"></div>
                <span>✅ ${p.package_id} → ${
                    p.destination.replace(/_/g,' ')}</span>
            </div>`;
    });

    // Score
    if (info && info.score !== undefined) {
        const s = info.score;
        document.getElementById("statScore").textContent =
            s.toFixed(4);
        document.getElementById("scoreBar").style.width =
            `${s * 100}%`;
    }

    // Log
    if (reward) {
        const log = document.getElementById("actionLog");
        const cls = reward.value > 0 ? "good" :
                    reward.value < -1 ? "bad" : "info";
        const entry = document.createElement("div");
        entry.className = `log-entry ${cls}`;
        entry.textContent =
            `[${reward.value > 0 ? "+" : ""}${
                reward.value.toFixed(2)}] ${reward.reason}`;
        log.insertBefore(entry, log.firstChild);
        if (done) {
            const doneEntry = document.createElement("div");
            doneEntry.className = "log-entry info";
            doneEntry.textContent = "── Episode Done ──";
            log.insertBefore(doneEntry, log.firstChild);
        }
    }
}

// ── API calls ──
async function resetEnv() {
    const task = document.getElementById("taskSelect").value;
    greedyState = null;
    const r = await fetch(
        `${BASE}/reset?task_id=${task}`, { method: "POST" }
    );
    const data = await r.json();
    updateUI(data.observation, null, false, {});
    addLog("Environment reset", "info");

    // Get initial score
    const gr = await fetch(`${BASE}/grader?task_id=${task}`);
    const gd = await gr.json();
    document.getElementById("statScore").textContent =
        gd.score.toFixed(4);
    document.getElementById("scoreBar").style.width =
        `${gd.score * 100}%`;
}

async function stepGreedy() {
    const task = document.getElementById("taskSelect").value;
    if (!state) { await resetEnv(); return; }

    const action = computeGreedy(state);
    if (!action) { addLog("No action available", "info"); return; }

    const r = await fetch(
        `${BASE}/step?task_id=${task}`,
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(action),
        }
    );
    const data = await r.json();

    // Get score
    const gr = await fetch(`${BASE}/grader?task_id=${task}`);
    const gd = await gr.json();

    updateUI(data.observation, data.reward, data.done, gd);

    if (data.done) stopAuto();
}

function computeGreedy(obs) {
    // Deliver if at destination
    for (const pkg of obs.carrying_packages) {
        if (pkg.destination === obs.current_location) {
            return { action_type: "deliver", target: pkg.package_id };
        }
    }
    // Move toward carrying destination
    if (obs.carrying_packages.length > 0) {
        return {
            action_type: "move",
            target: obs.carrying_packages[0].destination
        };
    }
    // Go to warehouse to pick up
    if (obs.undelivered_packages.length > 0) {
        const pkg = obs.undelivered_packages[0];
        const wh = obs.warehouses.find(w => w.name === pkg.warehouse);
        const whLoc = wh ? wh.location : obs.warehouses[0].location;
        if (obs.current_location === whLoc) {
            return { action_type: "pick_up", target: pkg.package_id };
        }
        return { action_type: "move", target: whLoc };
    }
    return null;
}

function runAuto() {
    if (autoInterval) return;
    autoInterval = setInterval(stepGreedy, 600);
}

function stopAuto() {
    clearInterval(autoInterval);
    autoInterval = null;
}

function addLog(msg, cls) {
    const log = document.getElementById("actionLog");
    const entry = document.createElement("div");
    entry.className = `log-entry ${cls}`;
    entry.textContent = msg;
    log.insertBefore(entry, log.firstChild);
}
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)